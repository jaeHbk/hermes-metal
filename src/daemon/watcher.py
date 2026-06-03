"""hermes-metal file-watcher daemon.

Watches an Obsidian vault for Markdown changes and keeps the LanceDB vector
index in sync by re-embedding modified files via a local llama-server running
in --embedding mode (default http://127.0.0.1:8081/embedding).

Notes on divergence from CLAUDE.md:
    * CLAUDE.md §3 describes the embedding step abstractly ("ARM-native
      nomic-embed-text"). Per the verified research, the actual transport is a
      second llama-server process exposing both the native /embedding endpoint
      and an OpenAI-compatible /v1/embeddings endpoint. This watcher only
      forwards a callable URL into ``indexer.index_file``; the embedding
      payload contract lives in ``src/backend/indexer.py``.
    * The default port for the embedding server is 8081 (chat is on 8080),
      matching the dual-server topology validated during research.

This module is the launchd entry-point for ``com.hermes.metal.watcher`` and
must shut down cleanly on SIGTERM so the agent's KeepAlive policy treats
``launchctl unload`` as an intentional stop, not a crash.
"""
from __future__ import annotations

import hashlib
import logging
import logging.handlers
import os
import signal
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileMovedEvent,
    PatternMatchingEventHandler,
)
from watchdog.observers import Observer

from src.backend.vault_filter import VaultFilter, build_filter

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from src.backend.database import LanceVault


logger = logging.getLogger("hermes.daemon.watcher")

# Markdown patterns we care about. Obsidian writes both .md and (rarely) .markdown.
_MARKDOWN_PATTERNS: list[str] = ["*.md", "*.markdown"]

# Default embedding endpoint. Matches the second llama-server instance launched
# with `--embedding --port 8081`. We use the OpenAI-compatible route because
# indexer.embed() posts the OpenAI request shape and parses body["data"];
# the native /embedding endpoint returns a different schema.
_DEFAULT_EMBED_URL = "http://127.0.0.1:8081/v1/embeddings"


class VaultWatcher(PatternMatchingEventHandler):
    """Per-path debounced filesystem handler for an Obsidian vault.

    Filesystem editors (Obsidian, vim, VS Code) frequently emit a burst of
    create/modify/rename events for a single logical save. We coalesce those
    bursts per path with a small ``threading.Timer`` so the indexer runs at
    most once per ``debounce_s`` window per file.

    Parameters
    ----------
    vault_path:
        Root directory of the Obsidian vault to watch (recursively).
    vault:
        Initialized ``LanceVault`` (see ``src/backend/database.py``) used for
        ``delete_by_source`` on remove/move events.
    embed_url:
        URL of the local llama-server embedding endpoint, forwarded into
        ``indexer.index_file``.
    debounce_s:
        Per-path coalescing window in seconds. 5s coalesces Obsidian's
        autosave-while-typing burst (events every ~2s) into a single re-index
        without making the index feel stale on save-then-ask workflows.
    """

    def __init__(
        self,
        vault_path: Path,
        vault: "LanceVault",
        embed_url: str,
        debounce_s: float = 5.0,
        vfilter: VaultFilter | None = None,
    ) -> None:
        # Derive watchdog patterns from the filter's include list so a user
        # who configures HERMES_VAULT_INCLUDE="*.md:*.txt" gets .txt events
        # delivered. Otherwise watchdog filters .txt at the OS-event layer and
        # the live index would only catch them via `hermes index --backfill`.
        # We DO NOT pre-filter slashed/non-slashless globs out: PatternMatching-
        # EventHandler accepts only basename-style globs, but our filter's
        # non-slashed patterns are exactly that, so the include list maps cleanly.
        resolved_filter = vfilter or build_filter(Path(vault_path).expanduser().resolve())
        watch_patterns = [p for p in resolved_filter.include if "/" not in p] or _MARKDOWN_PATTERNS
        super().__init__(
            patterns=watch_patterns,
            ignore_directories=True,
            case_sensitive=False,
        )
        self.vault_path = Path(vault_path).expanduser().resolve()
        self.vault = vault
        self.embed_url = embed_url
        self.debounce_s = float(debounce_s)
        # The pattern matcher above filters at the watchdog layer; the
        # VaultFilter additionally enforces path excludes (.obsidian, .trash,
        # attachments/, etc.) inside _schedule_index.
        self.vfilter = resolved_filter

        # Per-path debounce state. Guarded by ``_timers_lock`` because watchdog
        # dispatches events on its observer thread while timers fire on their
        # own ``threading.Timer`` threads.
        self._timers: dict[str, threading.Timer] = {}
        self._timers_lock = threading.Lock()

        # Per-path sha256 of the last successfully indexed content. Obsidian
        # touches mtime even on no-op saves; hashing lets us skip the embed
        # round-trip when bytes are unchanged. Lock-free: only ``_fire_index``
        # writes, and only on its own timer thread per path.
        self._hashes: dict[str, str] = {}

        # Lazy import: the indexer pulls in numpy/httpx, and we want
        # ``import watcher`` to be cheap (e.g. for unit tests that monkey-patch).
        from src.backend import indexer  # noqa: WPS433 (intentional local import)

        self._index_file = indexer.index_file

    # ------------------------------------------------------------------ utils

    def _schedule_index(self, path: str) -> None:
        """(Re)arm the per-path debounce timer for ``path``."""
        if not self.vfilter.accepts(path):
            logger.debug("filter rejected, skipping: %s", path)
            return
        with self._timers_lock:
            existing = self._timers.pop(path, None)
            if existing is not None:
                existing.cancel()
            timer = threading.Timer(self.debounce_s, self._fire_index, args=(path,))
            timer.daemon = True
            self._timers[path] = timer
            timer.start()

    def _fire_index(self, path: str) -> None:
        """Timer callback: actually re-index ``path``."""
        with self._timers_lock:
            # Drop our slot so a subsequent edit re-arms a fresh timer.
            self._timers.pop(path, None)

        try:
            target = Path(path)
            if not target.exists():
                # File was deleted between debounce arm and fire. Treat as a
                # delete to keep the index consistent.
                logger.info("debounced index target missing, deleting: %s", path)
                self.vault.delete_by_source(path)
                self._hashes.pop(path, None)
                return

            try:
                content = target.read_bytes()
            except OSError:
                logger.exception("failed to read %s for hashing", path)
                return
            digest = hashlib.sha256(content).hexdigest()
            if self._hashes.get(path) == digest:
                logger.debug("unchanged content, skipping re-index: %s", path)
                return

            logger.info("indexing %s", path)
            self._index_file(
                file_path=target,
                vault=self.vault,
                embed_url=self.embed_url,
            )
            self._hashes[path] = digest
        except Exception:  # noqa: BLE001 - we MUST never kill the timer thread
            logger.exception("failed to index %s", path)

    def _cancel_all_timers(self) -> None:
        with self._timers_lock:
            timers = list(self._timers.values())
            self._timers.clear()
        for timer in timers:
            timer.cancel()

    # ------------------------------------------------------------ watchdog API

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        logger.debug("created %s", event.src_path)
        self._schedule_index(event.src_path)

    def on_modified(self, event: FileModifiedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        logger.debug("modified %s", event.src_path)
        self._schedule_index(event.src_path)

    def on_deleted(self, event: FileDeletedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        logger.info("deleted %s", event.src_path)
        # Cancel any pending re-index for this path before purging the index.
        with self._timers_lock:
            pending = self._timers.pop(event.src_path, None)
        if pending is not None:
            pending.cancel()
        try:
            self.vault.delete_by_source(event.src_path)
        except Exception:  # noqa: BLE001
            logger.exception("failed to delete index rows for %s", event.src_path)
        self._hashes.pop(event.src_path, None)

    def on_moved(self, event: FileMovedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        src = event.src_path
        dst = event.dest_path
        logger.info("moved %s -> %s", src, dst)

        # Cancel any pending re-index keyed on the old path.
        with self._timers_lock:
            pending = self._timers.pop(src, None)
        if pending is not None:
            pending.cancel()

        try:
            self.vault.delete_by_source(src)
        except Exception:  # noqa: BLE001
            logger.exception("failed to delete index rows for %s", src)
        self._hashes.pop(src, None)

        # Only re-index destinations the user-configured filter accepts.
        # watchdog's PatternMatchingEventHandler filters src_path; on rename
        # the destination extension may differ (e.g. .md -> .tmp during a
        # save), so consult the filter directly. _schedule_index also rechecks,
        # but checking here avoids an empty-debounce noise log.
        if self.vfilter.accepts(dst):
            self._schedule_index(dst)


# ---------------------------------------------------------------- entry point


def _configure_logging(log_path: Path) -> None:
    """Wire RotatingFileHandler -> ``log_path`` plus a stderr handler.

    launchd captures stderr into ``StandardErrorPath``, so duplicating the
    stream there keeps the plist's log redirection useful while still giving
    us a self-rotating file for interactive ``tail -f``.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,  # 5 MiB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Replace any default handlers to avoid double-logging when launchd
    # restarts us into the same process tree.
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(file_handler)
    root.addHandler(stderr_handler)


def _resolve_working_dir() -> Path:
    """Locate the project root (where ``logs/`` and ``storage/`` live).

    launchd sets ``WorkingDirectory`` from the plist, but when the watcher is
    invoked manually (``python -m src.daemon.watcher``) we fall back to
    walking up from this file.
    """
    cwd = Path.cwd()
    if (cwd / "CLAUDE.md").exists():
        return cwd
    here = Path(__file__).resolve()
    # src/daemon/watcher.py -> project root is two parents up from src/.
    return here.parents[2]


def main() -> int:
    working_dir = _resolve_working_dir()
    log_path = working_dir / "logs" / "watcher.log"
    _configure_logging(log_path)

    vault_path_str = os.environ.get("HERMES_VAULT_PATH")
    if not vault_path_str:
        logger.error("HERMES_VAULT_PATH is not set; refusing to start")
        return 2
    vault_path = Path(vault_path_str).expanduser().resolve()
    if not vault_path.is_dir():
        logger.error("HERMES_VAULT_PATH does not exist or is not a directory: %s", vault_path)
        return 2

    lancedb_path = Path(
        os.environ.get(
            "HERMES_LANCEDB_PATH",
            str(working_dir / "storage" / "lancedb"),
        )
    ).expanduser().resolve()
    lancedb_path.mkdir(parents=True, exist_ok=True)

    embed_url = os.environ.get("HERMES_EMBED_URL", _DEFAULT_EMBED_URL)

    # Late import so logging is configured before the backend touches LanceDB
    # (which can emit its own log records on first open).
    from src.backend.database import LanceVault

    logger.info(
        "starting watcher: vault=%s lancedb=%s embed_url=%s",
        vault_path,
        lancedb_path,
        embed_url,
    )

    vault = LanceVault(path=lancedb_path)
    handler = VaultWatcher(
        vault_path=vault_path,
        vault=vault,
        embed_url=embed_url,
    )

    observer = Observer()
    observer.schedule(handler, str(vault_path), recursive=True)

    stop_event = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        logger.info("received signal %d, shutting down", signum)
        stop_event.set()

    # SIGTERM is what launchd sends on `launchctl unload` / stop.
    # SIGINT is for interactive ctrl-C.
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    observer.start()
    try:
        # Block on the stop event rather than observer.join() so signal
        # handlers run promptly. observer.join() on its own swallows signals
        # on some macOS Python builds.
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    finally:
        logger.info("stopping observer and cancelling pending timers")
        try:
            observer.stop()
        except Exception:  # noqa: BLE001
            logger.exception("observer.stop() raised")
        handler._cancel_all_timers()  # noqa: SLF001 (intentional teardown hook)
        try:
            observer.join(timeout=10.0)
        except Exception:  # noqa: BLE001
            logger.exception("observer.join() raised")
        # Best-effort vault close so LanceDB flushes any in-flight writes.
        close = getattr(vault, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                logger.exception("vault.close() raised")
        logger.info("watcher stopped cleanly")

    return 0


if __name__ == "__main__":
    sys.exit(main())
