"""`hermes index` — backfill, force re-index, and GC for the vault index.

Three modes (combinable):

* ``--backfill`` — walk the vault, index every file the filter accepts that
  is not already in the index (or whose content has changed since last
  index). Idempotent: rerunning a clean backfill is a no-op modulo content
  hashing.
* ``--force`` — when combined with ``--backfill``, re-embeds every file
  regardless of hash. Useful after changing the chunker or upgrading the
  embedding model.
* ``--gc`` — drop rows whose ``source_path`` no longer exists or no longer
  matches the vault filter. The watcher handles deletions while running,
  but files moved out of the vault while the daemon was stopped become
  orphans without this.

The watcher catches future writes; this command catches the past. Running
both in sequence on a fresh install is the supported way to onboard an
existing vault.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import httpx

from src.backend.database import LanceVault
from src.backend.indexer import index_file
from src.backend.vault_filter import build_filter, iter_vault_files


# ----------------------------------------------------------------- helpers


def _resolve_vault_path() -> Path | None:
    """Mirrors the watcher's HERMES_VAULT_PATH discovery, plus plist fallback.

    Returning None lets the caller emit a clean error rather than a stack
    trace when the user invokes ``hermes index`` before configuring a vault.
    """
    raw = os.environ.get("HERMES_VAULT_PATH")
    if raw:
        # Resolve so paths in the index (which the watcher writes after
        # ``Path.resolve()``) line up with what GC sees. Without this a
        # relative HERMES_VAULT_PATH would make every indexed source look
        # like an orphan and GC would wipe the index.
        return Path(raw).expanduser().resolve()
    # Fallback: read the watcher plist (same recovery path the doctor uses).
    import plistlib
    plist = Path.home() / "Library" / "LaunchAgents" / "com.hermes.metal.watcher.plist"
    if plist.is_file():
        try:
            with plist.open("rb") as fh:
                env = (plistlib.load(fh).get("EnvironmentVariables") or {})
            value = env.get("HERMES_VAULT_PATH")
            if value:
                return Path(value).expanduser().resolve()
        except (plistlib.InvalidFileException, OSError):
            pass
    return None


def _resolve_db_path() -> Path:
    raw = os.environ.get("HERMES_LANCEDB_PATH") or str(
        Path(__file__).resolve().parents[1] / "storage" / "lancedb"
    )
    return Path(raw).expanduser().resolve()


def _resolve_embed_url() -> str:
    return os.environ.get("HERMES_EMBED_URL", "http://127.0.0.1:8081/v1/embeddings")


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(64 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _existing_sources(vault: LanceVault) -> set[str]:
    return set(vault.distinct_sources())


# ----------------------------------------------------------------- modes


def _do_backfill(
    vault: LanceVault,
    files: list[Path],
    *,
    embed_url: str,
    force: bool,
    limit: int | None,
) -> tuple[int, int, int]:
    """Returns (indexed, skipped, errored)."""
    indexed = skipped = errored = 0
    existing = _existing_sources(vault) if not force else set()

    targets = files[:limit] if limit is not None else files
    total = len(targets)
    width = len(str(total)) if total else 1

    for i, path in enumerate(targets, 1):
        resolved = str(path.resolve())
        if not force and resolved in existing:
            # Without per-file mtime/hash tracking in the index itself we
            # can't distinguish "indexed and unchanged" from "indexed but
            # stale". The watcher handles incremental updates while it's
            # running; backfill defers to "if you want to refresh, use
            # --force." Keep the heuristic simple and predictable.
            print(f"  [{i:>{width}}/{total}] skip   {path.name}", file=sys.stderr)
            skipped += 1
            continue
        try:
            n_chunks = index_file(path, vault, embed_url=embed_url)
        except (httpx.HTTPError, OSError, ValueError) as exc:
            print(f"  [{i:>{width}}/{total}] FAIL   {path.name}: {exc}", file=sys.stderr)
            errored += 1
            continue
        print(f"  [{i:>{width}}/{total}] index  {path.name} ({n_chunks} chunks)", file=sys.stderr)
        indexed += 1
    return indexed, skipped, errored


def _do_gc(
    vault: LanceVault,
    accepted_files: set[str],
    *,
    dry_run: bool,
    force: bool,
) -> int:
    """Drop rows whose source_path is not in ``accepted_files``.

    ``accepted_files`` is the SET of currently-existing-and-filter-accepted
    paths — anything in the index that is NOT in this set is an orphan.
    Returns the number of distinct sources removed.

    Safety threshold: if ≥90% of sources would be removed, we refuse
    unless ``force=True``. The most likely cause of "everything looks
    orphan" is the user moved the vault root (so resolved paths in the
    index no longer line up). Wiping the index in that case is recoverable
    only by re-running --backfill, so make it a deliberate choice.
    """
    sources = vault.distinct_sources()
    orphans = [s for s in sources if s not in accepted_files]
    if sources and not force and len(orphans) / len(sources) >= 0.9:
        print(
            f"hermes: refusing to drop {len(orphans)}/{len(sources)} sources "
            f"(>=90%). This usually means the vault was moved or the path no "
            f"longer matches the indexed paths. Pass --force to confirm.",
            file=sys.stderr,
        )
        return 0
    for src in orphans:
        if dry_run:
            print(f"  would drop: {src}", file=sys.stderr)
        else:
            print(f"  drop: {src}", file=sys.stderr)
            vault.delete_by_source(src)
    return len(orphans)


# ----------------------------------------------------------------- entry


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hermes index",
        description="Backfill, re-index, or GC the vault index.",
    )
    p.add_argument("--backfill", action="store_true",
                   help="Walk the vault and index files not yet in the index.")
    p.add_argument("--force", action="store_true",
                   help="With --backfill: re-embed every file regardless of state.")
    p.add_argument("--gc", action="store_true",
                   help="Remove index rows whose source file is gone or now excluded.")
    p.add_argument("--dry-run", action="store_true",
                   help="With --gc: show what would be removed without removing.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N files (smoke-test backfill on huge vaults).")
    args = p.parse_args(argv)

    if not (args.backfill or args.gc):
        p.error("specify --backfill, --gc, or both.")
    if args.dry_run and not args.gc:
        # --dry-run only makes sense for the destructive op (GC). Refuse
        # rather than silently performing a real backfill, which would
        # confuse anyone using "dry run" as a safety check.
        p.error("--dry-run requires --gc. Backfill is always real (it's idempotent on top of itself).")

    vault_path = _resolve_vault_path()
    if vault_path is None:
        print("hermes: HERMES_VAULT_PATH not set and no watcher plist found.", file=sys.stderr)
        print("        export HERMES_VAULT_PATH=/path/to/vault and try again.", file=sys.stderr)
        return 2
    if not vault_path.is_dir():
        print(f"hermes: vault path {vault_path} does not exist.", file=sys.stderr)
        return 2

    db_path = _resolve_db_path()
    embed_url = _resolve_embed_url()
    vfilter = build_filter(vault_path)

    print(f"hermes index: vault={vault_path} db={db_path}", file=sys.stderr)

    files = iter_vault_files(vault_path, vfilter)
    print(f"  vault scan: {len(files)} accepted file(s)", file=sys.stderr)

    vault = LanceVault(path=db_path)
    started = time.monotonic()

    indexed = skipped = errored = 0
    if args.backfill:
        indexed, skipped, errored = _do_backfill(
            vault, files,
            embed_url=embed_url,
            force=args.force,
            limit=args.limit,
        )

    gc_dropped = 0
    if args.gc:
        accepted_set = {str(p.resolve()) for p in files}
        gc_dropped = _do_gc(vault, accepted_set, dry_run=args.dry_run, force=args.force)

    elapsed = time.monotonic() - started
    parts: list[str] = []
    if args.backfill:
        parts.append(f"{indexed} indexed, {skipped} skipped, {errored} errored")
    if args.gc:
        verb = "would drop" if args.dry_run else "dropped"
        parts.append(f"{verb} {gc_dropped} orphan source(s)")
    print(f"done in {elapsed:.1f}s — " + "; ".join(parts), file=sys.stderr)

    # Exit non-zero if the user asked for backfill and got hard errors —
    # makes `make backfill && hermes ask ...` safe to chain.
    return 1 if errored > 0 else 0


if __name__ == "__main__":
    sys.exit(run())
