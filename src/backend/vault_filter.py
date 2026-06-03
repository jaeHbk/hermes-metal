"""Vault inclusion/exclusion logic shared by the watcher and `hermes index`.

The watcher uses this to decide whether a filesystem event deserves
re-indexing; `hermes index --backfill` uses it to decide which files to
walk on a fresh install. Keeping a single source of truth means the live
index and a backfill always agree on what belongs.

Resolution order (highest to lowest priority):

1. ``HERMES_VAULT_EXCLUDE`` env var (colon-separated globs) — quick override
   for one-off invocations without editing config.
2. ``config/vault.yaml`` — user-editable file with ``include:`` and
   ``exclude:`` lists. YAML is optional; if PyYAML is unavailable or the
   file is missing, we fall back to defaults.
3. Built-in defaults (see ``DEFAULT_INCLUDE`` / ``DEFAULT_EXCLUDE``).

Globs are matched ``fnmatch``-style against the path *relative to the
vault root* (forward-slash-separated, no leading ``/``). A glob without a
``/`` matches against any path component, so ``.obsidian`` excludes
``foo/.obsidian/bar``. A glob with ``/`` is anchored: ``templates/*``
matches only the top-level templates directory.
"""
from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path


# These defaults reflect what Obsidian users almost never want indexed:
# the app's own metadata, soft-deleted notes, image/PDF attachments, and
# template stubs that aren't real notes. Anything not in include is
# excluded automatically (we whitelist file extensions).
DEFAULT_INCLUDE: tuple[str, ...] = ("*.md", "*.markdown")
DEFAULT_EXCLUDE: tuple[str, ...] = (
    ".obsidian",       # Obsidian config / workspace state
    ".obsidian/*",
    ".trash",          # Obsidian's soft-delete folder
    ".trash/*",
    ".git",
    ".git/*",
    "attachments",
    "attachments/*",
    "_attachments",
    "_attachments/*",
    "templates",
    "templates/*",
    "Templates",
    "Templates/*",
    ".DS_Store",
)


@dataclass(frozen=True)
class VaultFilter:
    """Decides whether a vault path should be indexed."""

    vault_root: Path
    include: tuple[str, ...]
    exclude: tuple[str, ...]

    # ----------------------------------------------------------- public

    def accepts(self, path: str | Path) -> bool:
        """True if ``path`` should be indexed.

        Accepts both absolute paths and paths relative to ``vault_root``.
        Returns False for paths outside the vault, directories, and any
        path that fails the include/exclude rules.
        """
        rel = self._relativize(path)
        if rel is None:
            return False
        # Check exclude first so an explicit excludes always wins. A glob
        # without a slash matches against any path component; a glob with
        # a slash is anchored to the relative path.
        if self._matches_any(rel, self.exclude):
            return False
        # File extensions only: include rules whitelist file shapes.
        return self._matches_any(rel, self.include)

    # --------------------------------------------------------- internal

    def _relativize(self, path: str | Path) -> str | None:
        p = Path(path)
        if p.is_absolute():
            try:
                rel = p.resolve().relative_to(self.vault_root.resolve())
            except ValueError:
                return None
        else:
            rel = p
        # Normalize to forward slashes so glob patterns are portable.
        return rel.as_posix()

    @staticmethod
    def _matches_any(rel: str, patterns: tuple[str, ...]) -> bool:
        # Build the candidate set: full relative path PLUS each path
        # component, so a slashless pattern like ".obsidian" matches a
        # nested ".obsidian/workspace.json".
        candidates = [rel, *rel.split("/")]
        for pat in patterns:
            if "/" in pat:
                # Anchored pattern: match against the full relative path only.
                if fnmatch.fnmatch(rel, pat):
                    return True
            else:
                # Slashless pattern: match against any path component or
                # against the basename.
                for c in candidates:
                    if fnmatch.fnmatch(c, pat):
                        return True
        return False


# ---------------------------------------------------------- construction


def _load_yaml_config(path: Path) -> dict[str, list[str]]:
    """Best-effort load. Returns {} on missing file or unavailable PyYAML."""
    if not path.is_file():
        return {}
    try:
        import yaml  # noqa: WPS433 — optional dependency
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for key in ("include", "exclude"):
        v = data.get(key)
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            out[key] = v
    return out


def build_filter(
    vault_root: str | Path,
    *,
    config_path: str | Path | None = None,
) -> VaultFilter:
    """Construct a VaultFilter applying env, YAML, and defaults in order.

    ``config_path`` defaults to ``<repo>/config/vault.yaml`` when omitted.
    The HERMES_VAULT_EXCLUDE env var, if set, REPLACES the YAML/default
    excludes (it's an override, not an additive). HERMES_VAULT_INCLUDE
    likewise replaces the include list. This avoids the surprise of an
    env-only setting being silently merged with a stale config file.
    """
    vault = Path(vault_root).expanduser().resolve()

    # Layer 1: defaults
    include: tuple[str, ...] = DEFAULT_INCLUDE
    exclude: tuple[str, ...] = DEFAULT_EXCLUDE

    # Layer 2: YAML config (optional)
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "config" / "vault.yaml"
    yaml_data = _load_yaml_config(Path(config_path))
    if "include" in yaml_data:
        include = tuple(yaml_data["include"])
    if "exclude" in yaml_data:
        exclude = tuple(yaml_data["exclude"])

    # Layer 3: env vars (highest priority)
    env_include = os.environ.get("HERMES_VAULT_INCLUDE")
    if env_include:
        include = tuple(p.strip() for p in env_include.split(":") if p.strip())
    env_exclude = os.environ.get("HERMES_VAULT_EXCLUDE")
    if env_exclude:
        exclude = tuple(p.strip() for p in env_exclude.split(":") if p.strip())

    return VaultFilter(vault_root=vault, include=include, exclude=exclude)


def iter_vault_files(vault_root: str | Path, vfilter: VaultFilter) -> list[Path]:
    """Walk ``vault_root`` and return all files the filter accepts.

    Returns a list (not a generator) so the caller can show a progress
    count up front. Skips entire directories that match an exclude pattern
    to avoid descending into ``attachments/`` etc. on huge vaults.
    """
    root = Path(vault_root).expanduser().resolve()
    if not root.is_dir():
        return []

    out: list[Path] = []
    # os.walk lets us prune dirs in-place via the dirnames mutation trick.
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dpath = Path(dirpath)
        # Prune excluded directories: build a relative dir path for each
        # subdir candidate and check if it would be excluded as a path.
        pruned: list[str] = []
        for d in dirnames:
            sub = dpath / d
            try:
                rel = sub.resolve().relative_to(root)
            except ValueError:
                pruned.append(d)
                continue
            rel_str = rel.as_posix()
            # Reuse VaultFilter._matches_any against the dir's path. Treat
            # a directory as "excluded" if either its full rel path or any
            # of its components matches an exclude.
            if VaultFilter._matches_any(rel_str, vfilter.exclude):  # noqa: SLF001
                pruned.append(d)
        for d in pruned:
            dirnames.remove(d)

        for fname in filenames:
            full = dpath / fname
            if vfilter.accepts(full):
                out.append(full)
    return out
