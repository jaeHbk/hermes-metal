"""Wiki-layer primitives: paths, init, page write, log/index update.

The wiki is a structured `<vault>/wiki/` subtree the LLM owns:

    <vault>/
    ├── (user's notes — never touched by the LLM)
    └── wiki/
        ├── .hermes-agents.md     ← schema; user-controlled, LLM reads it
        ├── index.md              ← content-oriented catalog (LLM-maintained)
        ├── log.md                ← chronological audit (LLM-appended)
        ├── sources/<stem>.md     ← one summary page per ingested raw source
        ├── topics/<name>.md      ← concept/entity pages built up over time
        └── digests/YYYY-MM-DD.md ← scheduled daily summaries (Phase D)

Design:

* **Stdlib only** — no httpx, lancedb, watchdog. Doctor and tests can
  exercise this without the daemons up.
* **Atomic writes** via temp+rename so a crash mid-write doesn't leave
  a half-written index.md or stub page that future runs can't parse.
* **YAML frontmatter** on every LLM-managed page with at minimum
  ``hermes-managed: true`` so the user (and future filters) can tell
  what's LLM-owned vs hand-written.
* **Wiki location**: ``$HERMES_VAULT_PATH/wiki/`` by default;
  ``HERMES_WIKI_PATH`` env var overrides for users who want it
  somewhere else (e.g. a separate vault).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ----------------------------------------------------------------- paths


@dataclass(frozen=True)
class WikiPaths:
    """Resolved wiki layout. All paths are absolute."""

    root: Path

    @property
    def index(self) -> Path:
        return self.root / "index.md"

    @property
    def log(self) -> Path:
        return self.root / "log.md"

    @property
    def schema(self) -> Path:
        return self.root / ".hermes-agents.md"

    @property
    def sources_dir(self) -> Path:
        return self.root / "sources"

    @property
    def topics_dir(self) -> Path:
        return self.root / "topics"

    @property
    def digests_dir(self) -> Path:
        return self.root / "digests"

    @property
    def conversations_dir(self) -> Path:
        return self.root / "conversations"


def resolve_wiki_path(vault_path: str | Path | None = None) -> Path:
    """Resolve the wiki root.

    Precedence: ``HERMES_WIKI_PATH`` env var > ``<vault>/wiki/``. Caller
    can pass ``vault_path`` explicitly; otherwise ``HERMES_VAULT_PATH``
    is consulted, falling back to ``~/Documents/Obsidian`` to match
    the rest of the project's defaults.
    """
    raw = os.environ.get("HERMES_WIKI_PATH")
    if raw:
        return Path(raw).expanduser().resolve()
    if vault_path is None:
        vault_path = os.environ.get("HERMES_VAULT_PATH") or "~/Documents/Obsidian"
    return (Path(vault_path).expanduser().resolve()) / "wiki"


def get_paths(vault_path: str | Path | None = None) -> WikiPaths:
    return WikiPaths(root=resolve_wiki_path(vault_path))


def is_initialized(paths: WikiPaths | None = None) -> bool:
    """True iff the wiki has been initialized via ``hermes wiki init``.

    We require both the index and the schema file because either alone
    can be created by an unrelated text editor.
    """
    p = paths or get_paths()
    return p.index.is_file() and p.schema.is_file()


# ------------------------------------------------------------ atomic write


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically (temp file + rename).

    On macOS rename within the same filesystem is atomic, so the user
    never sees a partially-written file. Same FS guarantee fails across
    mounts; we don't try to be clever about it — wiki files live inside
    the vault, so the temp file is always on the same FS.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


# --------------------------------------------------------------- init


_DEFAULT_SCHEMA = """\
# .hermes-agents.md

This file tells `hermes` how this vault is organized. The LLM reads it
at REPL/ask startup and threads it into the system prompt. Edit it to
match your conventions; the LLM never auto-edits this file.

## Vault layout

- Daily notes live at `journal/YYYY-MM-DD.md` (adjust if you use a
  different scheme).
- Class / course notes are tagged `#class/<course>` in frontmatter.
- The `wiki/` subtree is LLM-owned — do not hand-edit pages there.

## Citation conventions

- When citing in synthesis pages, prefer `[[wiki-link]]` over plain
  mentions so links show up in Obsidian's graph view.
- Cite raw notes by basename: `[Welcome.md]`.

## What to emphasize

(Add your own preferences here. Examples: "favor concrete examples over
abstract definitions", "always note disagreements between sources",
"prefer brevity in topic pages.")
"""

_DEFAULT_INDEX = """\
# Wiki Index

Catalog of everything in the wiki. The LLM updates this on every ingest
and `/file`. Don't edit by hand — your edits will be overwritten.

## Sources

(none yet — run `hermes ingest <path>` to add one)

## Topics

(none yet — use `/file <name>` in the REPL to promote an answer)

## Digests

(none yet — Phase D writes daily digests here)

## Conversations

(none yet — substantial REPL sessions are archived here)
"""

_DEFAULT_LOG = """\
# Wiki Log

Chronological audit trail. Append-only. Each entry begins with
`## [YYYY-MM-DDTHH:MM:SSZ] <op> | <subject>` so it's grep-able:

    grep '^## ' wiki/log.md | tail -20

"""


def init_wiki(paths: WikiPaths | None = None, *, force: bool = False) -> WikiPaths:
    """Create the wiki structure if absent.

    Idempotent: running it twice is a no-op unless ``force=True``, in
    which case index/log/schema get reset to defaults (sources/, topics/,
    digests/ are NEVER deleted — the user might already have content).
    """
    p = paths or get_paths()
    p.root.mkdir(parents=True, exist_ok=True)
    p.sources_dir.mkdir(exist_ok=True)
    p.topics_dir.mkdir(exist_ok=True)
    p.digests_dir.mkdir(exist_ok=True)
    p.conversations_dir.mkdir(exist_ok=True)

    if force or not p.schema.is_file():
        _atomic_write_text(p.schema, _DEFAULT_SCHEMA)
    if force or not p.index.is_file():
        _atomic_write_text(p.index, _DEFAULT_INDEX)
    if force or not p.log.is_file():
        _atomic_write_text(p.log, _DEFAULT_LOG)

    return p


# ------------------------------------------------------------ page write


@dataclass
class Page:
    """An LLM-managed wiki page."""

    title: str
    body: str  # markdown body, NO frontmatter (we add it)
    frontmatter: dict[str, str] = field(default_factory=dict)


_FRONTMATTER_KEY_RE = re.compile(r"^[A-Za-z][\w-]*$")


def _render_frontmatter(meta: dict[str, str]) -> str:
    """Render a flat dict as YAML frontmatter.

    Keys must be simple identifiers; values are quoted to survive
    colons, brackets, etc. We don't pull in PyYAML — the wiki layer
    stays stdlib-clean and our frontmatter is structurally trivial.
    """
    if not meta:
        return ""
    lines = ["---"]
    for k, v in meta.items():
        if not _FRONTMATTER_KEY_RE.match(k):
            raise ValueError(f"frontmatter key not safe for YAML: {k!r}")
        # Always quote string values so a value containing `:` (e.g. a
        # path) doesn't turn into a mapping. Booleans pass through.
        if isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        else:
            escaped = str(v).replace('"', '\\"')
            lines.append(f'{k}: "{escaped}"')
    lines.append("---\n")
    return "\n".join(lines)


def write_page(path: Path, page: Page) -> Path:
    """Render and atomically write ``page`` to ``path``.

    The frontmatter ALWAYS includes ``hermes-managed: true`` so any
    future filter can tell what's LLM-owned. ``hermes-updated`` is
    stamped at write time.
    """
    meta = dict(page.frontmatter)
    meta.setdefault("hermes-managed", "true")
    meta.setdefault("hermes-updated", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    rendered = _render_frontmatter(meta) + f"# {page.title}\n\n{page.body.rstrip()}\n"
    _atomic_write_text(path, rendered)
    return path


def is_managed(path: Path) -> bool:
    """True iff ``path`` is an LLM-owned wiki page.

    The check is bounded to the YAML frontmatter block (between the
    opening ``---`` and the next ``---`` line) so a user file that
    happens to contain the literal string ``hermes-managed: true`` in
    its body — say, a code example or a quoted line — does NOT get
    misidentified and overwritten.
    """
    if not path.is_file():
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:2048]
    except OSError:
        return False
    if not head.startswith("---"):
        return False
    # Find the frontmatter close: the next "---\n" after the opener.
    end = head.find("\n---", 3)
    if end == -1:
        return False
    block = head[3:end]
    return 'hermes-managed: "true"' in block or "hermes-managed: true" in block


# --------------------------------------------------- log + index updates


def append_log(paths: WikiPaths, op: str, subject: str, *, detail: str = "") -> None:
    """Append one entry to ``log.md``.

    Format: ``## [TS] op | subject\\n\\ndetail\\n\\n`` so simple shell
    tooling can parse it. ``op`` is short ("ingest", "file", "lint",
    "digest"); ``subject`` is the page name or filename.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"\n## [{ts}] {op} | {subject}\n"
    if detail:
        line += "\n" + detail.rstrip() + "\n"
    # Read-modify-write keeps the log a single file; atomic via temp+rename.
    existing = paths.log.read_text(encoding="utf-8") if paths.log.is_file() else _DEFAULT_LOG
    _atomic_write_text(paths.log, existing.rstrip() + line + "\n")


def update_index_row(
    paths: WikiPaths,
    section: str,                 # "Sources", "Topics", or "Digests"
    page_name: str,               # filename without extension
    summary: str,                 # one-line summary
) -> None:
    """Update or insert a row in the index for the named page.

    The index has three sections (## Sources / ## Topics / ## Digests),
    each containing a markdown list of ``- [name](rel/path) — summary``
    rows. We rewrite the list per section atomically to avoid duplicate
    rows after multiple ingests of the same page.
    """
    section = section.strip()
    if section not in ("Sources", "Topics", "Digests", "Conversations"):
        raise ValueError(f"unknown index section: {section!r}")

    # Map section -> subdirectory name for relative links.
    subdir_map = {
        "Sources": "sources",
        "Topics": "topics",
        "Digests": "digests",
        "Conversations": "conversations",
    }
    subdir = subdir_map[section]
    rel_link = f"{subdir}/{page_name}.md"
    new_row = f"- [{page_name}]({rel_link}) — {summary.strip()}"

    text = paths.index.read_text(encoding="utf-8") if paths.index.is_file() else _DEFAULT_INDEX
    sections = _split_index(text)
    rows = sections.get(section, [])
    # Drop any existing row with the same page name so updates replace
    # rather than duplicate. We compare on the link target.
    rows = [r for r in rows if not _row_targets(r, rel_link)]
    rows.append(new_row)
    rows.sort(key=str.lower)
    sections[section] = rows
    rebuilt = _join_index(sections, original=text)
    _atomic_write_text(paths.index, rebuilt)


# Heuristic: index pages are markdown lists under ## headers. We rebuild
# only the three known sections; everything else (page header, blurb)
# is preserved verbatim.
_SECTION_RE = re.compile(r"^## (Sources|Topics|Digests|Conversations)\s*$", re.MULTILINE)


def _split_index(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for match in _SECTION_RE.finditer(text):
        name = match.group(1)
        # Body runs from end of header line to next ## line (or EOF).
        start = match.end()
        next_match = _SECTION_RE.search(text, pos=start)
        end = next_match.start() if next_match else len(text)
        body = text[start:end].strip()
        rows = [
            line for line in body.splitlines()
            if line.startswith("- ") and "](" in line
        ]
        out[name] = rows
    return out


def _join_index(sections: dict[str, list[str]], *, original: str) -> str:
    """Rebuild the index, emitting any of the three known sections.

    Sections present in ``original`` are rebuilt in their original order;
    sections missing from ``original`` are appended at the end so a row
    written to a "Topics" section the user has deleted from index.md
    still ends up persisted (rather than silently dropping the update).
    """
    out_lines: list[str] = []
    last_end = 0
    seen: set[str] = set()
    for match in _SECTION_RE.finditer(original):
        name = match.group(1)
        seen.add(name)
        # The slice from last_end to match.end() is "<preamble/prev-body>\n##
        # Header". rstrip() it so blank lines the previous rebuild left between
        # the header and its body don't accumulate on every write (the section
        # regex's trailing \s* otherwise re-captures them, growing index.md
        # without bound across daily digests / per-session archives). We then
        # re-emit exactly one blank line before the body.
        out_lines.append(original[last_end:match.end()].rstrip())
        rows = sections.get(name, [])
        out_lines.append("\n\n")
        out_lines.append("\n".join(rows) if rows else "(none)")
        out_lines.append("\n\n")
        next_match = _SECTION_RE.search(original, pos=match.end())
        last_end = next_match.start() if next_match else len(original)
    # Trailing slice (after the last known section): collapse leading blank
    # lines for the same reason, then keep the remainder verbatim.
    tail = original[last_end:]
    out_lines.append(tail.lstrip("\n") if tail else tail)
    rebuilt = "".join(out_lines).rstrip()

    # Append any sections that didn't appear in the original. We only
    # emit a header for a missing section if it has rows to write —
    # otherwise we'd pollute the index with empty stubs.
    for name in ("Sources", "Topics", "Digests", "Conversations"):
        if name in seen:
            continue
        rows = sections.get(name, [])
        if not rows:
            continue
        rebuilt += f"\n\n## {name}\n\n" + "\n".join(rows)
    return rebuilt + "\n"


def _row_targets(row: str, rel_link: str) -> bool:
    """True if a markdown row points at ``rel_link``."""
    return f"]({rel_link})" in row


# ------------------------------------------------------------- lint helpers


_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
_MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def parse_links(text: str) -> set[str]:
    """Pull out all link targets from a markdown body.

    Returns target stems (no extension, no anchor). Both ``[[Foo]]``
    Obsidian-style links and ``[Foo](foo.md)`` markdown links count.
    Used by lint to detect orphans / stubs.
    """
    out: set[str] = set()
    for m in _WIKILINK_RE.finditer(text):
        out.add(_normalize_target(m.group(1)))
    for m in _MD_LINK_RE.finditer(text):
        target = m.group(1)
        # Skip absolute URLs.
        if "://" in target:
            continue
        out.add(_normalize_target(target))
    return out


def _normalize_target(s: str) -> str:
    s = s.strip()
    if s.endswith(".md"):
        s = s[:-3]
    # Drop a leading ./ or path components — we compare on stems for the
    # lint heuristic. False positives (two pages with the same stem in
    # different subdirs) are rare in practice and the lint output is
    # advisory anyway.
    if "/" in s:
        s = s.rsplit("/", 1)[1]
    return s


def all_pages(paths: WikiPaths) -> list[Path]:
    """Every .md file under the wiki root (excluding index/log/schema)."""
    if not paths.root.is_dir():
        return []
    out: list[Path] = []
    for sub in (paths.sources_dir, paths.topics_dir, paths.digests_dir, paths.conversations_dir):
        if sub.is_dir():
            for p in sub.iterdir():
                if p.is_file() and p.suffix == ".md":
                    out.append(p)
    return out


def page_stem(path: Path) -> str:
    return path.stem
