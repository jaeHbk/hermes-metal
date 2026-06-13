"""`hermes ingest <path>` — read a raw source, summarize via the chat
server, and write a wiki page under `wiki/sources/<stem>.md`.

This is the central operation in the wiki pattern: each source is read
once, distilled, and filed in a way the watcher then auto-indexes for
retrieval. Repeated ingest of the same source is rejected without
``--force``; repeated successful ingest does NOT generate duplicate
index rows because ``wiki.update_index_row`` is keyed on page name.

The summarization prompt is intentionally conservative: extract the
key claims, name the central entities/concepts, surface contradictions
the source flags, and leave a "questions to follow up on" section.
We don't try to extract *every* point — the LLM owns the trade-off.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from src import wiki
from src.server.client import HermesClient, HermesError

from dataclasses import dataclass


WROTE = "WROTE"
ALREADY_EXISTS = "ALREADY_EXISTS"
REFUSED_HANDWRITTEN = "REFUSED_HANDWRITTEN"

# Cap source text fed to the chat server so a huge file/page can't blow the
# context window. ingest_text is the single owner of this truncation (file,
# URL, and batch all flow through it).
MAX_SOURCE_CHARS = 32_000
_TRUNCATION_MARKER = "\n\n[truncated for length]"


@dataclass
class IngestResult:
    status: str               # WROTE | ALREADY_EXISTS | REFUSED_HANDWRITTEN
    page_path: Path
    summary: str


# ---------------------------------------------------------- prompt template


SYSTEM_PROMPT = """\
You are hermes-metal's wiki ingest worker. Your job is to read a raw
source document and produce a wiki summary page.

Output must be valid markdown with these sections, in order:

## Summary
A 2-3 sentence factual summary of what the source says.

## Key claims
A bulleted list of 3-7 specific claims, facts, or arguments. Each
claim should be precise enough that a future query like "what did X
say about Y?" can match it.

## Entities and concepts
A bulleted list of named entities (people, places, projects, ideas)
that appear. For each, one short sentence on its role in this source.
Use `[[wiki-link]]` syntax — e.g. `- [[Alan Turing]]: introduced...`
— so that future ingests linking the same name will graph-connect.

## Open questions
A bulleted list of 1-3 questions this source raises but does not
answer. These are leads for follow-up work.

Rules:
- Be specific. Avoid generic phrases like "discusses several topics."
- Cite the source by basename only when relevant: `[Welcome.md]`.
- Do NOT add a header above "## Summary" — the wiki layer adds that.
- Output ONLY the markdown body. No preamble, no apology, no closing.
"""


def _build_user_prompt(source_label: str, body: str) -> str:
    return (
        f"Source path: {source_label}\n\n"
        f"--- BEGIN SOURCE ---\n{body}\n--- END SOURCE ---\n\n"
        "Produce the wiki summary now."
    )


# ----------------------------------------------------------------- helpers


def _read_source(path: Path) -> str:
    """Read a source file as text. Truncation to the chat-server context cap
    is handled centrally in ``ingest_text`` (so URL/batch ingestion, which
    don't read files, get the same cap)."""
    return path.read_text(encoding="utf-8", errors="replace")


def _slugify(name: str) -> str:
    """Convert an arbitrary stem into a safe filename.

    Allows letters, digits, dot, dash, underscore. Anything else
    becomes ``_``. Empty result falls back to ``page``.
    """
    out = []
    for ch in name:
        if ch.isalnum() or ch in ".-_":
            out.append(ch)
        else:
            out.append("_")
    cleaned = "".join(out).strip("._-") or "page"
    # Avoid filenames like ``index`` / ``log`` that collide with wiki meta.
    # Compare case-insensitively because APFS (the project's sole target
    # filesystem) is case-insensitive by default — ``INDEX.md`` and
    # ``index.md`` are the same path, so we must guard either spelling.
    if cleaned.lower() in ("index", "log"):
        cleaned += "-source"
    return cleaned


def _extract_summary(body: str) -> str:
    """Pull the first non-empty line under '## Summary' for the index row.

    Falls back to the first 100 chars of ``body`` if the section is
    missing (the model misformatted its output). Index rows tolerate a
    little fuzziness — this is descriptive, not structural.
    """
    in_summary = False
    for line in body.splitlines():
        if line.strip().lower() == "## summary":
            in_summary = True
            continue
        if in_summary:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                break  # next section
            return stripped[:160]
    # Fallback.
    snippet = " ".join(body.split())[:120].strip()
    return snippet or "(no summary)"


# ------------------------------------------------------------------- run


def ingest_text(
    body: str,
    *,
    page_name: str,
    source_label: str,
    extra_frontmatter: dict[str, str] | None = None,
    max_tokens: int = 1024,
    force: bool = False,
) -> IngestResult:
    """Summarize ``body`` via the chat server and write a wiki source page.

    This is the shared write-path for every ingest entry point (file, URL,
    batch). ``page_name`` is the (already-derived) wiki page stem; it is
    slugified here so callers don't have to. ``source_label`` is the
    provenance string the LLM sees (a file path or a URL).
    ``extra_frontmatter`` is merged into the page frontmatter (e.g.
    ``{"source-url": ...}``).

    Returns an :class:`IngestResult`. Raises ``HermesError`` if the chat
    server fails, and re-raises any exception from the index/log write after
    rolling back the page.
    """
    stem = _slugify(page_name)
    paths = wiki.get_paths()
    page_path = paths.sources_dir / f"{stem}.md"

    # Hand-written-file guard MUST run regardless of --force.
    if page_path.exists() and not wiki.is_managed(page_path):
        return IngestResult(REFUSED_HANDWRITTEN, page_path, "")
    if page_path.exists() and not force:
        return IngestResult(ALREADY_EXISTS, page_path, "")

    if len(body) > MAX_SOURCE_CHARS:
        body = body[:MAX_SOURCE_CHARS] + _TRUNCATION_MARKER

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(source_label, body)},
    ]
    client = HermesClient()
    body_out = client.chat_sync(messages, max_tokens=max_tokens, temperature=0.3)

    meta = {"source-path": source_label}
    if extra_frontmatter:
        meta.update(extra_frontmatter)
    page = wiki.Page(title=stem, body=body_out, frontmatter=meta)
    summary = _extract_summary(body_out)

    wiki.write_page(page_path, page)
    try:
        wiki.update_index_row(paths, "Sources", stem, summary)
        wiki.append_log(
            paths, "ingest", stem,
            detail=f"Source: {source_label}\nWiki page: {page_path.relative_to(paths.root.parent)}",
        )
    except Exception:
        try:
            page_path.unlink()
        except OSError:
            pass
        raise
    return IngestResult(WROTE, page_path, summary)


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hermes ingest",
        description="Summarize a raw source into a wiki page (writes wiki/sources/<stem>.md).",
    )
    p.add_argument("path", help="Path to the source file (any text-readable format).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite an existing wiki/sources/<stem>.md.")
    p.add_argument("--name", default=None,
                   help="Override the wiki page name (default: source filename stem).")
    p.add_argument("--max-tokens", type=int, default=1024,
                   help="Cap on chat server response (default 1024).")
    args = p.parse_args(argv)

    src = Path(args.path).expanduser().resolve()
    if not src.is_file():
        print(f"hermes ingest: source not found: {src}", file=sys.stderr)
        return 2

    paths = wiki.get_paths()
    if not wiki.is_initialized(paths):
        print(f"hermes ingest: wiki not initialized at {paths.root}", file=sys.stderr)
        print(f"               run: hermes wiki init", file=sys.stderr)
        return 2

    print(f"hermes ingest: reading {src}", file=sys.stderr)
    body_in = _read_source(src)

    print(f"hermes ingest: calling chat server (this may take 30-60s)...", file=sys.stderr)
    try:
        res = ingest_text(
            body_in,
            page_name=args.name or src.stem,
            source_label=str(src),
            max_tokens=args.max_tokens,
            force=args.force,
        )
    except HermesError as exc:
        print(f"hermes ingest: chat server error: {exc}", file=sys.stderr)
        print(f"               (try: hermes doctor)", file=sys.stderr)
        return 1

    if res.status == REFUSED_HANDWRITTEN:
        print(f"hermes ingest: refusing to overwrite hand-written file: {res.page_path}",
              file=sys.stderr)
        return 1
    if res.status == ALREADY_EXISTS:
        print(f"hermes ingest: {res.page_path.name} already exists "
              f"(use --force to re-summarize).", file=sys.stderr)
        return 0

    print(f"hermes ingest: wrote {res.page_path.relative_to(paths.root.parent)}")
    print(f"               summary: {res.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
