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


def _build_user_prompt(source_path: Path, body: str) -> str:
    return (
        f"Source path: {source_path}\n\n"
        f"--- BEGIN SOURCE ---\n{body}\n--- END SOURCE ---\n\n"
        "Produce the wiki summary now."
    )


# ----------------------------------------------------------------- helpers


def _read_source(path: Path, *, max_chars: int = 32_000) -> str:
    """Read a source file. Truncates to ``max_chars`` to fit the chat
    server's context, with a note in the prompt so the model can warn
    if it sees the cut.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[truncated for length]"
    return text


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

    page_stem = _slugify(args.name or src.stem)
    page_path = paths.sources_dir / f"{page_stem}.md"
    # Hand-written-file guard MUST run regardless of --force. We never
    # clobber a file the LLM didn't author, even if the user explicitly
    # asks. They can delete it manually if they really mean it.
    if page_path.exists() and not wiki.is_managed(page_path):
        print(
            f"hermes ingest: refusing to overwrite hand-written file: {page_path}",
            file=sys.stderr,
        )
        return 1
    if page_path.exists() and not args.force:
        print(
            f"hermes ingest: {page_path.name} already exists "
            f"(use --force to re-summarize).",
            file=sys.stderr,
        )
        return 0

    print(f"hermes ingest: reading {src}", file=sys.stderr)
    body_in = _read_source(src)

    print(f"hermes ingest: calling chat server (this may take 30-60s)...", file=sys.stderr)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(src, body_in)},
    ]
    client = HermesClient()
    try:
        body_out = client.chat_sync(messages, max_tokens=args.max_tokens, temperature=0.3)
    except HermesError as exc:
        print(f"hermes ingest: chat server error: {exc}", file=sys.stderr)
        print(f"               (try: hermes doctor)", file=sys.stderr)
        return 1

    page = wiki.Page(
        title=page_stem,
        body=body_out,
        frontmatter={
            "source-path": str(src),
            "source-stem": src.stem,
        },
    )
    summary = _extract_summary(body_out)
    # Three-step write: page, then index, then log. If any step after the
    # first raises, the page is on disk but the index/log are out of sync.
    # Roll the page back so the wiki stays internally consistent; the user
    # can re-run ingest cleanly. This matters because a partial state
    # would survive into the next run, hiding the failure (the page
    # already exists → the early-return at line 156 returns 0).
    wiki.write_page(page_path, page)
    try:
        wiki.update_index_row(paths, "Sources", page_stem, summary)
        wiki.append_log(
            paths, "ingest", page_stem,
            detail=f"Source: {src}\nWiki page: {page_path.relative_to(paths.root.parent)}",
        )
    except Exception:
        # Best-effort rollback; if THIS fails too, the user gets the
        # original exception, not the rollback error.
        try:
            page_path.unlink()
        except OSError:
            pass
        raise

    print(f"hermes ingest: wrote {page_path.relative_to(paths.root.parent)}")
    print(f"               summary: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
