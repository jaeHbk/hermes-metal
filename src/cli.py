"""hermes-metal command-line interface.

Subcommands:
    hermes ask "<question>"     RAG: embed -> LanceDB top-k -> chat (streaming).
    hermes search "<query>"     Retrieval only; prints matched chunks + sources.
    hermes status               Health probes both servers and prints index stats.
    hermes doctor               End-to-end self-diagnostic with remediation.

Resolves config from environment variables (HERMES_CHAT_URL, HERMES_EMBED_URL,
HERMES_LANCEDB_PATH) so it works whether the daemons are running locally or
the user has pointed it at a remote box.
"""
from __future__ import annotations

import argparse
import sys


# `hermes doctor` is the one subcommand that must keep working when imports
# from src.backend (lancedb, pyarrow, httpx) fail — that's exactly when users
# reach for it. Route to src.doctor before any heavy import happens.
if len(sys.argv) >= 2 and sys.argv[1] == "doctor":
    from src import doctor as _doctor
    sys.exit(_doctor.run(sys.argv[2:]))

# `hermes notify` is similarly stdlib-only and independent of the daemons.
# Keep it bootable even when the venv's lancedb/httpx are broken so a user
# whose chat server is down can still set up their phone notifications.
# Wrap in a try/except KeyboardInterrupt because `--setup` calls input()
# and the user pressing ^C mid-prompt should exit cleanly, not traceback.
if len(sys.argv) >= 2 and sys.argv[1] == "notify":
    from src import notify as _notify
    try:
        sys.exit(_notify.run(sys.argv[2:]))
    except KeyboardInterrupt:
        print("\nhermes notify: interrupted.", file=sys.stderr)
        sys.exit(130)

# `hermes wiki` and `hermes lint` are stdlib-only (no LanceDB / no chat
# server). Early-route so a broken venv doesn't break the wiki bootstrap.
if len(sys.argv) >= 2 and sys.argv[1] == "wiki":
    from src import wiki_cmd as _wiki_cmd
    sys.exit(_wiki_cmd.run(sys.argv[2:]))
if len(sys.argv) >= 2 and sys.argv[1] == "lint":
    from src import lint_cmd as _lint_cmd
    sys.exit(_lint_cmd.run(sys.argv[2:]))

# `hermes digest` is the scheduled daily-summary entry point (also the
# LaunchAgent's ProgramArguments target). Early-route: it talks to the chat
# server and Telegram but needs neither lancedb nor the embed server, and a
# ^C during a long synthesis should exit cleanly rather than traceback.
if len(sys.argv) >= 2 and sys.argv[1] == "digest":
    from src import digest as _digest
    try:
        sys.exit(_digest.run(sys.argv[2:]))
    except KeyboardInterrupt:
        print("\nhermes digest: interrupted.", file=sys.stderr)
        sys.exit(130)


import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx

from src.backend.database import LanceVault
from src.backend.indexer import embed as embed_query
from src.server.client import HermesClient, HermesError


DEFAULT_K = 5
DEFAULT_MAX_CONTEXT_CHARS = 8000
DEFAULT_MAX_TOKENS = 512
# Vector search casts a wider net (top-N) so the reranker has candidates to
# reorder before we trim to top-k. 4× is enough headroom without bloating the
# rerank pass on a small index.
RERANK_FETCH_MULTIPLIER = 4

SYSTEM_PROMPT = (
    "You are hermes-metal, the user's local 'second brain'. Answer using ONLY "
    "the provided notes when they are relevant; if the notes do not contain "
    "the answer, say so plainly and answer from general knowledge while "
    "flagging that the notes did not cover it. Cite sources by their filename "
    "in square brackets, e.g. [Welcome.md], when you draw on them."
)


def _resolve_db_path() -> Path:
    raw = os.environ.get("HERMES_LANCEDB_PATH") or str(
        Path(__file__).resolve().parents[1] / "storage" / "lancedb"
    )
    return Path(raw).expanduser().resolve()


def _resolve_embed_url() -> str:
    return os.environ.get("HERMES_EMBED_URL", "http://127.0.0.1:8081/v1/embeddings")


def _format_context(hits: list[dict[str, Any]], max_chars: int) -> str:
    blocks: list[str] = []
    used = 0
    for h in hits:
        src = Path(h["source_path"]).name
        block = f"[{src} #chunk{h['chunk_idx']}]\n{h['text'].strip()}\n"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n---\n".join(blocks)


def _print_hits_summary(hits: list[dict[str, Any]], stream=sys.stderr) -> None:
    if not hits:
        print("(no matching notes found in vault)", file=stream)
        return
    print(f"retrieved {len(hits)} chunk(s):", file=stream)
    for h in hits:
        src = Path(h["source_path"]).name
        score = h.get("_distance")
        score_str = f"  d={score:.3f}" if isinstance(score, (int, float)) else ""
        print(f"  - {src} #chunk{h['chunk_idx']}{score_str}", file=stream)
    print("", file=stream)


def _retrieve(query: str, *, k: int) -> list[dict[str, Any]]:
    """Embed → (optional date filter) → vector top-N → rerank → top-k.

    Phase B (temporal): a high-precision date phrase in ``query`` scopes the
    vector search to an mtime/path window. Phase C (rerank): we over-fetch
    candidates and reorder by the heuristic reranker (semantic + recency +
    lexical) before trimming to ``k``. Both degrade gracefully — an
    un-migrated index simply has no mtime to filter/score on.
    """
    import time as _time
    from src.backend import temporal, reranker

    db_path = _resolve_db_path()
    if not db_path.exists():
        raise SystemExit(
            f"LanceDB path does not exist: {db_path}\n"
            "Run the watcher (it auto-creates on first event) or "
            "edit a note in your vault."
        )
    vault = LanceVault(path=db_path)
    if vault.count() == 0:
        return []
    qvec = embed_query([query], embed_url=_resolve_embed_url(), task="search_query")[0]

    # Temporal scoping only when the index actually has mtime to filter on.
    where = None
    window = temporal.parse_window(query) if vault.has_metadata else None
    if window is not None:
        where = window.where_clause()
        print(f"  ↳ scoped to {window.phrase}", file=sys.stderr)

    fetch_k = k * RERANK_FETCH_MULTIPLIER if vault.has_metadata else k
    hits = vault.search(qvec, k=fetch_k, filter=where)
    # If a temporal filter eliminated everything, fall back to an unfiltered
    # search rather than answering "nothing found" — the date scope is a
    # best-effort precision aid, not a hard constraint the user asked for.
    if window is not None and not hits:
        print(f"  ↳ no notes in {window.phrase}; widening to all notes", file=sys.stderr)
        hits = vault.search(qvec, k=fetch_k)
    if not vault.has_metadata:
        return hits[:k]
    return reranker.rerank(query, hits, k=k, now=_time.time(), metric="cosine")


async def _ask_async(question: str, *, k: int, max_tokens: int, no_rag: bool) -> int:
    hits: list[dict[str, Any]] = []
    if not no_rag:
        try:
            hits = _retrieve(question, k=k)
        except httpx.HTTPError as exc:
            print(f"hermes: embed server unreachable: {exc}", file=sys.stderr)
            print("hermes: continuing without retrieval (use --no-rag to suppress).", file=sys.stderr)
        _print_hits_summary(hits)

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if hits:
        context = _format_context(hits, DEFAULT_MAX_CONTEXT_CHARS)
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Notes from my vault that may be relevant:\n\n{context}\n\n"
                    f"Question: {question}"
                ),
            }
        )
    else:
        messages.append({"role": "user", "content": question})

    async with HermesClient() as client:
        try:
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.4,
                "stream": True,
            }
            async with client._chat.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    piece = delta.get("content")
                    if piece:
                        sys.stdout.write(piece)
                        sys.stdout.flush()
            sys.stdout.write("\n")
        except httpx.HTTPError as exc:
            print(f"\nhermes: chat server error: {exc}", file=sys.stderr)
            return 1
        except HermesError as exc:
            print(f"\nhermes: {exc}", file=sys.stderr)
            return 1
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    return asyncio.run(
        _ask_async(
            args.question,
            k=args.k,
            max_tokens=args.max_tokens,
            no_rag=args.no_rag,
        )
    )


def _cmd_search(args: argparse.Namespace) -> int:
    try:
        hits = _retrieve(args.query, k=args.k)
    except httpx.HTTPError as exc:
        print(f"hermes: embed server unreachable: {exc}", file=sys.stderr)
        return 1
    if not hits:
        print("(no matches)")
        return 0
    for i, h in enumerate(hits, 1):
        src = Path(h["source_path"]).name
        score = h.get("_distance")
        header = f"[{i}] {src} #chunk{h['chunk_idx']}"
        if isinstance(score, (int, float)):
            header += f"   distance={score:.4f}"
        print(header)
        print(f"    path: {h['source_path']}")
        snippet = h["text"].strip().replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:280] + "..."
        print(f"    {snippet}")
        print()
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    chat_url = os.environ.get("HERMES_CHAT_URL", "http://127.0.0.1:8080")
    embed_url = _resolve_embed_url().rsplit("/v1/", 1)[0].rsplit("/embedding", 1)[0]

    print("hermes-metal status")
    print(f"  chat server:   {chat_url}")
    with httpx.Client(timeout=3.0) as client:
        try:
            r = client.get(f"{chat_url}/health")
            print(f"    /health    {r.status_code} {r.text.strip()}")
        except httpx.HTTPError as exc:
            print(f"    /health    UNREACHABLE ({exc})")

    print(f"  embed server:  {embed_url}")
    with httpx.Client(timeout=3.0) as client:
        try:
            r = client.get(f"{embed_url}/health")
            print(f"    /health    {r.status_code} {r.text.strip()}")
        except httpx.HTTPError as exc:
            print(f"    /health    UNREACHABLE ({exc})")

    db_path = _resolve_db_path()
    print(f"  lancedb:       {db_path}")
    if db_path.exists():
        try:
            vault = LanceVault(path=db_path)
            print(f"    rows       {vault.count()}")
        except Exception as exc:  # noqa: BLE001
            print(f"    rows       ERROR ({exc})")
    else:
        print("    rows       (path does not exist yet)")

    vault_path = os.environ.get("HERMES_VAULT_PATH", "(unset)")
    print(f"  vault:         {vault_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hermes",
        description="Local-first second-brain CLI (hermes-metal).",
    )
    # cmd is OPTIONAL: bare `hermes` drops into the REPL (handled in main()).
    sub = p.add_subparsers(dest="cmd")

    ask = sub.add_parser("ask", help="RAG: retrieve from vault and chat (streaming).")
    ask.add_argument("question", help="Natural-language question.")
    ask.add_argument("-k", type=int, default=DEFAULT_K, help=f"top-k chunks (default {DEFAULT_K}).")
    ask.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                     help=f"max response tokens (default {DEFAULT_MAX_TOKENS}).")
    ask.add_argument("--no-rag", action="store_true",
                     help="Skip retrieval; chat with the model directly.")
    ask.set_defaults(func=_cmd_ask)

    search = sub.add_parser("search", help="Retrieval only; prints matched chunks.")
    search.add_argument("query", help="Search query.")
    search.add_argument("-k", type=int, default=DEFAULT_K, help=f"top-k results (default {DEFAULT_K}).")
    search.set_defaults(func=_cmd_search)

    status = sub.add_parser("status", help="Probe both servers and show index size.")
    status.set_defaults(func=_cmd_status)

    # `doctor` is intercepted before module-level imports (see top of file) so
    # it works when src.backend imports fail. This entry only exists so the
    # subcommand shows up in `hermes --help`.
    doctor = sub.add_parser(
        "doctor",
        help="Self-diagnostic: host, build, models, agents, servers, index.",
    )
    doctor.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON instead of the report.")
    doctor.set_defaults(func=lambda _a: _run_doctor_late())

    repl = sub.add_parser("repl", help="Interactive multi-turn RAG chat (default if no subcommand).")
    repl.add_argument("--no-rag", action="store_true", help="start with RAG disabled.")
    repl.add_argument("-k", type=int, default=DEFAULT_K,
                      help=f"top-k chunks per turn (default {DEFAULT_K}).")
    repl.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                      help=f"max response tokens per turn (default {DEFAULT_MAX_TOKENS}).")
    repl.set_defaults(func=_cmd_repl)

    # Intercepted at the top of the module before heavy imports — see the
    # `if sys.argv[1] == "notify"` block. This entry exists only so the
    # subcommand shows up in `hermes --help`. Detailed flag definitions live
    # in src/notify.py to avoid drift from a duplicate definition here.
    sub.add_parser(
        "notify",
        help="Send a Telegram message, or run --setup / --check. "
             "(See `hermes notify --help` for flags.)",
        add_help=False,  # actual help comes from notify's own parser
    )

    # Help-only stubs: real flags live in src/wiki_cmd.py and src/lint_cmd.py
    # to avoid drift from a duplicate definition.
    sub.add_parser(
        "wiki",
        help="Initialize / inspect the wiki layer. (See `hermes wiki --help`.)",
        add_help=False,
    )
    sub.add_parser(
        "lint",
        help="Wiki health-check: orphans, stubs, stale, unused sources.",
        add_help=False,
    )

    # Intercepted at the top of the module (see the `digest` early-route).
    # Help-only stub; real flags live in src/digest.py.
    sub.add_parser(
        "digest",
        help="Build a daily digest of vault activity and file it in the wiki. "
             "(See `hermes digest --help`.)",
        add_help=False,
    )

    ingest = sub.add_parser(
        "ingest",
        help="Summarize a raw source into a wiki page (writes wiki/sources/<stem>.md).",
    )
    ingest.add_argument("path", nargs="?", default=None,
                        help="Path to the source file.")
    ingest.add_argument("--url", default=None,
                        help="Fetch and summarize a web page instead of a local file.")
    ingest.add_argument("--force", action="store_true",
                        help="Overwrite an existing wiki page.")
    ingest.add_argument("--name", default=None,
                        help="Override the wiki page name.")
    ingest.add_argument("--max-tokens", type=int, default=1024,
                        help="Cap on chat server response.")
    ingest.set_defaults(func=_cmd_ingest)

    ingest_links = sub.add_parser(
        "ingest-links",
        help="Fetch + summarize every URL in a text file (one URL per line).",
    )
    ingest_links.add_argument("file", help="Path to a text file of URLs (one per line; "
                                           "blank lines and # comments ignored).")
    ingest_links.add_argument("--force", action="store_true",
                              help="Re-summarize URLs whose wiki page already exists.")
    ingest_links.add_argument("--max-tokens", type=int, default=1024,
                              help="Cap on chat server response per page.")
    ingest_links.add_argument("--no-index", action="store_true",
                              help="Skip auto-indexing the new pages at the end.")
    ingest_links.set_defaults(func=_cmd_ingest_links)

    index = sub.add_parser(
        "index",
        help="Backfill / GC the vault index (one-shot, complementary to the live watcher).",
    )
    index.add_argument("--backfill", action="store_true",
                       help="Index files not yet in the index.")
    index.add_argument("--force", action="store_true",
                       help="With --backfill: re-embed every file regardless of state.")
    index.add_argument("--gc", action="store_true",
                       help="Remove index rows whose source file is gone or now excluded.")
    index.add_argument("--migrate", action="store_true",
                       help="Upgrade an old index to the metadata-rich schema (re-indexes stale rows).")
    index.add_argument("--gc-chats", action="store_true",
                       help="Prune archived REPL conversations older than --older-than days.")
    index.add_argument("--older-than", type=int, default=None, metavar="DAYS",
                       help="With --gc-chats: age threshold in days (default 90).")
    index.add_argument("--dry-run", action="store_true",
                       help="With --gc / --gc-chats: show what would be removed without removing.")
    index.add_argument("--limit", type=int, default=None,
                       help="Process at most N files.")
    index.set_defaults(func=_cmd_index)

    return p


def _run_doctor_late() -> int:
    # Reachable only if the user passes `doctor` after some other command has
    # already triggered module-top imports (e.g. via Python `-c`). The normal
    # `hermes doctor` path exits inside the early-route block at the top of
    # this module and never gets here.
    from src import doctor as _doctor
    argv = sys.argv[2:] if len(sys.argv) > 2 else []
    return _doctor.run(argv)


def _cmd_ingest(args: argparse.Namespace) -> int:
    from src import ingest_cmd

    if args.url and args.path:
        print("hermes ingest: pass either a path or --url, not both.", file=sys.stderr)
        return 2
    if args.url:
        return _ingest_one_url(
            args.url, name=args.name, force=args.force, max_tokens=args.max_tokens,
        )
    if not args.path:
        print("hermes ingest: provide a file path or --url <url>.", file=sys.stderr)
        return 2

    flags: list[str] = [args.path]
    if args.force:
        flags.append("--force")
    if args.name:
        flags.extend(["--name", args.name])
    flags.extend(["--max-tokens", str(args.max_tokens)])
    return ingest_cmd.run(flags)


def _ingest_one_url(url: str, *, name: str | None, force: bool, max_tokens: int) -> int:
    from src import web, ingest_cmd, wiki

    paths = wiki.get_paths()
    if not wiki.is_initialized(paths):
        print(f"hermes ingest: wiki not initialized at {paths.root}", file=sys.stderr)
        print(f"               run: hermes wiki init", file=sys.stderr)
        return 2

    try:
        art = web.fetch_article(url)
    except web.WebError as exc:
        print(f"hermes ingest: {exc}", file=sys.stderr)
        return 1

    page_name = name or art.title or _name_from_url(url)
    extra = {"source-url": url, "ingested-via": "url"}
    if art.title:
        extra["source-title"] = art.title
    if art.date:
        extra["source-date"] = art.date

    print(f"hermes ingest: fetched {url} — summarizing (30-60s)...", file=sys.stderr)
    try:
        res = ingest_cmd.ingest_text(
            art.text, page_name=page_name, source_label=url,
            extra_frontmatter=extra, max_tokens=max_tokens, force=force,
        )
    except HermesError as exc:
        print(f"hermes ingest: chat server error: {exc}", file=sys.stderr)
        return 1

    if res.status == ingest_cmd.REFUSED_HANDWRITTEN:
        print(f"hermes ingest: refusing to overwrite hand-written file: {res.page_path}",
              file=sys.stderr)
        return 1
    if res.status == ingest_cmd.ALREADY_EXISTS:
        print(f"hermes ingest: {res.page_path.name} already exists "
              f"(use --force to re-summarize).", file=sys.stderr)
        return 0
    print(f"hermes ingest: wrote {res.page_path.name} — {res.summary}")
    return 0


def _name_from_url(url: str) -> str:
    """Derive a page name from a URL when there's no title: last non-empty
    path segment, else the host."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    segs = [s for s in parsed.path.split("/") if s]
    if segs:
        return segs[-1].rsplit(".", 1)[0]  # drop a trailing .html etc.
    return parsed.netloc or url


def _cmd_ingest_links(args: argparse.Namespace) -> int:
    from src import ingest_links_cmd
    flags = [args.file, "--max-tokens", str(args.max_tokens)]
    if args.force:
        flags.append("--force")
    if args.no_index:
        flags.append("--no-index")
    return ingest_links_cmd.run(flags)


def _cmd_index(args: argparse.Namespace) -> int:
    from src import index_cmd
    flags: list[str] = []
    if args.backfill:
        flags.append("--backfill")
    if args.force:
        flags.append("--force")
    if args.gc:
        flags.append("--gc")
    if args.migrate:
        flags.append("--migrate")
    if args.gc_chats:
        flags.append("--gc-chats")
    if args.older_than is not None:
        flags.extend(["--older-than", str(args.older_than)])
    if args.dry_run:
        flags.append("--dry-run")
    if args.limit is not None:
        flags.extend(["--limit", str(args.limit)])
    return index_cmd.run(flags)


def _cmd_repl(args: argparse.Namespace) -> int:
    from src import repl as _repl
    session = _repl.ChatSession(
        rag_enabled=not args.no_rag,
        k=args.k,
        max_tokens=args.max_tokens,
        context_window=_repl._read_context_window_default(),
    )
    _repl._setup_readline()
    try:
        return asyncio.run(_repl._async_run(session))
    finally:
        _repl._save_readline_history()


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # Bare `hermes` (no subcommand) → drop into REPL with defaults. We don't
    # mark `cmd` required so we can implement this fall-through cleanly.
    if args.cmd is None:
        from src import repl as _repl
        return _repl.run([])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
