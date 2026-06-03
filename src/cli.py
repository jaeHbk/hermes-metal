"""hermes-metal command-line interface.

Three subcommands:
    hermes ask "<question>"     RAG: embed -> LanceDB top-k -> chat (streaming).
    hermes search "<query>"     Retrieval only; prints matched chunks + sources.
    hermes status               Health probes both servers and prints index stats.

Resolves config from environment variables (HERMES_CHAT_URL, HERMES_EMBED_URL,
HERMES_LANCEDB_PATH) so it works whether the daemons are running locally or
the user has pointed it at a remote box.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from src.backend.database import LanceVault
from src.backend.indexer import embed as embed_query
from src.server.client import HermesClient, HermesError


DEFAULT_K = 5
DEFAULT_MAX_CONTEXT_CHARS = 8000
DEFAULT_MAX_TOKENS = 512

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
    return vault.search(qvec, k=k)


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
    sub = p.add_subparsers(dest="cmd", required=True)

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

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
