"""hermes-metal interactive REPL.

`hermes repl` (or bare `hermes`) drops the user into a multi-turn chat that
keeps conversation history, re-runs RAG retrieval each turn, and streams
tokens as they arrive.

Design choices (reasoned, not arbitrary):

1. Plain-history storage. The user's actual words go into ``history``; the
   retrieved chunks for THIS turn are injected only into the message we send
   to the server, never persisted. With k=5 chunks of ~1 KB each, persisting
   them across N turns would exhaust an 8K-token context window inside ~5
   turns. Keeping history lean lets the model see the actual conversation,
   not stale retrieval debris.

2. Per-turn retrieval, lazy. Each user turn triggers a fresh retrieval from
   the LATEST question. We don't re-rank historical context — that's a known
   research-grade rabbit hole and the streaming UX is the bigger win.

3. Token-budget trim before, not after. Before sending, if the rough token
   count exceeds ``ctx * TRIM_RATIO`` we drop the oldest (user, assistant)
   pair. The system message and the current user message are always kept.

4. Persistent event loop. We run the whole session inside a single
   ``asyncio.Runner`` so the ``HermesClient``'s httpx connection pool stays
   warm across turns — TCP keepalive matters more than you'd think for an
   8B model where first-token latency dominates.

5. SIGINT mid-stream cancels the generation, not the whole REPL. A second
   Ctrl-C at the prompt (with empty input) exits cleanly. EOF (Ctrl-D) on
   an empty prompt also exits.

6. ``readline`` (stdlib) gives line editing and persistent ↑-history at
   ``~/.hermes/repl_history`` for free. macOS' system Python links readline
   against libedit which has subtly different rc syntax — we tolerate
   either by guarding the ``parse_and_bind`` call.

This module is intentionally self-contained; it only depends on the same
public surface (``HermesClient``, ``LanceVault``, ``embed``) that the
``ask``/``search`` subcommands already use.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from src.backend.database import LanceVault
from src.backend.indexer import embed as embed_query
from src.server.client import HermesClient, HermesError


# ---------------------------------------------------------------- constants


_BASE_SYSTEM_PROMPT = (
    "You are hermes-metal, the user's local 'second brain'. Answer using ONLY "
    "the provided notes when they are relevant; if the notes do not contain "
    "the answer, say so plainly and answer from general knowledge while "
    "flagging that the notes did not cover it. Cite sources by their filename "
    "in square brackets, e.g. [Welcome.md], when you draw on them. The "
    "conversation may span multiple turns; refer back to earlier turns when "
    "useful, but trust the freshly-retrieved notes over your memory of them."
)


def _build_system_prompt() -> str:
    """Compose the live system prompt: base + today's date + wiki schema.

    Reading the wiki schema is best-effort — if the wiki isn't
    initialized or the file has been deleted, we just skip that block.
    Date is always included because temporal awareness is cheap and
    the model otherwise has no idea when "today" is.
    """
    from datetime import datetime
    parts = [_BASE_SYSTEM_PROMPT]
    today = datetime.now().astimezone()
    parts.append(f"\nToday is {today.strftime('%Y-%m-%d (%A)')}.")
    try:
        from src import wiki as _wiki
        paths = _wiki.get_paths()
        if paths.schema.is_file():
            schema_text = paths.schema.read_text(encoding="utf-8", errors="replace").strip()
            if schema_text:
                parts.append(
                    "\nThe user has provided this guide to their vault structure "
                    "(read it, follow it):\n\n"
                    + schema_text
                )
    except Exception:  # noqa: BLE001 — best-effort; never let prompt build fail
        pass
    return "\n".join(parts)


# Kept for backwards compat with tests / external callers; computed once
# per session by `_async_run` via _build_system_prompt().
SYSTEM_PROMPT = _BASE_SYSTEM_PROMPT

DEFAULT_K = 5
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_CONTEXT_CHARS = 8000

# Drop oldest turn pairs when approx-token estimate of the outgoing payload
# exceeds this fraction of the model's context window. Conservative because
# the context budget also has to fit max_tokens of generation.
TRIM_RATIO = 0.6

HISTORY_FILE = Path.home() / ".hermes" / "repl_history"
HISTORY_LIMIT = 1000

# Cross-session KV cache name. Per-vault hashed so two vaults on the same
# host don't clobber each other's caches. Saved on slot 0; the streamer
# explicitly passes id_slot=0 so this stays correct if the chat server is
# ever launched with --parallel >1.
#
# Best-effort: a restore failure on first run, after upgrading llama.cpp, or
# after changing the chat server's --slot-save-path is fully tolerable —
# correctness is unaffected, the model just reprocesses the prefix.
KV_SLOT_ID = 0


def _kv_slot_name() -> str:
    """Per-vault slot name. Different vaults → different cache files.

    We salt with the resolved vault path so the same user running
    `hermes` against two different vaults doesn't have one session
    restore the other's cache. Falls back to a constant when no vault
    path is configured (the cache will be effectively shared, which is
    acceptable for that degenerate case).
    """
    import hashlib
    raw = os.environ.get("HERMES_VAULT_PATH") or "default"
    digest = hashlib.sha1(str(Path(raw).expanduser().resolve()).encode("utf-8")).hexdigest()[:12]
    return f"hermes-repl-{digest}"


BANNER = """\
hermes-metal REPL — local second-brain chat.
  /help              show commands       /clear         reset conversation
  /sources           show last hits      /save P        save transcript
  /load P            load transcript     /file NAME     file last answer in wiki
  /wiki              wiki status         /forget-cache  drop persisted KV slot
  /norag             disable RAG         /rag           re-enable RAG
  Ctrl-C             cancel generation / exit at empty prompt
  Ctrl-D             exit
"""


# -------------------------------------------------------------------- state


@dataclass
class ChatSession:
    """Conversation state for one REPL session.

    The list of (role, content) pairs is the raw history we replay to the
    server each turn. Retrieval results are kept ONLY for ``/sources`` — we
    do not feed them back into the next turn from history; we re-retrieve.
    """

    history: list[dict[str, str]] = field(default_factory=list)
    last_hits: list[dict[str, Any]] = field(default_factory=list)
    last_assistant: str = ""              # for /file <name>
    last_user: str = ""                   # for /file <name>'s "filed from" backref
    rag_enabled: bool = True
    k: int = DEFAULT_K
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    context_window: int = 8192            # default; refined from host_topology.env
    system_prompt: str = SYSTEM_PROMPT    # live prompt; overwritten in _async_run

    def add_user(self, content: str) -> None:
        self.history.append({"role": "user", "content": content})
        # NOTE: ``last_user`` is intentionally NOT updated here. It pairs
        # with ``last_assistant``, which only gets set on a successful
        # assistant turn. If we updated ``last_user`` here, a cancelled
        # follow-up question would corrupt the (Q, A) pair that ``/file``
        # reads — filing answer A1 with backref to question Q2.

    def add_assistant(self, content: str) -> None:
        # Skip empty assistant entries (e.g. user cancelled mid-stream before
        # any token arrived) so we don't pollute history with blank turns.
        if content.strip():
            self.history.append({"role": "assistant", "content": content})
            self.last_assistant = content
            # Now that the assistant turn is durable, snapshot the user
            # message it answered. history[-2] is the corresponding
            # ``user`` entry (we just appended assistant at -1).
            if len(self.history) >= 2 and self.history[-2]["role"] == "user":
                self.last_user = self.history[-2]["content"]

    def reset(self) -> None:
        self.history.clear()
        self.last_hits.clear()
        self.last_assistant = ""
        self.last_user = ""


# ---------------------------------------------------------------- helpers


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _trim_history(history: list[dict[str, str]], budget: int) -> list[dict[str, str]]:
    """Drop oldest (user, assistant) pairs until under ``budget`` tokens.

    The most recent user turn is never dropped — it's what we need to answer.
    Operates on a COPY so the live ``history`` keeps full provenance for the
    transcript / ``/save``.
    """
    out = list(history)
    used = sum(_approx_tokens(m["content"]) for m in out)
    # Drop from the front in pairs so user/assistant alternation stays clean.
    # Stop as soon as we'd be left with fewer than 2 messages — the current
    # user message plus at most one prior assistant turn.
    while used > budget and len(out) > 2:
        # Drop oldest two messages (user, assistant) regardless of role; if
        # the oldest is an assistant orphan, drop just it.
        if len(out) >= 2 and out[0]["role"] == "user" and out[1]["role"] == "assistant":
            removed = _approx_tokens(out[0]["content"]) + _approx_tokens(out[1]["content"])
            del out[:2]
        else:
            removed = _approx_tokens(out[0]["content"])
            del out[:1]
        used -= removed
    return out


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


def _resolve_db_path() -> Path:
    raw = os.environ.get("HERMES_LANCEDB_PATH") or str(
        Path(__file__).resolve().parents[1] / "storage" / "lancedb"
    )
    return Path(raw).expanduser().resolve()


def _resolve_embed_url() -> str:
    return os.environ.get("HERMES_EMBED_URL", "http://127.0.0.1:8081/v1/embeddings")


def _slot_save_dir() -> Path:
    """Where llama-server writes saved slot files.

    Matches the daemon plist's --slot-save-path argument, which is rendered
    from ``$WORKING_DIR/$SLOT_SAVE_PATH`` in engine_flags.env. We resolve
    the same way at REPL time so /forget-cache can unlink the on-disk file
    without shelling out.
    """
    repo = Path(__file__).resolve().parents[1]
    flags = repo / "config" / "engine_flags.env"
    rel = "storage/slots"  # default if env file is missing
    if flags.is_file():
        for line in flags.read_text(encoding="utf-8").splitlines():
            if line.startswith("SLOT_SAVE_PATH="):
                rel = line.split("=", 1)[1].strip().strip('"').strip("'") or rel
                break
    return (repo / rel).resolve()


def _delete_slot_file_on_disk(slot_name: str) -> bool:
    """Unlink the slot's persisted .bin file (or any common variant).

    llama-server appends an extension to the slot stem; the canonical form
    in current builds is ``<name>.bin``. Older builds used no extension or
    ``.gguf``-suffixed forms — try a small set, succeed on whichever exists.
    Returns True iff at least one file was removed.
    """
    base = _slot_save_dir() / slot_name
    candidates = [base.with_suffix(s) for s in (".bin", ".gguf", "")] + [base]
    removed = False
    for c in candidates:
        try:
            c.unlink()
            removed = True
        except FileNotFoundError:
            continue
        except OSError:
            # Permission denied or other; surface only if nothing succeeded.
            continue
    return removed


def _read_context_window_default() -> int:
    """Pull CONTEXT_TOKENS from host_topology.env if present.

    The chat server is started with this exact value (see Makefile +
    daemon.plist.template), so trimming against it matches what the model
    can actually accept.
    """
    path = Path(__file__).resolve().parents[1] / "config" / "host_topology.env"
    if not path.is_file():
        return 8192
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("CONTEXT_TOKENS="):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                return 8192
    return 8192


# -------------------------------------------------------------- retrieval


async def _retrieve(
    vault: LanceVault | None, query: str, *, k: int, embed_url: str
) -> list[dict[str, Any]]:
    """Fetch top-k chunks for ``query``. Returns [] on any soft failure.

    Soft failures (embed server down, vault empty, db missing) MUST return []
    rather than raise — the REPL should keep going with a no-RAG turn rather
    than dying because the embed agent crashed.

    `embed_query` is a synchronous function (it spins up its own httpx.Client
    and blocks). Calling it directly inside the asyncio loop would freeze
    SIGINT delivery and any other awaitables for up to its 60s read timeout.
    Punt it to a worker thread so the loop stays responsive — same for the
    LanceDB calls, which can also block on disk I/O.
    """
    if vault is None:
        return []
    try:
        if await asyncio.to_thread(vault.count) == 0:
            return []
    except Exception:  # noqa: BLE001 — vault may have been closed mid-session
        return []
    try:
        vecs = await asyncio.to_thread(
            embed_query, [query], embed_url=embed_url, task="search_query"
        )
        qvec = vecs[0]
    except (httpx.HTTPError, ValueError, IndexError):
        return []
    try:
        return await asyncio.to_thread(vault.search, qvec, k=k)
    except Exception:  # noqa: BLE001
        return []


# ----------------------------------------------------------------- streaming


async def _stream_assistant_reply(
    client: HermesClient,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
) -> tuple[str, bool]:
    """Stream a chat completion to stdout. Returns (full_text, was_cancelled).

    Uses ``client._chat`` (the chat-server httpx pool) directly rather than
    going through ``HermesClient.chat`` because that helper doesn't expose
    streaming. We talk to the OpenAI-compatible /v1/chat/completions route
    (same shape ``src/cli.py`` uses for ``hermes ask``).

    Cancellation contract: install a loop-scoped SIGINT handler ONLY for the
    duration of the stream. While streaming, ^C trips the local cancel event
    and we exit the loop with whatever tokens we already printed. Outside
    streaming (i.e. at the prompt), the handler is uninstalled and asyncio's
    default behavior re-raises KeyboardInterrupt, which the prompt loop
    catches as "exit cleanly."
    """
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        # Pin to slot 0 so slot_save/restore in this module always target the
        # same slot the conversation actually used. Default behavior picks the
        # least-recently-used idle slot; that's correct only because the chat
        # server defaults to --parallel 1. Be explicit anyway.
        "id_slot": KV_SLOT_ID,
    }
    cancel = asyncio.Event()
    loop = asyncio.get_running_loop()
    sig_installed = False
    try:
        loop.add_signal_handler(signal.SIGINT, cancel.set)
        sig_installed = True
    except (NotImplementedError, RuntimeError):
        # Windows / non-main thread: stream cancellation will surface as a
        # KeyboardInterrupt in the awaiting task instead. The caller's
        # try/except KeyboardInterrupt path handles that.
        pass

    full: list[str] = []
    cancelled = False
    try:
        async with client._chat.stream(  # noqa: SLF001 — intentional access
            "POST", "/v1/chat/completions", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if cancel.is_set():
                    cancelled = True
                    break
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
                    full.append(piece)
    except httpx.HTTPError as exc:
        sys.stdout.write("\n")
        sys.stdout.flush()
        raise HermesError(f"chat: {exc}") from exc
    finally:
        if sig_installed:
            try:
                loop.remove_signal_handler(signal.SIGINT)
            except (NotImplementedError, RuntimeError):
                pass
    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(full), cancelled


# --------------------------------------------------------------- one turn


async def _do_turn(
    session: ChatSession,
    user_input: str,
    *,
    client: HermesClient,
    vault: LanceVault | None,
    embed_url: str,
) -> None:
    """Run one user→assistant exchange. Mutates ``session``."""
    # Retrieve before persisting the user message: if the user cancels mid-
    # retrieval (we don't currently support that, but cheap to be safe), we
    # haven't dirtied history.
    hits: list[dict[str, Any]] = []
    if session.rag_enabled and vault is not None:
        hits = await _retrieve(vault, user_input, k=session.k, embed_url=embed_url)
        session.last_hits = hits
        if hits:
            print(f"  ↳ retrieved {len(hits)} chunk(s):", file=sys.stderr)
            for h in hits:
                src = Path(h["source_path"]).name
                d = h.get("_distance")
                d_str = f"  d={d:.3f}" if isinstance(d, (int, float)) else ""
                print(f"     - {src} #chunk{h['chunk_idx']}{d_str}", file=sys.stderr)

    # Inject retrieval into THIS turn's user message; persist only the plain
    # question into history so future turns aren't bloated with stale chunks.
    if hits:
        ctx_block = _format_context(hits, DEFAULT_MAX_CONTEXT_CHARS)
        decorated = (
            f"Notes from my vault that may be relevant:\n\n{ctx_block}\n\n"
            f"Question: {user_input}"
        )
    else:
        decorated = user_input

    session.add_user(user_input)

    # Build outgoing messages: system + trimmed history with the LAST user
    # message swapped for the decorated form. Trim BEFORE the swap so the
    # decorated message (which can be much larger than the plain one) is
    # always counted against the budget.
    budget = int(session.context_window * TRIM_RATIO)
    sent_history = _trim_history(session.history, budget)
    # Replace the trailing user message with the decorated version. We do
    # this on the trimmed copy so persisted history stays plain.
    if sent_history and sent_history[-1]["role"] == "user":
        sent_history = sent_history[:-1] + [{"role": "user", "content": decorated}]

    messages: list[dict[str, Any]] = [{"role": "system", "content": session.system_prompt}]
    messages.extend(sent_history)

    text = ""
    cancelled = False
    try:
        text, cancelled = await _stream_assistant_reply(
            client, messages,
            max_tokens=session.max_tokens,
            temperature=session.temperature,
        )
    except HermesError as exc:
        print(f"hermes: {exc}", file=sys.stderr)
        # Roll back the user turn so retrying doesn't accumulate ghosts.
        if session.history and session.history[-1]["role"] == "user":
            session.history.pop()
        return
    except KeyboardInterrupt:
        # Reached on platforms where the in-stream signal handler couldn't be
        # installed (e.g. Windows). Treat as user-initiated cancel: keep any
        # already-printed text so the partial answer survives, and re-raise
        # so the caller can decide whether to exit the REPL.
        cancelled = True
        sys.stdout.write("\n")
        sys.stdout.flush()
        # Do NOT consume here — let the outer loop see KeyboardInterrupt and
        # decide whether to exit (e.g. second ^C exits the REPL).
        # But first, persist whatever we got so /save still has it.
        # (When raised, `text` stayed at "" because we never returned from
        # the streamer. There's no partial content to record on this path.)
        raise
    finally:
        if cancelled:
            print("[cancelled]", file=sys.stderr)
        # Always persist whatever assistant text was streamed. If empty
        # (immediate cancel before any token), roll back the user turn to
        # keep history balanced.
        if text:
            session.add_assistant(text)
        elif session.history and session.history[-1]["role"] == "user":
            session.history.pop()


# ---------------------------------------------------------- slash commands


def _cmd_help() -> None:
    print(BANNER, end="")


def _cmd_sources(session: ChatSession) -> None:
    if not session.last_hits:
        print("(no retrievals yet)")
        return
    for i, h in enumerate(session.last_hits, 1):
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
        print(f"    {snippet}\n")


_TRANSCRIPT_HEADER_RE = re.compile(r"^###\s+(user|assistant|system)\s*$")


def _cmd_save(session: ChatSession, arg: str) -> None:
    if not arg:
        print("usage: /save <path>")
        return
    target = Path(arg).expanduser()
    try:
        with target.open("w", encoding="utf-8") as fh:
            for m in session.history:
                fh.write(f"### {m['role']}\n{m['content']}\n\n")
        print(f"saved {len(session.history)} messages to {target}")
    except OSError as exc:
        print(f"hermes: failed to save: {exc}", file=sys.stderr)


def _parse_transcript(text: str) -> list[dict[str, str]]:
    """Parse a `/save`-format transcript back into history entries.

    Format expected (matches ``_cmd_save`` writer):

        ### user
        <content...>
        <blank line>
        ### assistant
        <content...>
        <blank line>

    The header line is matched by ``^### (user|assistant|system)$`` so that
    a content body containing a different ``###`` heading (e.g. user pasted
    a Markdown subsection) does not split a message. Trailing blank lines on
    each message are stripped to undo the writer's `\\n\\n` separator.
    Returns [] on a file that contains no recognizable headers.
    """
    out: list[dict[str, str]] = []
    role: str | None = None
    buf: list[str] = []

    def flush() -> None:
        if role is None:
            return
        # The writer emits "{content}\n\n" — that final "\n\n" lands as
        # exactly ONE empty entry at the end of buf (the blank separator
        # line). Strip only that one entry, not all trailing blanks, so
        # user content that legitimately ends with a blank line round-trips.
        local = list(buf)
        if local and local[-1] == "":
            local.pop()
        body = "\n".join(local)
        if body or role:
            # Preserve empty-but-real messages too; the writer is the source
            # of truth for what counts as a "real" message.
            out.append({"role": role, "content": body})

    for line in text.splitlines():
        m = _TRANSCRIPT_HEADER_RE.match(line)
        if m:
            flush()
            role = m.group(1)
            buf = []
        elif role is not None:
            buf.append(line)
    flush()
    return out


def _cmd_file(session: ChatSession, arg: str) -> None:
    """`/file <name>` — promote the last assistant turn into a wiki topic page.

    The point of this command (per the LLM-Wiki article): explorations
    should compound back into the knowledge base, not vanish into chat
    history. We write to ``wiki/topics/<name>.md`` with the answer as
    the body and the user's question retained in the frontmatter as a
    backref.

    Refuses to overwrite an existing page without ``--force``-style
    explicit replacement; users can `/file <name> --force` to override
    (we accept "--force" as a trailing token in arg).
    """
    if not arg:
        print("usage: /file <name> [--force]")
        return
    if not session.last_assistant.strip():
        print("(no answer to file yet — ask something first)")
        return

    # Recognize `--force` anywhere in the args (not just trailing) so
    # `/file foo --force` and `/file --force foo` both work, and an
    # unknown flag like `--verbose` doesn't silently slip into the name.
    raw_parts = arg.split()
    force = False
    name_parts: list[str] = []
    for part in raw_parts:
        if part == "--force":
            force = True
        elif part.startswith("--"):
            print(f"unknown flag: {part}  (try /file <name> [--force])")
            return
        else:
            name_parts.append(part)
    name = " ".join(name_parts).strip()
    if not name:
        print("usage: /file <name> [--force]")
        return

    try:
        from src import wiki as _wiki
        from src.ingest_cmd import _slugify
    except ImportError as exc:
        print(f"hermes: wiki module unavailable: {exc}", file=sys.stderr)
        return

    paths = _wiki.get_paths()
    if not _wiki.is_initialized(paths):
        print(f"hermes: wiki not initialized at {paths.root}.", file=sys.stderr)
        print(f"        run: hermes wiki init", file=sys.stderr)
        return

    stem = _slugify(name)
    page_path = paths.topics_dir / f"{stem}.md"
    # Hand-written guard MUST run regardless of --force: we never clobber
    # a file the LLM didn't author. If the user really wants to replace
    # one, they delete it manually.
    if page_path.exists() and not _wiki.is_managed(page_path):
        print(f"hermes: refusing to overwrite hand-written {page_path}", file=sys.stderr)
        return
    if page_path.exists() and not force:
        print(f"hermes: {page_path.name} already exists "
              f"(/file {name} --force to overwrite).", file=sys.stderr)
        return

    page = _wiki.Page(
        title=stem,
        body=session.last_assistant.strip(),
        frontmatter={
            "filed-from": "repl",
            "question": session.last_user.strip().replace("\n", " ")[:200] if session.last_user else "",
        },
    )
    _wiki.write_page(page_path, page)
    summary = session.last_user.strip().replace("\n", " ")[:120] if session.last_user else "filed from REPL"
    _wiki.update_index_row(paths, "Topics", stem, summary)
    _wiki.append_log(paths, "file", stem, detail=f"From REPL question: {session.last_user[:200]!r}")
    print(f"(filed → {page_path.relative_to(paths.root.parent)})")


def _cmd_wiki(_session: ChatSession) -> None:
    """`/wiki` — quick wiki status without leaving the REPL."""
    try:
        from src import wiki as _wiki
    except ImportError as exc:
        print(f"hermes: wiki module unavailable: {exc}", file=sys.stderr)
        return
    paths = _wiki.get_paths()
    if not _wiki.is_initialized(paths):
        print(f"(wiki not initialized at {paths.root}; run: hermes wiki init)")
        return
    sources = list(paths.sources_dir.glob("*.md")) if paths.sources_dir.is_dir() else []
    topics = list(paths.topics_dir.glob("*.md")) if paths.topics_dir.is_dir() else []
    digests = list(paths.digests_dir.glob("*.md")) if paths.digests_dir.is_dir() else []
    print(f"wiki: {paths.root}")
    print(f"  sources={len(sources)}  topics={len(topics)}  digests={len(digests)}")


def _cmd_load(session: ChatSession, arg: str) -> None:
    if not arg:
        print("usage: /load <path>")
        return
    target = Path(arg).expanduser()
    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"hermes: failed to read {target}: {exc}", file=sys.stderr)
        return
    parsed = _parse_transcript(text)
    if not parsed:
        print(f"hermes: {target} has no recognizable transcript headers "
              "(expected `### user` / `### assistant` blocks).", file=sys.stderr)
        return
    session.history = parsed
    session.last_hits = []  # retrievals from the prior session don't apply
    n_user = sum(1 for m in parsed if m["role"] == "user")
    n_asst = sum(1 for m in parsed if m["role"] == "assistant")
    print(f"loaded {len(parsed)} messages ({n_user} user / {n_asst} assistant) from {target}")


def _handle_slash(line: str, session: ChatSession) -> str:
    """Handle a /command. Returns 'continue', 'quit', or 'send' (= treat as text)."""
    parts = line[1:].split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1] if len(parts) > 1 else ""
    if cmd in ("help", "?"):
        _cmd_help()
    elif cmd == "clear":
        session.reset()
        print("(history cleared)")
    elif cmd == "sources":
        _cmd_sources(session)
    elif cmd == "save":
        _cmd_save(session, arg)
    elif cmd == "load":
        _cmd_load(session, arg)
    elif cmd == "file":
        _cmd_file(session, arg)
    elif cmd == "wiki":
        _cmd_wiki(session)
    elif cmd == "norag":
        session.rag_enabled = False
        print("(RAG disabled — chat with the model only)")
    elif cmd == "rag":
        session.rag_enabled = True
        print("(RAG re-enabled)")
    elif cmd == "forget-cache":
        # Erase the persisted KV slot. Use when restoring stale state
        # appears to confuse the model (e.g. after upgrading llama.cpp).
        return "forget-cache"
    elif cmd in ("exit", "quit", "q"):
        return "quit"
    else:
        print(f"unknown command: /{cmd}  (try /help)")
    return "continue"


# --------------------------------------------------------------- readline


def _setup_readline() -> None:
    """Best-effort line editing + persistent history. Never fatal."""
    try:
        import readline  # noqa: WPS433 — stdlib, optional
    except ImportError:
        return
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if HISTORY_FILE.is_file():
        try:
            readline.read_history_file(str(HISTORY_FILE))
        except OSError:
            pass
    readline.set_history_length(HISTORY_LIMIT)
    # macOS system Python links readline against libedit; the rc syntax for
    # libedit differs ("bind ^I rl_complete" vs "tab: complete"). Use the
    # __doc__ marker upstream readline / libedit both expose. Note that on
    # libedit-Pythons __doc__ can be None, so coerce to a string FIRST —
    # `"libedit" in None` raises TypeError otherwise.
    doc = readline.__doc__ or ""
    try:
        if "libedit" in doc:
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
    except Exception:  # noqa: BLE001
        pass


def _save_readline_history() -> None:
    try:
        import readline  # noqa: WPS433
        readline.write_history_file(str(HISTORY_FILE))
    except (ImportError, OSError):
        pass


# ----------------------------------------------------------------- driver


async def _async_run(session: ChatSession) -> int:
    """Main async loop. One HermesClient for the whole session keeps the
    httpx connection pool warm across turns."""
    # Compose the live system prompt once at session start: base + today's
    # date + the user's wiki schema (if initialized). Cached on the
    # session so every turn sees the same prompt — keeps KV-slot prefix
    # matching stable.
    session.system_prompt = _build_system_prompt()

    db_path = _resolve_db_path()
    embed_url = _resolve_embed_url()

    vault: LanceVault | None
    if db_path.exists():
        try:
            vault = LanceVault(path=db_path)
        except Exception as exc:  # noqa: BLE001
            print(f"hermes: lancedb open failed ({exc}); continuing without RAG", file=sys.stderr)
            vault = None
    else:
        print(f"hermes: lancedb not found at {db_path}; continuing without RAG", file=sys.stderr)
        vault = None

    async with HermesClient() as client:
        # Sanity probe: report on the chat server up-front so the user
        # doesn't type a paragraph and only then learn the daemon is down.
        chat_healthy = False
        try:
            health = await client.health()
            status = health.get("status", "?")
            chat_healthy = status == "ok"
            if not chat_healthy:
                print(f"hermes: chat server status={status}; will retry per turn", file=sys.stderr)
        except HermesError as exc:
            print(f"hermes: chat server health probe failed: {exc}", file=sys.stderr)
            print("       (try: hermes doctor)", file=sys.stderr)

        # Best-effort KV-cache restore. The chat server is launched with
        # --slot-save-path so the prefix from a prior session is on disk.
        # If restore fails (file missing on first run, schema changed, etc.)
        # we silently skip — the conversation just reprocesses from scratch.
        kv_slot_name = _kv_slot_name()
        if chat_healthy:
            try:
                await client.slot_restore(KV_SLOT_ID, kv_slot_name)
            except HermesError:
                pass

        # Signal handling is per-turn: SIGINT during a stream is intercepted
        # inside `_stream_assistant_reply` to cancel just the generation. At
        # the prompt, asyncio's default behavior re-raises KeyboardInterrupt
        # into the awaiting task — we catch it below and exit cleanly.

        print(BANNER)
        empty_streak = 0
        try:
            while True:
                try:
                    raw = await asyncio.to_thread(_prompt, "hermes> ")
                except (KeyboardInterrupt, EOFError):
                    # ^C or ^D at the prompt: exit.
                    print()
                    break

                line = raw.strip()
                if not line:
                    empty_streak += 1
                    if empty_streak >= 2:
                        # Two empty Enters in a row exits — a soft "I'm done".
                        break
                    continue
                empty_streak = 0

                if line.startswith("/"):
                    action = _handle_slash(line, session)
                    if action == "quit":
                        break
                    if action == "forget-cache":
                        # Two-phase: ask the server to clear the in-memory
                        # slot, THEN unlink the on-disk file. Server's
                        # action=erase only clears the slot's KV cells; the
                        # persisted .bin file from a prior save survives and
                        # would be re-loaded on next start without this.
                        try:
                            await client.slot_erase(KV_SLOT_ID, kv_slot_name)
                        except HermesError as exc:
                            if not ("404" in str(exc) or "no such" in str(exc).lower()):
                                print(f"hermes: erase failed: {exc}", file=sys.stderr)
                        removed = _delete_slot_file_on_disk(kv_slot_name)
                        if removed:
                            print("(KV cache erased; on-disk file removed)")
                        else:
                            print("(KV cache erased; no on-disk file to remove)")
                    continue

                try:
                    await _do_turn(
                        session, line,
                        client=client,
                        vault=vault,
                        embed_url=embed_url,
                    )
                except KeyboardInterrupt:
                    # Reaches here only when the in-stream signal handler
                    # could not be installed (Windows, etc.). Treat as
                    # cancel-the-turn, NOT exit-the-REPL — same UX as the
                    # signal-handler-installed path.
                    print("\n[cancelled]", file=sys.stderr)
        finally:
            # Persist the KV cache so the next REPL boot skips re-prefilling
            # the system prompt + early history. Best-effort: a save failure
            # is silenced so REPL exit is never blocked. Wrap in
            # asyncio.wait_for so a dead chat server doesn't drag REPL exit
            # out for the full 60s httpx read timeout.
            if chat_healthy and session.history:
                try:
                    await asyncio.wait_for(
                        client.slot_save(KV_SLOT_ID, kv_slot_name),
                        timeout=2.0,
                    )
                except (HermesError, asyncio.TimeoutError):
                    pass
            if vault is not None:
                close = getattr(vault, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:  # noqa: BLE001
                        pass

    return 0


def _prompt(prompt_str: str) -> str:
    """input() in a worker thread, with the prompt visible on stderr-as-stdout
    even when stdout is a pipe (so `echo q | hermes repl` still shows the
    banner)."""
    return input(prompt_str)


# ----------------------------------------------------------------- entry


def run(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        prog="hermes repl",
        description="Interactive multi-turn RAG chat with your vault.",
    )
    p.add_argument("--no-rag", action="store_true",
                   help="start with RAG disabled (chat-only).")
    p.add_argument("-k", type=int, default=DEFAULT_K,
                   help=f"top-k chunks per turn (default {DEFAULT_K}).")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help=f"max response tokens per turn (default {DEFAULT_MAX_TOKENS}).")
    args = p.parse_args(argv)

    _setup_readline()

    session = ChatSession(
        rag_enabled=not args.no_rag,
        k=args.k,
        max_tokens=args.max_tokens,
        context_window=_read_context_window_default(),
    )

    try:
        return asyncio.run(_async_run(session))
    finally:
        _save_readline_history()


if __name__ == "__main__":
    sys.exit(run())
