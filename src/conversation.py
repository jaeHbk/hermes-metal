"""Cross-session conversation memory (Phase E).

The wiki *is* the conversation memory: a REPL session worth keeping is written
as a managed page under ``wiki/conversations/<timestamp>.md``. Because that
lives inside the vault, the watcher indexes it like any other note, so a past
chat becomes retrievable on a future turn — and the daily digest can cite it
the same way it cites notes. No separate LanceDB table, no new index schema.

Two ways a conversation lands in the wiki:

* **Explicit** — the user runs ``/file <name>`` in the REPL, which promotes a
  single answer into ``topics/`` (that path already exists in repl.py).
* **Automatic (this module)** — on REPL exit, if the session was substantial
  (≥ ``MIN_TURNS`` exchanges and ≥ ``MIN_CHARS`` of content), the whole
  transcript is archived. This is opt-in via ``HERMES_REPL_ARCHIVE=1`` so a
  user who doesn't want every chat persisted isn't surprised.

Archives grow monotonically, so ``hermes index --gc-chats --older-than N``
(wired in index_cmd) prunes old ones — built in from day one, per the Phase E
risk note, so the directory can't metastasize.

Stdlib only.
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src import wiki


# Archiving is opt-in: persisting every REPL session is a privacy/space choice
# the user should make deliberately.
def archive_enabled() -> bool:
    return os.environ.get("HERMES_REPL_ARCHIVE", "").strip().lower() in ("1", "true", "yes", "on")


# Thresholds below which a session isn't worth archiving (a one-line "hi" test
# shouldn't become a permanent wiki page).
MIN_TURNS = 2          # at least 2 user+assistant exchanges
MIN_CHARS = 400        # total content across the transcript

_FILENAME_RE = re.compile(r"^conversation-(\d{8}T\d{6}Z)\.md$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def should_archive(history: list[dict[str, str]]) -> bool:
    """True iff the session clears the substance bar.

    Counts assistant turns as the "exchange" unit (a user turn with no answer
    isn't a real exchange). Empty / trivial sessions return False.
    """
    if not history:
        return False
    assistant_turns = sum(1 for m in history if m.get("role") == "assistant" and m.get("content", "").strip())
    if assistant_turns < MIN_TURNS:
        return False
    total = sum(len(m.get("content", "")) for m in history)
    return total >= MIN_CHARS


def _render_transcript(history: list[dict[str, str]]) -> str:
    """Render history as a readable markdown transcript (bold role labels)."""
    blocks: list[str] = []
    for m in history:
        role = m.get("role", "?")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            # System prompt isn't conversation content; skip it from the archive.
            continue
        label = {"user": "**You:**", "assistant": "**hermes:**"}.get(role, f"**{role}:**")
        blocks.append(f"{label}\n\n{content}")
    return "\n\n---\n\n".join(blocks)


def _first_user_question(history: list[dict[str, str]]) -> str:
    for m in history:
        if m.get("role") == "user" and m.get("content", "").strip():
            return m["content"].strip().replace("\n", " ")[:120]
    return "(conversation)"


def archive_session(
    history: list[dict[str, str]],
    *,
    paths: Optional[wiki.WikiPaths] = None,
    now: Optional[datetime] = None,
) -> Optional[Path]:
    """Write a substantial session to ``wiki/conversations/`` and index it.

    Returns the page path on success, ``None`` if the session didn't clear the
    bar or the wiki isn't initialized. Never raises on a normal write — REPL
    exit must not be blocked by an archive failure (callers wrap defensively
    too). Mirrors digest/ingest: page → index → log with rollback.
    """
    if not should_archive(history):
        return None
    p = paths or wiki.get_paths()
    if not wiki.is_initialized(p):
        return None
    p.conversations_dir.mkdir(parents=True, exist_ok=True)

    ts = (now or _now()).strftime("%Y%m%dT%H%M%SZ")
    stem = f"conversation-{ts}"
    page_path = p.conversations_dir / f"{stem}.md"
    # Extremely unlikely collision (same-second exit); bail rather than clobber.
    if page_path.exists():
        return None

    body = _render_transcript(history)
    if not body.strip():
        return None
    first_q = _first_user_question(history)
    page = wiki.Page(
        title=f"Conversation — {ts}",
        body=body,
        frontmatter={"filed-from": "repl-archive", "opening-question": first_q},
    )
    wiki.write_page(page_path, page)
    # Page and index must agree: if the index update fails, roll the page back
    # so we don't leave an unreferenced file. The log append, by contrast, is a
    # non-load-bearing audit trail — once the index row is committed, a log
    # failure must NOT trigger a page rollback (that would strand a dangling
    # index row pointing at a now-deleted page). So the log write is separate
    # and best-effort.
    try:
        wiki.update_index_row(p, "Conversations", stem, first_q)
    except Exception:
        try:
            page_path.unlink()
        except OSError:
            pass
        raise
    try:
        wiki.append_log(p, "archive", stem, detail=f"REPL session, {len(history)} messages")
    except Exception:  # noqa: BLE001 — audit log is best-effort; page+index already consistent
        pass
    return page_path


def gc_conversations(
    paths: wiki.WikiPaths,
    *,
    older_than_days: int,
    now: Optional[datetime] = None,
    dry_run: bool = False,
) -> list[Path]:
    """Delete archived conversations older than ``older_than_days``.

    Age is taken from the timestamp in the filename (the archive's own clock),
    not mtime, so re-indexing or a filesystem copy doesn't reset the age.
    Returns the list of removed (or, in dry-run, would-be-removed) paths.
    Only touches LLM-managed ``conversation-*.md`` files — a hand-dropped file
    in the directory is left alone.
    """
    if not paths.conversations_dir.is_dir():
        return []
    cutoff = now or _now()
    removed: list[Path] = []
    for f in sorted(paths.conversations_dir.glob("conversation-*.md")):
        m = _FILENAME_RE.match(f.name)
        if not m:
            continue
        try:
            stamp = datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        age_days = (cutoff - stamp).total_seconds() / 86400.0
        if age_days < older_than_days:
            continue
        if not wiki.is_managed(f):
            continue
        removed.append(f)
        if not dry_run:
            try:
                f.unlink()
            except OSError:
                removed.pop()  # couldn't remove; don't claim we did
    return removed
