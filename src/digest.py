"""`hermes digest` — scheduled daily summary of vault activity (Phase D).

Walks the vault for notes touched in a date window (default: yesterday),
builds a structured digest, writes it as a durable wiki page at
``wiki/digests/YYYY-MM-DD.md`` (so digests compound over time and show up in
Obsidian's graph view), and — only when the user has explicitly opted in —
pushes a headline + the full markdown to Telegram.

The digest has four sections:

* **Activity** — which notes changed, mechanically derived from mtime. Always
  present.
* **Learnings** — an LLM-synthesized conceptual extract of the day's notes.
  Best-effort: if the chat server is down, this section says so and the rest
  of the digest still ships.
* **Practice questions** — generated *only* when the day's notes look like
  class/course material (a ``#class/*`` tag or a course-like path). Gated so a
  digest of ordinary daily notes isn't padded with quiz questions.
* **Open questions** — ``TODO``s, unchecked ``- [ ]`` boxes, and lines ending
  in ``?``, surfaced mechanically from the day's notes.

Design posture:

* **Privacy is default-off.** Writing the local wiki page is always safe.
  Pushing summarized vault content off the machine to Telegram is a real
  choice, so it happens only when ``HERMES_DIGEST_PUSH`` is truthy AND a bot
  is configured. ``hermes doctor`` warns when push is on.
* **Idempotent.** The wiki page is the state: if a digest for the date already
  exists, we don't regenerate or re-send unless ``--force``.
* **Degrades gracefully.** Mechanical sections are pure stdlib; the LLM
  section is wrapped so a dead chat server never aborts the digest.
* **Deterministic under test.** ``now`` and the chat client are injectable.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from src import wiki


# ----------------------------------------------------------------- config


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def push_enabled() -> bool:
    """True iff the user opted into pushing digests to Telegram.

    Default-off privacy gate: requires the explicit env opt-in AND a
    configured bot. Either alone is not enough.
    """
    if not _env_truthy("HERMES_DIGEST_PUSH"):
        return False
    try:
        from src import notify
        return notify.is_configured()
    except Exception:  # noqa: BLE001 — notify import/Syntax issues must not crash digest
        return False


# Cap on how much of each note we feed the LLM, and how many notes, so a busy
# day doesn't blow the chat context. Mechanical sections see the full files.
_LLM_PER_NOTE_CHARS = 4000
_LLM_MAX_NOTES = 12

# Patterns that mark a note as class/course material (enables practice
# questions). Tag check is "starts with 'class'"; path check is a path segment.
_CLASS_TAG_PREFIXES = ("class", "course", "lecture")
_CLASS_PATH_RE = re.compile(r"(?:^|/)(class|classes|course|courses|lecture|lectures)(?:/|$)", re.IGNORECASE)

# Open-question harvesting.
_TODO_RE = re.compile(r"\b(TODO|FIXME|TBD)\b[:\s]\s*(.+)", re.IGNORECASE)
_UNCHECKED_BOX_RE = re.compile(r"^\s*[-*]\s*\[ \]\s+(.+\S)\s*$")
_QUESTION_LINE_RE = re.compile(r"^\s*(.{8,}\?)\s*$")


# ----------------------------------------------------------------- types


@dataclass
class DigestNote:
    path: Path
    rel: str          # path relative to vault root (display + citation)
    mtime: float
    text: str         # frontmatter-stripped body
    tags: list[str]


@dataclass
class DigestResult:
    date_iso: str
    weekday: str
    notes: list[DigestNote]
    learnings: str
    practice_questions: list[str]
    open_questions: list[str]
    llm_ok: bool
    is_class: bool

    @property
    def headline(self) -> str:
        n = len(self.notes)
        bits = [f"{n} note{'s' if n != 1 else ''}"]
        if self.open_questions:
            bits.append(f"{len(self.open_questions)} open question{'s' if len(self.open_questions) != 1 else ''}")
        if self.practice_questions:
            bits.append(f"{len(self.practice_questions)} practice Q")
        return f"hermes digest {self.date_iso}: " + ", ".join(bits) + "."

    def to_markdown(self) -> str:
        return _render_markdown(self)


# ------------------------------------------------------------- date window


def resolve_date(date_arg: Optional[str], *, now: Optional[datetime] = None) -> datetime:
    """Resolve the digest's target day.

    No ``--date`` → yesterday (the canonical "summarize what I did yesterday"
    use). An explicit ``YYYY-MM-DD`` is parsed in local time. ``now`` is
    injectable for tests.
    """
    if now is None:
        now = datetime.now().astimezone()
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    if date_arg:
        d = datetime.strptime(date_arg, "%Y-%m-%d")
        return d.replace(tzinfo=now.tzinfo)
    return now - timedelta(days=1)


def _day_window(day: datetime) -> tuple[float, float]:
    start = datetime.combine(day.date(), time.min, tzinfo=day.tzinfo)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


# ---------------------------------------------------------- vault scanning


def collect_notes(
    vault_path: Path,
    day: datetime,
    *,
    config_path: Optional[Path] = None,
) -> list[DigestNote]:
    """Return notes whose mtime falls within ``day`` (the wiki subtree excluded).

    Uses the shared vault filter so the digest sees the same files the index
    does. Wiki pages are skipped — a digest summarizing prior digests would be
    a feedback loop.
    """
    from src.backend.vault_filter import build_filter, iter_vault_files
    from src.backend.indexer import extract_tags, _FRONTMATTER_RE

    start, end = _day_window(day)
    vfilter = build_filter(vault_path, config_path=config_path)
    wiki_root = wiki.get_paths(vault_path).root.resolve()

    out: list[DigestNote] = []
    for p in iter_vault_files(vault_path, vfilter):
        rp = p.resolve()
        # Skip the wiki subtree (digests/sources/topics shouldn't feed digests).
        try:
            rp.relative_to(wiki_root)
            continue
        except ValueError:
            pass
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if not (start <= mtime < end):
            continue
        try:
            raw = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        body = _FRONTMATTER_RE.sub("", raw, count=1).lstrip("\n")
        try:
            rel = str(rp.relative_to(vault_path.resolve()))
        except ValueError:
            rel = p.name
        out.append(DigestNote(path=rp, rel=rel, mtime=mtime, text=body, tags=extract_tags(raw)))
    out.sort(key=lambda n: n.mtime)
    return out


def is_class_material(notes: list[DigestNote]) -> bool:
    for n in notes:
        for t in n.tags:
            tl = t.lower()
            if any(tl == pfx or tl.startswith(pfx + "/") for pfx in _CLASS_TAG_PREFIXES):
                return True
        if _CLASS_PATH_RE.search(n.rel):
            return True
    return False


def extract_open_questions(notes: list[DigestNote], *, limit: int = 12) -> list[str]:
    """Harvest TODOs, unchecked checkboxes, and ``?``-terminated lines.

    De-duplicated, source-attributed, capped. Skips fenced code blocks so a
    code comment ``# TODO`` in an example isn't surfaced as a real question.
    """
    seen: set[str] = set()
    out: list[str] = []
    for note in notes:
        in_fence = False
        for line in note.text.splitlines():
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            found: Optional[str] = None
            m = _TODO_RE.search(line)
            if m:
                found = f"{m.group(1).upper()}: {m.group(2).strip()}"
            if found is None:
                m = _UNCHECKED_BOX_RE.match(line)
                if m:
                    found = m.group(1).strip()
            if found is None:
                m = _QUESTION_LINE_RE.match(line)
                if m and not line.lstrip().startswith("#"):
                    found = m.group(1).strip()
            if found:
                key = found.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(f"{found}  [{Path(note.rel).name}]")
                    if len(out) >= limit:
                        return out
    return out


# ------------------------------------------------------------- LLM section


_LEARNINGS_SYSTEM = """\
You are hermes-metal's daily-digest worker. You are given the notes a user
touched on a single day. Write a concise "Learnings" section: 2-5 sentences
distilling what the day's notes are actually about and what was figured out
or decided. Be specific and factual; cite note filenames in square brackets
like [auth.md] when a point comes from one. Output ONLY the prose, no header,
no preamble.
"""

_PRACTICE_SYSTEM = """\
You are hermes-metal's study-aid worker. The user touched class/course notes
today. Produce 3-5 short practice questions that test understanding of the
material in these notes. Output ONLY a plain numbered list (1., 2., ...), one
question per line, no header and no answers.
"""


def _day_corpus(notes: list[DigestNote]) -> str:
    blocks: list[str] = []
    for n in notes[:_LLM_MAX_NOTES]:
        body = n.text[:_LLM_PER_NOTE_CHARS]
        if len(n.text) > _LLM_PER_NOTE_CHARS:
            body += "\n[...truncated...]"
        blocks.append(f"### {n.rel}\n{body}")
    return "\n\n".join(blocks)


def _parse_numbered(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^\s*\d+[.)]\s+(.*\S)\s*$", line)
        if m:
            out.append(m.group(1).strip())
    return out


# Chat callable signature: (messages, max_tokens) -> str. Default uses the real
# HermesClient; tests inject a stub. Returns None on any failure (degraded).
ChatFn = Callable[[list[dict[str, Any]], int], str]


def _default_chat_fn() -> ChatFn:
    from src.server.client import HermesClient, HermesError

    def _chat(messages: list[dict[str, Any]], max_tokens: int) -> str:
        client = HermesClient()
        try:
            return client.chat_sync(messages, max_tokens=max_tokens, temperature=0.3)
        except HermesError as exc:  # surfaced to caller as a degraded section
            raise RuntimeError(str(exc)) from exc

    return _chat


def synthesize(
    notes: list[DigestNote],
    *,
    is_class: bool,
    chat_fn: Optional[ChatFn] = None,
) -> tuple[str, list[str], bool]:
    """Return (learnings_prose, practice_questions, llm_ok).

    Best-effort: any chat failure yields a placeholder learnings string,
    empty practice list, and llm_ok=False so the digest still ships.
    """
    if not notes:
        return ("No notes were touched on this day.", [], True)
    fn = chat_fn or _default_chat_fn()
    corpus = _day_corpus(notes)
    try:
        learnings = fn(
            [
                {"role": "system", "content": _LEARNINGS_SYSTEM},
                {"role": "user", "content": f"The day's notes:\n\n{corpus}\n\nWrite the Learnings section."},
            ],
            512,
        ).strip()
    except Exception as exc:  # noqa: BLE001 — degrade, don't abort
        return (
            f"_(chat server unavailable — mechanical digest only: {exc})_",
            [],
            False,
        )

    practice: list[str] = []
    if is_class:
        try:
            raw = fn(
                [
                    {"role": "system", "content": _PRACTICE_SYSTEM},
                    {"role": "user", "content": f"The class notes:\n\n{corpus}\n\nWrite the practice questions."},
                ],
                512,
            )
            practice = _parse_numbered(raw)
        except Exception:  # noqa: BLE001 — practice is optional; keep learnings
            practice = []
    return (learnings, practice, True)


# --------------------------------------------------------------- rendering


def _render_markdown(r: DigestResult) -> str:
    lines: list[str] = []
    lines.append("## Activity")
    if r.notes:
        for n in r.notes:
            stamp = datetime.fromtimestamp(n.mtime).strftime("%H:%M")
            lines.append(f"- [{n.rel}] — last edited {stamp}")
    else:
        lines.append("_No notes were touched on this day._")
    lines.append("")
    lines.append("## Learnings")
    lines.append(r.learnings.strip() or "_(nothing to synthesize)_")
    lines.append("")
    if r.is_class:
        lines.append("## Practice questions")
        if r.practice_questions:
            for i, q in enumerate(r.practice_questions, 1):
                lines.append(f"{i}. {q}")
        else:
            lines.append("_(none generated)_")
        lines.append("")
    lines.append("## Open questions")
    if r.open_questions:
        for q in r.open_questions:
            lines.append(f"- {q}")
    else:
        lines.append("_None surfaced from the day's notes._")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------- build/run


def build_digest(
    vault_path: Path,
    day: datetime,
    *,
    chat_fn: Optional[ChatFn] = None,
    config_path: Optional[Path] = None,
) -> DigestResult:
    notes = collect_notes(vault_path, day, config_path=config_path)
    is_class = is_class_material(notes)
    learnings, practice, llm_ok = synthesize(notes, is_class=is_class, chat_fn=chat_fn)
    open_qs = extract_open_questions(notes)
    return DigestResult(
        date_iso=day.strftime("%Y-%m-%d"),
        weekday=day.strftime("%A"),
        notes=notes,
        learnings=learnings,
        practice_questions=practice,
        open_questions=open_qs,
        llm_ok=llm_ok,
        is_class=is_class,
    )


def _digest_page_path(paths: wiki.WikiPaths, date_iso: str) -> Path:
    return paths.digests_dir / f"{date_iso}.md"


def write_digest_page(paths: wiki.WikiPaths, result: DigestResult) -> Path:
    """Write the digest as a managed wiki page and update index + log.

    Three-step write with rollback, mirroring ``ingest_cmd``: if index/log
    update fails after the page lands, the page is unlinked so the wiki stays
    internally consistent and a re-run isn't blocked by a stranded file.

    The hand-written guard is re-checked HERE, immediately before the write,
    not only in ``run()``. ``run()``'s check happens before a multi-second LLM
    synthesis call, so a user could create a hand-authored file at the digest
    path during that window; re-checking at write time closes that gap. (A
    sub-millisecond TOCTOU still exists between this check and os.replace, but
    that requires a write landing in that exact instant — acceptable for a
    single-user local tool, and far tighter than the seconds-long LLM window.)
    """
    page_path = _digest_page_path(paths, result.date_iso)
    if page_path.exists() and not wiki.is_managed(page_path):
        raise RuntimeError(f"refusing to overwrite hand-written {page_path}")
    page = wiki.Page(
        title=f"Digest — {result.date_iso} ({result.weekday})",
        body=result.to_markdown(),
        frontmatter={
            "digest-date": result.date_iso,
            "note-count": str(len(result.notes)),
        },
    )
    wiki.write_page(page_path, page)
    try:
        wiki.update_index_row(
            paths, "Digests", result.date_iso,
            f"{len(result.notes)} note(s), {len(result.open_questions)} open question(s)",
        )
        wiki.append_log(
            paths, "digest", result.date_iso,
            detail=f"Notes: {len(result.notes)}; class={result.is_class}; llm_ok={result.llm_ok}",
        )
    except Exception:
        try:
            page_path.unlink()
        except OSError:
            pass
        raise
    return page_path


def _push(result: DigestResult, page_path: Path) -> bool:
    """Push headline + full digest document to Telegram. Returns True on send."""
    from src import notify
    notify.send(result.headline + f"\n\nFull digest: {page_path.name}")
    notify.send_document(
        f"digest-{result.date_iso}.md",
        result.to_markdown().encode("utf-8"),
        caption=result.headline,
    )
    return True


def run(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="hermes digest",
        description="Build a daily digest of vault activity; file it in the wiki and optionally push it.",
    )
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                   help="Day to digest (default: yesterday).")
    p.add_argument("--dry-run", action="store_true",
                   help="Build and print the digest; do not write the wiki page or push.")
    p.add_argument("--no-push", action="store_true",
                   help="Write the wiki page but never push to Telegram (overrides HERMES_DIGEST_PUSH).")
    p.add_argument("--force", action="store_true",
                   help="Regenerate even if a digest for the date already exists; re-send if push is on.")
    args = p.parse_args(argv)

    raw_vault = os.environ.get("HERMES_VAULT_PATH")
    if not raw_vault:
        print("hermes digest: HERMES_VAULT_PATH not set.", file=sys.stderr)
        print("               export HERMES_VAULT_PATH=/path/to/vault and try again.", file=sys.stderr)
        return 2
    vault_path = Path(raw_vault).expanduser().resolve()
    if not vault_path.is_dir():
        print(f"hermes digest: vault path {vault_path} does not exist.", file=sys.stderr)
        return 2

    try:
        day = resolve_date(args.date)
    except ValueError:
        print(f"hermes digest: bad --date {args.date!r} (expected YYYY-MM-DD).", file=sys.stderr)
        return 2
    date_iso = day.strftime("%Y-%m-%d")

    paths = wiki.get_paths(vault_path)
    # The digest lives in the wiki; bootstrap it if absent (idempotent — only
    # creates missing structure, never overwrites). Skip the disk touch on a
    # dry run so a preview is side-effect-free.
    if not args.dry_run and not wiki.is_initialized(paths):
        print(f"hermes digest: initializing wiki at {paths.root}", file=sys.stderr)
        wiki.init_wiki(paths)

    page_path = _digest_page_path(paths, date_iso)
    if page_path.exists() and not args.force and not args.dry_run:
        # Idempotency: a digest already exists for this date. Don't regenerate
        # or re-send. This is what makes a daily LaunchAgent safe to fire
        # repeatedly (e.g. after a missed run / catch-up).
        if page_path.exists() and not wiki.is_managed(page_path):
            print(f"hermes digest: refusing to touch hand-written {page_path}", file=sys.stderr)
            return 1
        print(f"hermes digest: {date_iso} already exists ({page_path}). Use --force to regenerate.")
        return 0
    if page_path.exists() and not wiki.is_managed(page_path):
        # --force can't override the hand-written guard.
        print(f"hermes digest: refusing to overwrite hand-written {page_path}", file=sys.stderr)
        return 1

    print(f"hermes digest: building digest for {date_iso} ({day.strftime('%A')})...", file=sys.stderr)
    result = build_digest(vault_path, day)
    if not result.notes:
        print(f"hermes digest: no notes touched on {date_iso}; nothing to summarize.", file=sys.stderr)
        # Still emit an (empty) digest on --force so a daily cadence has a page
        # for every day; otherwise skip to avoid noise.
        if not args.force:
            return 0

    md = result.to_markdown()
    if args.dry_run:
        print(f"# Digest — {date_iso} ({day.strftime('%A')})\n")
        print(md)
        print(f"\n[dry-run] would write {page_path}", file=sys.stderr)
        want_push = push_enabled() and not args.no_push
        print(f"[dry-run] push: {'YES' if want_push else 'no'}", file=sys.stderr)
        return 0

    try:
        written = write_digest_page(paths, result)
    except RuntimeError as exc:
        # Hand-written file appeared at the digest path during synthesis.
        print(f"hermes digest: {exc}", file=sys.stderr)
        return 1
    print(f"hermes digest: wrote {written.relative_to(paths.root.parent)}")
    if not result.llm_ok:
        print("hermes digest: note — chat server was unavailable; digest is mechanical-only.",
              file=sys.stderr)

    # Push only with explicit opt-in and a configured bot.
    if args.no_push:
        return 0
    if push_enabled():
        try:
            _push(result, written)
            print("hermes digest: pushed to Telegram.", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 — push failure must not fail the local write
            print(f"hermes digest: push failed (digest still saved locally): {exc}", file=sys.stderr)
            return 1
    else:
        print("hermes digest: push disabled (set HERMES_DIGEST_PUSH=1 and configure a bot to enable).",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
