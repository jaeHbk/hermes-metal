"""Tests for cross-session conversation memory (Phase E).

Covers the substance gate, transcript rendering (system prompt excluded),
archive→index→log write, the filename-timestamp GC, and the hand-written
guard in GC. ``now`` injected; wiki pointed at tmp_path.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src import conversation, wiki


UTC = timezone.utc


@pytest.fixture
def wpaths(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_WIKI_PATH", str(tmp_path / "wiki"))
    p = wiki.get_paths()
    wiki.init_wiki(p)
    return p


def _long_session():
    big = "x " * 300  # well over MIN_CHARS
    return [
        {"role": "system", "content": "SYSTEM PROMPT SHOULD BE SKIPPED"},
        {"role": "user", "content": "First substantial question about the reranker design."},
        {"role": "assistant", "content": "A detailed answer. " + big},
        {"role": "user", "content": "A follow-up question."},
        {"role": "assistant", "content": "Another detailed answer."},
    ]


def test_should_archive_gate():
    assert conversation.should_archive(_long_session())
    # Too few turns.
    assert not conversation.should_archive(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
    )
    # Empty.
    assert not conversation.should_archive([])


def test_archive_writes_managed_page(wpaths):
    now = datetime(2026, 6, 7, 9, 0, tzinfo=UTC)
    p = conversation.archive_session(_long_session(), paths=wpaths, now=now)
    assert p is not None
    assert p.name == "conversation-20260607T090000Z.md"
    assert wiki.is_managed(p)
    body = p.read_text()
    assert "SYSTEM PROMPT SHOULD BE SKIPPED" not in body  # system excluded
    assert "**You:**" in body and "**hermes:**" in body


def test_archive_updates_index_and_log(wpaths):
    now = datetime(2026, 6, 7, 9, 0, tzinfo=UTC)
    conversation.archive_session(_long_session(), paths=wpaths, now=now)
    assert "## Conversations" in wpaths.index.read_text()
    assert "conversation-20260607T090000Z" in wpaths.index.read_text()
    assert "archive" in wpaths.log.read_text()


def test_archive_skips_trivial(wpaths):
    p = conversation.archive_session(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}],
        paths=wpaths,
    )
    assert p is None


def test_archive_noop_without_wiki(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_WIKI_PATH", str(tmp_path / "nowiki"))
    p = wiki.get_paths()
    # Not initialized → archive returns None rather than creating junk.
    assert conversation.archive_session(_long_session(), paths=p) is None


def test_log_failure_does_not_strand_index_row(wpaths, monkeypatch):
    # Regression (adversarial review): if append_log fails AFTER the index row
    # is committed, the page must survive (best-effort log) so the index row
    # isn't left dangling at a deleted page.
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(wiki, "append_log", boom)
    now = datetime(2026, 6, 7, 9, 0, tzinfo=UTC)
    p = conversation.archive_session(_long_session(), paths=wpaths, now=now)
    assert p is not None and p.exists()  # page NOT rolled back
    assert "conversation-20260607T090000Z" in wpaths.index.read_text()  # index row consistent


def test_gc_removes_old_keeps_new(wpaths):
    # Two archives at different timestamps.
    old = conversation.archive_session(_long_session(), paths=wpaths,
                                       now=datetime(2026, 1, 1, tzinfo=UTC))
    new = conversation.archive_session(_long_session(), paths=wpaths,
                                       now=datetime(2026, 6, 1, tzinfo=UTC))
    assert old and new
    # GC as of mid-June, 90-day cutoff: Jan archive is >90d, June is not.
    removed = conversation.gc_conversations(
        wpaths, older_than_days=90, now=datetime(2026, 6, 15, tzinfo=UTC), dry_run=False
    )
    assert old in removed and new not in removed
    assert not old.exists() and new.exists()


def test_gc_dry_run_keeps_files(wpaths):
    old = conversation.archive_session(_long_session(), paths=wpaths,
                                       now=datetime(2026, 1, 1, tzinfo=UTC))
    removed = conversation.gc_conversations(
        wpaths, older_than_days=30, now=datetime(2026, 6, 1, tzinfo=UTC), dry_run=True
    )
    assert old in removed
    assert old.exists()  # dry run: nothing deleted


def test_gc_ignores_hand_written(wpaths):
    # A non-managed file matching the name pattern must NOT be GC'd.
    hand = wpaths.conversations_dir / "conversation-20200101T000000Z.md"
    hand.write_text("# my own file, no frontmatter\n")
    removed = conversation.gc_conversations(
        wpaths, older_than_days=1, now=datetime(2026, 6, 1, tzinfo=UTC), dry_run=False
    )
    assert hand not in removed
    assert hand.exists()
