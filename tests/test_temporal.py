"""Tests for the temporal query parser (Phase B).

The load-bearing property: high-precision phrases produce correct windows,
and anything ambiguous returns None (so retrieval falls through to an
unscoped vector search rather than silently answering from the wrong dates).
``now`` is injected so these are deterministic regardless of when they run.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.backend.temporal import parse_window, DateWindow


# A fixed reference moment: Sunday 2026-06-07 14:30 in UTC-7.
TZ = timezone(timedelta(hours=-7))
NOW = datetime(2026, 6, 7, 14, 30, tzinfo=TZ)


def _window(q):
    return parse_window(q, now=NOW)


def test_no_temporal_phrase_returns_none():
    assert _window("explain cosine similarity") is None
    assert _window("what is the reranker") is None
    assert _window("") is None


def test_vague_phrases_not_matched():
    # Deliberately NOT parsed — too imprecise to scope safely.
    assert _window("recently I was thinking") is None
    assert _window("a while ago") is None
    assert _window("the other day") is None


def test_yesterday_is_single_day_with_path_hint():
    w = _window("what did I write yesterday")
    assert w is not None
    assert w.phrase == "yesterday"
    assert w.iso_day == "2026-06-06"
    # Window is exactly one day wide.
    assert abs((w.end - w.start) - 86400) < 1
    # where_clause includes the path-date OR for daily notes.
    assert "source_path LIKE '%2026-06-06%'" in w.where_clause()
    assert "mtime >=" in w.where_clause()


def test_today_single_day():
    w = _window("notes from today")
    assert w is not None and w.iso_day == "2026-06-07"


def test_explicit_iso_date():
    w = _window("anything from 2026-06-01 please")
    assert w is not None
    assert w.iso_day == "2026-06-01"
    assert abs((w.end - w.start) - 86400) < 1


def test_invalid_iso_date_falls_through():
    # 2026-13-40 is not a real date; parser must not crash and must not
    # produce a bogus window (it falls through to None here).
    assert _window("notes from 2026-13-40") is None


def test_past_n_days_window():
    w = _window("show me the past 7 days")
    assert w is not None
    assert w.iso_day is None  # rolling window, not a single day
    assert abs((w.end - w.start) - 7 * 86400) < 1


def test_last_week_is_prior_iso_week():
    w = _window("last week's standups")
    assert w is not None and w.phrase == "last week"
    # last week = the 7 days ending at this week's Monday 00:00.
    span = w.end - w.start
    assert abs(span - 7 * 86400) < 1


def test_this_week():
    w = _window("this week")
    assert w is not None
    assert abs((w.end - w.start) - 7 * 86400) < 1


def test_iso_month_whole_month():
    w = _window("summary for 2026-05")
    assert w is not None and w.phrase == "2026-05"
    # May has 31 days.
    assert abs((w.end - w.start) - 31 * 86400) < 1


def test_iso_date_beats_iso_month():
    # A full ISO date present means we don't also match the YYYY-MM inside it.
    w = _window("on 2026-06-01")
    assert w is not None and w.iso_day == "2026-06-01"
    assert abs((w.end - w.start) - 86400) < 1  # one day, not a month


def test_last_month_and_this_month():
    lm = _window("last month")
    tm = _window("this month")
    assert lm is not None and tm is not None
    # June (this month) has 30 days; May (last month) has 31.
    assert abs((tm.end - tm.start) - 30 * 86400) < 1
    assert abs((lm.end - lm.start) - 31 * 86400) < 1


def test_where_clause_no_injection_surface():
    # The LIKE value is a fixed YYYY-MM-DD we control — no user text leaks in.
    w = _window("yesterday'; DROP TABLE vault_chunks; --")
    assert w is not None
    assert "DROP TABLE" not in w.where_clause()
    assert "2026-06-06" in w.where_clause()


def test_iso_date_not_matched_inside_identifiers():
    # Regression (adversarial review): a date embedded in a hyphenated
    # filename/identifier must NOT be parsed as a temporal phrase, or the
    # query gets mis-scoped to that date's window.
    assert _window("open report-2026-06-07.md") is None
    assert _window("project-id-2026-06-07-notes") is None
    assert _window("v1-2026-06-07-final build") is None
    # ...but a genuinely standalone date still matches, even with punctuation.
    assert _window("notes from (2026-06-07)") is not None
    assert _window("on 2026-06-07.") is not None


def test_iso_month_not_matched_inside_identifiers():
    assert _window("restore archive-2026-06-backup") is None
    assert _window("v2-2026-06-snapshot") is None
    # Standalone month still matches.
    assert _window("summary for 2026-06") is not None


def test_naive_now_is_tolerated():
    # A naive datetime shouldn't crash; it's treated as UTC.
    w = parse_window("yesterday", now=datetime(2026, 6, 7, 12, 0))
    assert w is not None and w.iso_day == "2026-06-06"
