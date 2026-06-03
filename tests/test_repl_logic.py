"""Pure-Python REPL logic: history trim, transcript round-trip.

These tests don't touch the network or asyncio — they exercise the
deterministic helpers that make the REPL safe to use across long
sessions.
"""
from __future__ import annotations

from src.repl import (
    ChatSession,
    _approx_tokens,
    _format_context,
    _parse_transcript,
    _trim_history,
    _cmd_save,
    _cmd_load,
)


def _msg(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


# ----------------------------------------------------------------- trim


def test_trim_keeps_short_history_intact():
    h = [_msg("user", "hi"), _msg("assistant", "hello"), _msg("user", "bye")]
    out = _trim_history(h, budget=10_000)
    assert out == h


def test_trim_drops_oldest_pair_when_over_budget():
    # Build history: 3 user/assistant pairs of ~100 tokens each, plus current user.
    big = "x" * 400  # ≈100 tokens (4 chars/token)
    h = [
        _msg("user", "Q1: " + big),
        _msg("assistant", "A1: " + big),
        _msg("user", "Q2: " + big),
        _msg("assistant", "A2: " + big),
        _msg("user", "Q3 current"),
    ]
    out = _trim_history(h, budget=200)
    # Current user must always survive.
    assert out[-1]["content"] == "Q3 current"
    # Older pairs should be dropped to fit the budget.
    assert len(out) < len(h)


def test_trim_never_strips_below_two_messages():
    # Even if the budget is impossibly small, we must keep at least the
    # current user message so the model has something to respond to.
    h = [
        _msg("user", "Q1: " + "x" * 4000),
        _msg("assistant", "A1: " + "x" * 4000),
        _msg("user", "Q2 current"),
    ]
    out = _trim_history(h, budget=1)
    assert len(out) >= 1
    assert any(m["content"] == "Q2 current" for m in out)


def test_trim_does_not_mutate_input():
    h = [_msg("user", "a"), _msg("assistant", "b"), _msg("user", "c")]
    snapshot = [dict(m) for m in h]
    _trim_history(h, budget=1)
    assert h == snapshot


# ----------------------------------------------------------------- transcript


def test_transcript_round_trip(tmp_path):
    sess = ChatSession()
    sess.history = [
        _msg("user", "What is 2+2?"),
        _msg("assistant", "4"),
        _msg("user", "And in binary?"),
        _msg("assistant", "100"),
    ]
    target = tmp_path / "t.txt"
    _cmd_save(sess, str(target))

    fresh = ChatSession()
    _cmd_load(fresh, str(target))
    assert fresh.history == sess.history


def test_transcript_load_preserves_markdown_content(tmp_path):
    """A user message containing a `### foo` line must NOT split a message."""
    sess = ChatSession()
    sess.history = [
        _msg("user", "Here's a section:\n### My subsection\nWith content."),
        _msg("assistant", "Got it."),
    ]
    target = tmp_path / "t.txt"
    _cmd_save(sess, str(target))

    fresh = ChatSession()
    _cmd_load(fresh, str(target))
    assert fresh.history == sess.history
    assert "### My subsection" in fresh.history[0]["content"]


def test_transcript_load_invalid_file(tmp_path, capsys):
    bogus = tmp_path / "not-a-transcript.txt"
    bogus.write_text("Just some plain text without headers.\n")
    sess = ChatSession()
    sess.history = [_msg("user", "should not be replaced")]
    _cmd_load(sess, str(bogus))
    # On parse failure, history must remain untouched.
    assert sess.history == [_msg("user", "should not be replaced")]


def test_parse_transcript_strips_trailing_blank_lines():
    text = "### user\nhi\n\n### assistant\nhello\n\n"
    parsed = _parse_transcript(text)
    assert parsed == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


# ----------------------------------------------------------------- helpers


def test_approx_tokens_min_one():
    assert _approx_tokens("") == 1
    assert _approx_tokens("x") == 1
    # 4 chars per token (rough).
    assert _approx_tokens("x" * 40) == 10


def test_format_context_truncates_at_max_chars():
    hits = [
        {"source_path": "/v/a.md", "chunk_idx": 0, "text": "alpha"},
        {"source_path": "/v/b.md", "chunk_idx": 0, "text": "beta"},
        {"source_path": "/v/c.md", "chunk_idx": 0, "text": "gamma"},
    ]
    # max_chars=30 fits the first block ("[a.md #chunk0]\nalpha\n" ≈ 21 chars)
    # but not a second one. The implementation breaks BEFORE adding a block
    # that would push past the cap, then joins selected blocks with "\n---\n".
    out = _format_context(hits, max_chars=30)
    assert "alpha" in out
    assert "beta" not in out  # second block was rejected by the cap
    assert "gamma" not in out


def test_format_context_takes_all_when_budget_large():
    hits = [
        {"source_path": "/v/a.md", "chunk_idx": 0, "text": "alpha"},
        {"source_path": "/v/b.md", "chunk_idx": 0, "text": "beta"},
    ]
    out = _format_context(hits, max_chars=10_000)
    assert "alpha" in out and "beta" in out
    assert "\n---\n" in out  # separator between blocks
