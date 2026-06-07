"""Tests for the daily digest (Phase D).

The chat client is stubbed so no server is needed; ``now``/``day`` are
injected for determinism. Covers note selection by mtime window, class
detection gating practice questions, mechanical open-question harvesting,
graceful degradation when the chat server is down, idempotency, and the
hand-written-file guard.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src import digest, wiki


TZ = timezone(timedelta(hours=-7))
DAY = datetime(2026, 6, 6, 12, 0, tzinfo=TZ)  # the day we digest


@pytest.fixture
def vault(tmp_path, monkeypatch):
    v = tmp_path / "vault"
    (v / "journal").mkdir(parents=True)
    (v / "class" / "cs").mkdir(parents=True)
    monkeypatch.setenv("HERMES_VAULT_PATH", str(v))
    monkeypatch.setenv("HERMES_WIKI_PATH", str(v / "wiki"))
    # Ensure push gate is off by default in tests.
    monkeypatch.delenv("HERMES_DIGEST_PUSH", raising=False)
    return v


def _touch(path: Path, content: str, when: datetime):
    path.write_text(content)
    ts = when.timestamp()
    os.utime(path, (ts, ts))


def _stub_chat(class_text="1. Q one?\n2. Q two?\n3. Q three?", learn="Learned things [a.md]."):
    def fn(messages, max_tokens):
        if "practice" in messages[0]["content"].lower() or "study-aid" in messages[0]["content"].lower():
            return class_text
        return learn
    return fn


def test_collect_notes_only_in_window(vault):
    _touch(vault / "journal" / "2026-06-06.md", "# Mon\n\ntoday's note", DAY)
    _touch(vault / "old.md", "# Old\n\nlast month", DAY - timedelta(days=30))
    _touch(vault / "tomorrow.md", "# Tmrw", DAY + timedelta(days=1))
    notes = digest.collect_notes(vault, DAY)
    rels = {n.rel for n in notes}
    assert any("2026-06-06.md" in r for r in rels)
    assert not any("old.md" in r for r in rels)
    assert not any("tomorrow.md" in r for r in rels)


def test_wiki_subtree_excluded_from_digest(vault):
    wiki.init_wiki(wiki.get_paths(vault))
    # A digest page from a prior day must not feed today's digest.
    _touch(wiki.get_paths(vault).digests_dir / "2026-06-06.md", "old digest", DAY)
    _touch(vault / "real.md", "# Real\n\nnote", DAY)
    notes = digest.collect_notes(vault, DAY)
    assert all("wiki" not in n.rel for n in notes)
    assert any("real.md" in n.rel for n in notes)


def test_class_detection_by_tag(vault):
    _touch(vault / "class" / "cs" / "l3.md", "---\ntags: [class/cs]\n---\n# B-trees\n\nsorted", DAY)
    notes = digest.collect_notes(vault, DAY)
    assert digest.is_class_material(notes)


def test_class_detection_by_path(vault):
    _touch(vault / "class" / "cs" / "l3.md", "# B-trees\n\nno tags but in class/ dir", DAY)
    notes = digest.collect_notes(vault, DAY)
    assert digest.is_class_material(notes)


def test_non_class_notes_no_practice(vault):
    _touch(vault / "journal" / "2026-06-06.md", "# Mon\n\njust a daily note", DAY)
    res = digest.build_digest(vault, DAY, chat_fn=_stub_chat())
    assert not res.is_class
    assert res.practice_questions == []
    assert "## Practice questions" not in res.to_markdown()


def test_class_notes_get_practice(vault):
    _touch(vault / "class" / "cs" / "l3.md", "---\ntags: [class/cs]\n---\n# B-trees\n\nsorted", DAY)
    res = digest.build_digest(vault, DAY, chat_fn=_stub_chat())
    assert res.is_class
    assert len(res.practice_questions) == 3
    assert "## Practice questions" in res.to_markdown()


def test_open_questions_harvested(vault):
    _touch(vault / "journal" / "2026-06-06.md",
           "# Mon\n\nTODO: ship phase D\n- [ ] write tests\n- [x] done thing\n\nWhy is the sky blue?\n", DAY)
    res = digest.build_digest(vault, DAY, chat_fn=_stub_chat())
    joined = " || ".join(res.open_questions)
    assert "ship phase D" in joined
    assert "write tests" in joined
    assert "done thing" not in joined        # checked box excluded
    assert "Why is the sky blue?" in joined


def test_open_questions_skip_code_fences(vault):
    _touch(vault / "journal" / "2026-06-06.md",
           "# Mon\n\n```python\n# TODO: not a real task\n```\n\nTODO: a real task\n", DAY)
    res = digest.build_digest(vault, DAY, chat_fn=_stub_chat())
    joined = " || ".join(res.open_questions)
    assert "a real task" in joined
    assert "not a real task" not in joined


def test_degraded_when_chat_unavailable(vault):
    _touch(vault / "journal" / "2026-06-06.md", "# Mon\n\nnote\n\nTODO: x", DAY)
    def boom(messages, max_tokens):
        raise RuntimeError("connection refused")
    res = digest.build_digest(vault, DAY, chat_fn=boom)
    assert res.llm_ok is False
    assert "unavailable" in res.learnings.lower()
    # Mechanical sections still present.
    assert "TODO: x" in " ".join(res.open_questions)


def test_write_digest_page_and_index(vault):
    wiki.init_wiki(wiki.get_paths(vault))
    _touch(vault / "journal" / "2026-06-06.md", "# Mon\n\nnote", DAY)
    res = digest.build_digest(vault, DAY, chat_fn=_stub_chat())
    paths = wiki.get_paths(vault)
    p = digest.write_digest_page(paths, res)
    assert p.exists()
    assert wiki.is_managed(p)
    assert "2026-06-06" in paths.index.read_text()


def test_run_idempotent(vault, monkeypatch, capsys):
    wiki.init_wiki(wiki.get_paths(vault))
    _touch(vault / "journal" / "2026-06-06.md", "# Mon\n\nnote", DAY)
    monkeypatch.setattr(digest, "_default_chat_fn", lambda: _stub_chat())
    rc1 = digest.run(["--date", "2026-06-06"])
    assert rc1 == 0
    # Second run must not regenerate.
    capsys.readouterr()
    rc2 = digest.run(["--date", "2026-06-06"])
    assert rc2 == 0
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "already exists" in (out)


def test_run_dry_run_no_write(vault, monkeypatch, capsys):
    _touch(vault / "journal" / "2026-06-06.md", "# Mon\n\nnote", DAY)
    monkeypatch.setattr(digest, "_default_chat_fn", lambda: _stub_chat())
    rc = digest.run(["--date", "2026-06-06", "--dry-run"])
    assert rc == 0
    # No wiki page written.
    paths = wiki.get_paths(vault)
    assert not (paths.digests_dir / "2026-06-06.md").exists() if paths.digests_dir.is_dir() else True


def test_push_gate_off_by_default(vault, monkeypatch):
    # No HERMES_DIGEST_PUSH → push disabled even if a bot were configured.
    monkeypatch.delenv("HERMES_DIGEST_PUSH", raising=False)
    assert digest.push_enabled() is False


def test_hand_written_digest_not_clobbered(vault, monkeypatch):
    wiki.init_wiki(wiki.get_paths(vault))
    paths = wiki.get_paths(vault)
    # A user-authored file at the digest path (no hermes-managed frontmatter).
    hand = paths.digests_dir / "2026-06-06.md"
    hand.write_text("# My own notes\n\nnot LLM-managed\n")
    _touch(vault / "journal" / "2026-06-06.md", "# Mon\n\nnote", DAY)
    monkeypatch.setattr(digest, "_default_chat_fn", lambda: _stub_chat())
    rc = digest.run(["--date", "2026-06-06", "--force"])
    assert rc == 1  # refused
    assert "not LLM-managed" in hand.read_text()  # untouched


def test_write_digest_page_rechecks_handwritten_guard(vault):
    # Regression (adversarial review TOCTOU): write_digest_page itself must
    # refuse a hand-written file at the path, not just run()'s pre-check —
    # a user file can appear during the seconds-long LLM synthesis window.
    wiki.init_wiki(wiki.get_paths(vault))
    paths = wiki.get_paths(vault)
    hand = paths.digests_dir / "2026-06-06.md"
    hand.write_text("# hand-written, not managed\n")
    _touch(vault / "journal" / "2026-06-06.md", "# Mon\n\nnote", DAY)
    res = digest.build_digest(vault, DAY, chat_fn=_stub_chat())
    with pytest.raises(RuntimeError, match="hand-written"):
        digest.write_digest_page(paths, res)
    assert "not managed" in hand.read_text()  # untouched


def test_resolve_date_defaults_to_yesterday():
    now = datetime(2026, 6, 7, 9, 0, tzinfo=TZ)
    d = digest.resolve_date(None, now=now)
    assert d.strftime("%Y-%m-%d") == "2026-06-06"
    d2 = digest.resolve_date("2026-01-15", now=now)
    assert d2.strftime("%Y-%m-%d") == "2026-01-15"
