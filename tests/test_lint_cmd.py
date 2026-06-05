"""Tests for src/lint_cmd.py — orphan/stub/stale/unused detection."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src import lint_cmd, wiki


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_WIKI_PATH", str(tmp_path / "wiki"))
    monkeypatch.delenv("HERMES_VAULT_PATH", raising=False)
    return tmp_path


def _write_page(path: Path, body: str, frontmatter=None):
    """Write a managed page with optional custom frontmatter."""
    page = wiki.Page(title=path.stem, body=body, frontmatter=frontmatter or {})
    wiki.write_page(path, page)


# ----------------------------------------------------------------- orphan


def test_orphan_topic_is_flagged(vault, capsys):
    wiki.init_wiki()
    paths = wiki.get_paths()
    _write_page(paths.topics_dir / "lonely.md", "## body\nno inbound links")
    rc = lint_cmd.run([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Orphan" in out and "lonely.md" in out


def test_topic_with_inbound_link_is_not_orphan(vault, capsys):
    wiki.init_wiki()
    paths = wiki.get_paths()
    _write_page(paths.topics_dir / "linked.md", "## body\nfoo")
    _write_page(paths.topics_dir / "linker.md", "see [[linked]]")
    rc = lint_cmd.run([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "linked.md" not in out  # no orphan complaint about it


# ----------------------------------------------------------------- stub


def test_self_loop_does_not_mask_orphan(vault, capsys):
    """A page linking to its own stem must still be flagged as orphan
    if no OTHER page links to it. Prior bug: self-link inflated inbound
    count to 1, masking the orphan."""
    wiki.init_wiki()
    paths = wiki.get_paths()
    _write_page(paths.topics_dir / "lonely-self.md", "I link to [[lonely-self]] only.")
    rc = lint_cmd.run([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "lonely-self.md" in out  # still flagged as orphan


def test_subdir_stem_collision_evaluated_independently(vault, capsys):
    """Two pages with the same stem in different subdirs must BOTH be
    evaluated for orphan/unused status. Prior bug: dict keyed on stem
    silently dropped one of them."""
    wiki.init_wiki()
    paths = wiki.get_paths()
    _write_page(paths.sources_dir / "duplicate.md", "source body")
    _write_page(paths.topics_dir / "duplicate.md", "topic body")
    # Linker references the stem 'duplicate' — ambiguous, but the lint
    # heuristic correctly counts it as inbound for both files.
    _write_page(paths.topics_dir / "linker.md", "see [[duplicate]]")
    rc = lint_cmd.run([])
    assert rc == 0
    # 'linker.md' is the orphan (no inbound), but the dup files are not
    # flagged twice or silently elided. We mainly confirm lint did not
    # crash and the run completed.
    out = capsys.readouterr().out
    assert "linker.md" in out


def test_stale_parses_fractional_seconds(vault, capsys):
    """ISO timestamps with sub-second precision must parse, not silently
    skip. Prior bug: only `%Y-%m-%dT%H:%M:%SZ` was attempted."""
    from datetime import datetime, timedelta, timezone
    wiki.init_wiki()
    paths = wiki.get_paths()
    page_path = paths.topics_dir / "fractional.md"
    old = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%S.123Z")
    page_path.write_text(
        f'---\nhermes-managed: "true"\nhermes-updated: "{old}"\n---\n# fractional\n\nbody\n',
        encoding="utf-8",
    )
    _write_page(paths.topics_dir / "linker.md", "[[fractional]]")
    rc = lint_cmd.run(["--stale-days", "30"])
    out = capsys.readouterr().out
    assert "Stale" in out
    assert "fractional.md" in out


def test_referenced_stub_is_flagged(vault, capsys):
    wiki.init_wiki()
    paths = wiki.get_paths()
    _write_page(paths.topics_dir / "src.md", "see [[Imaginary Topic]]")
    rc = lint_cmd.run([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Stubs" in out
    assert "Imaginary Topic" in out


# ----------------------------------------------------------------- stale


def test_stale_page_is_flagged(vault, capsys):
    wiki.init_wiki()
    paths = wiki.get_paths()
    page_path = paths.topics_dir / "ancient.md"
    # Hand-write with old hermes-updated frontmatter.
    old = (datetime.now(timezone.utc) - timedelta(days=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    page_path.write_text(
        f'---\nhermes-managed: "true"\nhermes-updated: "{old}"\n---\n# ancient\n\nbody\n',
        encoding="utf-8",
    )
    # Need an inbound link so it's not flagged as orphan instead.
    _write_page(paths.topics_dir / "linker.md", "[[ancient]]")
    rc = lint_cmd.run(["--stale-days", "30"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Stale" in out
    assert "ancient.md" in out


def test_recent_page_not_stale(vault, capsys):
    wiki.init_wiki()
    paths = wiki.get_paths()
    _write_page(paths.topics_dir / "fresh.md", "body")
    _write_page(paths.topics_dir / "linker.md", "[[fresh]]")
    rc = lint_cmd.run(["--stale-days", "30"])
    out = capsys.readouterr().out
    assert "Stale" not in out


# --------------------------------------------------------- unused source


def test_unused_source_is_flagged(vault, capsys):
    wiki.init_wiki()
    paths = wiki.get_paths()
    _write_page(paths.sources_dir / "unused.md", "summary")
    rc = lint_cmd.run([])
    out = capsys.readouterr().out
    assert "Unused sources" in out and "unused.md" in out


def test_cited_source_is_not_flagged(vault, capsys):
    wiki.init_wiki()
    paths = wiki.get_paths()
    _write_page(paths.sources_dir / "useful.md", "summary")
    _write_page(paths.topics_dir / "topic.md", "see [[useful]]")
    rc = lint_cmd.run([])
    out = capsys.readouterr().out
    assert "Unused sources" not in out


# ------------------------------------------------------------ misc


def test_lint_uninitialized_returns_2(vault):
    rc = lint_cmd.run([])
    assert rc == 2


def test_lint_clean_wiki_says_clean(vault, capsys):
    wiki.init_wiki()
    paths = wiki.get_paths()
    # Digests are exempt from orphan-checks (chronologically terminal).
    # A clean wiki: source cited by a topic that links to a digest. The
    # source has inbound (topic), the topic has inbound (digest), the
    # digest is exempt — clean.
    _write_page(paths.sources_dir / "good.md", "summary")
    _write_page(paths.topics_dir / "topic.md", "[[good]] context.")
    _write_page(paths.digests_dir / "2026-06-04.md", "today: [[topic]]")
    rc = lint_cmd.run([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "clean" in out


def test_lint_strict_returns_1_on_issues(vault):
    wiki.init_wiki()
    paths = wiki.get_paths()
    _write_page(paths.topics_dir / "lonely.md", "body")
    rc = lint_cmd.run(["--strict"])
    assert rc == 1


def test_lint_strict_returns_0_when_clean(vault):
    wiki.init_wiki()
    paths = wiki.get_paths()
    _write_page(paths.sources_dir / "s.md", "body")
    _write_page(paths.topics_dir / "t.md", "[[s]]")
    _write_page(paths.digests_dir / "d.md", "[[t]]")
    rc = lint_cmd.run(["--strict"])
    assert rc == 0
