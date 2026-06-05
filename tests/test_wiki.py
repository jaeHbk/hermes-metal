"""Tests for src/wiki.py — paths, init, page write, log/index update.

Covers the contract callers (ingest, lint, /file) depend on:
* init creates the structure idempotently
* write_page is atomic (no partial files on caller exception)
* update_index_row replaces existing rows by page name (no duplicates)
* parse_links extracts both [[wiki]] and [md](link) targets
* hermes-managed frontmatter survives round-trip
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src import wiki
from src.wiki import (
    Page,
    WikiPaths,
    all_pages,
    append_log,
    get_paths,
    init_wiki,
    is_initialized,
    is_managed,
    page_stem,
    parse_links,
    resolve_wiki_path,
    update_index_row,
    write_page,
)


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """Per-test vault with HERMES_WIKI_PATH pointed at it."""
    root = tmp_path / "wiki"
    monkeypatch.setenv("HERMES_WIKI_PATH", str(root))
    monkeypatch.delenv("HERMES_VAULT_PATH", raising=False)
    return root


# ----------------------------------------------------------- resolution


def test_resolve_uses_env_var(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_WIKI_PATH", str(tmp_path / "explicit"))
    monkeypatch.setenv("HERMES_VAULT_PATH", "/some/other/vault")
    assert resolve_wiki_path() == (tmp_path / "explicit").resolve()


def test_resolve_falls_back_to_vault(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_WIKI_PATH", raising=False)
    monkeypatch.setenv("HERMES_VAULT_PATH", str(tmp_path / "v"))
    assert resolve_wiki_path() == (tmp_path / "v").resolve() / "wiki"


# ----------------------------------------------------------------- init


def test_init_creates_structure(vault):
    paths = init_wiki()
    assert paths.root.is_dir()
    assert paths.schema.is_file()
    assert paths.index.is_file()
    assert paths.log.is_file()
    assert paths.sources_dir.is_dir()
    assert paths.topics_dir.is_dir()
    assert paths.digests_dir.is_dir()


def test_init_is_idempotent(vault):
    paths = init_wiki()
    # Edit the schema; second init should NOT clobber.
    paths.schema.write_text("CUSTOM SCHEMA", encoding="utf-8")
    init_wiki()
    assert paths.schema.read_text() == "CUSTOM SCHEMA"


def test_init_force_resets(vault):
    paths = init_wiki()
    paths.schema.write_text("CUSTOM SCHEMA", encoding="utf-8")
    init_wiki(force=True)
    assert "hermes-agents.md" in paths.schema.read_text()


def test_init_force_preserves_content_dirs(vault):
    paths = init_wiki()
    (paths.sources_dir / "important.md").write_text("user content")
    init_wiki(force=True)
    # Force resets meta files but MUST NOT touch content directories.
    assert (paths.sources_dir / "important.md").read_text() == "user content"


def test_is_initialized(vault):
    assert not is_initialized()
    init_wiki()
    assert is_initialized()


# --------------------------------------------------------------- write_page


def test_write_page_adds_managed_frontmatter(vault):
    init_wiki()
    paths = get_paths()
    page = Page(title="Test", body="body content")
    target = paths.topics_dir / "test.md"
    write_page(target, page)
    text = target.read_text()
    assert text.startswith("---\n")
    assert 'hermes-managed: "true"' in text
    assert 'hermes-updated:' in text
    assert "# Test" in text
    assert "body content" in text


def test_write_page_atomic_no_partial_on_error(vault, monkeypatch):
    init_wiki()
    paths = get_paths()
    target = paths.topics_dir / "atomic.md"

    real_replace = os.replace
    calls = {"n": 0}

    def fail_replace(*a, **k):
        calls["n"] += 1
        raise OSError("simulated FS error")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        write_page(target, Page(title="X", body="y"))
    # No partial file at the target path; temp file also cleaned up.
    assert not target.exists()
    leftovers = list(paths.topics_dir.glob("atomic.md.tmp.*"))
    assert leftovers == []


def test_is_managed(vault):
    init_wiki()
    paths = get_paths()
    managed = paths.topics_dir / "m.md"
    write_page(managed, Page(title="M", body="b"))
    assert is_managed(managed)

    handwritten = paths.topics_dir / "h.md"
    handwritten.write_text("# Just a markdown file\n\nNo frontmatter.\n")
    assert not is_managed(handwritten)


def test_frontmatter_rejects_unsafe_keys(vault):
    init_wiki()
    paths = get_paths()
    bad = Page(title="x", body="y", frontmatter={"a key with spaces": "v"})
    with pytest.raises(ValueError):
        write_page(paths.topics_dir / "bad.md", bad)


# --------------------------------------------------------------- index update


def test_update_index_row_inserts(vault):
    init_wiki()
    paths = get_paths()
    update_index_row(paths, "Topics", "alpha", "first topic")
    text = paths.index.read_text()
    assert "[alpha](topics/alpha.md)" in text
    assert "first topic" in text


def test_update_index_row_replaces_existing(vault):
    init_wiki()
    paths = get_paths()
    update_index_row(paths, "Topics", "alpha", "first version")
    update_index_row(paths, "Topics", "alpha", "second version")
    text = paths.index.read_text()
    # Only one row for `alpha`.
    assert text.count("[alpha](topics/alpha.md)") == 1
    assert "second version" in text
    assert "first version" not in text


def test_update_index_keeps_alphabetical_order(vault):
    init_wiki()
    paths = get_paths()
    update_index_row(paths, "Topics", "zeta", "z")
    update_index_row(paths, "Topics", "alpha", "a")
    update_index_row(paths, "Topics", "mu", "m")
    text = paths.index.read_text()
    a_idx = text.find("[alpha]")
    m_idx = text.find("[mu]")
    z_idx = text.find("[zeta]")
    assert 0 < a_idx < m_idx < z_idx


def test_update_index_emits_missing_section(vault):
    """If the user deleted ## Topics from index.md, a Topics row must
    still land — appended at the end, not silently dropped."""
    init_wiki()
    paths = get_paths()
    # Strip the Topics section out of index.md.
    text = paths.index.read_text()
    # Remove "## Topics" and its body up to the next "## " or EOF.
    # Replace with just the remaining content (no Topics section at all).
    paths.index.write_text("# Wiki Index\n\n## Sources\n\n(none yet)\n")
    update_index_row(paths, "Topics", "alpha", "first topic")
    after = paths.index.read_text()
    assert "## Topics" in after
    assert "[alpha](topics/alpha.md)" in after


def test_is_managed_only_inspects_frontmatter_block(vault):
    """A user file with `hermes-managed: true` in its BODY (e.g. quoted
    in a code block) must not be misidentified as LLM-owned."""
    init_wiki()
    paths = get_paths()
    page_path = paths.topics_dir / "tricky.md"
    page_path.write_text(
        '---\nuser-meta: "1"\n---\n\n'
        '# A user note\n\nExample frontmatter:\n```\nhermes-managed: true\n```\n',
        encoding="utf-8",
    )
    assert not is_managed(page_path)


def test_update_index_unknown_section_raises(vault):
    init_wiki()
    paths = get_paths()
    with pytest.raises(ValueError):
        update_index_row(paths, "Bogus", "x", "y")


# ----------------------------------------------------------------- log


def test_append_log_creates_grep_friendly_entry(vault):
    init_wiki()
    paths = get_paths()
    append_log(paths, "ingest", "alpha", detail="extra")
    text = paths.log.read_text()
    # Grep-friendly: starts with "## [TS] op | subject"
    found = [ln for ln in text.splitlines() if ln.startswith("## [")]
    assert len(found) == 1
    assert "ingest | alpha" in found[0]
    assert "extra" in text


def test_append_log_preserves_prior_entries(vault):
    init_wiki()
    paths = get_paths()
    append_log(paths, "ingest", "first")
    append_log(paths, "ingest", "second")
    found = [ln for ln in paths.log.read_text().splitlines() if ln.startswith("## [")]
    assert len(found) == 2
    assert "first" in found[0] and "second" in found[1]


# --------------------------------------------------------------- parse_links


def test_parse_links_wiki_style():
    text = "see [[Alan Turing]] and also [[Computability#section|Comp]]."
    out = parse_links(text)
    assert "Alan Turing" in out
    assert "Computability" in out


def test_parse_links_markdown_style():
    text = "see [Foo](foo.md) and [Bar](../topics/bar.md) but not [Web](https://x.com)"
    out = parse_links(text)
    assert "foo" in out
    assert "bar" in out
    assert all("x.com" not in t for t in out)


def test_parse_links_empty():
    assert parse_links("") == set()


def test_all_pages_skips_meta_files(vault):
    init_wiki()
    paths = get_paths()
    write_page(paths.topics_dir / "t.md", Page(title="T", body="b"))
    write_page(paths.sources_dir / "s.md", Page(title="S", body="b"))
    pages = all_pages(paths)
    stems = {page_stem(p) for p in pages}
    assert stems == {"t", "s"}
    # index/log/schema must not be in the list.
    assert "index" not in stems
    assert "log" not in stems
