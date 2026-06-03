"""Tests for src/backend/vault_filter.py.

The filter is the single source of truth shared by the watcher and
``hermes index``. Every regression here would silently mis-index files.
"""
from __future__ import annotations

import os

import pytest

from src.backend.vault_filter import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    VaultFilter,
    build_filter,
    iter_vault_files,
)


@pytest.fixture
def vault(tmp_path):
    """A small representative vault."""
    (tmp_path / "Welcome.md").write_text("hi")
    (tmp_path / "design").mkdir()
    (tmp_path / "design" / "auth.md").write_text("auth notes")
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "daily.md").write_text("template stub")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "workspace.json").write_text("{}")
    (tmp_path / "attachments").mkdir()
    (tmp_path / "attachments" / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / ".trash").mkdir()
    (tmp_path / ".trash" / "deleted.md").write_text("oops")
    (tmp_path / "notes.markdown").write_text("alt extension")
    (tmp_path / "todo.txt").write_text("not markdown")
    return tmp_path


def test_default_filter_accepts_top_level_md(vault):
    f = build_filter(vault)
    assert f.accepts(vault / "Welcome.md")
    assert f.accepts(vault / "design" / "auth.md")
    assert f.accepts(vault / "notes.markdown")


def test_default_filter_excludes_obsidian_meta(vault):
    f = build_filter(vault)
    assert not f.accepts(vault / ".obsidian" / "workspace.json")


def test_default_filter_excludes_trash_and_templates(vault):
    f = build_filter(vault)
    assert not f.accepts(vault / ".trash" / "deleted.md")
    assert not f.accepts(vault / "templates" / "daily.md")


def test_default_filter_excludes_attachments(vault):
    f = build_filter(vault)
    assert not f.accepts(vault / "attachments" / "image.png")


def test_default_filter_excludes_non_markdown(vault):
    f = build_filter(vault)
    assert not f.accepts(vault / "todo.txt")


def test_filter_rejects_paths_outside_vault(vault, tmp_path):
    other = tmp_path.parent / "outside.md"
    f = build_filter(vault)
    # Path outside the vault root should be rejected even if extension matches.
    assert not f.accepts(other)


def test_iter_vault_files_returns_only_accepted(vault):
    f = build_filter(vault)
    files = iter_vault_files(vault, f)
    rels = sorted(p.relative_to(vault).as_posix() for p in files)
    assert rels == ["Welcome.md", "design/auth.md", "notes.markdown"]


def test_env_exclude_replaces_yaml_and_defaults(vault, monkeypatch):
    # Drop "templates" from excludes via env override (env REPLACES, not merges).
    monkeypatch.setenv("HERMES_VAULT_EXCLUDE", "*.png:.obsidian/*")
    f = build_filter(vault)
    files = iter_vault_files(vault, f)
    rels = sorted(p.relative_to(vault).as_posix() for p in files)
    # `.trash/deleted.md` and `templates/daily.md` should now appear because
    # the override removed those default exclusions.
    assert "templates/daily.md" in rels
    assert ".trash/deleted.md" in rels


def test_yaml_config_overrides_defaults(vault, tmp_path, monkeypatch):
    config = tmp_path / "vault.yaml"
    config.write_text("include:\n  - '*.md'\nexclude:\n  - 'design/*'\n")
    f = build_filter(vault, config_path=config)
    assert f.accepts(vault / "Welcome.md")
    assert not f.accepts(vault / "design" / "auth.md")
    # .markdown is NO LONGER accepted because YAML's include omitted it.
    assert not f.accepts(vault / "notes.markdown")


def test_anchored_glob_matches_full_path():
    # Anchored pattern with "/" matches full rel path only.
    f = VaultFilter(
        vault_root=__import__("pathlib").Path("/tmp"),
        include=("*.md",),
        exclude=("templates/*",),
    )
    # `templates/daily.md` excluded; a top-level `daily.md` is accepted.
    assert not f.accepts("templates/daily.md")
    assert f.accepts("daily.md")


def test_slashless_glob_matches_any_component():
    # Slashless excludes match any path component.
    f = VaultFilter(
        vault_root=__import__("pathlib").Path("/tmp"),
        include=("*.md",),
        exclude=(".obsidian",),
    )
    # Both top-level and nested .obsidian dirs (whole component) are excluded.
    assert not f.accepts(".obsidian/foo.md")
    assert not f.accepts("nested/.obsidian/foo.md")


def test_iter_vault_files_handles_missing_vault(tmp_path):
    nonexistent = tmp_path / "no-such-dir"
    f = build_filter(tmp_path)
    assert iter_vault_files(nonexistent, f) == []


def test_default_constants_in_sync():
    # Sanity guard: the README and example YAML reflect these defaults.
    assert "*.md" in DEFAULT_INCLUDE
    assert ".obsidian" in DEFAULT_EXCLUDE
    assert "attachments" in DEFAULT_EXCLUDE
