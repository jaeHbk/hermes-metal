"""Watcher × VaultFilter integration tests.

The watcher daemon is the live-index half of the system. These tests verify
the filter's verdict actually short-circuits the indexing pipeline, without
a real LanceDB or embed server.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.backend.vault_filter import build_filter
from src.daemon.watcher import VaultWatcher


@pytest.fixture
def vault(tmp_path):
    (tmp_path / "good.md").write_text("real note")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "workspace.json").write_text("{}")
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "stub.md").write_text("template")
    return tmp_path


def test_excluded_path_does_not_schedule(vault, monkeypatch):
    """Editing an excluded file must not arm the debounce timer."""
    fake_index = MagicMock()
    monkeypatch.setattr("src.daemon.watcher.indexer", MagicMock(index_file=fake_index), raising=False)

    w = VaultWatcher(
        vault_path=vault,
        vault=MagicMock(),
        embed_url="http://0/",
        debounce_s=0.01,
        vfilter=build_filter(vault),
    )
    # Patch index_file on the instance after construction
    w._index_file = fake_index

    # Excluded path: should be a no-op
    w._schedule_index(str(vault / "templates" / "stub.md"))
    assert len(w._timers) == 0
    fake_index.assert_not_called()

    # Accepted path: should arm a timer
    w._schedule_index(str(vault / "good.md"))
    assert len(w._timers) == 1
    # Cancel before it fires so the test doesn't actually try to embed.
    for t in w._timers.values():
        t.cancel()


def test_filter_derived_from_include_drives_watch_patterns(vault, monkeypatch):
    """If HERMES_VAULT_INCLUDE adds *.txt, the watcher's pattern matcher
    must accept .txt events too — not just the hardcoded markdown set."""
    monkeypatch.setenv("HERMES_VAULT_INCLUDE", "*.md:*.txt")
    w = VaultWatcher(
        vault_path=vault,
        vault=MagicMock(),
        embed_url="http://0/",
    )
    assert "*.txt" in w.patterns
    assert "*.md" in w.patterns


def test_on_moved_uses_filter_not_hardcoded_extensions(vault, monkeypatch):
    """A file moved from .md to a non-markdown extension should NOT be
    re-indexed; the previous source must still be deleted from the index."""
    fake_vault = MagicMock()
    w = VaultWatcher(
        vault_path=vault,
        vault=fake_vault,
        embed_url="http://0/",
        debounce_s=0.01,
        vfilter=build_filter(vault),
    )

    # A .md → .tmp move (Obsidian's atomic write pattern): destination is
    # filter-rejected, so no re-index should be scheduled. Old source should
    # still be deleted so the index stays consistent.
    src_path = str(vault / "good.md")
    dst_path = str(vault / "good.tmp")
    event = MagicMock(
        is_directory=False,
        src_path=src_path,
        dest_path=dst_path,
    )
    w.on_moved(event)
    fake_vault.delete_by_source.assert_called_once_with(src_path)
    assert len(w._timers) == 0
