"""End-to-end tests for `hermes index` (backfill + GC).

We mock the embed HTTP call so tests don't need a running llama-server.
LanceDB is real but pointed at a tmp_path so each test gets a clean DB.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src import index_cmd
from src.backend import database, indexer
from src.backend.database import LanceVault


@pytest.fixture
def fake_embed(monkeypatch):
    """Replace the synchronous httpx.post call with a deterministic stub.

    Returns a 768-dim vector that varies by input length so different chunks
    don't collide in the index.
    """
    def _fake_embed(texts, embed_url=None, **_kwargs):
        return [[float(i + len(t)) for i in range(database.DEFAULT_EMBED_DIM)] for t in texts]

    monkeypatch.setattr(indexer, "embed", _fake_embed)


@pytest.fixture
def vault_dir(tmp_path):
    v = tmp_path / "vault"
    v.mkdir()
    (v / "a.md").write_text("alpha content here")
    (v / "b.md").write_text("beta content here")
    (v / "c.md").write_text("gamma content here")
    (v / ".obsidian").mkdir()
    (v / ".obsidian" / "workspace.json").write_text("{}")
    return v


@pytest.fixture
def db_dir(tmp_path):
    return tmp_path / "lancedb"


@pytest.fixture
def env(monkeypatch, vault_dir, db_dir):
    monkeypatch.setenv("HERMES_VAULT_PATH", str(vault_dir))
    monkeypatch.setenv("HERMES_LANCEDB_PATH", str(db_dir))
    return None


def test_backfill_indexes_all_md_files(fake_embed, env, db_dir, vault_dir):
    rc = index_cmd.run(["--backfill"])
    assert rc == 0
    v = LanceVault(path=db_dir)
    assert v.count() >= 3  # one chunk per short file
    sources = set(v.distinct_sources())
    assert {str((vault_dir / f).resolve()) for f in ("a.md", "b.md", "c.md")} <= sources


def test_backfill_skips_already_indexed(fake_embed, env, db_dir, vault_dir, capsys):
    index_cmd.run(["--backfill"])
    capsys.readouterr()  # clear
    rc = index_cmd.run(["--backfill"])
    assert rc == 0
    err = capsys.readouterr().err
    # Second run should mark every file as skipped — no "index NAME" line
    # should appear (those are emitted only when a file is actually written).
    assert err.count("skip") >= 3
    assert err.count("index  ") == 0  # the per-file "index  <name>" lines


def test_force_reindexes_everything(fake_embed, env, db_dir, vault_dir, capsys):
    index_cmd.run(["--backfill"])
    capsys.readouterr()
    rc = index_cmd.run(["--backfill", "--force"])
    assert rc == 0
    err = capsys.readouterr().err
    assert err.count("index ") >= 3  # leading word "index", not "indexed"


def test_gc_drops_orphan_sources(fake_embed, env, db_dir, vault_dir):
    index_cmd.run(["--backfill"])
    v = LanceVault(path=db_dir)
    initial = v.count()

    # Delete one note from disk; the index still has it.
    (vault_dir / "b.md").unlink()
    rc = index_cmd.run(["--gc"])
    assert rc == 0
    v2 = LanceVault(path=db_dir)
    assert v2.count() < initial
    sources = set(v2.distinct_sources())
    assert not any(s.endswith("b.md") for s in sources)


def test_gc_dry_run_keeps_rows(fake_embed, env, db_dir, vault_dir):
    index_cmd.run(["--backfill"])
    (vault_dir / "b.md").unlink()

    v = LanceVault(path=db_dir)
    before = v.count()
    rc = index_cmd.run(["--gc", "--dry-run"])
    assert rc == 0
    v2 = LanceVault(path=db_dir)
    assert v2.count() == before  # nothing actually removed


def test_filter_excluded_files_not_indexed(fake_embed, env, db_dir, vault_dir):
    # The .obsidian/workspace.json should never enter the index.
    index_cmd.run(["--backfill"])
    v = LanceVault(path=db_dir)
    sources = v.distinct_sources()
    assert not any(".obsidian" in s for s in sources)


def test_limit_caps_processed_files(fake_embed, env, db_dir, vault_dir, capsys):
    rc = index_cmd.run(["--backfill", "--limit", "2"])
    assert rc == 0
    v = LanceVault(path=db_dir)
    assert len(v.distinct_sources()) == 2


def test_no_flags_errors():
    with pytest.raises(SystemExit):
        index_cmd.run([])


def test_missing_vault_path_returns_2(monkeypatch, db_dir):
    monkeypatch.delenv("HERMES_VAULT_PATH", raising=False)
    monkeypatch.setenv("HERMES_LANCEDB_PATH", str(db_dir))
    # Also avoid the watcher-plist fallback by stubbing the home path.
    monkeypatch.setattr(index_cmd, "_resolve_vault_path", lambda: None)
    rc = index_cmd.run(["--backfill"])
    assert rc == 2
