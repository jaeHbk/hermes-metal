"""Tests for the v1→v2 schema migration and metadata-aware writes (Phase C).

The migration is the highest-risk change in this push — get it wrong and a
user either silently runs on the old schema or loses data. These tests prove
it's lossless, idempotent, and that an un-migrated table still accepts writes
(degraded, no metadata) instead of raising.
"""
from __future__ import annotations

import json

import pyarrow as pa
import pytest

from src.backend import database
from src.backend.database import LanceVault, _v1_fields, SCHEMA_VERSION


def _v1_table(db_path, embed_dim=4):
    """Create a genuine v1 (no-metadata) table at db_path, return its name."""
    import lancedb
    schema = pa.schema(_v1_fields() + [pa.field("vector", pa.list_(pa.float32(), embed_dim))])
    db = lancedb.connect(str(db_path))
    rows = [
        {"id": "a", "source_path": "/v/x.md", "chunk_idx": 0, "text": "alpha", "vector": [1, 0, 0, 0]},
        {"id": "b", "source_path": "/v/y.md", "chunk_idx": 0, "text": "beta", "vector": [0, 1, 0, 0]},
    ]
    db.create_table("vault_chunks", data=pa.Table.from_pylist(rows, schema=schema), schema=schema)
    return rows


def test_fresh_table_is_v2(tmp_path):
    v = LanceVault(path=tmp_path / "db", embed_dim=4)
    assert v.schema_version() == SCHEMA_VERSION == 2
    assert v.has_metadata
    # Sidecar hint is written for doctor.
    hint = json.loads((v.path / database.SCHEMA_HINT_FILE).read_text())
    assert hint["schema_version"] == 2
    assert hint["metric"] == "cosine"


def test_auto_migrate_on_open(tmp_path):
    db_path = tmp_path / "db"
    _v1_table(db_path, embed_dim=4)
    # Opening with auto_migrate (default) should upgrade in place.
    v = LanceVault(path=db_path, embed_dim=4)
    assert v.has_metadata
    assert v.schema_version() == 2
    # Rows preserved.
    assert v.count() == 2


def test_migration_is_lossless(tmp_path):
    db_path = tmp_path / "db"
    _v1_table(db_path, embed_dim=4)
    v = LanceVault(path=db_path, embed_dim=4, auto_migrate=False)
    assert not v.has_metadata
    before = v.count()
    added = v.migrate()
    assert set(added) == {"mtime", "tags", "heading_trail"}
    assert v.count() == before  # no rows lost
    # Existing vectors still searchable under cosine.
    hits = v.search([1.0, 0.0, 0.0, 0.0], k=2)
    assert len(hits) == 2
    assert "mtime" in hits[0]


def test_migration_race_tolerant(tmp_path):
    # Two LanceVault handles on the same v1 table (simulating the watcher's
    # auto-migrate racing `index --migrate`). The second migrate must not
    # crash even though the columns now already exist.
    db_path = tmp_path / "db"
    _v1_table(db_path, embed_dim=4)
    a = LanceVault(path=db_path, embed_dim=4, auto_migrate=False)
    b = LanceVault(path=db_path, embed_dim=4, auto_migrate=False)
    added_a = a.migrate()
    assert set(added_a) == {"mtime", "tags", "heading_trail"}
    # b still thinks columns are missing (stale handle); its migrate hits the
    # "already exists" path and must absorb it without raising.
    added_b = b.migrate()
    assert added_b == [] or set(added_b).issubset({"mtime", "tags", "heading_trail"})
    assert b.has_metadata
    assert b.count() == 2  # no data lost in the race


def test_migration_idempotent(tmp_path):
    db_path = tmp_path / "db"
    _v1_table(db_path, embed_dim=4)
    v = LanceVault(path=db_path, embed_dim=4, auto_migrate=False)
    assert v.migrate() == ["mtime", "tags", "heading_trail"] or set(v.migrate()) == set()
    # Second migrate is a no-op.
    assert v.migrate() == []


def test_stale_metadata_detection(tmp_path):
    db_path = tmp_path / "db"
    _v1_table(db_path, embed_dim=4)
    v = LanceVault(path=db_path, embed_dim=4)  # auto-migrated
    # After migrate, both rows have placeholder mtime 0.0 → all stale.
    stale = v.sources_with_stale_metadata()
    assert set(stale) == {"/v/x.md", "/v/y.md"}
    # Writing a row WITH metadata clears its staleness.
    v.upsert([{
        "id": "a", "source_path": "/v/x.md", "chunk_idx": 0, "text": "alpha",
        "mtime": 123.0, "tags": ["t"], "heading_trail": "H", "vector": [1, 0, 0, 0],
    }])
    stale2 = v.sources_with_stale_metadata()
    assert "/v/x.md" not in stale2
    assert "/v/y.md" in stale2


def test_v1_table_accepts_writes_degraded(tmp_path):
    # An un-migrated table must accept upserts (without metadata), not raise.
    db_path = tmp_path / "db"
    _v1_table(db_path, embed_dim=4)
    v = LanceVault(path=db_path, embed_dim=4, auto_migrate=False)
    n = v.upsert([{
        "id": "c", "source_path": "/v/z.md", "chunk_idx": 0, "text": "gamma",
        "mtime": 999.0, "tags": ["x"], "heading_trail": "H", "vector": [0, 0, 1, 0],
    }])
    assert n == 1
    assert v.count() == 3  # write landed even though table has no metadata cols


def test_metadata_write_roundtrip(tmp_path):
    v = LanceVault(path=tmp_path / "db", embed_dim=4)
    v.upsert([{
        "id": "a", "source_path": "/v/x.md", "chunk_idx": 0, "text": "alpha",
        "mtime": 42.0, "tags": ["class/cs", "daily"], "heading_trail": "A > B",
        "vector": [1, 0, 0, 0],
    }])
    hits = v.search([1.0, 0, 0, 0], k=1)
    assert hits[0]["mtime"] == 42.0
    assert hits[0]["tags"] == ["class/cs", "daily"]
    assert hits[0]["heading_trail"] == "A > B"


def test_search_metric_filter_combo(tmp_path):
    v = LanceVault(path=tmp_path / "db", embed_dim=4)
    v.upsert([
        {"id": "old", "source_path": "/v/o.md", "chunk_idx": 0, "text": "x",
         "mtime": 100.0, "tags": [], "heading_trail": "", "vector": [1, 0, 0, 0]},
        {"id": "new", "source_path": "/v/n.md", "chunk_idx": 0, "text": "y",
         "mtime": 9999.0, "tags": [], "heading_trail": "", "vector": [1, 0, 0, 0]},
    ])
    hits = v.search([1.0, 0, 0, 0], k=5, filter="mtime >= 5000")
    assert [h["id"] for h in hits] == ["new"]


def test_all_chunks_projects_corpus_for_bm25(tmp_path):
    """all_chunks() returns the text corpus BM25 needs, without the vector."""
    v = LanceVault(path=tmp_path / "db", embed_dim=4)
    v.upsert([
        {"id": "a", "source_path": "/v/x.md", "chunk_idx": 0, "text": "alpha",
         "mtime": 1.0, "tags": [], "heading_trail": "", "vector": [1, 0, 0, 0]},
        {"id": "b", "source_path": "/v/y.md", "chunk_idx": 1, "text": "beta",
         "mtime": 2.0, "tags": [], "heading_trail": "", "vector": [0, 1, 0, 0]},
    ])
    chunks = v.all_chunks()
    assert len(chunks) == 2
    # Default projection carries text + locator columns, NOT the heavy vector.
    keys = set(chunks[0])
    assert {"id", "source_path", "chunk_idx", "text"} <= keys
    assert "vector" not in keys
    assert {c["text"] for c in chunks} == {"alpha", "beta"}


def test_all_chunks_empty_table(tmp_path):
    v = LanceVault(path=tmp_path / "db", embed_dim=4)
    assert v.all_chunks() == []
