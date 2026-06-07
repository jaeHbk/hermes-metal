from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

DEFAULT_EMBED_DIM = 768
TABLE_NAME = "vault_chunks"
# Repo-local default keeps storage co-located with logs/ and storage/slots/
# (matches watcher.main() and README). Override with HERMES_LANCEDB_PATH.
DEFAULT_DB_PATH = str(Path(__file__).resolve().parents[2] / "storage" / "lancedb")

# Schema version. v1 = id/source_path/chunk_idx/text/vector (the original
# retrieval-only schema). v2 adds the Phase C metadata columns (mtime, tags,
# heading_trail) that power date-filtered retrieval and reranking. Bumping
# this and adding the column to ``_META_FIELDS`` is the supported way to
# evolve the schema; ``migrate()`` reconciles an on-disk table to the
# current version in place.
SCHEMA_VERSION = 2

# Default vector metric. Nomic embeddings are L2-normalized, so cosine
# distance is the correct similarity measure; the original code used
# lancedb's L2 default, which is measurably worse on normalized vectors.
# This is a *query-time* choice (brute-force search picks the metric per
# query) so changing it needs no migration — old indexes get the better
# metric for free on the next query.
DEFAULT_METRIC = "cosine"

# Sidecar written next to the lance table so `hermes doctor` — which is
# stdlib-only and never imports lancedb — can report the schema version
# without opening the table. The lance table's actual columns remain the
# source of truth; this file is a cheap hint, reconciled on every migrate.
SCHEMA_HINT_FILE = ".hermes-schema.json"


def _v1_fields() -> list[pa.Field]:
    return [
        pa.field("id", pa.string(), nullable=False),
        pa.field("source_path", pa.string(), nullable=False),
        pa.field("chunk_idx", pa.int32(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
    ]


# Phase C metadata columns, in canonical order. Kept separate from the v1
# fields so migration knows exactly which columns to add to an old table.
_META_FIELDS: list[pa.Field] = [
    # File mtime as a POSIX timestamp (seconds). float64 so date-window
    # filters (`mtime >= X AND mtime < Y`) work in a lancedb `where` clause.
    pa.field("mtime", pa.float64(), nullable=False),
    # Tags harvested from frontmatter `tags:` and inline `#tag` markers.
    # list<string> so `array_has(tags, 'class/cs')` filters work.
    pa.field("tags", pa.list_(pa.string()), nullable=False),
    # The markdown heading hierarchy active at the chunk's start, e.g.
    # "Auth > Token rotation". Empty string when the chunk precedes any
    # heading. Used by the reranker for term-overlap scoring.
    pa.field("heading_trail", pa.string(), nullable=False),
]

_META_NAMES: tuple[str, ...] = tuple(f.name for f in _META_FIELDS)


def _build_schema(embed_dim: int) -> pa.Schema:
    """The full current (v2) schema. The vector column always comes last so
    the column order is stable for human inspection of the arrow table."""
    return pa.schema(
        _v1_fields()
        + _META_FIELDS
        + [
            pa.field(
                "vector",
                pa.list_(pa.float32(), list_size=embed_dim),
                nullable=False,
            )
        ]
    )


class LanceVault:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        embed_dim: int = DEFAULT_EMBED_DIM,
        table_name: str = TABLE_NAME,
        auto_migrate: bool = True,
    ) -> None:
        resolved = path or os.environ.get("HERMES_LANCEDB_PATH") or DEFAULT_DB_PATH
        self.path = Path(resolved).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.embed_dim = embed_dim
        self.table_name = table_name
        self._schema = _build_schema(embed_dim)
        self._db = lancedb.connect(str(self.path))
        self._table = self._open_or_create_table()
        # Reconcile an old (v1) table to the current schema on open so the
        # live system is never silently running on a stale schema. The add
        # is in-place, lossless, and instant (it appends columns with
        # defaults — existing vectors are untouched). Opt out with
        # auto_migrate=False for callers that must not mutate on connect.
        if auto_migrate and self._missing_meta_columns():
            self.migrate()

    # ------------------------------------------------------------ schema

    def _present_columns(self) -> set[str]:
        return {f.name for f in self._table.schema}

    def _missing_meta_columns(self) -> list[str]:
        present = self._present_columns()
        return [n for n in _META_NAMES if n not in present]

    @property
    def has_metadata(self) -> bool:
        """True iff the on-disk table carries the v2 metadata columns."""
        return not self._missing_meta_columns()

    def schema_version(self) -> int:
        return SCHEMA_VERSION if self.has_metadata else 1

    def _open_or_create_table(self):
        existing = set(self._db.table_names())
        if self.table_name in existing:
            return self._db.open_table(self.table_name)
        empty = pa.Table.from_pylist([], schema=self._schema)
        tbl = self._db.create_table(self.table_name, data=empty, schema=self._schema)
        # Assign before writing the hint: _write_schema_hint -> schema_version
        # -> _present_columns reads self._table, which must already be set.
        self._table = tbl
        self._write_schema_hint()
        return tbl

    def migrate(self) -> list[str]:
        """Add any missing v2 metadata columns to the on-disk table in place.

        Returns the list of columns actually added (empty if already current).
        The scalar columns get SQL-expression defaults; the list<string>
        ``tags`` column is added via a pyarrow field (lancedb's SQL parser
        can't express an empty list literal). Existing rows get the column
        defaults — real metadata only lands when those files are re-indexed
        (``hermes index --migrate`` does that repopulation).

        **Race-tolerant.** The watcher auto-migrates on open and a user may run
        ``hermes index --migrate`` at the same time — two separate processes,
        each opening the same v1 table. ``add_columns`` raises "Column already
        exists" for the loser of that race. We add each column independently
        and treat an already-exists failure as success (re-checking the live
        schema after each attempt), so a concurrent migration never crashes
        the watcher daemon or the CLI.
        """
        missing = self._missing_meta_columns()
        if not missing:
            self._write_schema_hint()
            return []

        added: list[str] = []
        # One column per add_columns call so a race on column A doesn't block
        # column B. Each is wrapped: if it raises because the column already
        # exists (another process won), re-open and re-check rather than fail.
        plan: list[tuple[str, Any]] = []
        if "mtime" in missing:
            plan.append(("mtime", {"mtime": "CAST(0.0 AS double)"}))
        if "heading_trail" in missing:
            plan.append(("heading_trail", {"heading_trail": "''"}))
        if "tags" in missing:
            # list<string> can't be a SQL literal default; add via arrow field.
            plan.append(("tags", pa.field("tags", pa.list_(pa.string()))))

        for name, spec in plan:
            try:
                self._table.add_columns(spec)
                added.append(name)
            except Exception as exc:  # noqa: BLE001 — tolerate concurrent add
                # Re-open and check: if the column exists now, a concurrent
                # migrate added it — that's fine. Otherwise it's a real error.
                self._table = self._db.open_table(self.table_name)
                if name in self._present_columns():
                    continue
                raise RuntimeError(
                    f"migrate: failed to add column {name!r}: {exc}"
                ) from exc

        # Reopen so the in-memory schema reflects all new columns.
        self._table = self._db.open_table(self.table_name)
        self._write_schema_hint()
        return added

    def _write_schema_hint(self) -> None:
        """Best-effort: drop a stdlib-readable version hint next to the table.

        Doctor reads this without importing lancedb. A write failure is
        non-fatal — the lance table's columns remain the real source of truth.
        """
        try:
            hint = {
                "schema_version": self.schema_version(),
                "embed_dim": self.embed_dim,
                "metric": DEFAULT_METRIC,
                "table": self.table_name,
            }
            (self.path / SCHEMA_HINT_FILE).write_text(
                json.dumps(hint, indent=2) + "\n", encoding="utf-8"
            )
        except OSError:
            pass

    # ------------------------------------------------------------ writes

    def upsert(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        # Write only the columns the on-disk table actually has, so an
        # un-migrated (v1) table still accepts writes (degraded: no metadata)
        # rather than raising a schema-mismatch from merge_insert.
        present = self._present_columns()
        write_fields = [f for f in self._schema if f.name in present]
        write_schema = pa.schema(write_fields)
        normalized = [
            self._normalize_record(r, columns={f.name for f in write_fields})
            for r in records
        ]
        table = pa.Table.from_pylist(normalized, schema=write_schema)
        (
            self._table.merge_insert("id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(table)
        )
        return len(normalized)

    def search(
        self,
        query_vector: list[float],
        k: int = 8,
        filter: str | None = None,
        *,
        metric: str | None = DEFAULT_METRIC,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.embed_dim:
            raise ValueError(
                f"query_vector dim {len(query_vector)} != embed_dim {self.embed_dim}"
            )
        query = self._table.search(query_vector)
        if metric:
            query = query.metric(metric)
        query = query.limit(k)
        if filter:
            query = query.where(filter)
        return query.to_list()

    def delete_by_source(self, path: str) -> None:
        escaped = path.replace("'", "''")
        self._table.delete(f"source_path = '{escaped}'")

    def count(self) -> int:
        return self._table.count_rows()

    def distinct_sources(self) -> list[str]:
        """Return every distinct ``source_path`` currently in the table.

        Used by ``hermes index --gc`` to find rows whose source file has
        been deleted/moved out of the vault. ``to_arrow()`` ships with the
        lancedb wheel; ``to_lance()`` would require pylance separately.
        """
        if self._table.count_rows() == 0:
            return []
        arrow_table = self._table.to_arrow()
        seen: set[str] = set()
        for v in arrow_table.column("source_path").to_pylist():
            if v is not None:
                seen.add(v)
        return sorted(seen)

    def sources_with_stale_metadata(self) -> list[str]:
        """Distinct source_paths whose rows carry placeholder metadata.

        After an in-place ``migrate()`` the new columns hold their defaults
        (mtime == 0.0) until the file is re-indexed. ``hermes index
        --migrate`` uses this to repopulate only what needs it, so a migration
        on a large vault doesn't have to re-embed rows that are already
        current. Returns [] on a v1 table (no mtime column to inspect).
        """
        if not self.has_metadata or self._table.count_rows() == 0:
            return []
        arrow_table = self._table.to_arrow()
        names = set(arrow_table.column_names)
        if "mtime" not in names:
            return []
        srcs = arrow_table.column("source_path").to_pylist()
        mtimes = arrow_table.column("mtime").to_pylist()
        seen: set[str] = set()
        for src, mt in zip(srcs, mtimes):
            if src is not None and (mt is None or mt == 0.0):
                seen.add(src)
        return sorted(seen)

    def close(self) -> None:
        self._table = None
        self._db = None

    def _normalize_record(
        self, record: dict[str, Any], *, columns: set[str] | None = None
    ) -> dict[str, Any]:
        vector = record["vector"]
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        vector = [float(v) for v in vector]
        if len(vector) != self.embed_dim:
            raise ValueError(
                f"vector dim {len(vector)} != embed_dim {self.embed_dim} "
                f"for id={record.get('id')!r}"
            )
        out: dict[str, Any] = {
            "id": str(record["id"]),
            "source_path": str(record["source_path"]),
            "chunk_idx": int(record["chunk_idx"]),
            "text": str(record["text"]),
            "vector": vector,
        }
        # Metadata columns are written only when the table has them. Defaults
        # mirror the migrate() defaults so a row written pre- and post-migrate
        # is indistinguishable once repopulated.
        cols = columns or set()
        if "mtime" in cols:
            out["mtime"] = float(record.get("mtime") or 0.0)
        if "tags" in cols:
            tags = record.get("tags") or []
            out["tags"] = [str(t) for t in tags]
        if "heading_trail" in cols:
            out["heading_trail"] = str(record.get("heading_trail") or "")
        return out
