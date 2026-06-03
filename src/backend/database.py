from __future__ import annotations

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


def _build_schema(embed_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string(), nullable=False),
            pa.field("source_path", pa.string(), nullable=False),
            pa.field("chunk_idx", pa.int32(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field(
                "vector",
                pa.list_(pa.float32(), list_size=embed_dim),
                nullable=False,
            ),
        ]
    )


class LanceVault:
    def __init__(
        self,
        path: str | Path | None = None,
        *,
        embed_dim: int = DEFAULT_EMBED_DIM,
        table_name: str = TABLE_NAME,
    ) -> None:
        resolved = path or os.environ.get("HERMES_LANCEDB_PATH") or DEFAULT_DB_PATH
        self.path = Path(resolved).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.embed_dim = embed_dim
        self.table_name = table_name
        self._schema = _build_schema(embed_dim)
        self._db = lancedb.connect(str(self.path))
        self._table = self._open_or_create_table()

    def _open_or_create_table(self):
        existing = set(self._db.table_names())
        if self.table_name in existing:
            return self._db.open_table(self.table_name)
        empty = pa.Table.from_pylist([], schema=self._schema)
        return self._db.create_table(self.table_name, data=empty, schema=self._schema)

    def upsert(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        normalized = [self._normalize_record(r) for r in records]
        table = pa.Table.from_pylist(normalized, schema=self._schema)
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
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.embed_dim:
            raise ValueError(
                f"query_vector dim {len(query_vector)} != embed_dim {self.embed_dim}"
            )
        query = self._table.search(query_vector).limit(k)
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

    def close(self) -> None:
        self._table = None
        self._db = None

    def _normalize_record(self, record: dict[str, Any]) -> dict[str, Any]:
        vector = record["vector"]
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        vector = [float(v) for v in vector]
        if len(vector) != self.embed_dim:
            raise ValueError(
                f"vector dim {len(vector)} != embed_dim {self.embed_dim} "
                f"for id={record.get('id')!r}"
            )
        return {
            "id": str(record["id"]),
            "source_path": str(record["source_path"]),
            "chunk_idx": int(record["chunk_idx"]),
            "text": str(record["text"]),
            "vector": vector,
        }
