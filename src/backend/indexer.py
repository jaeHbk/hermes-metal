from __future__ import annotations

import hashlib
import re
from pathlib import Path

import httpx

from .database import LanceVault

DEFAULT_EMBED_URL = "http://127.0.0.1:8081/v1/embeddings"
DEFAULT_EMBED_MODEL = "nomic-embed-text-v1.5"
EMBED_DIM = 768
DOCUMENT_TASK_PREFIX = "search_document"
QUERY_TASK_PREFIX = "search_query"

_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"^```")
_HTTP_TIMEOUT = httpx.Timeout(60.0, connect=5.0)


def stable_chunk_id(file_path: str | Path, chunk_idx: int) -> str:
    raw = f"{Path(file_path).resolve()}::{chunk_idx}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def read_markdown(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return _FRONTMATTER_RE.sub("", text, count=1).lstrip("\n")


def _split_paragraphs_preserving_fences(text: str) -> list[str]:
    lines = text.splitlines()
    blocks: list[str] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        if buf:
            block = "\n".join(buf).strip("\n")
            if block:
                blocks.append(block)
            buf.clear()

    for line in lines:
        if _CODE_FENCE_RE.match(line):
            buf.append(line)
            in_fence = not in_fence
            if not in_fence:
                flush()
            continue
        if in_fence:
            buf.append(line)
            continue
        if line.strip() == "":
            flush()
        else:
            buf.append(line)
    flush()
    return blocks


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 64) -> list[str]:
    if not text.strip():
        return []
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError("overlap must be >= 0 and < max_tokens")

    paragraphs = _split_paragraphs_preserving_fences(text)
    if not paragraphs:
        return []

    max_chars = max_tokens * 4
    overlap_chars = overlap * 4

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def joined(parts: list[str]) -> str:
        return "\n\n".join(parts)

    for para in paragraphs:
        para_len = len(para)
        sep = 2 if current else 0
        if current and current_len + sep + para_len > max_chars:
            chunks.append(joined(current))
            if overlap_chars > 0:
                tail = joined(current)[-overlap_chars:]
                current = [tail]
                current_len = len(tail)
            else:
                current = []
                current_len = 0

        if para_len > max_chars:
            if current:
                chunks.append(joined(current))
                current = []
                current_len = 0
            start = 0
            step = max_chars - overlap_chars if max_chars > overlap_chars else max_chars
            while start < para_len:
                end = min(start + max_chars, para_len)
                chunks.append(para[start:end])
                if end == para_len:
                    break
                start += step
            continue

        if current:
            current.append(para)
            current_len += sep + para_len
        else:
            current = [para]
            current_len = para_len

    if current:
        chunks.append(joined(current))

    return [c for c in chunks if c.strip()]


def embed(
    texts: list[str],
    embed_url: str = DEFAULT_EMBED_URL,
    *,
    task: str = DOCUMENT_TASK_PREFIX,
    model: str = DEFAULT_EMBED_MODEL,
    client: httpx.Client | None = None,
) -> list[list[float]]:
    if not texts:
        return []

    prefixed = [f"{task}: {t}" for t in texts]
    payload = {"model": model, "input": prefixed, "encoding_format": "float"}

    owns_client = client is None
    http = client or httpx.Client(timeout=_HTTP_TIMEOUT)
    try:
        resp = http.post(embed_url, json=payload)
        resp.raise_for_status()
        body = resp.json()
    finally:
        if owns_client:
            http.close()

    rows = sorted(body["data"], key=lambda r: r["index"])
    vectors = [[float(v) for v in r["embedding"]] for r in rows]

    if len(vectors) != len(texts):
        raise ValueError(
            f"embedding count mismatch: got {len(vectors)}, expected {len(texts)}"
        )
    for i, vec in enumerate(vectors):
        if len(vec) != EMBED_DIM:
            raise ValueError(
                f"embedding[{i}] dim {len(vec)} != {EMBED_DIM}"
            )
    return vectors


def index_file(
    file_path: str | Path,
    vault: LanceVault,
    embed_url: str = DEFAULT_EMBED_URL,
) -> int:
    resolved = str(Path(file_path).resolve())
    text = read_markdown(file_path)
    chunks = chunk_text(text)

    vault.delete_by_source(resolved)
    if not chunks:
        return 0

    vectors = embed(chunks, embed_url, task=DOCUMENT_TASK_PREFIX)
    records = [
        {
            "id": stable_chunk_id(resolved, idx),
            "source_path": resolved,
            "chunk_idx": idx,
            "text": chunk,
            "vector": vec,
        }
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    vault.upsert(records)
    return len(records)


class Indexer:
    def __init__(
        self,
        vault: LanceVault,
        embed_url: str = DEFAULT_EMBED_URL,
        *,
        model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        self.vault = vault
        self.embed_url = embed_url
        self.model = model

    def index_file(self, file_path: str | Path) -> int:
        return index_file(file_path, self.vault, self.embed_url)

    def remove_file(self, file_path: str | Path) -> None:
        self.vault.delete_by_source(str(Path(file_path).resolve()))

    def embed_query(self, text: str) -> list[float]:
        return embed(
            [text],
            self.embed_url,
            task=QUERY_TASK_PREFIX,
            model=self.model,
        )[0]
