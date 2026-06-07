from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import httpx

from .database import LanceVault

DEFAULT_EMBED_URL = "http://127.0.0.1:8081/v1/embeddings"
DEFAULT_EMBED_MODEL = "nomic-embed-text-v1.5"
EMBED_DIM = 768
DOCUMENT_TASK_PREFIX = "search_document"
QUERY_TASK_PREFIX = "search_query"

_FRONTMATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)
_FRONTMATTER_CAPTURE_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"^```")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Inline #tag: a '#' at start-of-line or after whitespace, followed by a
# letter then word/slash/dash chars. The leading non-'#' guard means markdown
# headings ('## Foo' → '#' then '#', no letter) never match as tags.
_INLINE_TAG_RE = re.compile(r"(?:^|\s)#([A-Za-z][\w/-]*)")
_HTTP_TIMEOUT = httpx.Timeout(60.0, connect=5.0)


def stable_chunk_id(file_path: str | Path, chunk_idx: int) -> str:
    raw = f"{Path(file_path).resolve()}::{chunk_idx}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def read_markdown(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return _FRONTMATTER_RE.sub("", text, count=1).lstrip("\n")


# --------------------------------------------------------------- metadata


def extract_tags(raw_text: str) -> list[str]:
    """Harvest tags from frontmatter ``tags:`` and inline ``#tag`` markers.

    Frontmatter forms handled (no YAML dependency — the parse is deliberately
    small and tolerant):

    * ``tags: [a, b, c]``           inline flow list
    * ``tags: a, b`` / ``tags: a``  scalar / comma list
    * a ``tags:`` line followed by ``  - item`` block-list lines

    Inline ``#tag`` markers anywhere in the body are added too. Returns a
    de-duplicated, order-preserving list. Never raises on malformed input.
    """
    tags: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        t = t.strip().strip("\"'").lstrip("#").strip()
        if t and t not in seen:
            seen.add(t)
            tags.append(t)

    fm = _FRONTMATTER_CAPTURE_RE.match(raw_text)
    if fm:
        lines = fm.group(1).splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            m = re.match(r"^\s*tags\s*:\s*(.*)$", line, re.IGNORECASE)
            if not m:
                i += 1
                continue
            rest = m.group(1).strip()
            if rest.startswith("[") and rest.endswith("]"):
                for part in rest[1:-1].split(","):
                    _add(part)
            elif rest:
                # scalar or comma/space separated on the same line
                for part in re.split(r"[,\s]+", rest):
                    _add(part)
            else:
                # block list: subsequent '  - item' lines
                j = i + 1
                while j < len(lines):
                    bm = re.match(r"^\s*-\s+(.*\S)\s*$", lines[j])
                    if not bm:
                        break
                    _add(bm.group(1))
                    j += 1
                i = j
                continue
            i += 1

    # Inline tags from the body (frontmatter stripped so we don't double-scan).
    body = _FRONTMATTER_RE.sub("", raw_text, count=1)
    for m in _INLINE_TAG_RE.finditer(body):
        _add(m.group(1))
    return tags


def file_mtime(path: str | Path) -> float:
    """POSIX mtime (seconds) of ``path``; 0.0 if the file is gone."""
    try:
        return float(Path(path).stat().st_mtime)
    except OSError:
        return 0.0


# --------------------------------------------------------------- chunking


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


def _heading_trail_for_block(block: str, stack: list[tuple[int, str]]) -> str:
    """Update ``stack`` for any heading lines in ``block`` and return the
    heading trail (``"H1 > H2"``) in effect for the block.

    A block that begins with a heading includes that heading in its own
    trail (so a chunk starting at ``## Auth`` is tagged under Auth). The
    stack persists across blocks so body paragraphs inherit the nearest
    preceding heading. Code fences never start with a bare ``#␣`` after the
    fence marker, so headings inside fences are not matched here.
    """
    for line in block.splitlines():
        m = _HEADING_RE.match(line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        # Pop deeper-or-equal levels, then push this heading.
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return " > ".join(title for _lvl, title in stack)


def chunk_markdown(
    text: str, max_tokens: int = 512, overlap: int = 64
) -> list[dict[str, str]]:
    """Chunk markdown, tracking the heading trail active at each chunk start.

    Returns a list of ``{"text": ..., "heading_trail": ...}`` dicts. This is
    the single grouping implementation; :func:`chunk_text` is a thin wrapper
    that drops the heading trail, so the two can never drift.
    """
    if not text.strip():
        return []
    if overlap < 0 or overlap >= max_tokens:
        raise ValueError("overlap must be >= 0 and < max_tokens")

    # Build (block_text, heading_trail) pairs in document order.
    stack: list[tuple[int, str]] = []
    blocks: list[tuple[str, str]] = []
    for block in _split_paragraphs_preserving_fences(text):
        trail = _heading_trail_for_block(block, stack)
        blocks.append((block, trail))
    if not blocks:
        return []

    max_chars = max_tokens * 4
    overlap_chars = overlap * 4

    chunks: list[dict[str, str]] = []
    current: list[str] = []
    current_len = 0
    current_trail = ""           # heading trail of the first real block in `current`
    have_real_block = False      # an overlap tail seed doesn't count as real

    def joined(parts: list[str]) -> str:
        return "\n\n".join(parts)

    def emit(parts: list[str], trail: str) -> None:
        body = joined(parts)
        if body.strip():
            chunks.append({"text": body, "heading_trail": trail})

    for para, trail in blocks:
        para_len = len(para)
        sep = 2 if current else 0
        if current and current_len + sep + para_len > max_chars:
            emit(current, current_trail)
            if overlap_chars > 0:
                tail = joined(current)[-overlap_chars:]
                current = [tail]
                current_len = len(tail)
            else:
                current = []
                current_len = 0
            have_real_block = False
            current_trail = ""

        if para_len > max_chars:
            if current:
                emit(current, current_trail)
                current = []
                current_len = 0
                have_real_block = False
                current_trail = ""
            start = 0
            step = max_chars - overlap_chars if max_chars > overlap_chars else max_chars
            while start < para_len:
                end = min(start + max_chars, para_len)
                emit([para[start:end]], trail)
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
        if not have_real_block:
            current_trail = trail
            have_real_block = True

    if current:
        emit(current, current_trail)

    return chunks


def chunk_text(text: str, max_tokens: int = 512, overlap: int = 64) -> list[str]:
    """Backward-compatible chunker: chunk text, discarding heading trails.

    Delegates to :func:`chunk_markdown` so the grouping logic lives in one
    place. Existing callers and tests that expect ``list[str]`` are unaffected.
    """
    return [c["text"] for c in chunk_markdown(text, max_tokens=max_tokens, overlap=overlap)]


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
    raw = Path(file_path).read_text(encoding="utf-8")
    text = _FRONTMATTER_RE.sub("", raw, count=1).lstrip("\n")
    chunks = chunk_markdown(text)

    vault.delete_by_source(resolved)
    if not chunks:
        return 0

    # File-level metadata is shared by every chunk; chunk-level heading_trail
    # comes from the chunker.
    mtime = file_mtime(file_path)
    tags = extract_tags(raw)

    vectors = embed([c["text"] for c in chunks], embed_url, task=DOCUMENT_TASK_PREFIX)
    records = [
        {
            "id": stable_chunk_id(resolved, idx),
            "source_path": resolved,
            "chunk_idx": idx,
            "text": chunk["text"],
            "mtime": mtime,
            "tags": tags,
            "heading_trail": chunk["heading_trail"],
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
