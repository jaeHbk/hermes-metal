"""Tests for the markdown chunker in src/backend/indexer.py.

Chunking logic is the most invariant-heavy code in the project: getting
chunk boundaries wrong silently degrades retrieval quality. These tests
cover the documented behaviors and the three edge cases that previously
broke during development:

* Code fences must not split mid-fence.
* A single paragraph larger than ``max_tokens`` is sliced with overlap.
* Empty / whitespace-only input returns no chunks (not one empty chunk).
"""
from __future__ import annotations

from src.backend.indexer import chunk_text, read_markdown, _split_paragraphs_preserving_fences


def test_empty_input_returns_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\n   \n") == []


def test_short_input_returns_one_chunk():
    text = "Just a single paragraph that fits comfortably."
    chunks = chunk_text(text, max_tokens=512, overlap=64)
    assert chunks == [text]


def test_paragraphs_separated_by_blank_lines():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    blocks = _split_paragraphs_preserving_fences(text)
    assert blocks == ["First paragraph.", "Second paragraph.", "Third paragraph."]


def test_code_fence_kept_intact():
    text = (
        "Intro paragraph.\n\n"
        "```python\n"
        "def f():\n"
        "    pass\n"
        "\n"
        "    return 42\n"
        "```\n\n"
        "Tail paragraph."
    )
    blocks = _split_paragraphs_preserving_fences(text)
    # The code fence (with its internal blank line) must be ONE block, not split.
    fence_blocks = [b for b in blocks if "```" in b]
    assert len(fence_blocks) == 1
    assert "def f():" in fence_blocks[0]
    assert "return 42" in fence_blocks[0]


def test_oversized_paragraph_is_sliced_with_overlap():
    # max_chars = max_tokens * 4 → 40 chars per chunk.
    long_para = "x" * 200
    chunks = chunk_text(long_para, max_tokens=10, overlap=2)
    assert len(chunks) > 1
    # No chunk should exceed max_chars.
    assert all(len(c) <= 40 for c in chunks)
    # Overlap means concatenating chunks yields strictly MORE than the
    # original — if joined == 200, there was no overlap at all.
    joined = "".join(chunks)
    assert len(joined) > 200


def test_overlap_invariants_rejected():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("hello world", max_tokens=10, overlap=10)
    with pytest.raises(ValueError):
        chunk_text("hello world", max_tokens=10, overlap=-1)


def test_frontmatter_stripped_by_read_markdown(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("---\ntitle: Foo\ntag: bar\n---\n\nActual body.\n")
    assert read_markdown(f) == "Actual body.\n"


def test_no_frontmatter_passthrough(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Heading\n\nBody.\n")
    assert read_markdown(f) == "# Heading\n\nBody.\n"
