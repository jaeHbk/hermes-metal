"""Tests for src/ingest_cmd.py — happy path and the foot-guns.

We mock HermesClient.chat_sync so the suite doesn't need a chat server.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src import ingest_cmd, wiki


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_WIKI_PATH", str(tmp_path / "wiki"))
    monkeypatch.delenv("HERMES_VAULT_PATH", raising=False)
    return tmp_path


@pytest.fixture
def fake_chat(monkeypatch):
    """Replace HermesClient.chat_sync with a deterministic stub."""
    def _stub_chat_sync(self, messages, **_kwargs):
        return (
            "## Summary\nA short factual summary line.\n\n"
            "## Key claims\n- The sky is blue.\n- Water is wet.\n\n"
            "## Entities and concepts\n- [[Sky]]: overhead.\n\n"
            "## Open questions\n- Why?\n"
        )
    monkeypatch.setattr(
        "src.server.client.HermesClient.chat_sync", _stub_chat_sync
    )


def _make_source(path: Path, body: str = "Some source content.") -> Path:
    path.write_text(body, encoding="utf-8")
    return path


# ----------------------------------------------------------- happy path


def test_ingest_writes_page_index_log(vault, fake_chat):
    wiki.init_wiki()
    paths = wiki.get_paths()
    src = _make_source(vault / "doc1.md")

    rc = ingest_cmd.run([str(src)])
    assert rc == 0

    page = paths.sources_dir / "doc1.md"
    assert page.is_file()
    assert wiki.is_managed(page)
    body = page.read_text()
    assert "## Summary" in body
    assert "[[Sky]]" in body
    # Frontmatter holds the source path.
    assert f'source-path: "{src}"' in body or f'source-path: "{src.resolve()}"' in body

    # Index has a Sources row.
    assert "[doc1](sources/doc1.md)" in paths.index.read_text()
    # Log has an ingest entry.
    log = paths.log.read_text()
    assert "ingest | doc1" in log


def test_ingest_refuses_existing_page(vault, fake_chat, capsys):
    wiki.init_wiki()
    paths = wiki.get_paths()
    src = _make_source(vault / "doc.md")
    ingest_cmd.run([str(src)])

    # Second ingest without --force is a no-op.
    rc = ingest_cmd.run([str(src)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "already exists" in err


def test_ingest_force_overwrites(vault, fake_chat):
    wiki.init_wiki()
    paths = wiki.get_paths()
    src = _make_source(vault / "doc.md")
    ingest_cmd.run([str(src)])

    page = paths.sources_dir / "doc.md"
    first_content = page.read_text()

    # Sleep would be needed for ts to differ, but we just check overwrite.
    rc = ingest_cmd.run([str(src), "--force"])
    assert rc == 0
    # Page still exists, write succeeded.
    assert page.is_file()
    # Index didn't duplicate the row.
    idx = paths.index.read_text()
    assert idx.count("[doc](sources/doc.md)") == 1


def test_ingest_refuses_handwritten_target(vault, fake_chat, capsys):
    """A pre-existing non-managed file must not be overwritten even by name collision."""
    wiki.init_wiki()
    paths = wiki.get_paths()
    src = _make_source(vault / "doc.md")
    # User pre-wrote a file at the target path WITHOUT hermes-managed.
    handwritten = paths.sources_dir / "doc.md"
    handwritten.write_text("# Hand-written\n\nMy own content.\n")

    rc = ingest_cmd.run([str(src)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "hand-written" in err
    # User's file is untouched.
    assert handwritten.read_text() == "# Hand-written\n\nMy own content.\n"


def test_ingest_refuses_handwritten_even_with_force(vault, fake_chat, capsys):
    """The hand-written guard must apply REGARDLESS of --force. Prior
    bug: the guard was nested inside `not args.force`, so --force
    silently overwrote user files."""
    wiki.init_wiki()
    paths = wiki.get_paths()
    src = _make_source(vault / "doc.md")
    handwritten = paths.sources_dir / "doc.md"
    handwritten.write_text("# Hand-written\n\nMy own content.\n")

    rc = ingest_cmd.run([str(src), "--force"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "hand-written" in err
    assert handwritten.read_text() == "# Hand-written\n\nMy own content.\n"


def test_ingest_rolls_back_page_on_index_failure(vault, fake_chat, monkeypatch):
    """If update_index_row raises, the just-written page must be unlinked
    so a retry isn't blocked by an orphan file with no log entry."""
    from src import wiki as _wiki
    wiki.init_wiki()
    paths = wiki.get_paths()
    src = _make_source(vault / "doc.md")

    def boom(*a, **k):
        raise OSError("simulated index write failure")
    monkeypatch.setattr(_wiki, "update_index_row", boom)

    with pytest.raises(OSError):
        ingest_cmd.run([str(src)])
    # Page was unlinked; the wiki is internally consistent.
    assert not (paths.sources_dir / "doc.md").exists()


def test_ingest_missing_source_returns_2(vault, fake_chat):
    wiki.init_wiki()
    rc = ingest_cmd.run([str(vault / "no-such-file.md")])
    assert rc == 2


def test_ingest_uninitialized_wiki_returns_2(vault, fake_chat):
    src = _make_source(vault / "doc.md")
    # Wiki NOT initialized.
    rc = ingest_cmd.run([str(src)])
    assert rc == 2


# ----------------------------------------------------------- name override


def test_name_override_slugifies(vault, fake_chat):
    wiki.init_wiki()
    paths = wiki.get_paths()
    src = _make_source(vault / "doc.md")
    rc = ingest_cmd.run([str(src), "--name", "Wild Spaces & Things!"])
    assert rc == 0
    # Trailing underscores/dashes/dots get stripped by _slugify.
    page = paths.sources_dir / "Wild_Spaces___Things.md"
    assert page.is_file()


# ----------------------------------------------------------- chat error


def test_ingest_chat_error_returns_1(vault, monkeypatch):
    from src.server.client import HermesError
    wiki.init_wiki()
    src = _make_source(vault / "doc.md")

    def boom(self, *a, **k):
        raise HermesError("server is down")
    monkeypatch.setattr("src.server.client.HermesClient.chat_sync", boom)

    rc = ingest_cmd.run([str(src)])
    assert rc == 1


def test_slugify_protects_meta_filenames():
    assert ingest_cmd._slugify("index") == "index-source"
    assert ingest_cmd._slugify("log") == "log-source"
    assert ingest_cmd._slugify("normal-name") == "normal-name"
    assert ingest_cmd._slugify("") == "page"


def test_extract_summary_finds_section():
    body = "## Summary\nThis is the summary.\n\n## Key claims\n- x\n"
    assert ingest_cmd._extract_summary(body).startswith("This is the summary")


def test_extract_summary_falls_back_when_section_missing():
    body = "Something else entirely.\n"
    out = ingest_cmd._extract_summary(body)
    assert "Something else" in out


def test_ingest_text_truncates_long_body_once(vault, fake_chat, monkeypatch):
    """A source over the cap is truncated exactly once, with a single marker.
    Guards against re-introducing a second truncation pass."""
    import src.ingest_cmd as ic
    captured = {}

    def _capture_chat(self, messages, **_kw):
        captured["user"] = messages[1]["content"]
        return ("## Summary\nS.\n\n## Key claims\n- c\n\n"
                "## Entities and concepts\n- [[E]]: x.\n\n## Open questions\n- q\n")
    monkeypatch.setattr("src.server.client.HermesClient.chat_sync", _capture_chat)

    wiki.init_wiki()
    big = "x" * 50_000
    res = ic.ingest_text(big, page_name="big", source_label="big-src")
    assert res.status == ic.WROTE
    # Exactly one truncation marker in the prompt sent to the chat server.
    assert captured["user"].count("[truncated for length]") == 1


# ----------------------------------------------------------- ingest_text core


def test_ingest_text_writes_with_extra_frontmatter(vault, fake_chat):
    """The shared core writes a page and threads extra frontmatter (the
    URL-provenance path uses this for source-url)."""
    wiki.init_wiki()
    paths = wiki.get_paths()

    res = ingest_cmd.ingest_text(
        "Raw article body text.",
        page_name="My Article",
        source_label="https://example.com/my-article",
        extra_frontmatter={"source-url": "https://example.com/my-article",
                           "ingested-via": "url"},
    )
    assert res.status == ingest_cmd.WROTE
    page = paths.sources_dir / "My_Article.md"
    assert page.is_file()
    body = page.read_text()
    assert 'source-url: "https://example.com/my-article"' in body
    assert 'ingested-via: "url"' in body


def test_ingest_text_already_exists_is_idempotent(vault, fake_chat):
    wiki.init_wiki()
    ingest_cmd.ingest_text("body", page_name="dup", source_label="x")
    res = ingest_cmd.ingest_text("body", page_name="dup", source_label="x")
    assert res.status == ingest_cmd.ALREADY_EXISTS


def test_ingest_text_refuses_handwritten(vault, fake_chat):
    wiki.init_wiki()
    paths = wiki.get_paths()
    (paths.sources_dir / "hand.md").write_text("# mine\n\nhand-written\n")
    res = ingest_cmd.ingest_text("body", page_name="hand", source_label="x", force=True)
    assert res.status == ingest_cmd.REFUSED_HANDWRITTEN
    assert (paths.sources_dir / "hand.md").read_text() == "# mine\n\nhand-written\n"
