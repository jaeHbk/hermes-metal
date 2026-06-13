"""Tests for src/ingest_links_cmd.py — the batch URL driver.

web.fetch_article and the chat client are monkeypatched; index_cmd.run is
monkeypatched so the batch test needs no embed server. All-mocked, no network.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src import ingest_links_cmd, ingest_cmd, web, wiki


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_WIKI_PATH", str(tmp_path / "wiki"))
    monkeypatch.delenv("HERMES_VAULT_PATH", raising=False)
    return tmp_path


@pytest.fixture
def fake_chat(monkeypatch):
    def _stub(self, messages, **_kw):
        return ("## Summary\nS.\n\n## Key claims\n- c\n\n"
                "## Entities and concepts\n- [[E]]: x.\n\n## Open questions\n- q\n")
    monkeypatch.setattr("src.server.client.HermesClient.chat_sync", _stub)


@pytest.fixture
def no_index(monkeypatch):
    """Stub index_cmd.run so auto-index doesn't hit the embed server.
    Returns a list that records calls."""
    calls = []
    def _stub(argv=None):
        calls.append(argv)
        return 0
    monkeypatch.setattr("src.index_cmd.run", _stub)
    return calls


def _fake_fetch(mapping):
    """Build a fetch_article stand-in from {url: Article-or-WebError}."""
    def _f(url, **_kw):
        v = mapping[url]
        if isinstance(v, Exception):
            raise v
        return v
    return _f


def _links_file(tmp_path: Path, urls: list[str]) -> Path:
    f = tmp_path / "links.txt"
    f.write_text("\n".join(urls) + "\n")
    return f


def test_batch_ingests_all_good(vault, fake_chat, no_index, monkeypatch):
    wiki.init_wiki()
    paths = wiki.get_paths()
    urls = ["https://a.example/one", "https://b.example/two", "https://c.example/three"]
    mapping = {u: web.Article(url=u, title=u.rsplit("/", 1)[1], text="body " * 50)
               for u in urls}
    monkeypatch.setattr("src.web.fetch_article", _fake_fetch(mapping))

    f = _links_file(vault, urls)
    rc = ingest_links_cmd.run([str(f)])
    assert rc == 0
    idx = paths.index.read_text()
    assert "[one](sources/one.md)" in idx
    assert "[two](sources/two.md)" in idx
    assert "[three](sources/three.md)" in idx
    # Auto-index ran once with --backfill.
    assert no_index and "--backfill" in no_index[0]


def test_batch_skips_failures_and_writes_failed_file(vault, fake_chat, no_index, monkeypatch):
    wiki.init_wiki()
    good = "https://a.example/ok"
    bad = "https://paywall.example/x"
    mapping = {
        good: web.Article(url=good, title="ok", text="body " * 50),
        bad: web.WebError("HTTP 403 fetching " + bad),
    }
    monkeypatch.setattr("src.web.fetch_article", _fake_fetch(mapping))

    f = _links_file(vault, [good, bad])
    rc = ingest_links_cmd.run([str(f)])
    # Non-zero because at least one failed.
    assert rc == 1
    failed = f.with_suffix(".failed.txt")
    assert failed.is_file()
    assert bad in failed.read_text()
    assert good not in failed.read_text()


def test_batch_rerun_is_idempotent(vault, fake_chat, no_index, monkeypatch):
    wiki.init_wiki()
    u = "https://a.example/one"
    mapping = {u: web.Article(url=u, title="one", text="body " * 50)}
    monkeypatch.setattr("src.web.fetch_article", _fake_fetch(mapping))
    f = _links_file(vault, [u])

    assert ingest_links_cmd.run([str(f)]) == 0
    # Second run: page already exists → ALREADY_EXISTS, no duplicate row.
    assert ingest_links_cmd.run([str(f)]) == 0
    idx = wiki.get_paths().index.read_text()
    assert idx.count("[one](sources/one.md)") == 1


def test_batch_skips_comments_blanks_and_nonhttp(vault, fake_chat, no_index, monkeypatch):
    wiki.init_wiki()
    u = "https://a.example/one"
    mapping = {u: web.Article(url=u, title="one", text="body " * 50)}
    fetched = []
    def _f(url, **_kw):
        fetched.append(url)
        return mapping[url]
    monkeypatch.setattr("src.web.fetch_article", _f)

    f = vault / "links.txt"
    f.write_text("\n".join([
        "# a comment", "", u, "  ", "ftp://nope.example/x", u,  # dup at end
    ]) + "\n")
    rc = ingest_links_cmd.run([str(f)])
    assert rc == 0
    # Only the one valid http URL fetched, and only once (de-duped).
    assert fetched == [u]


def test_batch_no_index_flag_skips_indexing(vault, fake_chat, no_index, monkeypatch):
    wiki.init_wiki()
    u = "https://a.example/one"
    mapping = {u: web.Article(url=u, title="one", text="body " * 50)}
    monkeypatch.setattr("src.web.fetch_article", _fake_fetch(mapping))
    f = _links_file(vault, [u])
    rc = ingest_links_cmd.run([str(f), "--no-index"])
    assert rc == 0
    assert no_index == []  # index_cmd.run never called


def test_batch_uninitialized_wiki_fails_before_fetch(vault, monkeypatch):
    # Wiki NOT initialized.
    called = []
    monkeypatch.setattr("src.web.fetch_article",
                        lambda *a, **k: called.append(1))
    f = _links_file(vault, ["https://a.example/one"])
    rc = ingest_links_cmd.run([str(f)])
    assert rc == 2
    assert called == []  # never fetched


def test_batch_missing_file_returns_2(vault):
    rc = ingest_links_cmd.run([str(vault / "nope.txt")])
    assert rc == 2
