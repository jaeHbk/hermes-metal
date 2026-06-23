"""Watcher auto-ingest-of-pasted-URLs tests (bead hermes_metal-19t).

When a watched note's body contains http(s) URLs, the watcher fetches +
summarizes each NEW url into ``wiki/sources/<name>.md`` via the SAME write
path single/batch ingest use (``web.fetch_article`` → ``ingest_cmd.ingest_text``).

Everything is mocked: ``web.fetch_article`` is monkeypatched, the chat server
is stubbed at ``HermesClient.chat_sync``, and the embed step (``index_file``)
is a MagicMock so the suite needs no LanceDB or llama servers.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src import ingest_cmd, web, wiki
from src.backend.vault_filter import build_filter
from src.daemon.watcher import VaultWatcher


@pytest.fixture
def fake_chat(monkeypatch):
    def _stub(self, messages, **_kw):
        return ("## Summary\nS.\n\n## Key claims\n- c\n\n"
                "## Entities and concepts\n- [[E]]: x.\n\n## Open questions\n- q\n")
    monkeypatch.setattr("src.server.client.HermesClient.chat_sync", _stub)


@pytest.fixture
def vault(tmp_path, monkeypatch):
    """A vault with the wiki initialized inside it."""
    monkeypatch.setenv("HERMES_WIKI_PATH", str(tmp_path / "wiki"))
    monkeypatch.delenv("HERMES_VAULT_PATH", raising=False)
    monkeypatch.delenv("HERMES_AUTO_INGEST_LINKS", raising=False)
    wiki.init_wiki(wiki.get_paths(tmp_path))
    return tmp_path


def _make_watcher(vault: Path) -> VaultWatcher:
    w = VaultWatcher(
        vault_path=vault,
        vault=MagicMock(),
        embed_url="http://0/",
        debounce_s=0.01,
        vfilter=build_filter(vault),
    )
    # Replace the embed step with a mock so we never touch a real embed server.
    w._index_file = MagicMock(return_value=1)
    return w


def test_pasted_url_is_ingested(vault, fake_chat, monkeypatch):
    """A note containing an http URL → ingest_text writes wiki/sources/<name>.md."""
    url = "https://blog.example/great-article"
    art = web.Article(url=url, title="great-article", text="body " * 50)
    monkeypatch.setattr("src.web.fetch_article", lambda u, **_k: art)

    note = vault / "note.md"
    note.write_text(f"# My note\n\nCheck this out: {url}\n")

    w = _make_watcher(vault)
    w._fire_index(str(note))

    # Embed still happened.
    w._index_file.assert_called_once()
    # The source page was written via the shared path.
    page = wiki.get_paths(vault).sources_dir / "great-article.md"
    assert page.is_file()
    assert wiki.is_managed(page)
    idx = wiki.get_paths(vault).index.read_text()
    assert "great-article.md" in idx


def test_existing_source_not_refetched(vault, fake_chat, monkeypatch):
    """A URL whose source page already exists is skipped without re-fetching."""
    url = "https://blog.example/dup"
    art = web.Article(url=url, title="dup", text="body " * 50)
    fetched: list[str] = []

    def _fetch(u, **_k):
        fetched.append(u)
        return art

    monkeypatch.setattr("src.web.fetch_article", _fetch)

    note = vault / "note.md"
    note.write_text(f"see {url}\n")
    w = _make_watcher(vault)

    w._fire_index(str(note))
    assert fetched == [url]  # fetched once

    # Re-fire (e.g. another debounce / mtime touch). The pre-check sees the
    # existing source page and skips the fetch entirely.
    note.write_text(f"see {url}\n\nedited\n")
    w._fire_index(str(note))
    assert fetched == [url]  # still only one fetch


def test_fetch_failure_is_swallowed(vault, fake_chat, monkeypatch):
    """A WebError during auto-ingest must not kill the embed or the timer thread."""
    url = "https://paywall.example/x"
    monkeypatch.setattr(
        "src.web.fetch_article",
        lambda u, **_k: (_ for _ in ()).throw(web.WebError("HTTP 403")),
    )

    note = vault / "note.md"
    note.write_text(f"blocked: {url}\n")
    w = _make_watcher(vault)

    # Must NOT raise.
    w._fire_index(str(note))
    # Embed still happened despite the fetch failure.
    w._index_file.assert_called_once()
    # No source page written.
    assert not (wiki.get_paths(vault).sources_dir / "x.md").is_file()


def test_chat_server_down_is_swallowed(vault, monkeypatch):
    """Chat-server-down (HermesError) during summarize must not abort the embed."""
    from src.server.client import HermesError

    url = "https://blog.example/down"
    art = web.Article(url=url, title="down", text="body " * 50)
    monkeypatch.setattr("src.web.fetch_article", lambda u, **_k: art)

    def _boom(self, messages, **_kw):
        raise HermesError("connection refused")

    monkeypatch.setattr("src.server.client.HermesClient.chat_sync", _boom)

    note = vault / "note.md"
    note.write_text(f"link: {url}\n")
    w = _make_watcher(vault)

    w._fire_index(str(note))
    w._index_file.assert_called_once()
    assert not (wiki.get_paths(vault).sources_dir / "down.md").is_file()


def test_toggle_off_disables_autolink(vault, fake_chat, monkeypatch):
    """HERMES_AUTO_INGEST_LINKS=0 disables auto-ingest entirely."""
    monkeypatch.setenv("HERMES_AUTO_INGEST_LINKS", "0")
    url = "https://blog.example/off"
    fetched: list[str] = []
    monkeypatch.setattr("src.web.fetch_article",
                        lambda u, **_k: fetched.append(u))

    note = vault / "note.md"
    note.write_text(f"link: {url}\n")
    w = _make_watcher(vault)

    w._fire_index(str(note))
    w._index_file.assert_called_once()  # embed still happens
    assert fetched == []  # never fetched


def test_wiki_notes_are_not_self_ingested(vault, fake_chat, monkeypatch):
    """A note under wiki/ must not have its URLs auto-ingested (no self-loop)."""
    url = "https://blog.example/inner"
    fetched: list[str] = []
    monkeypatch.setattr("src.web.fetch_article",
                        lambda u, **_k: fetched.append(u))

    # A managed source page that itself references a URL.
    inner = wiki.get_paths(vault).sources_dir / "inner.md"
    inner.write_text(f"---\nhermes-managed: true\n---\n# inner\n\nrefs {url}\n")

    w = _make_watcher(vault)
    w._fire_index(str(inner))
    # The embed of the wiki page itself still runs (watcher indexes wiki pages),
    # but no auto-ingest fetch happens for its URLs.
    assert fetched == []


def test_no_urls_means_no_ingest(vault, fake_chat, monkeypatch):
    """A plain note with no URLs triggers no fetch at all."""
    fetched: list[str] = []
    monkeypatch.setattr("src.web.fetch_article",
                        lambda u, **_k: fetched.append(u))
    note = vault / "plain.md"
    note.write_text("# Plain\n\nNo links here, just prose.\n")
    w = _make_watcher(vault)
    w._fire_index(str(note))
    w._index_file.assert_called_once()
    assert fetched == []
