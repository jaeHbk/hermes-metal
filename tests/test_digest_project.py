"""Tests for the digest's project-correlation section (bead hermes_metal-1ej).

A designated "project" (env HERMES_PROJECT_NOTE or a ``project:`` key in
config/vault.yaml, path relative to the vault root, a note OR a folder) drives
a "How this connects to <project>" section that ranks the day's changed notes
+ newly-ingested sources by relevance to the project's content.

All scoring is exercised in the *mechanical* (no-embed-server) path here by
passing ``embed_fn=None``-equivalent stubs; an embed-up path is covered with a
stub embedder. No network, no LanceDB, no llama servers.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src import digest, wiki


TZ = timezone(timedelta(hours=-7))
DAY = datetime(2026, 6, 6, 12, 0, tzinfo=TZ)


@pytest.fixture
def vault(tmp_path, monkeypatch):
    v = tmp_path / "vault"
    (v / "journal").mkdir(parents=True)
    monkeypatch.setenv("HERMES_VAULT_PATH", str(v))
    monkeypatch.setenv("HERMES_WIKI_PATH", str(v / "wiki"))
    monkeypatch.delenv("HERMES_DIGEST_PUSH", raising=False)
    monkeypatch.delenv("HERMES_PROJECT_NOTE", raising=False)
    return v


def _touch(path: Path, content: str, when: datetime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    ts = when.timestamp()
    os.utime(path, (ts, ts))


def _stub_chat(learn="Learned things."):
    def fn(messages, max_tokens):
        return learn
    return fn


def test_no_project_configured_omits_section(vault):
    """With no project configured, the correlation section is absent entirely."""
    _touch(vault / "journal" / "2026-06-06.md", "# Mon\n\nnote about widgets", DAY)
    res = digest.build_digest(vault, DAY, chat_fn=_stub_chat())
    assert res.project is None
    assert res.project_correlations == []
    md = res.to_markdown()
    assert "How this connects to" not in md


def test_project_section_ranks_relevant_above_irrelevant(vault, monkeypatch):
    """Mechanical fallback: a note sharing terms with the project ranks above
    an unrelated note."""
    # Project note is about distributed consensus / raft.
    _touch(vault / "project.md",
           "# Raft project\n\nImplementing the raft consensus algorithm: "
           "leader election, log replication, and term voting.", DAY - timedelta(days=10))
    monkeypatch.setenv("HERMES_PROJECT_NOTE", "project.md")

    _touch(vault / "journal" / "relevant.md",
           "# Notes\n\nDebugged raft leader election and log replication today.", DAY)
    _touch(vault / "journal" / "irrelevant.md",
           "# Groceries\n\nBought milk, eggs, and a loaf of sourdough bread.", DAY)

    res = digest.build_digest(vault, DAY, chat_fn=_stub_chat())
    assert res.project is not None
    rels = [rel for rel, _score in res.project_correlations]
    assert rels, "expected at least one correlation"
    # The raft note must rank above the groceries note.
    assert "relevant.md" in rels[0]
    md = res.to_markdown()
    assert "How this connects to" in md
    assert "relevant.md" in md


def test_project_from_yaml_config(vault, monkeypatch, tmp_path):
    """A ``project:`` key in a vault.yaml config is honored (env unset)."""
    cfg = tmp_path / "vault.yaml"
    cfg.write_text("project: \"project.md\"\n")
    _touch(vault / "project.md", "# P\n\nkanban board sprints velocity", DAY - timedelta(days=5))
    _touch(vault / "journal" / "2026-06-06.md",
           "# Mon\n\nplanned sprint velocity for the kanban board", DAY)

    res = digest.build_digest(vault, DAY, chat_fn=_stub_chat(), config_path=cfg)
    assert res.project is not None
    assert res.project_correlations


def test_project_folder_aggregates_notes(vault, monkeypatch):
    """A project pointing at a folder aggregates all notes inside it."""
    _touch(vault / "proj" / "a.md", "# A\n\nembeddings vector search lancedb", DAY - timedelta(days=3))
    _touch(vault / "proj" / "b.md", "# B\n\nreranker cosine similarity", DAY - timedelta(days=3))
    monkeypatch.setenv("HERMES_PROJECT_NOTE", "proj")

    _touch(vault / "journal" / "2026-06-06.md",
           "# Mon\n\ntuned the reranker cosine similarity over embeddings", DAY)
    res = digest.build_digest(vault, DAY, chat_fn=_stub_chat())
    assert res.project is not None
    assert res.project_correlations


def test_project_uses_embeddings_when_available(vault, monkeypatch):
    """When an embed_fn is supplied, similarity ranking uses it. The stub
    returns vectors that make 'relevant.md' nearest to the project."""
    _touch(vault / "project.md", "# P\n\nanything", DAY - timedelta(days=10))
    monkeypatch.setenv("HERMES_PROJECT_NOTE", "project.md")
    _touch(vault / "journal" / "relevant.md", "# R\n\nclose", DAY)
    _touch(vault / "journal" / "far.md", "# F\n\ndistant", DAY)

    # Deterministic stub embedder: map by content keyword to fixed vectors.
    def embed_fn(texts):
        out = []
        for t in texts:
            if "anything" in t:       # project
                out.append([1.0, 0.0])
            elif "close" in t:
                out.append([0.9, 0.1])   # near project
            else:
                out.append([0.0, 1.0])   # far from project
        return out

    res = digest.build_digest(vault, DAY, chat_fn=_stub_chat(), embed_fn=embed_fn)
    rels = [rel for rel, _ in res.project_correlations]
    assert "relevant.md" in rels[0]


def test_project_embed_failure_falls_back_to_mechanical(vault, monkeypatch):
    """If the embedder raises (server down), we degrade to term overlap and
    still produce a section — never crash."""
    _touch(vault / "project.md",
           "# P\n\nraft consensus leader election", DAY - timedelta(days=10))
    monkeypatch.setenv("HERMES_PROJECT_NOTE", "project.md")
    _touch(vault / "journal" / "relevant.md",
           "# R\n\nraft consensus leader election notes", DAY)
    _touch(vault / "journal" / "irrelevant.md", "# I\n\nunrelated cooking recipe", DAY)

    def boom_embed(texts):
        raise RuntimeError("embed server connection refused")

    res = digest.build_digest(vault, DAY, chat_fn=_stub_chat(), embed_fn=boom_embed)
    rels = [rel for rel, _ in res.project_correlations]
    assert rels and "relevant.md" in rels[0]


def test_project_section_in_telegram_push(vault, monkeypatch):
    """The correlation section must appear in the Telegram push text too
    (the push sends result.to_markdown(), which includes the section)."""
    _touch(vault / "project.md", "# P\n\nraft consensus", DAY - timedelta(days=10))
    monkeypatch.setenv("HERMES_PROJECT_NOTE", "project.md")
    _touch(vault / "journal" / "2026-06-06.md", "# Mon\n\nraft consensus log", DAY)

    res = digest.build_digest(vault, DAY, chat_fn=_stub_chat())
    # The full markdown (what send_document and the headline+body push use)
    # carries the section.
    assert "How this connects to" in res.to_markdown()


def test_missing_project_path_is_safe(vault, monkeypatch):
    """A configured project path that does not exist degrades to no section,
    not a crash."""
    monkeypatch.setenv("HERMES_PROJECT_NOTE", "does/not/exist.md")
    _touch(vault / "journal" / "2026-06-06.md", "# Mon\n\nnote", DAY)
    res = digest.build_digest(vault, DAY, chat_fn=_stub_chat())
    assert res.project is None
    assert "How this connects to" not in res.to_markdown()
