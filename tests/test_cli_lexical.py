"""Tests for `hermes search --lexical` CLI wiring.

Confirms the flag parses, routes to the BM25 path (not the vector path), and
that the lexical path never touches the embed server. The vault is mocked so
these stay daemon-free and network-free.
"""
from __future__ import annotations

import pytest

from src import cli


def test_search_parser_accepts_lexical_flag():
    parser = cli._build_parser()
    args = parser.parse_args(["search", "kerberos ticket", "--lexical"])
    assert args.lexical is True
    assert args.query == "kerberos ticket"


def test_search_lexical_defaults_off():
    parser = cli._build_parser()
    args = parser.parse_args(["search", "anything"])
    assert args.lexical is False


def test_cmd_search_routes_to_lexical(monkeypatch, capsys):
    """--lexical calls the BM25 retriever, NOT the vector retriever."""
    called = {"lexical": False, "vector": False}

    def fake_lexical(query, *, k):
        called["lexical"] = True
        return [{"source_path": "/n/auth.md", "chunk_idx": 2,
                 "text": "kerberos ticket lifetime", "_bm25": 3.14}]

    def fake_vector(query, *, k):  # must not be called
        called["vector"] = True
        return []

    monkeypatch.setattr(cli, "_lexical_retrieve", fake_lexical)
    monkeypatch.setattr(cli, "_retrieve", fake_vector)

    parser = cli._build_parser()
    args = parser.parse_args(["search", "kerberos", "--lexical"])
    rc = cli._cmd_search(args)

    assert rc == 0
    assert called["lexical"] is True
    assert called["vector"] is False
    out = capsys.readouterr().out
    assert "auth.md" in out
    assert "bm25=3.1400" in out  # BM25 score rendered, not a vector distance


def test_cmd_search_default_routes_to_vector(monkeypatch):
    """Without --lexical, search uses the vector retriever."""
    called = {"lexical": False, "vector": False}
    monkeypatch.setattr(cli, "_lexical_retrieve",
                        lambda q, *, k: called.__setitem__("lexical", True) or [])
    monkeypatch.setattr(cli, "_retrieve",
                        lambda q, *, k: called.__setitem__("vector", True) or [])

    parser = cli._build_parser()
    args = parser.parse_args(["search", "kerberos"])
    cli._cmd_search(args)

    assert called["vector"] is True
    assert called["lexical"] is False
