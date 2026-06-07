"""Tests for the heuristic reranker (Phase C).

Verifies each signal in isolation and the combined ordering, plus the
graceful-degradation paths (missing distance, missing mtime) that keep the
reranker working on a partially-migrated index. ``now`` is injected.
"""
from __future__ import annotations

import time

from src.backend.reranker import (
    rerank,
    RerankWeights,
    _semantic_score,
    _recency_score,
    _lexical_score,
    _terms,
)


NOW = 1_780_000_000.0  # fixed reference timestamp
DAY = 86400.0


# ------------------------------------------------------------- unit: signals


def test_semantic_cosine_mapping():
    # cosine distance 0 → similarity 1; distance 2 → 0; distance 1 → 0.5.
    assert _semantic_score(0.0, "cosine") == 1.0
    assert _semantic_score(2.0, "cosine") == 0.0
    assert abs(_semantic_score(1.0, "cosine") - 0.5) < 1e-9


def test_semantic_missing_distance_is_neutral():
    assert _semantic_score(None, "cosine") == 0.5
    assert _semantic_score("nan-ish", "cosine") == 0.5


def test_recency_decays_with_age():
    fresh = _recency_score(NOW, NOW, 30.0)
    month = _recency_score(NOW - 30 * DAY, NOW, 30.0)
    year = _recency_score(NOW - 365 * DAY, NOW, 30.0)
    assert fresh > month > year
    assert abs(fresh - 1.0) < 1e-6
    assert abs(month - 0.5) < 1e-6  # exactly one half-life


def test_recency_missing_mtime_is_neutral_not_zero():
    # An un-migrated row (mtime 0.0) must not be penalized to the floor.
    assert _recency_score(0.0, NOW, 30.0) == 0.5
    assert _recency_score(None, NOW, 30.0) == 0.5


def test_lexical_heading_weighted_over_body():
    qterms = _terms("token rotation")
    heading_hit = {"heading_trail": "Auth > Token rotation", "text": "unrelated"}
    body_hit = {"heading_trail": "Misc", "text": "token rotation appears here"}
    assert _lexical_score(qterms, heading_hit) > _lexical_score(qterms, body_hit)


def test_lexical_empty_query_is_zero():
    assert _lexical_score(set(), {"heading_trail": "x", "text": "y"}) == 0.0


# ------------------------------------------------------------- integration


def _hit(id_, dist, mtime, trail="", text=""):
    return {"id": id_, "_distance": dist, "mtime": mtime, "heading_trail": trail, "text": text}


def test_recency_breaks_near_semantic_tie():
    hits = [
        _hit("old", 0.10, NOW - 365 * DAY, "Misc", "auth token"),
        _hit("new", 0.12, NOW - 3600, "Auth > Tokens", "auth token rotation"),
    ]
    out = rerank("auth token rotation", hits, k=2, now=NOW, metric="cosine")
    # The slightly-farther-but-recent-and-heading-matching chunk wins.
    assert out[0]["id"] == "new"


def test_pure_semantic_still_wins_when_recency_equal():
    hits = [
        _hit("close", 0.05, NOW, "A", "x"),
        _hit("far", 0.80, NOW, "A", "x"),
    ]
    out = rerank("anything", hits, k=2, now=NOW)
    assert out[0]["id"] == "close"


def test_rerank_trims_to_k():
    hits = [_hit(f"h{i}", 0.1 * i, NOW) for i in range(10)]
    out = rerank("q", hits, k=3, now=NOW)
    assert len(out) == 3


def test_rerank_empty_input():
    assert rerank("q", [], k=5, now=NOW) == []


def test_rerank_is_stable_on_ties():
    # Identical scores → input order preserved (vector order from LanceDB).
    hits = [_hit("a", 0.2, NOW), _hit("b", 0.2, NOW), _hit("c", 0.2, NOW)]
    out = rerank("zzz", hits, k=3, now=NOW)
    assert [h["id"] for h in out] == ["a", "b", "c"]


def test_rerank_enriches_with_component_scores():
    out = rerank("token", [_hit("a", 0.3, NOW, "Token", "token")], k=1, now=NOW)
    rr = out[0]["_rerank"]
    assert set(rr) == {"semantic", "recency", "lexical", "score"}
    assert 0.0 <= rr["score"] <= 1.0


def test_custom_weights_can_disable_recency():
    w = RerankWeights(semantic=1.0, recency=0.0, lexical=0.0)
    hits = [
        _hit("old_close", 0.05, NOW - 999 * DAY),
        _hit("new_far", 0.5, NOW),
    ]
    out = rerank("q", hits, k=2, now=NOW, weights=w)
    # With recency off, the semantically closer (older) one wins.
    assert out[0]["id"] == "old_close"


def test_missing_metadata_does_not_crash():
    # Hits straight from a v1 table: no mtime, no heading_trail.
    hits = [{"id": "a", "_distance": 0.2, "text": "hi"}]
    out = rerank("hi", hits, k=1, now=NOW)
    assert out[0]["id"] == "a"
