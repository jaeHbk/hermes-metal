"""Tests for the BM25 lexical retrieval path (synto-inspired, zero-RAM).

Verifies tokenization, BM25 ranking properties (rare-term boost, term-freq
saturation, length normalization), the drop-zero-overlap behavior, and
determinism. Pure-Python: no daemon, no network, no LanceDB.
"""
from __future__ import annotations

from src.backend.lexical import BM25Index, bm25_search, tokenize


def _docs(*texts: str) -> list[dict]:
    return [
        {"id": str(i), "source_path": f"/n/{i}.md", "chunk_idx": 0, "text": t}
        for i, t in enumerate(texts)
    ]


# --------------------------------------------------------------- tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    toks = tokenize("The Quick brown FOX")
    assert "the" not in toks  # stopword
    assert toks == ["quick", "brown", "fox"]


def test_tokenize_drops_single_char_tokens():
    assert tokenize("a I x ab 9 42") == ["ab", "42"]


def test_tokenize_keeps_alphanumeric_identifiers():
    # Error codes / identifiers are exactly what lexical search is for.
    assert tokenize("error E0501 in main.rs") == ["error", "e0501", "main", "rs"]


def test_tokenize_empty_and_none_safe():
    assert tokenize("") == []
    assert tokenize(None) == []


# ----------------------------------------------------------------- ranking


def test_matching_doc_ranks_above_nonmatching():
    docs = _docs(
        "token rotation policy for auth",
        "unrelated note about cooking pasta",
    )
    hits = bm25_search("token rotation", docs, k=5)
    assert len(hits) == 1  # the pasta note shares no query term → dropped
    assert hits[0]["source_path"] == "/n/0.md"
    assert hits[0]["_bm25"] > 0


def test_zero_overlap_docs_are_dropped_not_returned_as_filler():
    docs = _docs("apples and oranges", "bananas")
    hits = bm25_search("quantum chromodynamics", docs, k=5)
    assert hits == []


def test_rare_term_outweighs_common_term():
    # "auth" appears in every doc (low IDF); "kerberos" is rare (high IDF).
    docs = _docs(
        "auth notes general",
        "auth notes general too",
        "auth kerberos ticket details",
    )
    hits = bm25_search("auth kerberos", docs, k=3)
    assert hits[0]["source_path"] == "/n/2.md"  # the rare-term doc wins


def test_length_normalization_prefers_concise_match():
    # Same single occurrence of the query term, but one doc is padded with
    # lots of unrelated text. BM25's b-normalization should favor the concise.
    short = "kerberos"
    long = "kerberos " + " ".join(f"filler{i}" for i in range(200))
    docs = _docs(short, long)
    hits = bm25_search("kerberos", docs, k=2)
    assert hits[0]["source_path"] == "/n/0.md"


def test_term_frequency_saturates():
    # A doc mentioning the term many times should not score unboundedly; the
    # k1 saturation means 10x the term is far less than 10x the score.
    idx = BM25Index(_docs("x", "kerberos", "kerberos " * 10))
    s_once = idx.score(["kerberos"], 1)
    s_many = idx.score(["kerberos"], 2)
    assert s_many > s_once
    assert s_many < 10 * s_once  # saturation, not linear growth


def test_empty_query_returns_nothing():
    docs = _docs("some content here")
    assert bm25_search("the and of", docs, k=5) == []  # all stopwords
    assert bm25_search("", docs, k=5) == []


def test_empty_corpus_returns_nothing():
    assert bm25_search("anything", [], k=5) == []


def test_results_are_deterministic_and_capped_at_k():
    docs = _docs(*[f"shared term doc number {i}" for i in range(10)])
    a = bm25_search("shared term", docs, k=3)
    b = bm25_search("shared term", docs, k=3)
    assert len(a) == 3
    assert [h["source_path"] for h in a] == [h["source_path"] for h in b]


def test_hit_carries_bm25_not_distance():
    hits = bm25_search("kerberos", _docs("kerberos ticket"), k=1)
    assert "_bm25" in hits[0]
    assert "_distance" not in hits[0]


def test_idf_non_negative_for_common_terms():
    # A term in every document must not produce a negative IDF (which would
    # let a common term lower a doc's rank). The +1 smoothing guards this.
    idx = BM25Index(_docs("common", "common", "common"))
    assert idx._idf("common") >= 0.0
