"""Lexical (BM25) retrieval — a zero-RAM, server-free fallback path.

This is the one idea hermes-metal borrows from kytmanov/synto, whose whole
thesis is "no embeddings, no vector database" — BM25 over source text, routed
without a resident model. hermes keeps its vector pipeline as the default
(it's better at paraphrase/synonym recall), but adds BM25 as an **opt-in**
alternative for two cases the vector path can't serve:

* **Embed server down.** Vector ``hermes search`` needs the nomic embed
  server on :8081. When it's not running (or on a CPU host where the Metal
  servers don't exist at all), BM25 still answers from the text already in
  LanceDB. No network, no model.
* **Exact-term / rare-token queries.** Error codes, file names, identifiers,
  quoted phrases — where literal term match beats semantic nearness.

**Why this respects the minimal-RAM ethos.** BM25 here adds *nothing* to
steady-state memory: there is no persisted lexical index and no resident
model. The index is built transiently from chunk text the vault already
stores, used for one query, and discarded. The default daemon footprint
(``ProcessType=Background``, three llama/embed/watcher agents) is untouched —
this only materializes when a user explicitly runs ``hermes search --lexical``.

Pure-Python, stdlib + ``math`` only (mirrors :mod:`src.backend.reranker`), so
it's unit-testable with no daemon, no network, and no LanceDB up.
"""
from __future__ import annotations

import math
import re
from typing import Any, Iterable


# BM25 parameters (Robertson/Okapi defaults). ``k1`` controls term-frequency
# saturation; ``b`` controls length normalization. These defaults are the
# widely-used standard and suit short note chunks well.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# Same small stopword set as the reranker, kept deliberately tiny: aggressive
# stopwording hurts more than it helps on technical notes (error codes,
# acronyms, identifiers). Mirrors src/backend/reranker.py._STOPWORDS.
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it its of on or "
    "that the their to was were what when where which who why with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, minus 1-char tokens and stopwords.

    Returns a *list* (not a set) because BM25 needs term frequencies — how
    many times each term occurs in a document, not just whether it appears.
    """
    return [
        w.lower()
        for w in _WORD_RE.findall(text or "")
        if len(w) > 1 and w.lower() not in _STOPWORDS
    ]


class BM25Index:
    """An in-memory BM25 index over a fixed corpus of documents.

    Built once from an iterable of ``(doc, text)`` pairs, queried, then
    dropped. Holds only token lists + per-term document frequencies — no
    vectors, no model. For a personal vault (thousands of chunks) this is a
    few MB of transient Python objects, freed as soon as the query returns.
    """

    def __init__(
        self,
        docs: list[dict[str, Any]],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
        text_key: str = "text",
    ) -> None:
        self.k1 = k1
        self.b = b
        self.docs = docs
        # Tokenize each document once.
        self._tokens: list[list[str]] = [tokenize(d.get(text_key, "")) for d in docs]
        self._doc_len: list[int] = [len(t) for t in self._tokens]
        n = len(docs)
        self.avgdl: float = (sum(self._doc_len) / n) if n else 0.0

        # Per-term frequency within each doc, and document frequency (how many
        # docs contain the term) for IDF.
        self._tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for toks in self._tokens:
            counts: dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            self._tf.append(counts)
            for term in counts:
                df[term] = df.get(term, 0) + 1
        self._df = df
        self._n = n

    def _idf(self, term: str) -> float:
        """Okapi BM25 IDF with the standard +0.5 smoothing.

        ``log((N - df + 0.5) / (df + 0.5) + 1)``. The ``+ 1`` inside the log
        keeps the IDF non-negative even for terms that appear in more than
        half the corpus (the classic unsmoothed form can go negative and let
        a common term *lower* a document's score, which is never what we want
        for ranking).
        """
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_terms: Iterable[str], doc_idx: int) -> float:
        """BM25 score of one document against the (already-tokenized) query."""
        tf = self._tf[doc_idx]
        dl = self._doc_len[doc_idx]
        if dl == 0 or self.avgdl == 0:
            return 0.0
        norm = self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
        total = 0.0
        for term in query_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            total += self._idf(term) * (f * (self.k1 + 1.0)) / (f + norm)
        return total

    def search(self, query: str, *, k: int = 5) -> list[dict[str, Any]]:
        """Return the top-``k`` documents for ``query``, ranked by BM25.

        Each returned hit is a shallow copy of the source doc with a
        ``_bm25`` score added (and ``_distance`` left absent — these hits did
        not come from the vector index). Documents that share no query term
        score 0 and are dropped, so a query for a rare token returns only the
        chunks that actually contain it rather than ``k`` arbitrary rows.
        Ties keep corpus order, so the result is deterministic for tests.
        """
        # De-dup query terms: a term repeated in the query shouldn't multiply
        # its own contribution (BM25 scores per distinct query term).
        qterms = list(dict.fromkeys(tokenize(query)))
        if not qterms or self._n == 0:
            return []
        scored: list[tuple[float, int]] = []
        for idx in range(self._n):
            s = self.score(qterms, idx)
            if s > 0.0:
                scored.append((s, idx))
        # Sort by score desc, then corpus index asc (stable, deterministic).
        scored.sort(key=lambda t: (-t[0], t[1]))
        out: list[dict[str, Any]] = []
        for s, idx in scored[: max(0, k)]:
            hit = dict(self.docs[idx])
            hit["_bm25"] = round(s, 4)
            out.append(hit)
        return out


def bm25_search(query: str, docs: list[dict[str, Any]], *, k: int = 5) -> list[dict[str, Any]]:
    """Convenience one-shot: build a transient index, query it, drop it.

    This is the intended entry point for ``hermes search --lexical``: the
    index lives only for the duration of the call, so nothing lingers in
    memory after the results are returned.
    """
    return BM25Index(docs).search(query, k=k)
