"""Heuristic reranker for hermes-metal retrieval (Phase C).

The retrieval pipeline became: embed → vector top-N → **rerank → top-k**.
This module is the rerank stage. We deliberately use a *heuristic* combining
three signals rather than a learned cross-encoder:

* **Why not a cross-encoder?** A local ``bge-reranker-base`` (~280 MB, plus
  its own llama-server slot or a torch dependency) would buy marginal quality
  at real cost to an always-on background daemon — more RAM, another model to
  fetch, a new failure mode. The whole project's thesis is small footprint
  (see the benchmark story in the README), so a zero-dependency heuristic is
  the right trade. The pipeline is structured so a cross-encoder could drop in
  later behind the same ``rerank()`` signature.

The three signals, each normalized to [0, 1] and linearly combined:

1. **Semantic** — the vector similarity already computed by LanceDB. Cosine
   ``_distance`` in [0, 2] is mapped to similarity ``1 - d/2``. This is the
   dominant signal (highest default weight).
2. **Recency** — exponential decay on file mtime with a configurable
   half-life. A note edited today outranks an identical one from last year.
   Rows with no mtime (un-migrated index) contribute a neutral 0.5 so the
   reranker degrades gracefully on a v1 table.
3. **Lexical** — Jaccard-ish term overlap between the query and each chunk's
   ``heading_trail`` + a capped scan of its text. Cheap to compute, and it
   rescues the case where the right chunk is semantically near-tied with
   several others but is literally *about* the queried heading.

Pure-Python, stdlib + ``math`` only. No numpy, no model, no network — so it's
unit-testable without any daemon up.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


# Default signal weights. Semantic dominates; recency and lexical are
# tie-breakers. They need not sum to 1 (the final score is just a weighted
# sum used for ordering), but keeping them ~1 makes the score interpretable.
DEFAULT_WEIGHTS = {
    "semantic": 0.70,
    "recency": 0.15,
    "lexical": 0.15,
}

# Recency half-life in days: a file this old contributes half the recency
# signal of a brand-new file. 30 days suits a notes vault where "recent"
# means "this month."
DEFAULT_HALFLIFE_DAYS = 30.0

# Cap how much chunk body we scan for lexical overlap — heading_trail is the
# high-signal field; scanning the whole chunk would let long chunks dominate.
_LEXICAL_TEXT_SCAN_CHARS = 400

_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# Tiny stopword set so lexical overlap isn't dominated by "the/of/a". Kept
# small on purpose — aggressive stopwording hurts more than it helps on
# technical notes.
_STOPWORDS = frozenset(
    "a an and are as at be by для for from has have in into is it its of on or "
    "that the their to was were what when where which who why with you your".split()
)


def _terms(text: str) -> set[str]:
    return {
        w.lower()
        for w in _WORD_RE.findall(text or "")
        if len(w) > 1 and w.lower() not in _STOPWORDS
    }


def _semantic_score(distance: Any, metric: str) -> float:
    """Map a LanceDB ``_distance`` to a [0, 1] similarity.

    cosine distance ∈ [0, 2] → similarity = 1 - d/2.
    L2 distance ∈ [0, ∞)    → similarity = 1 / (1 + d) (monotonic fallback).
    A missing/non-numeric distance yields a neutral 0.5.
    """
    if not isinstance(distance, (int, float)) or math.isnan(float(distance)):
        return 0.5
    d = float(distance)
    if metric == "cosine":
        return max(0.0, min(1.0, 1.0 - d / 2.0))
    return 1.0 / (1.0 + max(0.0, d))


def _recency_score(mtime: Any, now: float, halflife_days: float) -> float:
    """Exponential decay on age. Neutral 0.5 when mtime is missing/placeholder.

    A row with mtime == 0.0 (un-migrated, or a file whose stat failed) is
    treated as "unknown age" → 0.5, never penalized to 0. ``now`` is passed
    in (not read from the clock) so the function is deterministic for tests.
    """
    if not isinstance(mtime, (int, float)) or mtime <= 0.0:
        return 0.5
    age_days = max(0.0, (now - float(mtime)) / 86400.0)
    if halflife_days <= 0:
        return 1.0
    return math.pow(0.5, age_days / halflife_days)


def _lexical_score(query_terms: set[str], hit: dict[str, Any]) -> float:
    """Overlap of query terms with the hit's heading trail + capped text.

    Heading-trail terms are weighted double (a heading match is a strong
    topical signal). Score is overlap / |query_terms|, clamped to [0, 1].
    Returns 0.0 when the query has no usable terms.
    """
    if not query_terms:
        return 0.0
    heading_terms = _terms(hit.get("heading_trail", ""))
    text_terms = _terms((hit.get("text", "") or "")[:_LEXICAL_TEXT_SCAN_CHARS])
    if not heading_terms and not text_terms:
        return 0.0
    matched = 0.0
    for qt in query_terms:
        if qt in heading_terms:
            matched += 1.0
        elif qt in text_terms:
            matched += 0.5
    return min(1.0, matched / len(query_terms))


@dataclass(frozen=True)
class RerankWeights:
    semantic: float = DEFAULT_WEIGHTS["semantic"]
    recency: float = DEFAULT_WEIGHTS["recency"]
    lexical: float = DEFAULT_WEIGHTS["lexical"]
    halflife_days: float = DEFAULT_HALFLIFE_DAYS


def rerank(
    query: str,
    hits: list[dict[str, Any]],
    *,
    k: int,
    now: float,
    metric: str = "cosine",
    weights: RerankWeights | None = None,
) -> list[dict[str, Any]]:
    """Reorder ``hits`` by the combined heuristic score and return the top-k.

    Each returned hit gains a ``_rerank`` sub-dict (the component scores and
    the combined value) for transparency in ``/sources`` and tests. The
    original ``_distance`` is preserved. Stable: ties keep their input order
    (which is the vector-score order LanceDB returned), so the reranker never
    *worsens* a clear semantic win on a tie.

    ``now`` is an explicit POSIX timestamp (caller passes ``time.time()``) so
    this stays deterministic under test.
    """
    if not hits:
        return []
    w = weights or RerankWeights()
    qterms = _terms(query)

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, hit in enumerate(hits):
        sem = _semantic_score(hit.get("_distance"), metric)
        rec = _recency_score(hit.get("mtime"), now, w.halflife_days)
        lex = _lexical_score(qterms, hit)
        combined = w.semantic * sem + w.recency * rec + w.lexical * lex
        enriched = dict(hit)
        enriched["_rerank"] = {
            "semantic": round(sem, 4),
            "recency": round(rec, 4),
            "lexical": round(lex, 4),
            "score": round(combined, 4),
        }
        # Negate combined for ascending sort = descending score; idx as the
        # secondary key makes the sort stable on ties.
        scored.append((-combined, idx, enriched))

    scored.sort(key=lambda t: (t[0], t[1]))
    return [hit for _neg, _idx, hit in scored[: max(0, k)]]
