"""Retrieval-quality metrics + latency statistics.

All quality metrics are computed at *document* granularity with binary
relevance: a retrieved item counts as relevant iff its document identifier is in
the ground-truth relevant set. The ranked list passed in must already be
ordered best-first (that is exactly what the retrieval pipeline returns).

Because a KB may embed several chunks from the same document, the ranked list is
de-duplicated to first occurrence before scoring, so `recall@5` means "5 distinct
documents", matching how `grouping_enabled` KBs behave in production.

No third-party deps — pure functions, trivially unit-testable.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _dedupe(ranked_docs: Iterable[str]) -> list[str]:
    """Keep first occurrence of each doc id, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for d in ranked_docs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def recall_at_k(ranked_docs: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of ground-truth documents found within the top-k results."""
    if not relevant:
        return 0.0
    topk = set(_dedupe(ranked_docs)[:k])
    return len(topk & relevant) / len(relevant)


def precision_at_k(ranked_docs: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k results that are relevant (k = requested cutoff)."""
    if k <= 0:
        return 0.0
    topk = _dedupe(ranked_docs)[:k]
    if not topk:
        return 0.0
    hit = sum(1 for d in topk if d in relevant)
    return hit / k


def hit_at_k(ranked_docs: Sequence[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant document is in the top-k, else 0.0."""
    topk = _dedupe(ranked_docs)[:k]
    return 1.0 if any(d in relevant for d in topk) else 0.0


def reciprocal_rank(ranked_docs: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant document (rank is 1-based); 0 if none."""
    for i, d in enumerate(_dedupe(ranked_docs), start=1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_docs: Sequence[str], relevant: set[str], k: int) -> float:
    """Normalized DCG@k with binary relevance.

    DCG@k = sum over positions i (1-based) of rel_i / log2(i + 1).
    IDCG@k is the DCG of the ideal ranking (all relevant docs first, capped at
    the number of relevant docs and at k).
    """
    if not relevant or k <= 0:
        return 0.0
    topk = _dedupe(ranked_docs)[:k]
    dcg = sum(
        (1.0 / math.log2(i + 1)) for i, d in enumerate(topk, start=1) if d in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Latency statistics
# ---------------------------------------------------------------------------
def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (matches numpy's default 'linear').

    `p` is in [0, 100]. Returns 0.0 for an empty input.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (p / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def latency_stats(values: Sequence[float]) -> dict[str, float]:
    """Summary stats for a list of millisecond latencies."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
        }
    vals = [float(v) for v in values]
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
    }