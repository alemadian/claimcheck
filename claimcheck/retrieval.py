"""
Layer 2: retrieval evaluation.

Scores the retriever against the gold passages for each claim, isolated from
the verdict step so a regression points at the right layer. All deterministic.

For an UNSUPPORTED claim there is nothing to retrieve (gold is empty), so recall
is vacuously perfect; the interesting retrieval signal lives on the
supported / contradicted claims, where a passage genuinely exists to find.

Metrics
-------
- recall@k     : fraction of gold passages present in the top-k retrieved.
- precision@k  : fraction of the top-k that are gold.
- mrr          : reciprocal rank of the first gold hit (1.0 = gold at rank 1).
- ndcg@k       : ordering-aware gain (binary relevance = is-gold).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .schema import ClaimCase, ReviewOutput


@dataclass(frozen=True)
class RetrievalScore:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int


def _recall(gold: set[str], retrieved: list[str]) -> float:
    if not gold:
        return 1.0  # nothing to retrieve (unsupported claim): vacuously perfect
    hit = sum(1 for g in gold if g in retrieved)
    return hit / len(gold)


def _precision(gold: set[str], retrieved: list[str]) -> float:
    if not retrieved:
        return 1.0 if not gold else 0.0
    hit = sum(1 for r in retrieved if r in gold)
    return hit / len(retrieved)


def _mrr(gold: set[str], retrieved: list[str]) -> float:
    if not gold:
        return 1.0
    for rank, rid in enumerate(retrieved, start=1):
        if rid in gold:
            return 1.0 / rank
    return 0.0


def _ndcg(gold: set[str], retrieved: list[str], k: int) -> float:
    if not gold:
        return 1.0
    dcg = 0.0
    for i, rid in enumerate(retrieved[:k]):
        rel = 1.0 if rid in gold else 0.0
        dcg += rel / math.log2(i + 2)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def score_retrieval(case: ClaimCase, out: ReviewOutput, k: int | None = None) -> RetrievalScore:
    gold = set(case.gold_doc_ids)
    retrieved = out.retrieved_ids
    eff_k = k if k is not None else max(len(retrieved), 1)
    topk = retrieved[:eff_k]
    return RetrievalScore(
        recall_at_k=_recall(gold, topk),
        precision_at_k=_precision(gold, topk),
        mrr=_mrr(gold, retrieved),
        ndcg_at_k=_ndcg(gold, retrieved, eff_k),
        k=eff_k,
    )
