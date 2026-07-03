"""Non-RRF fusion baselines: the score-normalization family, two further
rank-based rules, and an oracle-weighted RRF ceiling.

CombSUM and CombMNZ (Fox and Shaw) are the classic score-based fusions. Both
need each channel's scores brought onto a shared scale first, here the standard
min-max normalization over the retrieved list, which is exactly the per-channel
calibration step Ruffle exists to avoid; they are the contrast class, not just
another baseline. Borda and ISR are rank-based rules like RRF itself, included
as completeness rows.

The oracle is not a competitor: it grid-searches fixed per-channel RRF weights
on the evaluation split using the relevance judgments, so it bounds what any
per-channel weighting scheme (label-free or not) could achieve with this run
data. Reported alongside the label-free conditions, it turns the table into a
bracket: the RRF floor, Ruffle between, the oracle ceiling.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from itertools import product

import numpy as np

from ruffle_evals.channels import CHANNEL_KEYS, Run

__all__ = ["borda", "combmnz", "combsum", "isr", "oracle_rrf"]

Rankings = dict[str, list[tuple[str, float]]]


def _fuse_by(
    runs: dict[str, Run],
    qids: Iterable[str],
    keys: Sequence[str],
    contribution,
) -> Rankings:
    """Fuses each query by summing ``contribution(rank, score, items)`` per
    channel, with the harness's deterministic first-seen tie order."""
    fused: Rankings = {}
    for qid in qids:
        first_seen: dict[str, int] = {}
        scores: dict[str, float] = {}
        for key in keys:
            items = runs[key].get(qid, [])
            for rank, (doc_id, score) in enumerate(items, start=1):
                first_seen.setdefault(doc_id, len(first_seen))
                scores[doc_id] = scores.get(doc_id, 0.0) + contribution(rank, score, items)
        fused[qid] = sorted(scores.items(), key=lambda it: (-it[1], first_seen[it[0]]))
    return fused


def combsum(
    runs: dict[str, Run], qids: Iterable[str], keys: Sequence[str] = CHANNEL_KEYS
) -> Rankings:
    """CombSUM over min-max normalized scores: each channel's retrieved list is
    rescaled to [0, 1], then summed. A degenerate list (one score value)
    contributes 1.0 for its members."""

    def contribution(rank: int, score: float, items) -> float:
        lo = items[-1][1]
        hi = items[0][1]
        return 1.0 if hi == lo else (score - lo) / (hi - lo)

    return _fuse_by(runs, qids, keys, contribution)


def combmnz(
    runs: dict[str, Run], qids: Iterable[str], keys: Sequence[str] = CHANNEL_KEYS
) -> Rankings:
    """CombMNZ: CombSUM multiplied by the number of channels that retrieved the
    document."""
    base = combsum(runs, qids, keys)
    fused: Rankings = {}
    for qid, items in base.items():
        counts: dict[str, int] = {}
        for key in keys:
            for doc_id, _ in runs[key].get(qid, []):
                counts[doc_id] = counts.get(doc_id, 0) + 1
        order = {doc_id: i for i, (doc_id, _) in enumerate(items)}
        rescored = [(doc_id, score * counts[doc_id]) for doc_id, score in items]
        fused[qid] = sorted(rescored, key=lambda it: (-it[1], order[it[0]]))
    return fused


def borda(
    runs: dict[str, Run], qids: Iterable[str], keys: Sequence[str] = CHANNEL_KEYS
) -> Rankings:
    """Borda count over the truncated lists: rank r in a list of n scores
    n - r + 1 points, absence scores zero."""
    return _fuse_by(runs, qids, keys, lambda rank, score, items: float(len(items) - rank + 1))


def isr(
    runs: dict[str, Run], qids: Iterable[str], keys: Sequence[str] = CHANNEL_KEYS
) -> Rankings:
    """Inverse square rank: rank r contributes 1 / r^2, a steeper decay than
    RRF's 1 / (eta + r)."""
    return _fuse_by(runs, qids, keys, lambda rank, score, items: 1.0 / (rank * rank))


def _ndcg10(ranking: list[str], rels: dict[str, int]) -> float:
    dcg = sum(
        rels.get(doc_id, 0) / math.log2(i + 2.0) for i, doc_id in enumerate(ranking[:10])
    )
    ideal = sorted(rels.values(), reverse=True)[:10]
    idcg = sum(rel / math.log2(i + 2.0) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def oracle_rrf(
    runs: dict[str, Run],
    qrels: dict[str, dict[str, int]],
    qids: list[str],
    keys: Sequence[str] = CHANNEL_KEYS,
    eta: float = 60.0,
    step: float = 0.1,
) -> tuple[Rankings, dict[str, float]]:
    """Weighted RRF with fixed per-channel weights grid-searched on the
    evaluation split against the relevance judgments.

    The weights live on the unit simplex (a common rescaling leaves an RRF
    ranking unchanged), searched at the given step. The search objective is an
    in-harness linear-gain nDCG@10; the returned rankings are then scored by the
    same ir_measures path as every other condition. Judgments are used to choose
    the weights, so this is a ceiling on per-channel weighting, not a
    label-free competitor.
    """
    # Per query: the candidate list (first-seen order) and a (docs x channels)
    # matrix of unweighted RRF contributions, so each weight combination is one
    # matrix-vector product.
    per_query: dict[str, tuple[list[str], np.ndarray]] = {}
    for qid in qids:
        first_seen: dict[str, int] = {}
        for key in keys:
            for doc_id, _ in runs[key].get(qid, []):
                first_seen.setdefault(doc_id, len(first_seen))
        docs = list(first_seen)
        matrix = np.zeros((len(docs), len(keys)))
        for col, key in enumerate(keys):
            for rank, (doc_id, _) in enumerate(runs[key].get(qid, []), start=1):
                matrix[first_seen[doc_id], col] = 1.0 / (eta + rank)
        per_query[qid] = (docs, matrix)

    steps = int(round(1.0 / step))
    grid = [
        np.array([a, b, steps - a - b], dtype=float) / steps
        for a, b in product(range(steps + 1), repeat=2)
        if a + b <= steps
    ]

    best_weights = grid[0]
    best_score = -1.0
    for weights in grid:
        total = 0.0
        for qid in qids:
            docs, matrix = per_query[qid]
            if not docs:
                continue
            fused = matrix @ weights
            top = np.argsort(-fused, kind="stable")[:10]
            total += _ndcg10([docs[int(i)] for i in top], qrels.get(qid, {}))
        score = total / max(len(qids), 1)
        if score > best_score:
            best_score = score
            best_weights = weights

    rankings: Rankings = {}
    for qid in qids:
        docs, matrix = per_query[qid]
        fused = matrix @ best_weights
        order = np.argsort(-fused, kind="stable")
        rankings[qid] = [(docs[int(i)], float(fused[int(i)])) for i in order]
    return rankings, {key: float(w) for key, w in zip(keys, best_weights)}
