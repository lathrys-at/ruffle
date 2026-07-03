"""Metric computation over fused rankings, via ir_measures.

The reported measures are the BEIR standard nDCG@10 plus Recall@100 and MRR@10.
Significance against the plain-RRF baseline is a paired t-test on per-query
nDCG@10 over the evaluation queries.
"""

from __future__ import annotations

import ir_measures
from ir_measures import RR, R, nDCG
from scipy import stats

__all__ = ["MEASURES", "evaluate", "paired_p"]

MEASURES = [nDCG @ 10, R @ 100, RR @ 10]

Ranking = dict[str, list[tuple[str, float]]]


def _as_run(ranking: Ranking) -> dict[str, dict[str, float]]:
    return {qid: dict(items) for qid, items in ranking.items()}


def evaluate(
    qrels: dict[str, dict[str, int]], ranking: Ranking
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate metrics plus per-query nDCG@10, for the significance test.

    The qrels are restricted to the queries the ranking covers: the harness
    evaluates conditions on the held-out split only, and an aggregate over the
    full judged set would score every warmup query as a zero.
    """
    run = _as_run(ranking)
    qrels = {qid: rels for qid, rels in qrels.items() if qid in run}
    aggregate = {
        str(measure): value
        for measure, value in ir_measures.calc_aggregate(MEASURES, qrels, run).items()
    }
    per_query = {
        m.query_id: m.value for m in ir_measures.iter_calc([nDCG @ 10], qrels, run)
    }
    return aggregate, per_query


def paired_p(baseline: dict[str, float], condition: dict[str, float]) -> float | None:
    """Two-sided paired t-test p-value on per-query nDCG@10, over the queries both
    conditions scored. Identical vectors have no variance to test, reported as
    ``None`` rather than a p-value."""
    qids = sorted(set(baseline) & set(condition))
    a = [baseline[q] for q in qids]
    b = [condition[q] for q in qids]
    if len(qids) < 2 or a == b:
        return None
    return float(stats.ttest_rel(a, b).pvalue)
