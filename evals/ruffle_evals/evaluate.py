"""Metric computation over fused rankings, via ir_measures.

The headline measures are the BEIR standard nDCG@10 plus Recall@100 and MRR@10;
AP@100 and Recall@10 are computed into the result files without a table column.
Significance against the plain-RRF baseline is a paired t-test on per-query
nDCG@10 over the evaluation queries, and each fused condition carries a
per-query delta profile against the baseline: aggregates hide whether a mean
gain is a small improvement everywhere or large wins bought with real damage,
and the profile's loss tail is the empirical per-query do-no-harm statement.
"""

from __future__ import annotations

import ir_measures
from ir_measures import AP, RR, R, nDCG
from scipy import stats

__all__ = ["MEASURES", "delta_profile", "evaluate", "paired_p"]

MEASURES = [nDCG @ 10, R @ 100, RR @ 10, AP @ 100, R @ 10]

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


def delta_profile(
    baseline: dict[str, float], condition: dict[str, float]
) -> dict[str, float] | None:
    """The per-query nDCG@10 delta distribution against the baseline: the
    win/loss/tie rates, the mean delta, and the 5th-percentile delta (how badly
    the condition loses on its worst tail of queries)."""
    qids = sorted(set(baseline) & set(condition))
    if not qids:
        return None
    deltas = sorted(condition[q] - baseline[q] for q in qids)
    n = len(deltas)
    wins = sum(d > 1e-9 for d in deltas)
    losses = sum(d < -1e-9 for d in deltas)
    return {
        "win": wins / n,
        "loss": losses / n,
        "tie": (n - wins - losses) / n,
        "mean": sum(deltas) / n,
        "p5": deltas[int(0.05 * (n - 1))],
    }


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
