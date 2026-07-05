"""The warmup learning curve.

Warms a fresh stateful fuser on increasing prefixes of the warmup split and
scores the same evaluation split each time, tracing the climb from the cold
floor toward the fully warmed numbers. It reuses the cached channel runs, so it
costs fusion and metrics only. Size zero is online-from-cold: the fuser still
adapts across the evaluation queries themselves.
"""

from __future__ import annotations

from ruffle_evals.channels import CHANNEL_KEYS, Run
from ruffle_evals.evaluate import evaluate, paired_p
from ruffle_evals.fusion import rrf, ruffle_warm

__all__ = ["CURVE_SIZES", "learning_curve"]

CURVE_SIZES = (0, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)


def learning_curve(
    runs: dict[str, Run],
    qrels: dict[str, dict[str, int]],
    warm_qids: list[str],
    eval_qids: list[str],
) -> dict:
    """nDCG@10 and R@100 on the fixed evaluation split as a function of warmup
    size, with the RRF floor alongside."""
    baseline = rrf(runs, eval_qids)
    base_metrics, base_ndcg = evaluate(qrels, baseline.rankings)
    sizes = sorted({s for s in CURVE_SIZES if s <= len(warm_qids)} | {len(warm_qids)})
    points = []
    for size in sizes:
        outcome = ruffle_warm(runs, warm_qids[:size], eval_qids)
        metrics, ndcg = evaluate(qrels, outcome.rankings)
        points.append(
            {
                "warmup": size,
                "metrics": metrics,
                "p_vs_rrf": paired_p(base_ndcg, ndcg),
                "mean_weights": outcome.mean_weights(CHANNEL_KEYS),
            }
        )
    return {"rrf": {"metrics": base_metrics}, "points": points}
