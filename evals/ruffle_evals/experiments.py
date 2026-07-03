"""Targeted experiments beyond the headline comparison: a degraded fourth
channel, and the warmup learning curve.

Both reuse the cached channel runs, so they cost fusion and metrics only.

The degraded experiment adds a broken fourth channel derived from the BM25 run
and measures what it costs each fusion. Two failure modes, chosen because they
sit on opposite sides of what label-free weighting can see:

- ``wrong-query``: the channel serves another query's BM25 results (a rotation
  over the query set). The scores are internally healthy, the content is
  irrelevant. Per-channel statistics read a channel at its own norm, so this mode
  is designed to be invisible to them; the honest expectation is that Ruffle
  matches RRF's damage rather than recovering it, while the conflict diagnostic
  is what should move.
- ``flaky``: on a seeded half of the queries the channel serves the tail of its
  own result list (ranks 51-100 with their true low scores), simulating
  intermittent retrieval failure. Here the failure is visible per query: the top
  scores sit at bulk level and below the channel's learned reference, which is
  exactly the departure-from-own-norm that discrimination weighting reads.

The learning curve warms a fresh stateful fuser on increasing prefixes of the
warmup split and scores the same evaluation split each time, tracing the climb
from the cold floor toward the fully warmed numbers. Size zero is online-from-
cold: the fuser still adapts across the evaluation queries themselves.
"""

from __future__ import annotations

import random

from ruffle_evals import SEED
from ruffle_evals.channels import CHANNEL_KEYS, Run
from ruffle_evals.evaluate import evaluate, paired_p
from ruffle_evals.fusion import channel_configs, rrf, ruffle_warm

__all__ = ["CURVE_SIZES", "DEGRADED_MODES", "degraded", "learning_curve"]

DEGRADED_MODES = ("wrong-query", "flaky")

CURVE_SIZES = (0, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000)

_BROKEN = "broken"

_FLAKY_TAIL_FROM = 50


def _wrong_query_run(base: Run, qids: list[str]) -> Run:
    """Each query serves the next query's results: internally plausible scores
    over irrelevant content."""
    ordered = sorted(qids)
    rotated = ordered[1:] + ordered[:1]
    return {qid: list(base.get(other, [])) for qid, other in zip(ordered, rotated)}


def _flaky_run(base: Run, qids: list[str]) -> tuple[Run, set[str]]:
    """On a seeded half of the queries the channel serves its own tail (true low
    scores); on the rest it is the healthy BM25 run."""
    rng = random.Random(SEED)
    failures = {qid for qid in sorted(qids) if rng.random() < 0.5}
    run: Run = {}
    for qid in qids:
        items = base.get(qid, [])
        if qid in failures:
            cut = min(_FLAKY_TAIL_FROM, len(items) // 2)
            items = items[cut:]
        run[qid] = list(items)
    return run, failures


def degraded(
    runs: dict[str, Run],
    qrels: dict[str, dict[str, int]],
    warm_qids: list[str],
    eval_qids: list[str],
) -> dict:
    """The degraded-channel experiment for both failure modes.

    Conditions per mode: RRF over the three healthy channels (the ceiling), RRF
    and Ruffle warm over the four channels including the broken one. The paired
    test compares each four-channel condition against four-channel RRF, so it
    reads "what does this fusion recover of the damage", not "is fusion useful".
    """
    all_qids = warm_qids + eval_qids
    result: dict = {"modes": {}}
    clean_rrf = rrf(runs, eval_qids)
    clean_metrics, clean_ndcg = evaluate(qrels, clean_rrf.rankings)

    for mode in DEGRADED_MODES:
        failures: set[str] = set()
        if mode == "wrong-query":
            broken = _wrong_query_run(runs["bm25"], all_qids)
        else:
            broken, failures = _flaky_run(runs["bm25"], all_qids)
        runs4 = {**runs, _BROKEN: broken}
        keys4 = (*CHANNEL_KEYS, _BROKEN)
        configs4 = channel_configs(keys4, tags={_BROKEN: f"bm25s-{mode}"})

        broken_rrf = rrf(runs4, eval_qids, keys=keys4)
        warm = ruffle_warm(runs4, warm_qids, eval_qids, configs=configs4)

        conditions: dict = {}
        broken_metrics, broken_ndcg = evaluate(qrels, broken_rrf.rankings)
        warm_metrics, warm_ndcg = evaluate(qrels, warm.rankings)
        conditions["rrf-clean"] = {
            "metrics": clean_metrics,
            "p_vs_rrf_broken": paired_p(broken_ndcg, clean_ndcg),
        }
        conditions["rrf"] = {
            "metrics": broken_metrics,
            "mean_weights": broken_rrf.mean_weights(keys4),
        }
        conditions["ruffle-warm"] = {
            "metrics": warm_metrics,
            "p_vs_rrf_broken": paired_p(broken_ndcg, warm_ndcg),
            "mean_weights": warm.mean_weights(keys4),
            "mean_conflict": warm.mean_conflict(),
        }
        entry: dict = {"conditions": conditions}
        if failures:
            eval_failed = [q for q in eval_qids if q in failures]
            eval_healthy = [q for q in eval_qids if q not in failures]
            entry["broken_weight_on_failed"] = sum(
                warm.weights[q][_BROKEN] for q in eval_failed
            ) / max(len(eval_failed), 1)
            entry["broken_weight_on_healthy"] = sum(
                warm.weights[q][_BROKEN] for q in eval_healthy
            ) / max(len(eval_healthy), 1)
            entry["failed_eval_queries"] = len(eval_failed)
        result["modes"][mode] = entry
    return result


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
