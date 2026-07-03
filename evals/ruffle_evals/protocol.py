"""The main condition comparison, shared by the per-collection CLI runs and the
composite/heavy runners: single channels, the fusion baselines, the Ruffle
conditions, and the oracle ceiling, all scored on one evaluation split."""

from __future__ import annotations

from collections.abc import Sequence

import ruffle

from ruffle_evals.baselines import borda, combmnz, combsum, isr, oracle_rrf
from ruffle_evals.channels import Channels, Run
from ruffle_evals.evaluate import delta_profile, evaluate, paired_p
from ruffle_evals.fusion import (
    FusionOutcome,
    aggressive_config,
    channel_configs,
    rrf,
    ruffle_cold,
    ruffle_warm,
)

__all__ = ["BASELINE", "main_conditions"]

BASELINE = "rrf"


def main_conditions(
    runs: dict[str, Run],
    channels: Channels | None,
    qrels: dict[str, dict[str, int]],
    warm_qids: list[str],
    eval_qids: list[str],
    refreshes: int,
    configs: list[ruffle.ChannelConfig] | None = None,
    warm_outcomes: dict[str, FusionOutcome] | None = None,
) -> tuple[dict, dict]:
    """Every main condition evaluated on one split: returns the persistable
    condition entries and the per-query nDCG@10 vectors behind the paired tests.

    ``warm_outcomes`` (name -> outcome) lets a caller substitute externally
    warmed conditions (the multi-evaluation-set path); otherwise the warm
    conditions are computed here from the warmup split.
    """
    configs = channel_configs() if configs is None else configs
    keys = [c.id.key for c in configs]

    rankings: dict[str, dict] = {}
    outcomes: dict[str, FusionOutcome | None] = {}
    for key in keys:
        rankings[key] = {qid: runs[key].get(qid, []) for qid in eval_qids}
        outcomes[key] = None
    for name, ranking in (
        ("borda", borda(runs, eval_qids, keys=keys)),
        ("isr", isr(runs, eval_qids, keys=keys)),
        ("combsum", combsum(runs, eval_qids, keys=keys)),
        ("combmnz", combmnz(runs, eval_qids, keys=keys)),
    ):
        rankings[name] = ranking
        outcomes[name] = None
    engine_conditions = [
        (BASELINE, rrf(runs, eval_qids, keys=keys)),
        ("ruffle-cold", ruffle_cold(runs, eval_qids, configs=configs)),
    ]
    if warm_outcomes is None:
        engine_conditions.extend(
            [
                ("ruffle-warm", ruffle_warm(runs, warm_qids, eval_qids, configs=configs)),
                (
                    "ruffle-warm-coupled",
                    ruffle_warm(
                        runs,
                        warm_qids,
                        eval_qids,
                        configs=configs,
                        channels=channels,
                        coupling=True,
                        refreshes=refreshes,
                    ),
                ),
                (
                    "ruffle-warm-aggressive",
                    ruffle_warm(
                        runs,
                        warm_qids,
                        eval_qids,
                        configs=configs,
                        channels=channels,
                        refreshes=refreshes,
                        config=aggressive_config(),
                    ),
                ),
            ]
        )
    else:
        engine_conditions.extend(warm_outcomes.items())
    for name, outcome in engine_conditions:
        rankings[name] = outcome.rankings
        outcomes[name] = outcome
    oracle_rankings, oracle_weights = oracle_rrf(runs, qrels, eval_qids, keys=keys)
    rankings["rrf-oracle"] = oracle_rankings
    outcomes["rrf-oracle"] = None

    conditions: dict = {}
    per_queries: dict = {}
    baseline_per_query = None
    for condition, ranking in rankings.items():
        aggregate, per_query = evaluate(qrels, ranking)
        outcome = outcomes[condition]
        # The per-query nDCG vector feeds the paired test but is not persisted:
        # it is thousands of lines per collection in the committed results, and
        # regenerable exactly from the fixed seed and the cached runs.
        per_queries[condition] = per_query
        conditions[condition] = {
            "metrics": aggregate,
            "mean_weights": None if outcome is None else outcome.mean_weights(keys),
            "mean_conflict": None if outcome is None else outcome.mean_conflict(),
        }
        if condition == BASELINE:
            baseline_per_query = per_query
    # The oracle's fixed simplex weights render in the weights column; they are
    # fitted on the judgments, which is what makes the row a ceiling.
    conditions["rrf-oracle"]["mean_weights"] = oracle_weights
    for condition, entry in conditions.items():
        if condition == BASELINE:
            entry["p_vs_rrf"] = None
            entry["delta_vs_rrf"] = None
        else:
            entry["p_vs_rrf"] = paired_p(baseline_per_query, per_queries[condition])
            entry["delta_vs_rrf"] = delta_profile(baseline_per_query, per_queries[condition])
    return conditions, per_queries
