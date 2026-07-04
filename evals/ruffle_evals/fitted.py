"""Approximate-oracle weights: fixed per-channel RRF weights fitted on a small
graded subsample of the warmup split, and their composition with Ruffle.

The oracle condition answers "what could any fixed weighting achieve with the
evaluation labels themselves". These conditions answer the deployable version:
an operator grades a few dozen queries, grid-searches fixed RRF weights on
them, and either uses those weights as-is (``rrf-fitted``) or declares them as
``base_weight`` on the channel registrations so Ruffle's per-query adaptation
composes on top (``ruffle-warm-fitted``). The subsample comes from the warmup
split only; the evaluation split stays untouched by any label.

The fit is repeated over a few seeded draws because a 10-to-100-query fit is
noisy. The first draw feeds the condition rows (a fixed, selection-free
choice); every draw's weights and evaluation nDCG@10 are stored so the spread
is visible in the result files.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from ruffle_evals import SEED
from ruffle_evals.baselines import oracle_rrf
from ruffle_evals.channels import Run

__all__ = ["DRAWS", "fit_budget", "fitted_weight_draws", "fixed_rrf"]

DRAWS = 3


def fit_budget(judged: int) -> int:
    """The grading budget: 5% of the judged warmup queries, floored at 10 (a
    percentage of a small collection is meaninglessly few) and capped at 100 (a
    grading session nobody extends past)."""
    return min(100, max(10, round(0.05 * judged)))


def fitted_weight_draws(
    runs: dict[str, Run],
    qrels: dict[str, dict[str, int]],
    warm_qids: list[str],
    keys: Sequence[str],
) -> tuple[int, list[dict[str, float]]]:
    """Simplex-grid weights fitted on ``DRAWS`` seeded subsamples of the judged
    warmup queries. Returns the budget and one weight map per draw."""
    judged = sorted(q for q in warm_qids if qrels.get(q))
    budget = fit_budget(len(judged))
    draws = []
    for d in range(DRAWS):
        rng = random.Random(SEED + d)
        sample = list(judged) if len(judged) <= budget else rng.sample(judged, budget)
        _, weights = oracle_rrf(runs, qrels, sample, keys=keys)
        draws.append(weights)
    return budget, draws


def fixed_rrf(
    runs: dict[str, Run],
    qids: Sequence[str],
    weights: dict[str, float],
    keys: Sequence[str],
    eta: float = 60.0,
) -> dict[str, list[tuple[str, float]]]:
    """Weighted RRF at fixed per-channel weights, under the oracle's rank and
    tie conventions so the fitted row is exactly the oracle procedure at a
    smaller labeling budget."""
    rankings: dict[str, list[tuple[str, float]]] = {}
    for qid in qids:
        first_seen: dict[str, int] = {}
        for key in keys:
            for doc_id, _ in runs[key].get(qid, []):
                first_seen.setdefault(doc_id, len(first_seen))
        scores = dict.fromkeys(first_seen, 0.0)
        for key in keys:
            w = weights[key]
            for rank, (doc_id, _) in enumerate(runs[key].get(qid, []), start=1):
                scores[doc_id] += w / (eta + rank)
        rankings[qid] = sorted(scores.items(), key=lambda it: (-it[1], first_seen[it[0]]))
    return rankings
