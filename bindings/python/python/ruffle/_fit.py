"""Offline base-weight fitting from a small graded sample.

This module is the labeled escape hatch. Ruffle's engine is calibration-free
by contract: it never learns that one channel is globally better than
another, because that is cross-channel information only relevance labels can
establish. An operator who holds a small set of graded queries can convert
exactly that information into :attr:`ruffle.ChannelConfig.base_weight`
declarations with :func:`fit_base_weights`, and the engine's per-query
adaptation composes on top. Nothing here touches the engine, its
configuration defaults, or its persisted state, and the module has no
dependencies beyond the standard library.

The procedure is a joint grid search over the weight simplex (step 0.1) with
a minimum weight per channel, optimizing linear-gain nDCG@10 at the deployed
RRF rank constant, guarded by a cross-fitted split-sample acceptance test:
the sample is split in half, each half's best fit is scored against uniform
weights on the opposite half, and the full-sample fit is returned only when
the pooled held-out estimate of its benefit is positive. The joint fit
handles redundant channels by construction (it optimizes the ensemble, not
per-channel credit); the minimum weight, not the acceptance test, is what
bounds the damage of an accepted-but-wrong fit, since no channel can be
silenced below the floor.

On the evaluation harness (nineteen collections, two-fold crossfit) the
guarded fit at a 64-query budget improved mean nDCG@10 on every collection
when composed with the engine, with the largest gains where one channel
dominates; the trade is a real per-query loss tail on queries where a
down-weighted channel was right, inherent to any static tilt. Fits are made
with coupling off; an enabled coupling discount additionally suppresses
redundant channels the fit may already have floored.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import NamedTuple, cast

from ruffle._frozen import FrozenMap

__all__ = ["FittedBaseWeights", "fit_base_weights"]

# One channel's items for one query: ranked ids best-first, or (id, score)
# pairs ordered by descending score with ties sharing their midrank, matching
# the engine's tie convention.
RankedItems = Sequence[str]
ScoredItems = Sequence[tuple[str, float]]
QueryItems = RankedItems | ScoredItems

_GRID_STEPS = 10  # simplex resolution: weights move in steps of 0.1
_NDCG_DEPTH = 10


class FittedBaseWeights(NamedTuple):
    """The outcome of :func:`fit_base_weights`.

    ``weights`` maps each channel key to its fitted base weight on the unit
    simplex (they sum to 1; only the ratios matter to the engine's
    renormalization). ``fell_back`` reports that the acceptance test rejected
    the fit and ``weights`` is uniform. ``heldout_fitted`` and
    ``heldout_uniform`` are the cross-fitted held-out mean nDCG@10 of the
    half-sample fits and of uniform weights, an honest estimate uncontaminated
    by the fit's own selection; their difference is what the acceptance test
    checks against zero. ``n`` is the number of graded queries the fit used;
    ``n_dropped`` counts queries excluded because no judged document carried a
    positive grade.
    """

    weights: Mapping[str, float]
    fell_back: bool
    heldout_fitted: float
    heldout_uniform: float
    n: int
    n_dropped: int


def _ranks(items: QueryItems) -> dict[str, float]:
    """Document id to RRF rank. Ranked input ranks by position; scored input
    orders by descending score with input order breaking exact ties for
    position, and tied scores share their midrank (the engine's convention)."""
    if not items:
        return {}
    if isinstance(items[0], str):
        ranked = cast(RankedItems, items)
        return {doc: float(i + 1) for i, doc in enumerate(ranked)}
    scored = cast(ScoredItems, items)
    order = sorted(enumerate(scored), key=lambda pair: (-pair[1][1], pair[0]))
    ranks: dict[str, float] = {}
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and order[j + 1][1][1] == order[i][1][1]:
            j += 1
        midrank = (i + j) / 2.0 + 1.0
        for _, (doc, _score) in order[i : j + 1]:
            ranks[doc] = midrank
        i = j + 1
    return ranks


def _ndcg10(order: Sequence[str], rels: Mapping[str, int]) -> float:
    """Linear-gain nDCG@10: gain is the raw grade, discount 1/log2(i + 2).

    This is the harness convention, not pytrec_eval's exponential-gain
    variant; a fit optimized here and measured there optimizes a slightly
    different objective.
    """
    dcg = 0.0
    for i, doc in enumerate(order[:_NDCG_DEPTH]):
        rel = rels.get(doc, 0)
        if rel > 0:
            dcg += rel / math.log2(i + 2.0)
    ideal = sorted((r for r in rels.values() if r > 0), reverse=True)[:_NDCG_DEPTH]
    idcg = sum(rel / math.log2(i + 2.0) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0.0 else 0.0


def _simplex(n: int, floor_steps: int) -> list[tuple[float, ...]]:
    """Every weight vector on the step-0.1 simplex with each component at or
    above the floor, in a deterministic order."""
    points: list[tuple[int, ...]] = []

    def rec(prefix: list[int], left: int, parts: int) -> None:
        if parts == 1:
            if left >= floor_steps:
                points.append((*prefix, left))
            return
        for c in range(floor_steps, left - floor_steps * (parts - 1) + 1):
            rec([*prefix, c], left - c, parts - 1)

    rec([], _GRID_STEPS, n)
    return [tuple(c / _GRID_STEPS for c in point) for point in points]


class _Query(NamedTuple):
    docs: tuple[str, ...]
    contribs: tuple[tuple[float, ...], ...]  # per doc, per channel: 1/(eta+rank)
    rels: Mapping[str, int]


def _prepare(
    runs: Mapping[str, Mapping[str, QueryItems]],
    qrels: Mapping[str, Mapping[str, int]],
    keys: Sequence[str],
    eta: float,
) -> tuple[list[_Query], int]:
    """Per-query contribution tables over the qrels/runs intersection, in
    sorted query order; zero-gain queries are dropped and counted."""
    queries: list[_Query] = []
    dropped = 0
    for qid in sorted(qrels):
        rels = qrels[qid]
        if not any(r > 0 for r in rels.values()):
            dropped += 1
            continue
        per_channel = [_ranks(runs[k].get(qid, ())) for k in keys]
        first_seen: dict[str, int] = {}
        for ranks in per_channel:
            for doc in ranks:
                first_seen.setdefault(doc, len(first_seen))
        if not first_seen:
            dropped += 1
            continue
        docs = tuple(first_seen)
        contribs = tuple(
            tuple((1.0 / (eta + ranks[doc])) if doc in ranks else 0.0 for ranks in per_channel)
            for doc in docs
        )
        queries.append(_Query(docs=docs, contribs=contribs, rels=rels))
    return queries, dropped


def _score(query: _Query, weights: Sequence[float]) -> float:
    fused = [
        (sum(w * c for w, c in zip(weights, row, strict=True)), i)
        for i, row in enumerate(query.contribs)
    ]
    fused.sort(key=lambda pair: (-pair[0], pair[1]))  # first-seen breaks ties
    order = [query.docs[i] for _, i in fused[:_NDCG_DEPTH]]
    return _ndcg10(order, query.rels)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _argmax(queries: Sequence[_Query], grid: Sequence[tuple[float, ...]]) -> tuple[float, ...]:
    best = grid[0]
    best_mean = -1.0
    for w in grid:
        m = _mean([_score(q, w) for q in queries])
        if m > best_mean:
            best_mean, best = m, w
    return best


def fit_base_weights(
    runs: Mapping[str, Mapping[str, QueryItems]],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    eta: float,
    floor: float = 0.2,
    guard: bool = True,
) -> FittedBaseWeights:
    """Fits static per-channel base weights on a graded sample.

    ``runs`` maps each channel key to its per-query retrieved items: either
    ranked document ids best-first, or ``(id, score)`` pairs (scores are used
    only for ordering; tied scores share their midrank, the engine's
    convention). ``qrels`` is the graded sample, query id to document id to
    integer grade; grades above zero are gains. The queries used are the
    intersection of ``qrels`` with the runs, minus queries with no positive
    grade; both counts are reported on the result. The sample must be
    representative traffic: fitting on queries selected for difficulty or
    disagreement biases the weights toward that slice.

    ``eta`` is REQUIRED and must equal the ``rrf_eta`` the weights will be
    deployed under; RRF weight fitting is eta-sensitive, and fitting at one
    constant while fusing at another measurably degrades the fit.

    ``floor`` is the minimum weight per channel on the simplex (default
    ``0.2``, supported band 0.1 to 0.25). The floor is the do-no-harm
    mechanism: no fit can silence a channel below it, which bounds the loss
    on queries where a down-weighted channel was right, and caps the
    achievable tilt at ``(1 - floor) / floor`` (4:1 at the default) at some
    cost where one channel is overwhelmingly dominant. A floor that leaves no
    non-uniform grid point for the channel count is refused.

    With ``guard`` on (default), acceptance is a cross-fitted split-sample
    test: the sample is split deterministically in half (alternating over
    sorted query ids), each half's best fit is scored against uniform weights
    on the opposite half, and the full-sample fit is returned only when the
    pooled held-out estimate of its benefit is positive; otherwise uniform
    weights are returned with ``fell_back`` set. The held-out estimate is
    honest (selection and evaluation never share queries), so the zero bar is
    a genuine expected-do-no-harm criterion; a high fallback rate on a
    deployment with little cross-channel quality difference is the intended
    safe outcome, not a failure. Sixty-four graded queries is the reference
    budget; sixteen is the guarded minimum, at roughly half the expected
    gain.

    The result's ``weights`` sum to 1 and only their ratios matter; apply
    them through :attr:`ruffle.ChannelConfig.base_weight`. Refit when a
    channel's model changes (the tag bump) or the corpus drifts. The fit is
    deterministic: no randomness anywhere, ties in the grid search resolve to
    the first maximum in a fixed order.

    Raises :class:`ValueError` for fewer than two channels, a non-finite or
    negative ``eta``, a ``floor`` outside ``[0, 0.5)`` or one that admits no
    non-uniform grid point for the channel count, or a usable sample smaller
    than two queries.
    """
    keys = sorted(runs)
    if len(keys) < 2:
        raise ValueError(
            "fit_base_weights needs at least two channels; a single channel's "
            "base weight renormalizes away"
        )
    if not (math.isfinite(eta) and eta >= 0.0):
        raise ValueError("eta must be finite and non-negative")
    if not (math.isfinite(floor) and 0.0 <= floor < 0.5):
        raise ValueError("floor must be finite and in [0, 0.5)")
    floor_steps = round(floor * _GRID_STEPS)
    if len(keys) * floor_steps >= _GRID_STEPS:
        raise ValueError(
            f"floor {floor} admits no non-uniform weights for {len(keys)} "
            f"channels; lower the floor or reduce the channel count"
        )

    queries, dropped = _prepare(runs, qrels, keys, eta)
    if len(queries) < 2:
        raise ValueError(
            "the usable graded sample has fewer than two queries after "
            "intersecting qrels with the runs and dropping zero-gain queries"
        )

    grid = _simplex(len(keys), floor_steps)
    uniform = tuple(1.0 / len(keys) for _ in keys)
    full = _argmax(queries, grid)

    # Cross-fitted held-out estimate, computed for reporting even when the
    # guard is off: deterministic alternating split over the sorted sample.
    half_a = [q for i, q in enumerate(queries) if i % 2 == 0]
    half_b = [q for i, q in enumerate(queries) if i % 2 == 1]
    heldout_fitted: list[float] = []
    heldout_uniform: list[float] = []
    for selected, held in ((half_a, half_b), (half_b, half_a)):
        if not selected or not held:
            continue
        w = _argmax(selected, grid)
        heldout_fitted.extend(_score(q, w) for q in held)
        heldout_uniform.extend(_score(q, uniform) for q in held)
    fitted_mean = _mean(heldout_fitted)
    uniform_mean = _mean(heldout_uniform)

    fell_back = guard and fitted_mean <= uniform_mean
    chosen = uniform if fell_back else full
    return FittedBaseWeights(
        weights=FrozenMap(dict(zip(keys, chosen, strict=True))),
        fell_back=fell_back,
        heldout_fitted=fitted_mean,
        heldout_uniform=uniform_mean,
        n=len(queries),
        n_dropped=dropped,
    )
