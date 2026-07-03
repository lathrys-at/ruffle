"""The fusion conditions: plain RRF, Ruffle cold, and Ruffle warm.

Every condition consumes the same per-query top-k channel runs, so differences in
the metrics come from the fusion alone. The cold condition calls
``Fuser.fuse_stateless`` with an empty prior, which reduces to unweighted RRF by
construction; it is kept as a separate condition because that reduction is a
recall-safety claim worth verifying on real runs rather than assuming.

The warm conditions replay the warmup split through a stateful fuser, in one
variant also refreshing the pairwise redundancy baselines from full-scored
anchors, then fuse the held-out evaluation split with the accumulated state. No
relevance judgment is ever visible to the engine; warming is unsupervised, and
the split exists so the evaluated queries are not the ones the baselines were
first formed on.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import ruffle

from ruffle_evals import SEED
from ruffle_evals.channels import CHANNEL_KEYS, Channels, Run
from ruffle_evals.datasets import Dataset

__all__ = [
    "FusionOutcome",
    "channel_configs",
    "rrf",
    "ruffle_cold",
    "ruffle_warm",
    "ruffle_warm_multi",
    "split_queries",
]

_TAGS = {"bm25": "bm25s-lucene", "tfidf": "char-wb-3-5", "dense": "all-MiniLM-L6-v2"}

_ANCHOR_CANDIDATES = 256


@dataclass(frozen=True)
class FusionOutcome:
    """One condition's fused rankings plus the engine readings behind them.

    ``weights`` holds the per-channel weight used on each query (1.0 for every
    channel under plain RRF and, by construction, under the cold condition).
    ``conflict`` holds the per-query conflict diagnostic where the engine ran,
    ``None`` for the harness's own RRF baseline.
    """

    rankings: dict[str, list[tuple[str, float]]]
    weights: dict[str, dict[str, float]]
    conflict: dict[str, float] | None

    def mean_weights(self, keys: Sequence[str]) -> dict[str, float]:
        n = max(len(self.weights), 1)
        return {k: sum(w[k] for w in self.weights.values()) / n for k in keys}

    def mean_conflict(self) -> float | None:
        if self.conflict is None or not self.conflict:
            return None
        return sum(self.conflict.values()) / len(self.conflict)


def channel_configs(
    keys: Sequence[str] = CHANNEL_KEYS, tags: dict[str, str] | None = None
) -> list[ruffle.ChannelConfig]:
    """Channel registrations for the given keys. No good-score reference is
    declared: the harness measures the calibration-free path, where every
    reference is learned from traffic."""
    all_tags = {**_TAGS, **(tags or {})}
    return [
        ruffle.ChannelConfig(
            ruffle.ChannelId(key, all_tags[key]), ruffle.Direction.HIGHER_IS_BETTER
        )
        for key in keys
    ]


def split_queries(dataset: Dataset, warm_frac: float = 0.5) -> tuple[list[str], list[str]]:
    """A seeded shuffle of the query ids, split into warmup and evaluation."""
    qids = sorted(dataset.queries.keys())
    random.Random(SEED).shuffle(qids)
    cut = int(len(qids) * warm_frac)
    return qids[:cut], qids[cut:]


def _inputs(
    runs: dict[str, Run], configs: list[ruffle.ChannelConfig], qid: str
) -> list[ruffle.ChannelInput]:
    return [
        ruffle.ChannelInput.scored(config, runs[config.id.key].get(qid, []))
        for config in configs
    ]


def rrf(
    runs: dict[str, Run],
    qids: Iterable[str],
    keys: Sequence[str] = CHANNEL_KEYS,
    eta: float = 60.0,
) -> FusionOutcome:
    """Plain unweighted reciprocal-rank fusion, implemented independently of the
    engine so the cold condition has something external to agree with.

    The tie conventions match the engine's (tied scores within a channel share
    their midrank; fused-score ties fall back to first-seen order), so agreement
    is checkable ranking for ranking rather than only metric for metric.
    """
    fused: dict[str, list[tuple[str, float]]] = {}
    weights: dict[str, dict[str, float]] = {}
    for qid in qids:
        first_seen: dict[str, int] = {}
        for key in keys:
            for doc_id, _ in runs[key].get(qid, []):
                first_seen.setdefault(doc_id, len(first_seen))
        scores = dict.fromkeys(first_seen, 0.0)
        for key in keys:
            items = runs[key].get(qid, [])
            order = sorted(
                range(len(items)), key=lambda i: (-items[i][1], first_seen[items[i][0]])
            )
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and items[order[j + 1]][1] == items[order[i]][1]:
                    j += 1
                midrank = (i + j) / 2.0 + 1.0
                for p in order[i : j + 1]:
                    scores[items[p][0]] += 1.0 / (eta + midrank)
                i = j + 1
        fused[qid] = sorted(scores.items(), key=lambda it: (-it[1], first_seen[it[0]]))
        weights[qid] = {key: 1.0 for key in keys}
    return FusionOutcome(rankings=fused, weights=weights, conflict=None)


def ruffle_cold(
    runs: dict[str, Run], qids: Iterable[str], configs: list[ruffle.ChannelConfig] | None = None
) -> FusionOutcome:
    """Ruffle stateless with an empty prior: per-query fusion with no accumulated
    baseline, the mode a deployment starts in."""
    configs = channel_configs() if configs is None else configs
    keys = [c.id.key for c in configs]
    prior = ruffle.Fuser(configs).state
    fused: dict[str, list[tuple[str, float]]] = {}
    weights: dict[str, dict[str, float]] = {}
    conflict: dict[str, float] = {}
    for qid in qids:
        result = ruffle.Fuser.fuse_stateless(_inputs(runs, configs, qid), configs, prior)
        fused[qid] = [(item.id, item.score) for item in result.ranking]
        weights[qid] = {key: result.weights[key] for key in keys}
        conflict[qid] = result.conflict
    return FusionOutcome(rankings=fused, weights=weights, conflict=conflict)


def _warm_fuser(
    runs: dict[str, Run],
    warm_qids: list[str],
    configs: list[ruffle.ChannelConfig],
    config: ruffle.FuseConfig | None,
    channels: Channels | None,
    coupling: bool,
    refreshes: int,
) -> ruffle.Fuser:
    """A stateful fuser with baselines accumulated over the warmup queries, and,
    under ``coupling``, redundancy baselines from interleaved anchor refreshes.

    Each anchor is one warmup query scored by every channel over a seeded random
    draw of corpus documents; a random draw rather than any channel's top-k,
    because a top-k pool is a selected sample that biases the correlation
    estimate.
    """
    keys = [c.id.key for c in configs]
    fuser = ruffle.Fuser(configs, config)
    anchor_qids = set(warm_qids[:: max(1, len(warm_qids) // refreshes)][:refreshes]) if coupling else set()
    rng = random.Random(SEED)
    for qid in warm_qids:
        fuser.fuse(_inputs(runs, configs, qid))
        if qid in anchor_qids:
            assert channels is not None
            candidates = rng.sample(list(channels.doc_ids), min(_ANCHOR_CANDIDATES, len(channels.doc_ids)))
            scored = {key: channels.score_candidates(qid, candidates, key) for key in keys}
            index = {doc_id: i for i, doc_id in enumerate(candidates)}
            anchor = ruffle.Anchor.build(
                candidates, configs, lambda doc_id, key: scored[key][index[doc_id]]
            )
            fuser.refresh_coupling(anchor)
    return fuser


def _fuse_eval(
    fuser: ruffle.Fuser,
    runs: dict[str, Run],
    eval_qids: list[str],
    configs: list[ruffle.ChannelConfig],
) -> FusionOutcome:
    keys = [c.id.key for c in configs]
    fused: dict[str, list[tuple[str, float]]] = {}
    weights: dict[str, dict[str, float]] = {}
    conflict: dict[str, float] = {}
    for qid in eval_qids:
        result = fuser.fuse(_inputs(runs, configs, qid))
        fused[qid] = [(item.id, item.score) for item in result.ranking]
        weights[qid] = {key: result.weights[key] for key in keys}
        conflict[qid] = result.conflict
    return FusionOutcome(rankings=fused, weights=weights, conflict=conflict)


def ruffle_warm(
    runs: dict[str, Run],
    warm_qids: list[str],
    eval_qids: list[str],
    configs: list[ruffle.ChannelConfig] | None = None,
    channels: Channels | None = None,
    coupling: bool = False,
    refreshes: int = 10,
) -> FusionOutcome:
    """Ruffle stateful: baselines accumulate over the warmup queries, then the
    evaluation queries are fused (still stateful, as a deployment would run)."""
    configs = channel_configs() if configs is None else configs
    config = ruffle.FuseConfig(coupling=ruffle.CouplingConfig(enabled=True)) if coupling else None
    fuser = _warm_fuser(runs, warm_qids, configs, config, channels, coupling, refreshes)
    return _fuse_eval(fuser, runs, eval_qids, configs)


def ruffle_warm_multi(
    runs: dict[str, Run],
    warm_qids: list[str],
    eval_sets: dict[str, list[str]],
    configs: list[ruffle.ChannelConfig],
    channels: Channels | None = None,
    coupling: bool = False,
    refreshes: int = 10,
) -> dict[str, FusionOutcome]:
    """One warmup, several evaluation sets: each set is fused by a fuser resumed
    from the same warm-state snapshot, so no evaluation set's queries leak into
    another's baselines."""
    config = ruffle.FuseConfig(coupling=ruffle.CouplingConfig(enabled=True)) if coupling else None
    warm_state = _warm_fuser(runs, warm_qids, configs, config, channels, coupling, refreshes).state
    outcomes: dict[str, FusionOutcome] = {}
    for name, eval_qids in eval_sets.items():
        fuser = ruffle.Fuser.resume(configs, warm_state, config)
        outcomes[name] = _fuse_eval(fuser, runs, eval_qids, configs)
    return outcomes
