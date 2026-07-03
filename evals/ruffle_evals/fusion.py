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
    "aggressive_config",
    "build_anchor_data",
    "channel_configs",
    "rrf",
    "ruffle_cold",
    "ruffle_warm",
    "ruffle_warm_multi",
    "split_queries",
]


def aggressive_config() -> "ruffle.FuseConfig":
    """The harness's aggressive profile: the same estimators with the
    conservatism turned down, every evidence gate left intact.

    Discrimination reacts more sharply to a departure from a channel's own norm
    (g_slope 1.0 -> 2.5) and can push an underperforming channel further down
    (g_floor 0.25 -> 0.1). The redundancy discount, once its reliability,
    refresh, and stability gates pass, may remove most of a duplicated signal
    (discount_cap 0.5 -> 0.9) with less mandatory shrinkage toward independence
    (shrink_to_identity 0.5 -> 0.2). What this profile deliberately cannot do is
    learn that one channel is globally better than another: that information is
    cross-channel and label-bound, outside the engine's contract at any setting.
    """
    return ruffle.FuseConfig(
        discrimination=ruffle.DiscriminationConfig(g_slope=2.5, g_floor=0.1),
        coupling=ruffle.CouplingConfig(enabled=True, discount_cap=0.9, shrink_to_identity=0.2),
    )

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


def build_anchor_data(
    channels: Channels,
    warm_qids: list[str],
    configs: list[ruffle.ChannelConfig],
    refreshes: int,
) -> dict[str, ruffle.Anchor]:
    """The anchor refreshes for one warmup pass, prebuilt: one full-scored
    anchor per selected warmup query, over a seeded random draw of corpus
    documents. A random draw rather than any channel's top-k, because a top-k
    pool is a selected sample that biases the correlation estimate.

    Anchors depend on the channels and the draw, never on the fusion
    configuration, so one payload serves every configuration candidate.
    """
    keys = [c.id.key for c in configs]
    anchor_qids = warm_qids[:: max(1, len(warm_qids) // refreshes)][:refreshes]
    rng = random.Random(SEED)
    anchors: dict[str, ruffle.Anchor] = {}
    # The draw order follows warmup order, keeping the rng sequence identical to
    # the previous inline construction.
    for qid in warm_qids:
        if qid not in anchor_qids:
            continue
        candidates = rng.sample(list(channels.doc_ids), min(_ANCHOR_CANDIDATES, len(channels.doc_ids)))
        scored = {key: channels.score_candidates(qid, candidates, key) for key in keys}
        index = {doc_id: i for i, doc_id in enumerate(candidates)}
        anchors[qid] = ruffle.Anchor.build(
            candidates, configs, lambda doc_id, key: scored[key][index[doc_id]]
        )
    return anchors


def _warm_fuser(
    runs: dict[str, Run],
    warm_qids: list[str],
    configs: list[ruffle.ChannelConfig],
    config: ruffle.FuseConfig | None,
    channels: Channels | None,
    coupling: bool,
    refreshes: int,
    anchor_data: dict[str, ruffle.Anchor] | None = None,
) -> ruffle.Fuser:
    """A stateful fuser with baselines accumulated over the warmup queries, and,
    under ``coupling``, redundancy baselines from interleaved anchor refreshes,
    prebuilt or computed here from the live channels."""
    fuser = ruffle.Fuser(configs, config)
    anchors: dict[str, ruffle.Anchor] = {}
    if coupling:
        if anchor_data is not None:
            anchors = anchor_data
        else:
            assert channels is not None
            anchors = build_anchor_data(channels, warm_qids, configs, refreshes)
    for qid in warm_qids:
        fuser.fuse(_inputs(runs, configs, qid))
        anchor = anchors.get(qid)
        if anchor is not None:
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


def _resolve_config(
    coupling: bool, config: ruffle.FuseConfig | None
) -> tuple[ruffle.FuseConfig | None, bool]:
    """The effective configuration and whether anchor refreshes run: an explicit
    ``config`` wins, and its coupling switch decides the anchors."""
    if config is not None:
        return config, config.coupling.enabled
    if coupling:
        return ruffle.FuseConfig(coupling=ruffle.CouplingConfig(enabled=True)), True
    return None, False


def ruffle_warm(
    runs: dict[str, Run],
    warm_qids: list[str],
    eval_qids: list[str],
    configs: list[ruffle.ChannelConfig] | None = None,
    channels: Channels | None = None,
    coupling: bool = False,
    refreshes: int = 10,
    config: ruffle.FuseConfig | None = None,
    anchor_data: dict[str, "ruffle.Anchor"] | None = None,
) -> FusionOutcome:
    """Ruffle stateful: baselines accumulate over the warmup queries, then the
    evaluation queries are fused (still stateful, as a deployment would run)."""
    configs = channel_configs() if configs is None else configs
    fuse_config, anchors = _resolve_config(coupling, config)
    # Without live channel models or prebuilt anchors there is nothing to score
    # an anchor with; the coupling switch then stays on but its gates never
    # pass, which reads as "enabled, no evidence yet", the recall-safe direction.
    anchors = anchors and (channels is not None or anchor_data is not None)
    fuser = _warm_fuser(
        runs, warm_qids, configs, fuse_config, channels, anchors, refreshes, anchor_data
    )
    return _fuse_eval(fuser, runs, eval_qids, configs)


def ruffle_warm_multi(
    runs: dict[str, Run],
    warm_qids: list[str],
    eval_sets: dict[str, list[str]],
    configs: list[ruffle.ChannelConfig],
    channels: Channels | None = None,
    coupling: bool = False,
    refreshes: int = 10,
    config: ruffle.FuseConfig | None = None,
) -> dict[str, FusionOutcome]:
    """One warmup, several evaluation sets: each set is fused by a fuser resumed
    from the same warm-state snapshot, so no evaluation set's queries leak into
    another's baselines."""
    fuse_config, anchors = _resolve_config(coupling, config)
    anchors = anchors and channels is not None
    warm_state = _warm_fuser(runs, warm_qids, configs, fuse_config, channels, anchors, refreshes).state
    outcomes: dict[str, FusionOutcome] = {}
    for name, eval_qids in eval_sets.items():
        fuser = ruffle.Fuser.resume(configs, warm_state, fuse_config)
        outcomes[name] = _fuse_eval(fuser, runs, eval_qids, configs)
    return outcomes
