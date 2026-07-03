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
from collections.abc import Iterable

import ruffle

from ruffle_evals import SEED
from ruffle_evals.channels import CHANNEL_KEYS, Channels, Run
from ruffle_evals.datasets import Dataset

__all__ = ["FusionResult", "channel_configs", "rrf", "ruffle_cold", "ruffle_warm", "split_queries"]

# Fused rankings by query id, plus the mean per-channel weights the engine used on
# the evaluated queries (1.0 for every channel under plain RRF and, by
# construction, under the cold condition).
FusionResult = tuple[dict[str, list[tuple[str, float]]], dict[str, float]]

_TAGS = {"bm25": "bm25s-lucene", "tfidf": "char-wb-3-5", "dense": "all-MiniLM-L6-v2"}

_ANCHOR_CANDIDATES = 256


def channel_configs() -> list[ruffle.ChannelConfig]:
    """The three channel registrations. No good-score reference is declared: the
    harness measures the calibration-free path, where every reference is learned
    from traffic."""
    return [
        ruffle.ChannelConfig(
            ruffle.ChannelId(key, _TAGS[key]), ruffle.Direction.HIGHER_IS_BETTER
        )
        for key in CHANNEL_KEYS
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


def rrf(runs: dict[str, Run], qids: Iterable[str], eta: float = 60.0) -> FusionResult:
    """Plain unweighted reciprocal-rank fusion, implemented independently of the
    engine so the cold condition has something external to agree with.

    The tie conventions match the engine's (tied scores within a channel share
    their midrank; fused-score ties fall back to first-seen order), so agreement
    is checkable ranking for ranking rather than only metric for metric.
    """
    fused: dict[str, list[tuple[str, float]]] = {}
    for qid in qids:
        first_seen: dict[str, int] = {}
        for key in CHANNEL_KEYS:
            for doc_id, _ in runs[key].get(qid, []):
                first_seen.setdefault(doc_id, len(first_seen))
        scores = dict.fromkeys(first_seen, 0.0)
        for key in CHANNEL_KEYS:
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
    return fused, {key: 1.0 for key in CHANNEL_KEYS}


def ruffle_cold(runs: dict[str, Run], qids: Iterable[str]) -> FusionResult:
    """Ruffle stateless with an empty prior: per-query fusion with no accumulated
    baseline, the mode a deployment starts in."""
    configs = channel_configs()
    prior = ruffle.Fuser(configs).state
    fused: dict[str, list[tuple[str, float]]] = {}
    weight_sums = {key: 0.0 for key in CHANNEL_KEYS}
    count = 0
    for qid in qids:
        result = ruffle.Fuser.fuse_stateless(_inputs(runs, configs, qid), configs, prior)
        fused[qid] = [(item.id, item.score) for item in result.ranking]
        for key in CHANNEL_KEYS:
            weight_sums[key] += result.weights[key]
        count += 1
    return fused, {key: s / max(count, 1) for key, s in weight_sums.items()}


def ruffle_warm(
    runs: dict[str, Run],
    warm_qids: list[str],
    eval_qids: list[str],
    channels: Channels | None = None,
    coupling: bool = False,
    refreshes: int = 10,
) -> FusionResult:
    """Ruffle stateful: baselines accumulate over the warmup queries, then the
    evaluation queries are fused (still stateful, as a deployment would run).

    With ``coupling`` set, the warmup also interleaves anchor refreshes so the
    redundancy discount has the reliability, refresh count, and stability evidence
    it is gated on. Each anchor is one warmup query scored by every channel over a
    seeded random draw of corpus documents; a random draw rather than any
    channel's top-k, because a top-k pool is a selected sample that biases the
    correlation estimate.
    """
    configs = channel_configs()
    config = ruffle.FuseConfig(coupling=ruffle.CouplingConfig(enabled=True)) if coupling else None
    fuser = ruffle.Fuser(configs, config)

    anchor_qids = set(warm_qids[:: max(1, len(warm_qids) // refreshes)][:refreshes]) if coupling else set()
    rng = random.Random(SEED)

    for qid in warm_qids:
        fuser.fuse(_inputs(runs, configs, qid))
        if qid in anchor_qids:
            assert channels is not None
            candidates = rng.sample(list(channels.doc_ids), min(_ANCHOR_CANDIDATES, len(channels.doc_ids)))
            lookups = {key: channels.score_lookup(qid, key) for key in CHANNEL_KEYS}
            anchor = ruffle.Anchor.build(
                candidates, configs, lambda doc_id, key: lookups[key](doc_id)
            )
            fuser.refresh_coupling(anchor)

    fused: dict[str, list[tuple[str, float]]] = {}
    weight_sums = {key: 0.0 for key in CHANNEL_KEYS}
    for qid in eval_qids:
        result = fuser.fuse(_inputs(runs, configs, qid))
        fused[qid] = [(item.id, item.score) for item in result.ranking]
        for key in CHANNEL_KEYS:
            weight_sums[key] += result.weights[key]
    n = max(len(eval_qids), 1)
    return fused, {key: s / n for key, s in weight_sums.items()}
