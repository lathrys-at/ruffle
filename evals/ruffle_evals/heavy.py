"""The large-corpus runners: cqadupstack (a 12-subforum composite) and MS MARCO
passage with the TREC-DL 2019/2020 query sets.

Both run the main condition comparison only; the degraded-channel and
learning-curve experiments answer mechanism questions already covered on the
four standard collections.

cqadupstack follows the BEIR reporting convention: each subforum is its own
corpus with its own channels, warmed and evaluated independently, and the
reported metrics are macro-averages over the twelve subforums. The paired test
pools the per-query values across subforums, each query being one paired
observation regardless of forum.

MS MARCO uses two channels (BM25 and dense, the canonical hybrid pair; a
character-ngram TF-IDF matrix is not workable at 8.8M passages). The dev/small
queries split into an unsupervised warmup half and an evaluation half, and the
TREC-DL 2019/2020 judged sets are evaluated with fusers resumed from the same
dev-warmed state snapshot: the transfer of warm baselines to a foreign query
set over the same corpus is exactly the deployment story.
"""

from __future__ import annotations

import random

import ir_datasets

from ruffle_evals import SEED
from ruffle_evals.channels import CHANNEL_KEYS, Channels
from ruffle_evals.datasets import load_id
from ruffle_evals.evaluate import paired_p
from ruffle_evals.fusion import (
    aggressive_config,
    channel_configs,
    ruffle_warm_multi,
    split_queries,
)
from ruffle_evals.protocol import BASELINE, main_conditions

__all__ = ["MSMARCO_KEYS", "SUBFORUMS", "run_cqadupstack", "run_msmarco"]

SUBFORUMS = (
    "android",
    "english",
    "gaming",
    "gis",
    "mathematica",
    "physics",
    "programmers",
    "stats",
    "tex",
    "unix",
    "webmasters",
    "wordpress",
)

MSMARCO_KEYS = ("bm25", "dense")

_MSMARCO_SETS = {
    "dev": "msmarco-passage/dev/small",
    "dl19": "msmarco-passage/trec-dl-2019/judged",
    "dl20": "msmarco-passage/trec-dl-2020/judged",
}


def run_cqadupstack(k: int, refreshes: int) -> dict:
    """The composite cqadupstack run: per-subforum pipelines, macro-averaged."""
    per_sub: dict[str, dict] = {}
    pooled: dict[str, dict[str, float]] = {}
    totals = {"warm": 0, "eval": 0}
    for sub in SUBFORUMS:
        name = f"cqadupstack-{sub}"
        print(f"[cqadupstack] {sub}: loading", flush=True)
        dataset = load_id(f"beir/cqadupstack/{sub}", name)
        channels = Channels.for_dataset(dataset)
        runs = channels.runs(k)
        warm_qids, eval_qids = split_queries(dataset)
        totals["warm"] += len(warm_qids)
        totals["eval"] += len(eval_qids)
        conditions, per_queries = main_conditions(
            runs, channels, dataset.qrels, warm_qids, eval_qids, refreshes
        )
        per_sub[sub] = {"eval_queries": len(eval_qids), "conditions": conditions}
        for condition, values in per_queries.items():
            pooled.setdefault(condition, {}).update(
                {f"{sub}:{qid}": v for qid, v in values.items()}
            )
        print(
            f"[cqadupstack] {sub}: {len(eval_qids)} eval queries, "
            f"warm nDCG@10 {conditions['ruffle-warm']['metrics']['nDCG@10']:.4f}",
            flush=True,
        )

    conditions = _macro_aggregate(per_sub, pooled)
    return {
        "dataset": "cqadupstack",
        "ir_datasets_id": "beir/cqadupstack/*",
        "note": (
            "Macro-averaged over the 12 subforums, each with its own corpus, "
            "channels, warmup, and oracle; the paired test pools per-query "
            "nDCG@10 across subforums."
        ),
        "channel_keys": list(CHANNEL_KEYS),
        "k": k,
        "seed": SEED,
        "coupling_refreshes": refreshes,
        "warm_queries": totals["warm"],
        "eval_queries": totals["eval"],
        "conditions": conditions,
        "subforums": {
            sub: {
                "eval_queries": entry["eval_queries"],
                "ndcg10": {
                    condition: entry["conditions"][condition]["metrics"]["nDCG@10"]
                    for condition in entry["conditions"]
                },
            }
            for sub, entry in per_sub.items()
        },
    }


def _macro_aggregate(per_sub: dict[str, dict], pooled: dict[str, dict[str, float]]) -> dict:
    subs = list(per_sub)
    names = list(per_sub[subs[0]]["conditions"])
    weights_total = sum(per_sub[s]["eval_queries"] for s in subs)
    aggregated: dict = {}
    for condition in names:
        entries = [per_sub[s]["conditions"][condition] for s in subs]
        metrics = {
            m: sum(e["metrics"][m] for e in entries) / len(entries)
            for m in entries[0]["metrics"]
        }
        if entries[0]["mean_weights"] is None:
            mean_weights = None
        else:
            # Weights are per-query means, so the aggregate weighs each subforum
            # by its evaluation-query count (a pooled mean), unlike the metrics,
            # which follow BEIR's equal-weight macro-average.
            mean_weights = {
                key: sum(
                    e["mean_weights"][key] * per_sub[s]["eval_queries"]
                    for s, e in zip(subs, entries)
                )
                / weights_total
                for key in entries[0]["mean_weights"]
            }
        aggregated[condition] = {
            "metrics": metrics,
            "mean_weights": mean_weights,
            "mean_conflict": None
            if entries[0]["mean_conflict"] is None
            else sum(
                e["mean_conflict"] * per_sub[s]["eval_queries"]
                for s, e in zip(subs, entries)
            )
            / weights_total,
            "p_vs_rrf": None
            if condition == BASELINE
            else paired_p(pooled[BASELINE], pooled[condition]),
        }
    return aggregated


def _load_msmarco_queryset(dsid: str, prefix: str) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    ds = ir_datasets.load(dsid)
    queries = {f"{prefix}:{q.query_id}": q.text for q in ds.queries_iter()}
    qrels: dict[str, dict[str, int]] = {}
    for qrel in ds.qrels_iter():
        qrels.setdefault(f"{prefix}:{qrel.query_id}", {})[qrel.doc_id] = qrel.relevance
    return queries, qrels


def run_msmarco(k: int, refreshes: int) -> dict:
    """MS MARCO passage: dev/small split into warmup and evaluation halves, and
    TREC-DL 2019/2020 evaluated from the same dev-warmed state snapshot."""
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    set_qids: dict[str, list[str]] = {}
    for prefix, dsid in _MSMARCO_SETS.items():
        qs, qr = _load_msmarco_queryset(dsid, prefix)
        queries.update(qs)
        qrels.update(qr)
        set_qids[prefix] = sorted(qs)
        print(f"[msmarco] {prefix}: {len(qs)} queries", flush=True)

    print("[msmarco] loading corpus", flush=True)
    corpus = ir_datasets.load("msmarco-passage")
    doc_ids: list[str] = []
    texts: list[str] = []
    for doc in corpus.docs_iter():
        doc_ids.append(doc.doc_id)
        texts.append(doc.text)
    print(f"[msmarco] {len(doc_ids)} passages", flush=True)

    channels = Channels("msmarco", doc_ids, texts, queries, keys=MSMARCO_KEYS)
    del texts
    runs = channels.runs(k)
    print("[msmarco] runs ready", flush=True)

    dev_qids = list(set_qids["dev"])
    random.Random(SEED).shuffle(dev_qids)
    cut = len(dev_qids) // 2
    warm_qids, dev_eval = dev_qids[:cut], dev_qids[cut:]
    eval_sets = {"dev": dev_eval, "dl19": set_qids["dl19"], "dl20": set_qids["dl20"]}
    configs = channel_configs(MSMARCO_KEYS)

    print("[msmarco] warming (plain, coupled, aggressive)", flush=True)
    warm_plain = ruffle_warm_multi(runs, warm_qids, eval_sets, configs)
    warm_coupled = ruffle_warm_multi(
        runs, warm_qids, eval_sets, configs, channels=channels, coupling=True, refreshes=refreshes
    )
    warm_aggressive = ruffle_warm_multi(
        runs,
        warm_qids,
        eval_sets,
        configs,
        channels=channels,
        refreshes=refreshes,
        config=aggressive_config(),
    )

    results_sets: dict[str, dict] = {}
    for name, eval_qids in eval_sets.items():
        conditions, _ = main_conditions(
            runs,
            channels,
            qrels,
            [],
            eval_qids,
            refreshes,
            configs=configs,
            warm_outcomes={
                "ruffle-warm": warm_plain[name],
                "ruffle-warm-coupled": warm_coupled[name],
                "ruffle-warm-aggressive": warm_aggressive[name],
            },
        )
        results_sets[name] = {"eval_queries": len(eval_qids), "conditions": conditions}
        print(f"[msmarco] {name} conditions done", flush=True)

    return {
        "dataset": "msmarco",
        "ir_datasets_id": "msmarco-passage (dev/small, trec-dl-2019/judged, trec-dl-2020/judged)",
        "note": (
            "Two channels (the canonical BM25 + dense hybrid pair). Warmed on "
            f"{len(warm_qids)} dev queries; dl19/dl20 are fused from the same "
            "dev-warmed state snapshot, a cross-query-set transfer over the "
            "shared corpus."
        ),
        "channel_keys": list(MSMARCO_KEYS),
        "k": k,
        "seed": SEED,
        "coupling_refreshes": refreshes,
        "warm_queries": len(warm_qids),
        "corpus_docs": len(doc_ids),
        "eval_sets": results_sets,
    }
