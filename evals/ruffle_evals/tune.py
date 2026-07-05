"""Configuration search: better shipped defaults for the warm path.

The search tunes the discrimination and coupling knobs against the cached
channel runs. Hygiene: candidates are scored on the REVERSED splits (the fuser
warms on the standard evaluation half and is scored on the standard warmup
half's judgments), so the benchmark's reported evaluation halves stay untouched
until one final validation of the winner. rrf_eta stays at 60: retuning the
rank-decay constant would change the RRF family baseline itself rather than
Ruffle's adaptive machinery.

The objective is the macro-mean nDCG@10 gain over plain RRF across collection
groups (the 12 cqadupstack subforums form one group), under a hard constraint
that no group falls below the RRF floor.

Anchors are precomputed once per collection and direction and cached: an anchor
depends on the channels and the seeded draw, never on the candidate
configuration, so one payload serves every candidate.
"""

from __future__ import annotations

import json
import math
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import ruffle

from ruffle_evals import CACHE_DIR, RESULTS_DIR, SEED
from ruffle_evals.baselines import _ndcg10
from ruffle_evals.channels import CHANNEL_KEYS, DENSE_SLUG, Channels, run_filename
from ruffle_evals.datasets import load, load_id
from ruffle_evals.evaluate import evaluate, paired_p
from ruffle_evals.fusion import _ANCHOR_CANDIDATES, channel_configs, rrf, ruffle_warm
from ruffle_evals.heavy import MSMARCO_KEYS, SUBFORUMS, _load_msmarco_queryset

__all__ = ["main"]

N_RANDOM = 400
N_REFINE_ROUNDS = 3
N_PERTURB = 40
REFRESHES = 10
K = 100
FLOOR_TOLERANCE = -0.001

# knob -> (section, kind, spec). Floats sample uniformly in their bounds
# (g_slope log-uniformly); choice knobs sample from their list. Evidence gates
# are searched too: "better tuned" may legitimately mean more or less patient.
_SPACE: dict[str, tuple[str, str, object]] = {
    "top_eps": ("discrimination", "float", (0.02, 0.15)),
    "top_m": ("discrimination", "choice", [5, 8, 10, 15, 20]),
    "min_distinct_values": ("discrimination", "choice", [5, 8, 12]),
    "denom_floor_frac": ("discrimination", "float", (0.2, 0.8)),
    "winsor_z": ("discrimination", "float", (2.5, 6.0)),
    "min_count_for_z": ("discrimination", "float", (3.0, 12.0)),
    "shrink_pool_size": ("discrimination", "choice", [10, 20, 40, 80]),
    "g_upper_bound": ("discrimination", "float", (2.0, 8.0)),
    "g_floor": ("discrimination", "float", (0.05, 0.5)),
    "g_slope": ("discrimination", "logfloat", (0.5, 4.0)),
    "enabled": ("coupling", "bool", None),
    "discount_cap": ("coupling", "float", (0.2, 0.95)),
    "shrink_to_identity": ("coupling", "float", (0.1, 0.7)),
    "min_overlap": ("coupling", "choice", [10, 20, 30, 50]),
    "min_reliability": ("coupling", "float", (5.0, 50.0)),
    "min_refreshes": ("coupling", "float", (1.0, 8.0)),
    "stratum_stability_max_var": ("coupling", "float", (0.1, 0.5)),
}


def _sample(rng: random.Random) -> dict:
    out: dict = {"discrimination": {}, "coupling": {}}
    for knob, (section, kind, spec) in _SPACE.items():
        if kind == "float":
            lo, hi = spec
            out[section][knob] = rng.uniform(lo, hi)
        elif kind == "logfloat":
            lo, hi = spec
            out[section][knob] = math.exp(rng.uniform(math.log(lo), math.log(hi)))
        elif kind == "choice":
            out[section][knob] = rng.choice(spec)
        else:
            out[section][knob] = rng.random() < 0.6
    return out


def _perturb(base: dict, rng: random.Random) -> dict:
    out = {"discrimination": dict(base["discrimination"]), "coupling": dict(base["coupling"])}
    for knob, (section, kind, spec) in _SPACE.items():
        if rng.random() > 0.35:
            continue
        value = out[section][knob]
        if kind in ("float", "logfloat"):
            lo, hi = spec
            out[section][knob] = min(hi, max(lo, value * math.exp(rng.gauss(0.0, 0.25))))
        elif kind == "choice":
            i = spec.index(value)
            out[section][knob] = spec[min(len(spec) - 1, max(0, i + rng.choice((-1, 1))))]
        else:
            out[section][knob] = value if rng.random() > 0.15 else not value
    return out


def _to_config(candidate: dict) -> ruffle.FuseConfig:
    return ruffle.FuseConfig(
        discrimination=ruffle.DiscriminationConfig(**candidate["discrimination"]),
        coupling=ruffle.CouplingConfig(**candidate["coupling"]),
    )


def _full_defaults(**coupling_overrides) -> dict:
    """A candidate dict carrying every searched knob at its engine default, so
    reference configurations survive the perturbation step."""
    defaults = ruffle.FuseConfig()
    candidate: dict = {"discrimination": {}, "coupling": {}}
    for knob, (section, _, _) in _SPACE.items():
        source = defaults.discrimination if section == "discrimination" else defaults.coupling
        candidate[section][knob] = getattr(source, knob)
    candidate["coupling"].update(coupling_overrides)
    return candidate


@dataclass
class Bundle:
    name: str
    group: str
    keys: tuple
    configs: list
    runs: dict
    qrels: dict
    warm_std: list
    eval_std: list
    inputs: dict = field(default_factory=dict)
    anchors_std: dict = field(default_factory=dict)
    anchors_rev: dict = field(default_factory=dict)
    rrf_rev_mean: float = 0.0

    # Tuning direction: warm on the standard evaluation half, score on the
    # standard warmup half's judgments.
    @property
    def warm_rev(self):
        return self.eval_std

    @property
    def eval_rev(self):
        return self.warm_std


def _prebuild_inputs(bundle: Bundle) -> None:
    for qid in (*bundle.warm_std, *bundle.eval_std):
        bundle.inputs[qid] = [
            ruffle.ChannelInput.scored(config, bundle.runs[config.id.key].get(qid, []))
            for config in bundle.configs
        ]


def _anchor_payloads(channels: Channels, warm_qids: list, keys: tuple) -> dict:
    """Raw anchor payloads with the same seeded draw sequence as
    fusion.build_anchor_data, serializable for the cache."""
    anchor_qids = warm_qids[:: max(1, len(warm_qids) // REFRESHES)][:REFRESHES]
    rng = random.Random(SEED)
    payloads = {}
    for qid in warm_qids:
        if qid not in anchor_qids:
            continue
        candidates = rng.sample(
            list(channels.doc_ids), min(_ANCHOR_CANDIDATES, len(channels.doc_ids))
        )
        payloads[qid] = {
            "candidates": candidates,
            "scores": {key: channels.score_candidates(qid, candidates, key) for key in keys},
        }
    return payloads


def _payload_to_anchors(payloads: dict, configs: list) -> dict:
    anchors = {}
    for qid, payload in payloads.items():
        index = {doc_id: i for i, doc_id in enumerate(payload["candidates"])}
        scores = payload["scores"]
        anchors[qid] = ruffle.Anchor.build(
            payload["candidates"], configs, lambda d, k: scores[k][index[d]]
        )
    return anchors


def _load_anchors(bundle: Bundle, make_channels) -> None:
    """Anchor payloads for both directions, from the cache or one live-channels
    pass."""
    path = CACHE_DIR / "anchors" / f"{bundle.name}-{DENSE_SLUG}-r{REFRESHES}.json"
    if path.exists():
        raw = json.loads(path.read_text())
    else:
        channels = make_channels()
        raw = {
            "std": _anchor_payloads(channels, bundle.warm_std, bundle.keys),
            "rev": _anchor_payloads(channels, bundle.warm_rev, bundle.keys),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw))
        del channels
    bundle.anchors_std = _payload_to_anchors(raw["std"], bundle.configs)
    bundle.anchors_rev = _payload_to_anchors(raw["rev"], bundle.configs)


def _split(qids: list) -> tuple[list, list]:
    ordered = sorted(qids)
    random.Random(SEED).shuffle(ordered)
    cut = len(ordered) // 2
    return ordered[:cut], ordered[cut:]


def _cached_runs(name: str, keys: tuple, k: int) -> dict | None:
    """Runs straight from the disk cache, without building channel models;
    ``None`` when any channel's cache is missing."""
    out = {}
    for key in keys:
        path = CACHE_DIR / "runs" / name / run_filename(key, k)
        if not path.exists():
            return None
        out[key] = {
            qid: [(d, float(s)) for d, s in items]
            for qid, items in json.loads(path.read_text()).items()
        }
    return out


def _standard_bundle(name: str, group: str, dataset) -> Bundle:
    """One BEIR bundle, building live channel models only when the runs or
    anchor caches are missing."""
    warm, ev = _split(list(dataset.queries))
    bundle = Bundle(
        name=name,
        group=group,
        keys=CHANNEL_KEYS,
        configs=channel_configs(),
        runs={},
        qrels=dataset.qrels,
        warm_std=warm,
        eval_std=ev,
    )
    runs = _cached_runs(name, CHANNEL_KEYS, K)
    channels = None
    if runs is None:
        channels = Channels.for_dataset(dataset)
        runs = channels.runs(K)
    bundle.runs = runs
    _prebuild_inputs(bundle)
    _load_anchors(
        bundle, lambda: channels if channels is not None else Channels.for_dataset(dataset)
    )
    return bundle


def _load_bundles() -> list[Bundle]:
    bundles: list[Bundle] = []
    for name in ("scifact", "nfcorpus", "fiqa", "quora"):
        bundles.append(_standard_bundle(name, name, load(name)))
        print(f"[tune] bundle {name} ready", flush=True)
    for sub in SUBFORUMS:
        name = f"cqadupstack-{sub}"
        bundles.append(_standard_bundle(name, "cqadupstack", load_id(f"beir/cqadupstack/{sub}", name)))
    print("[tune] cqadupstack bundles ready", flush=True)

    queries, qrels = _load_msmarco_queryset("msmarco-passage/dev/small", "dev")
    warm, ev = _split(list(queries))
    runs_dir = CACHE_DIR / "runs" / "msmarco"
    runs = {
        key: {
            qid: [(d, float(s)) for d, s in items]
            for qid, items in json.loads((runs_dir / run_filename(key, K)).read_text()).items()
        }
        for key in MSMARCO_KEYS
    }
    bundle = Bundle(
        name="msmarco",
        group="msmarco",
        keys=MSMARCO_KEYS,
        configs=channel_configs(MSMARCO_KEYS),
        runs=runs,
        qrels=qrels,
        warm_std=warm,
        eval_std=ev,
    )
    _prebuild_inputs(bundle)

    def _msmarco_channels() -> Channels:
        import ir_datasets

        corpus = ir_datasets.load("msmarco-passage")
        doc_ids, texts = [], []
        for doc in corpus.docs_iter():
            doc_ids.append(doc.doc_id)
            texts.append(doc.text)
        return Channels("msmarco", doc_ids, texts, queries, keys=MSMARCO_KEYS)

    _load_anchors(bundle, _msmarco_channels)
    bundles.append(bundle)
    print("[tune] msmarco bundle ready", flush=True)

    for b in bundles:
        baseline = rrf(b.runs, b.eval_rev, keys=b.keys)
        values = [
            _ndcg10([d for d, _ in baseline.rankings[qid]], b.qrels.get(qid, {}))
            for qid in b.eval_rev
        ]
        b.rrf_rev_mean = sum(values) / max(len(values), 1)
    return bundles


def _bundle_mean(bundle: Bundle, config: ruffle.FuseConfig, direction: str) -> float:
    warm = bundle.warm_rev if direction == "rev" else bundle.warm_std
    ev = bundle.eval_rev if direction == "rev" else bundle.eval_std
    anchors = (
        (bundle.anchors_rev if direction == "rev" else bundle.anchors_std)
        if config.coupling.enabled
        else {}
    )
    fuser = ruffle.Fuser(bundle.configs, config)
    for qid in warm:
        fuser.fuse(bundle.inputs[qid])
        anchor = anchors.get(qid)
        if anchor is not None:
            fuser.refresh_coupling(anchor)
    total = 0.0
    for qid in ev:
        fused = fuser.fuse(bundle.inputs[qid])
        total += _ndcg10([item.id for item in fused.ranking], bundle.qrels.get(qid, {}))
    return total / max(len(ev), 1)


def _evaluate_candidate(candidate: dict, bundles: list[Bundle]) -> dict:
    try:
        config = _to_config(candidate)
        by_group: dict[str, list[float]] = {}
        for bundle in bundles:
            mean = _bundle_mean(bundle, config, "rev")
            by_group.setdefault(bundle.group, []).append(mean - bundle.rrf_rev_mean)
        deltas = {group: sum(v) / len(v) for group, v in by_group.items()}
        objective = sum(deltas.values()) / len(deltas)
        return {"candidate": candidate, "deltas": deltas, "objective": objective,
                "floor": min(deltas.values())}
    except ruffle.ConfigError as e:
        return {"candidate": candidate, "error": str(e), "objective": None, "floor": None}


def main() -> int:
    bundles = _load_bundles()
    log_path = CACHE_DIR / "tuning-search.jsonl"
    log = log_path.open("a")

    def run_phase(phase: str, candidates: list[dict]) -> list[dict]:
        results = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            for i, result in enumerate(pool.map(lambda c: _evaluate_candidate(c, bundles), candidates)):
                result["phase"] = phase
                results.append(result)
                log.write(json.dumps(result) + "\n")
                log.flush()
                if (i + 1) % 20 == 0:
                    best = max(
                        (r for r in results if r["objective"] is not None),
                        key=lambda r: r["objective"],
                        default=None,
                    )
                    print(
                        f"[tune] {phase}: {i + 1}/{len(candidates)}, best so far "
                        f"{best['objective']:.5f}" if best else f"[tune] {phase}: {i + 1}",
                        flush=True,
                    )
        return results

    defaults = ruffle.FuseConfig()
    reference = [_full_defaults(), _full_defaults(enabled=True)]
    all_results = run_phase("reference", reference)
    print(f"[tune] references: {[r['objective'] for r in all_results]}", flush=True)

    rng = random.Random(SEED)
    all_results += run_phase("random", [_sample(rng) for _ in range(N_RANDOM)])

    def admissible(r):
        return r["objective"] is not None and r["floor"] >= FLOOR_TOLERANCE

    admissible_results = [r for r in all_results if admissible(r)]
    if not admissible_results:
        print("[tune] no candidate cleared the floor constraint; ranking by objective alone", flush=True)
        admissible_results = [r for r in all_results if r["objective"] is not None]
    incumbent = max(admissible_results, key=lambda r: r["objective"])
    for round_index in range(N_REFINE_ROUNDS):
        perturbed = [_perturb(incumbent["candidate"], rng) for _ in range(N_PERTURB)]
        round_results = run_phase(f"refine-{round_index}", perturbed)
        all_results += round_results
        best_round = max(
            (r for r in round_results if admissible(r)),
            key=lambda r: r["objective"],
            default=None,
        )
        if best_round and best_round["objective"] > incumbent["objective"]:
            incumbent = best_round
        print(f"[tune] refine {round_index}: incumbent {incumbent['objective']:.5f}", flush=True)
    log.close()

    winner = incumbent["candidate"]
    print(f"[tune] winner objective {incumbent['objective']:.5f}; validating", flush=True)
    validation = _validate(winner, bundles)

    summary = {
        "protocol": {
            "objective": "macro-mean nDCG@10 delta vs plain RRF on reversed splits",
            "floor_constraint": FLOOR_TOLERANCE,
            "candidates_evaluated": len(all_results),
            "space": {k: [s[0], s[1], s[2]] for k, s in _SPACE.items()},
            "rrf_eta": "fixed at 60",
            "seed": SEED,
        },
        "reference_objectives": {
            "defaults": all_results[0].get("objective"),
            "defaults_coupled": all_results[1].get("objective"),
        },
        "winner": {
            "knobs": winner,
            "objective": incumbent["objective"],
            "deltas_rev": incumbent["deltas"],
            "changed_vs_defaults": _diff_vs_defaults(winner, defaults),
        },
        "validation_std_direction": validation,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tuning.json").write_text(json.dumps(_round6(summary), indent=2, sort_keys=True) + "\n")
    print(f"[tune] wrote {RESULTS_DIR / 'tuning.json'}", flush=True)
    return 0


def _round6(value):
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _round6(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round6(v) for v in value]
    return value


def _diff_vs_defaults(winner: dict, defaults: ruffle.FuseConfig) -> dict:
    changed = {}
    for section, obj in (("discrimination", defaults.discrimination), ("coupling", defaults.coupling)):
        for knob, value in winner[section].items():
            default = getattr(obj, knob)
            if isinstance(value, float) and abs(value - float(default)) > 1e-9 or value != default:
                changed[f"{section}.{knob}"] = {"default": default, "tuned": value}
    return changed


def _validate(winner: dict, bundles: list[Bundle]) -> dict:
    """One pass on the standard direction: the tuned config against plain RRF
    and the default warm conditions, with ir_measures metrics and paired tests,
    macro-aggregated like the benchmark tables."""
    config = _to_config(winner)
    out: dict = {}
    by_group: dict[str, list[dict]] = {}
    for bundle in bundles:
        conditions = {}
        baseline = rrf(bundle.runs, bundle.eval_std, keys=bundle.keys)
        base_metrics, base_pq = evaluate(bundle.qrels, baseline.rankings)
        for label, cfg, coupling in (
            ("ruffle-warm", None, False),
            ("ruffle-warm-coupled", None, True),
            ("ruffle-warm-tuned", config, config.coupling.enabled),
        ):
            outcome = ruffle_warm(
                bundle.runs,
                bundle.warm_std,
                bundle.eval_std,
                configs=bundle.configs,
                coupling=coupling,
                refreshes=REFRESHES,
                config=cfg,
                anchor_data=bundle.anchors_std if coupling else None,
            )
            metrics, pq = evaluate(bundle.qrels, outcome.rankings)
            conditions[label] = {
                "metrics": metrics,
                "p_vs_rrf": paired_p(base_pq, pq),
                "mean_weights": outcome.mean_weights(bundle.keys),
            }
        conditions["rrf"] = {"metrics": base_metrics}
        by_group.setdefault(bundle.group, []).append(conditions)
    for group, entries in by_group.items():
        out[group] = {
            label: {
                "metrics": {
                    m: sum(e[label]["metrics"][m] for e in entries) / len(entries)
                    for m in entries[0][label]["metrics"]
                },
                **(
                    {"p_vs_rrf": entries[0][label]["p_vs_rrf"]}
                    if len(entries) == 1 and "p_vs_rrf" in entries[0][label]
                    else {}
                ),
            }
            for label in entries[0]
        }
    return out


if __name__ == "__main__":
    import sys

    sys.exit(main())
