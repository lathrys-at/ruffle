"""The harness entry point: ``python -m ruffle_evals [dataset ...]``.

Each named collection is loaded, its channel runs are computed or read from the
cache, and three protocols run on the same held-out evaluation split: the main
condition comparison, the degraded-channel experiment, and the warmup learning
curve. The numbers land in ``results/<dataset>.json``, ``<dataset>-degraded.json``,
and ``<dataset>-curve.json``; after every run the summary in
``results/RESULTS.md`` is regenerated from all result files present.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys

import ruffle

from ruffle_evals import RESULTS_DIR, SEED
from ruffle_evals.channels import CHANNEL_KEYS, Channels
from ruffle_evals.datasets import DATASETS, DEFAULT_DATASETS, load
from ruffle_evals.experiments import DEGRADED_MODES, degraded, learning_curve
from ruffle_evals.fusion import split_queries
from ruffle_evals.heavy import run_cqadupstack, run_msmarco
from ruffle_evals.protocol import main_conditions

__all__ = ["main"]

# The heavy collections run through dedicated runners (composite aggregation,
# multi-evaluation-set warm transfer) rather than the generic per-dataset path,
# and skip the degraded/curve experiments.
_HEAVY = {"cqadupstack": run_cqadupstack, "msmarco": run_msmarco}

_SUMMARY_ORDER = (*DATASETS, *_HEAVY)


def _rounded(value):
    """Floats rounded to six decimals for the committed result files; nested
    containers rounded recursively."""
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rounded(v) for v in value]
    return value


def _environment() -> dict:
    # The four searched discrimination defaults are recorded alongside the
    # version: results must say which defaults produced them even when the
    # version string alone does not.
    d = ruffle.DiscriminationConfig()
    return {
        "ruffle_version": ruffle.__version__,
        "python": platform.python_version(),
        "engine_defaults": {
            "top_eps": d.top_eps,
            "top_m": d.top_m,
            "winsor_z": d.winsor_z,
            "denom_floor_frac": d.denom_floor_frac,
        },
    }


def _run_dataset(name: str, k: int, warm_frac: float, refreshes: int) -> None:
    if name in _HEAVY:
        result = _HEAVY[name](k, refreshes)
        _write(name, "", {**result, **_environment()})
        return

    print(f"[{name}] loading collection", flush=True)
    dataset = load(name)
    print(
        f"[{name}] {len(dataset.docs)} docs, {len(dataset.queries)} queries, "
        f"{len(dataset.qrels)} judged",
        flush=True,
    )
    channels = Channels.for_dataset(dataset)
    runs = channels.runs(k)
    warm_qids, eval_qids = split_queries(dataset, warm_frac)
    print(f"[{name}] split: {len(warm_qids)} warmup, {len(eval_qids)} evaluation", flush=True)

    envelope = {
        "dataset": name,
        "ir_datasets_id": DATASETS[name],
        "k": k,
        "seed": SEED,
        "warm_queries": len(warm_qids),
        "eval_queries": len(eval_qids),
        **_environment(),
    }

    conditions, _ = main_conditions(runs, channels, dataset.qrels, warm_qids, eval_qids, refreshes)
    _write(name, "", {**envelope, "coupling_refreshes": refreshes, "conditions": conditions})

    print(f"[{name}] degraded-channel experiment", flush=True)
    _write(name, "-degraded", {**envelope, **degraded(runs, dataset.qrels, warm_qids, eval_qids)})

    print(f"[{name}] warmup learning curve", flush=True)
    _write(name, "-curve", {**envelope, **learning_curve(runs, dataset.qrels, warm_qids, eval_qids)})


def _write(name: str, suffix: str, result: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}{suffix}.json"
    path.write_text(json.dumps(_rounded(result), indent=2, sort_keys=True) + "\n")
    print(f"[{name}] wrote {path}", flush=True)


def _fmt(value: float | None, digits: int = 4) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def _weights_cell(mean_weights: dict[str, float] | None, keys) -> str:
    if mean_weights is None:
        return ""
    return " / ".join(f"{mean_weights[key]:.3f}" for key in keys)


def _delta_cell(profile: dict | None) -> str:
    if profile is None:
        return ""
    return f"{profile['win'] * 100:.0f}% / {profile['loss'] * 100:.0f}%"


def _conditions_table(conditions: dict, keys: list[str]) -> list[str]:
    lines = [
        f"| condition | nDCG@10 | R@100 | MRR@10 | p vs RRF | win/loss vs RRF | mean weights ({' / '.join(keys)}) |",
        "|---|---|---|---|---|---|---|",
    ]
    order = [
        *keys,
        "rrf",
        "borda",
        "isr",
        "combsum",
        "combmnz",
        "ruffle-cold",
        "ruffle-warm",
        "ruffle-warm-coupled",
        "ruffle-warm-aggressive",
        "rrf-oracle",
    ]
    for condition in order:
        entry = conditions.get(condition)
        if entry is None:
            continue
        metrics = entry["metrics"]
        lines.append(
            f"| {condition} | {_fmt(metrics.get('nDCG@10'))} | {_fmt(metrics.get('R@100'))} "
            f"| {_fmt(metrics.get('RR@10'))} | {_fmt(entry.get('p_vs_rrf'), 3)} "
            f"| {_delta_cell(entry.get('delta_vs_rrf'))} "
            f"| {_weights_cell(entry.get('mean_weights'), keys)} |"
        )
    lines.append("")
    return lines


def _main_table(result: dict) -> list[str]:
    keys = list(result.get("channel_keys", CHANNEL_KEYS))
    lines = [
        f"### {result['dataset']}",
        "",
        f"{result['eval_queries']} evaluation queries "
        f"({result['warm_queries']} warmup), top-{result['k']} per channel, "
        f"ruffle {result['ruffle_version']}.",
        "",
    ]
    if result.get("note"):
        lines.extend([result["note"], ""])
    lines.extend(_conditions_table(result["conditions"], keys))
    return lines


def _msmarco_tables(result: dict) -> list[str]:
    keys = list(result.get("channel_keys", CHANNEL_KEYS))
    lines = [
        f"### {result['dataset']}",
        "",
        f"{result['corpus_docs']} passages, top-{result['k']} per channel, "
        f"ruffle {result['ruffle_version']}.",
        "",
    ]
    if result.get("note"):
        lines.extend([result["note"], ""])
    for name, block in result["eval_sets"].items():
        lines.extend([f"#### {name} ({block['eval_queries']} evaluation queries)", ""])
        lines.extend(_conditions_table(block["conditions"], keys))
    return lines


def _degraded_table(result: dict) -> list[str]:
    lines = [
        "#### Degraded fourth channel",
        "",
        "A broken channel derived from the BM25 run joins the three healthy ones.",
        "`wrong-query` serves another query's results (healthy-looking scores,",
        "irrelevant content); `flaky` serves the tail of its own results (ranks",
        "51-100) on a seeded half of the queries. The p column compares against",
        "four-channel RRF, so it reads what each fusion recovers of the damage.",
        "",
        "| mode | condition | nDCG@10 | R@100 | MRR@10 | p vs RRF+broken | broken weight | mean conflict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode in DEGRADED_MODES:
        entry = result["modes"].get(mode)
        if entry is None:
            continue
        for condition in ("rrf-clean", "rrf", "ruffle-warm", "ruffle-warm-aggressive"):
            data = entry["conditions"].get(condition)
            if data is None:
                continue
            metrics = data["metrics"]
            weights = data.get("mean_weights")
            broken = "" if weights is None else f"{weights['broken']:.3f}"
            lines.append(
                f"| {mode} | {condition} | {_fmt(metrics.get('nDCG@10'))} "
                f"| {_fmt(metrics.get('R@100'))} | {_fmt(metrics.get('RR@10'))} "
                f"| {_fmt(data.get('p_vs_rrf_broken'), 3)} | {broken} "
                f"| {_fmt(data.get('mean_conflict'), 3)} |"
            )
    flaky = result["modes"].get("flaky", {})
    if "broken_weight_on_failed" in flaky:
        lines.extend(
            [
                "",
                f"In the flaky mode the broken channel's mean weight on the "
                f"{flaky['failed_eval_queries']} failed evaluation queries is "
                f"{flaky['broken_weight_on_failed']:.3f}, against "
                f"{flaky['broken_weight_on_healthy']:.3f} on the healthy ones.",
            ]
        )
    if "aggressive_broken_weight_on_failed" in flaky:
        lines.append(
            f"Under the aggressive profile that split widens to "
            f"{flaky['aggressive_broken_weight_on_failed']:.3f} against "
            f"{flaky['aggressive_broken_weight_on_healthy']:.3f}."
        )
    lines.append("")
    return lines


def _curve_table(result: dict) -> list[str]:
    base = result["rrf"]["metrics"]
    lines = [
        "#### Warmup learning curve",
        "",
        "Ruffle warm (default configuration) on the fixed evaluation split, warmed",
        "on increasing prefixes of the warmup split. Size zero is online-from-cold:",
        "the fuser adapts across the evaluation queries themselves. The RRF floor",
        f"on this split is nDCG@10 {_fmt(base.get('nDCG@10'))}, "
        f"R@100 {_fmt(base.get('R@100'))}.",
        "",
        "| warmup queries | nDCG@10 | R@100 | p vs RRF | mean weights (bm25 / tfidf / dense) |",
        "|---|---|---|---|---|",
    ]
    for point in result["points"]:
        metrics = point["metrics"]
        lines.append(
            f"| {point['warmup']} | {_fmt(metrics.get('nDCG@10'))} "
            f"| {_fmt(metrics.get('R@100'))} | {_fmt(point.get('p_vs_rrf'), 3)} "
            f"| {_weights_cell(point.get('mean_weights'), CHANNEL_KEYS)} |"
        )
    lines.append("")
    return lines


def _regenerate_summary() -> None:
    lines = [
        "# Results",
        "",
        "Generated by `python -m ruffle_evals`; the protocol is described in the",
        "harness [README](../README.md). Single-channel rows cover the same",
        "evaluation queries the fused conditions are scored on. The p column is a",
        "two-sided paired t-test on per-query nDCG@10 against the plain-RRF",
        "baseline; a blank cell means the rankings were identical to the baseline.",
        "",
    ]
    wrote_any = False
    for name in _SUMMARY_ORDER:
        main_path = RESULTS_DIR / f"{name}.json"
        if not main_path.exists():
            continue
        wrote_any = True
        result = json.loads(main_path.read_text())
        lines.extend(_msmarco_tables(result) if "eval_sets" in result else _main_table(result))
        degraded_path = RESULTS_DIR / f"{name}-degraded.json"
        if degraded_path.exists():
            lines.extend(_degraded_table(json.loads(degraded_path.read_text())))
        curve_path = RESULTS_DIR / f"{name}-curve.json"
        if curve_path.exists():
            lines.extend(_curve_table(json.loads(curve_path.read_text())))
    if not wrote_any:
        return
    (RESULTS_DIR / "RESULTS.md").write_text("\n".join(lines))
    print(f"regenerated {RESULTS_DIR / 'RESULTS.md'}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ruffle_evals",
        description="BEIR evaluation of Ruffle against plain RRF and single channels.",
    )
    # No argparse `choices`: with nargs="*" it validates the default list itself
    # as one candidate value and rejects every bare invocation.
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help=f"collections to run (default: {', '.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument("--k", type=int, default=100, help="run depth per channel")
    parser.add_argument(
        "--warm-frac", type=float, default=0.5, help="fraction of queries used for warmup"
    )
    parser.add_argument(
        "--refreshes", type=int, default=10, help="anchor refreshes in the coupled condition"
    )
    args = parser.parse_args(argv)
    known = {**DATASETS, **_HEAVY}
    unknown = [name for name in args.datasets if name not in known]
    if unknown:
        parser.error(
            f"unknown collection(s) {', '.join(unknown)}; available: {', '.join(known)}"
        )

    for name in args.datasets:
        _run_dataset(name, args.k, args.warm_frac, args.refreshes)
    _regenerate_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
