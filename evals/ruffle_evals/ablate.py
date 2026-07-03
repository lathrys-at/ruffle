"""The natural-values discrimination profile: ablation and validation.

The configuration search's winner rounds to a handful of natural settings, all
of them statistics-hardening rather than amplitude-raising. This module
isolates the discrimination knobs from coupling entirely (coupling stays off,
as shipped) and answers two questions: which of the rounded changes carry the
gain (per-knob ablation on the reversed tuning splits), and what the profile is
worth on the benchmark's standard direction (one validation pass, ir_measures
metrics, paired tests against plain RRF and the default warm condition).
"""

from __future__ import annotations

import json

import ruffle

from ruffle_evals import RESULTS_DIR
from ruffle_evals.evaluate import evaluate, paired_p
from ruffle_evals.fusion import rrf, ruffle_warm
from ruffle_evals.tune import (
    Bundle,
    _evaluate_candidate,
    _load_bundles,
    _round6,
)

__all__ = ["NATURAL", "main"]

# The search winner's discrimination knobs, rounded to natural values. g_slope,
# g_upper_bound, and min_count_for_z revert to their defaults: the search left
# the slope at 1.0, the upper bound never binds at observed weights, and the
# z-count gate moved within noise.
NATURAL: dict[str, object] = {
    "top_eps": 0.10,
    "top_m": 5,
    "min_distinct_values": 12,
    "denom_floor_frac": 0.75,
    "winsor_z": 2.5,
    "shrink_pool_size": 80,
    "g_floor": 0.20,
}


def _candidate(discrimination: dict) -> dict:
    return {"discrimination": dict(discrimination), "coupling": {}}


def main() -> int:
    bundles = _load_bundles()
    defaults = ruffle.DiscriminationConfig()

    variants: dict[str, dict] = {
        "default": _candidate({}),
        "natural": _candidate(NATURAL),
    }
    for knob in NATURAL:
        without = {k: v for k, v in NATURAL.items() if k != knob}
        variants[f"natural-without-{knob}"] = _candidate(without)
        variants[f"default-plus-{knob}"] = _candidate({knob: NATURAL[knob]})

    print(f"[ablate] {len(variants)} variants on the reversed splits", flush=True)
    ablation = {}
    for label, candidate in variants.items():
        result = _evaluate_candidate(candidate, bundles)
        ablation[label] = {
            "objective": result["objective"],
            "deltas": result["deltas"],
            "floor": result["floor"],
        }
        print(f"[ablate] {label}: {result['objective']:.5f}", flush=True)

    print("[ablate] validating natural profile, standard direction", flush=True)
    validation = _validate_natural(bundles)

    summary = {
        "natural_profile": {
            "discrimination": NATURAL,
            "coupling": "disabled (as shipped)",
            "reverted_to_default": {
                "g_slope": defaults.g_slope,
                "g_upper_bound": defaults.g_upper_bound,
                "min_count_for_z": defaults.min_count_for_z,
            },
        },
        "ablation_reversed_splits": ablation,
        "validation_std_direction": validation,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "tuning-natural.json"
    path.write_text(json.dumps(_round6(summary), indent=2, sort_keys=True) + "\n")
    print(f"[ablate] wrote {path}", flush=True)
    return 0


def _validate_natural(bundles: list[Bundle]) -> dict:
    """The standard-direction comparison: plain RRF, default warm, and the
    natural profile, macro-aggregated per collection group like the benchmark
    tables. The paired test against the default warm condition asks the
    defaults-change question directly."""
    config = ruffle.FuseConfig(discrimination=ruffle.DiscriminationConfig(**NATURAL))
    by_group: dict[str, list[dict]] = {}
    for bundle in bundles:
        baseline = rrf(bundle.runs, bundle.eval_std, keys=bundle.keys)
        base_metrics, base_pq = evaluate(bundle.qrels, baseline.rankings)
        warm_default = ruffle_warm(
            bundle.runs, bundle.warm_std, bundle.eval_std, configs=bundle.configs
        )
        default_metrics, default_pq = evaluate(bundle.qrels, warm_default.rankings)
        warm_natural = ruffle_warm(
            bundle.runs, bundle.warm_std, bundle.eval_std, configs=bundle.configs, config=config
        )
        natural_metrics, natural_pq = evaluate(bundle.qrels, warm_natural.rankings)
        by_group.setdefault(bundle.group, []).append(
            {
                "rrf": {"metrics": base_metrics},
                "ruffle-warm": {
                    "metrics": default_metrics,
                    "p_vs_rrf": paired_p(base_pq, default_pq),
                },
                "ruffle-warm-natural": {
                    "metrics": natural_metrics,
                    "p_vs_rrf": paired_p(base_pq, natural_pq),
                    "p_vs_warm_default": paired_p(default_pq, natural_pq),
                    "mean_weights": warm_natural.mean_weights(bundle.keys),
                },
            }
        )
    out = {}
    for group, entries in by_group.items():
        out[group] = {}
        for label in entries[0]:
            aggregated = {
                "metrics": {
                    m: sum(e[label]["metrics"][m] for e in entries) / len(entries)
                    for m in entries[0][label]["metrics"]
                }
            }
            if len(entries) == 1:
                for extra in ("p_vs_rrf", "p_vs_warm_default", "mean_weights"):
                    if extra in entries[0][label]:
                        aggregated[extra] = entries[0][label][extra]
            out[group][label] = aggregated
    return out


if __name__ == "__main__":
    import sys

    sys.exit(main())
