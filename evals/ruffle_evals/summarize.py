"""The headline summary: one nDCG@10 table across collections and fusion
methods, the Ruffle-versus-RRF delta profiles, and a delta chart.

``results/SUMMARY.md`` and ``results/summary-ndcg.png`` are regenerated from
whatever result files are present, so the summary grows as collections finish;
``results/RESULTS.md`` keeps the full per-collection tables. Run directly with
``python -m ruffle_evals.summarize``.
"""

from __future__ import annotations

import json
import sys

from ruffle_evals import RESULTS_DIR

__all__ = ["write_summary"]

# Columns, in presentation order. msmarco expands into its three query sets.
_COLLECTIONS = ("scifact", "nfcorpus", "fiqa", "quora", "cqadupstack", "msmarco")

_SINGLES = ("bm25", "tfidf", "dense")

_METHODS = (
    ("rrf", "RRF"),
    ("borda", "Borda"),
    ("isr", "ISR"),
    ("combsum", "CombSUM"),
    ("combmnz", "CombMNZ"),
    ("ruffle-warm", "Ruffle warm"),
    ("ruffle-warm-coupled", "Ruffle warm + coupling"),
    ("rrf-fitted", "RRF + fitted weights"),
    ("ruffle-warm-fitted", "Ruffle warm + fitted weights"),
    ("rrf-oracle", "Oracle-weighted RRF"),
)

_RUFFLE_ROWS = ("ruffle-warm", "ruffle-warm-coupled", "ruffle-warm-fitted")

# The chart shows every non-baseline method as a delta against RRF.
_CHART_METHODS = _METHODS[1:]
_CHART_COLORS = {
    "borda": "#b0b0b0",
    "isr": "#909090",
    "combsum": "#c9b8a8",
    "combmnz": "#a89078",
    "ruffle-warm": "#2f6db3",
    "ruffle-warm-coupled": "#1f4e87",
    "rrf-fitted": "#a58cc4",
    "ruffle-warm-fitted": "#6b4fa1",
    "rrf-oracle": "#d4a017",
}

_PREAMBLE = """\
# Benchmark summary

Ruffle fuses ranked lists from several retrieval channels and adaptively
reweights them per query, without relevance labels and without score
calibration. This page compares it on standard BEIR collections against plain
reciprocal-rank fusion (RRF), the other classic fusion rules, and each
collection's own label-fitted ceiling. The full per-collection tables,
including recall, MRR, single-channel rows, and the targeted experiments, are
in [RESULTS.md](RESULTS.md); the protocol is in the harness
[README](../README.md).

Channels are BM25, character-ngram TF-IDF, and dense retrieval over
`Alibaba-NLP/gte-modernbert-base` embeddings (MS MARCO uses the canonical
BM25 + dense pair). Half of each collection's queries warm Ruffle's baselines
unsupervised; every method is scored on the other half. `Oracle-weighted RRF`
is RRF with fixed per-channel weights grid-searched against the evaluation
judgments themselves. The labels choose its weights, so it is not a
competitor: it is the ceiling for any fixed per-channel weighting, and the
table reads as a bracket from the RRF floor to that ceiling.

Two conditions sit between the floor and the ceiling by construction.
`RRF + fitted weights` runs the same grid search on a small graded subsample
of the warmup split (5% of its judged queries, at least 10 and at most 100),
the deployable version of the oracle: what an operator gets by grading a few
dozen queries once. `Ruffle warm + fitted weights` declares those same
weights as `base_weight` on the channel registrations, so the static tilt
from labels and the label-free per-query adaptation compose. The fit is
repeated over three seeded draws; the tables show the first draw, and the
result files record every draw's weights and score.
"""

_READING = """\
## Reading the results

Two regimes appear in the table. Where the channels are comparably strong
(scifact, nfcorpus), fusion beats every single channel, and warm Ruffle sits
between the RRF floor and the oracle ceiling. The headroom in this regime is
small: even the label-fitted ceiling is only a point or two of nDCG above
plain RRF, so no reweighting scheme, labeled or not, can move the aggregate
much. Where the dense channel dominates (fiqa sharply; quora and cqadupstack
more moderately), dense alone beats every label-free fusion of it with the
weaker lexical channels, and the oracle converges on the dominant channel.
Ruffle narrows the gap to the oracle but cannot close it: knowing that one
channel is globally better than another requires labels, which is exactly the
information the engine's contract excludes. The fitted rows are the
operational answer: with a few dozen graded queries the fit recovers most of
the oracle's gain, and on the larger budgets nearly all of it.

The per-draw records in the result files show what composing adds: when a
draw fits well, the composed condition tracks the static fit closely; when a
draw fits badly, the adaptation pulls the result back toward plain warm
Ruffle instead of following the bad fit down. The exception is a fitted
weight of exactly zero, which silences a channel outright, so no per-query
evidence can revive it; the nfcorpus fitted row, where a 10-query fit chose
a single channel and landed below the RRF floor, is that failure mode in the
table. An operator declaring weights from a small fit should floor them at a
small positive value rather than zero, unless exclusion is the intent.

Across the label-free rules, no method wins everywhere. ISR's steeper rank
discount and the score-based CombSUM profit in the dominant-channel regime,
where top-heavy discounting and raw score magnitudes both lean toward the
strong channel, and several of their columns beat Ruffle there; on balanced
nfcorpus both fall back to the RRF baseline or below it. Ruffle is the one
method in the table that improves on RRF in every column. That consistency,
rather than the largest single number, is the designed behavior: the engine
is RRF plus per-query evidence, tilting only when a channel's own statistics
support it, so its floor is the baseline rather than the worst case of a
fixed convention. The delta profiles say the same thing per query: wins
outnumber losses in every column and the 5th-percentile delta stays near
zero, so the mean gains are not bought with per-query damage.

The clean-benchmark setting is also the regime where adaptive weighting has
the least to offer: healthy channels reading the same text rise and fall
together, so there is little per-query variation in channel usefulness for
the weighting to work with. Deployments with a more heterogeneous channel mix
present more of that variation; measuring Ruffle under such a mix is future
work.
"""


def _load(name: str) -> dict | None:
    path = RESULTS_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _columns() -> list[tuple[str, dict]]:
    """(column label, conditions dict) per available result set."""
    columns: list[tuple[str, dict]] = []
    for name in _COLLECTIONS:
        result = _load(name)
        if result is None:
            continue
        if "eval_sets" in result:
            for set_name, entry in result["eval_sets"].items():
                columns.append((f"msmarco-{set_name}", entry["conditions"]))
        else:
            columns.append((name, result["conditions"]))
    return columns


def _fmt_p(p: float | None) -> str:
    if p is None:
        return ""
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _best_single(conditions: dict) -> str:
    present = [(k, conditions[k]["metrics"]["nDCG@10"]) for k in _SINGLES if k in conditions]
    if not present:
        return ""
    key, value = max(present, key=lambda kv: kv[1])
    return f"{value:.4f} ({key})"


def _ndcg_table(columns: list[tuple[str, dict]]) -> list[str]:
    lines = [
        "## nDCG@10",
        "",
        "| method | " + " | ".join(label for label, _ in columns) + " |",
        "|---|" + "---|" * len(columns),
        "| best single channel | "
        + " | ".join(_best_single(conds) for _, conds in columns)
        + " |",
    ]
    for key, label in _METHODS:
        cells = []
        for _, conds in columns:
            entry = conds.get(key)
            cells.append("" if entry is None else f"{entry['metrics']['nDCG@10']:.4f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _delta_table(columns: list[tuple[str, dict]]) -> list[str]:
    lines = [
        "## Ruffle against RRF, per query",
        "",
        "Per-query nDCG@10 deltas against the RRF baseline on the same queries:",
        "the share of queries won and lost, the two-sided paired-t p-value, and",
        "the 5th-percentile delta (how much the worst tail of queries loses).",
        "",
        "| condition | collection | delta | p | win / loss | p5 |",
        "|---|---|---|---|---|---|",
    ]
    for key, label in _METHODS:
        if key not in _RUFFLE_ROWS:
            continue
        for column, conds in columns:
            entry = conds.get(key)
            if entry is None or entry.get("delta_vs_rrf") is None:
                continue
            profile = entry["delta_vs_rrf"]
            lines.append(
                f"| {label} | {column} | {profile['mean']:+.4f} "
                f"| {_fmt_p(entry.get('p_vs_rrf'))} "
                f"| {profile['win'] * 100:.0f}% / {profile['loss'] * 100:.0f}% "
                f"| {profile['p5']:+.4f} |"
            )
    lines.append("")
    return lines


def _chart(columns: list[tuple[str, dict]]) -> str | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    drawable = [
        (label, conds)
        for label, conds in columns
        if any(key in conds for key, _ in _CHART_METHODS)
    ]
    if not drawable:
        return None
    fig, ax = plt.subplots(figsize=(max(7.0, 1.9 * len(drawable)), 4.2))
    group_width = 0.84
    bar_width = group_width / len(_CHART_METHODS)
    baseline_ndcg = [conds["rrf"]["metrics"]["nDCG@10"] for _, conds in drawable]
    for i, (key, label) in enumerate(_CHART_METHODS):
        offsets = [
            x - group_width / 2 + (i + 0.5) * bar_width for x in range(len(drawable))
        ]
        deltas = [
            conds[key]["metrics"]["nDCG@10"] - base if key in conds else 0.0
            for (_, conds), base in zip(drawable, baseline_ndcg)
        ]
        ax.bar(
            offsets,
            deltas,
            bar_width * 0.92,
            label=label,
            color=_CHART_COLORS[key],
            edgecolor="white",
            linewidth=0.4,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(drawable)))
    ax.set_xticklabels([label for label, _ in drawable])
    ax.set_ylabel("nDCG@10 delta vs RRF")
    ax.set_title("Fusion methods against the RRF baseline (0 = RRF)")
    ax.legend(fontsize=8, ncol=4, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = RESULTS_DIR / "summary-ndcg.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out.name


def write_summary() -> bool:
    """Regenerates SUMMARY.md and the chart; returns False if no results exist."""
    columns = _columns()
    if not columns:
        return False
    lines = [_PREAMBLE]
    chart = _chart(columns)
    if chart is not None:
        lines.append(f"![nDCG@10 delta over RRF by collection]({chart})")
        lines.append("")
    lines.extend(_ndcg_table(columns))
    lines.extend(_delta_table(columns))
    lines.append(_READING)
    if _load("msmarco") is None:
        lines.append(
            "MS MARCO passage (8.8M documents, dev plus TREC-DL 2019/2020) is "
            "rerunning under the current dense model and will be added when it "
            "completes.\n"
        )
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines))
    print(f"regenerated {RESULTS_DIR / 'SUMMARY.md'}", flush=True)
    return True


if __name__ == "__main__":
    sys.exit(0 if write_summary() else 1)
