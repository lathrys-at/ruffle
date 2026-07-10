# Changelog

All notable changes to this project are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ruffle.fit_base_weights` (Python package): offline fitting of
  `ChannelConfig.base_weight` declarations from a small graded sample, the
  labeled complement to the label-free engine. A joint grid search over the
  weight simplex (step 0.1, minimum weight 0.2 per channel by default)
  maximizes linear-gain nDCG@10 at the deployed RRF constant, guarded by a
  cross-fitted split-sample acceptance test whose held-out estimate is
  honest by construction: the fit is returned only when its estimated
  benefit over uniform weights is positive, else uniform is returned with
  the fallback flagged. The floor, not the guard, is the do-no-harm
  mechanism (no fit can silence a channel), and it caps the achievable tilt
  at 4:1 by default. On the evaluation harness (nineteen collections,
  two-fold crossfit, 64-query budget) the fitted weights composed with the
  engine improved mean nDCG@10 on every collection, with the largest gains
  where one channel dominates and a documented per-query loss tail inherent
  to any static tilt. Pure Python, no new dependencies, deterministic;
  golden-pinned against the harness's numpy reference including tied fused
  scores straddling the metric cutoff. Python-only: fitting happens where
  operators hold graded data, and the tuning guide documents the algorithm
  to reimplementation precision for other languages. No engine, state, or
  format change.

- Within-query dispersion gate: `RrfConfig.min_g_dispersion` (Python
  `min_g_dispersion`, TypeScript `minGDispersion`), default `0.45`, validated
  finite and non-negative, `0` disables. The per-query weighting now acts only
  when the channels' level-normalized discrimination reads disperse beyond
  estimation noise, measured as the sample standard deviation of the normalized
  reads across the channels scored on the query. Below the threshold every
  adaptive weight is exactly `1`, and with coupling off and no `base_weight`
  tilt the fused ranking is byte-identical to plain RRF; an enabled redundancy
  discount and declared base weights still apply. Within-noise weight
  differences are coin-flip bets whose measured expected payoff is at best
  zero: on nineteen evaluation collections the gate removes most of the
  per-query loss tail (5th-percentile per-query loss at or near zero on
  eleven of nineteen) and the regressions a sharp rank discount would
  otherwise admit, at a small mean cost on collections where ungated bets were
  profitable. A channel whose weight level baseline has not warmed past
  `min_count_for_z` contributes exact neutral to the dispersion read, so a
  cold system fuses at the RRF floor and warms toward weighting. The default
  is the conservative point of the supported 0.40 to 0.50 band, tuned at two
  and three channels. Configuration only, no state change; `Fused` gains
  `g_dispersion` and `gated` so the firing rate is observable.

- Per-channel `g` level normalization. Each channel's state gains a `level`
  baseline (a `MeanVar` of its raw per-query discrimination weight), and the
  weight the fusion uses is `g` divided by that running mean once it has enough
  observations. The map behind `g` is neutral at the norm but nonlinear, so the
  shape of a channel's score distribution leaks a persistent level into its
  average `g` independent of retrieval quality: on a visual-document benchmark a
  peaky lexical scorer held mean weight 1.09 at nDCG 0.18 while the smooth
  late-interaction channel held 0.94 at nDCG 0.60, and the sum-to-N
  renormalization taxed the better channel for it. Both statistics behind `g`
  are standardized against the channel's own baselines, so the level carries no
  cross-channel information; removing it is another own-normalization, and
  persistent cross-channel preference remains `base_weight`'s job. The baseline
  accumulates raw `g` only (never the normalized value, which would self-cancel),
  merges exactly, and decays in lockstep with the other baselines. State format
  version 2 → 3; a version-2 state is accepted on load and upgraded in place, the
  new baseline starting empty while the carried separation and reference
  baselines stay intact.
- `DiscriminationConfig.g_deviation_keep` (Python `g_deviation_keep`, TypeScript
  `gDeviationKeep`): the fraction of the per-query weight deviation from neutral
  kept after the level normalization, default `0.6`, validated in `[0, 1]`. `1.0`
  uses the normalized deviation as is; `0.0` reduces the weighting to plain RRF,
  so lower is strictly more conservative. The default sits below `1.0` because
  the per-query signal's informativeness varies by corpus and scorer family: on
  six evaluation datasets (four BEIR text, two ViDoRe visual-document) the
  per-dataset optimum ranged from `0` (channel-dominated visual, where the
  engine's unshrunk bets regressed nDCG@10 by 0.020 against plain RRF,
  p = 0.0007) to `1` (text sets, where the bets were profitable), and `0.6`
  neutralizes the regression while keeping the text gains within noise of their
  unshrunk values and improving the 5th-percentile per-query loss tail on every
  dataset. Tuned in-sample at three channels; the value is a configuration knob,
  and deployment guidance lives with the field's documentation.

- `ChannelConfig.base_weight` (Rust `with_base_weight`, Python `base_weight`,
  TypeScript `baseWeight`): an operator-declared static weight multiplier on the
  channel's adaptive per-query weight. The fused weight is `base_weight * g`,
  renormalized over the channels present on the query, with the redundancy discount
  still operating on the adaptive part alone. The engine never learns that one
  channel is globally better than another, since that is cross-channel information
  only relevance labels can establish; the field is where an operator who holds such
  labels declares the tilt, and the per-query adaptation composes on top. Defaults
  to `1.0` (no declaration); `0.0` legally silences a channel's votes while its
  baselines keep updating; non-finite and negative values are refused at
  construction. Configuration, not persisted state: no state format change.

### Changed

- `RrfConfig.rrf_eta` default 60 → 20. The literature constant 60 is
  calibrated on 1000-deep TREC pools; at the pool depths the harness measures
  (up to 100 candidates per channel) it discounts rank 1 against rank 100 by
  only a factor of 2.6, diluting a strong channel's top hits with weak
  channels' mid-list votes. At 20, macro nDCG@10 over nineteen evaluation
  collections rises by about 0.012 with every collection improved or flat and
  Recall@100 unchanged; the plain-RRF optimum sat at 5 to 10 at every measured
  depth (20, 50, and 100 by truncation, and 1000 by native retrieval on three
  collections, which reproduces the shallower response curve with the optimum
  unmoved), so the constant is not depth-linked in the measured range. The
  gain is a fusion-rule constant, orthogonal to the adaptive weighting, and
  the dispersion gate above is what keeps the weighting from giving it back
  at the sharper discount. For pools deeper than 1000 items, or channel mixes
  very unlike the measured ones, 60 remains the literature reference point,
  as the field documentation states. The supported band is 10 to 30.
- Four discrimination defaults are retuned from a configuration search against the
  BEIR evaluation harness (`evals/`): `top_eps` 0.05 → 0.10, `top_m` 10 → 5,
  `winsor_z` 4.0 → 2.5, and `denom_floor_frac` 0.5 → 0.75. All four change how the
  statistics are measured, none change how strongly the weights respond: the top
  slice is wider, the goodness read averages a sharper fixed count, outlier reads
  are clamped earlier, and a near-tied bulk is floored harder. On the harness's six
  collection groups the retuned defaults match or beat the previous ones on every
  group (macro nDCG@10 0.4853 against 0.4838, with the largest gain on MS MARCO
  dev, p = 0.0004), with the same recall-safe floor: stateless fusion with an empty
  prior still reduces exactly to unweighted RRF. Persisted state is unaffected; the
  values are configuration, not format.

## [0.2.0] - 2026-07-03

The first bindings release: the same engine, callable from Python and TypeScript,
with byte-identical persisted state across all three.

### Added

- In-tree Python bindings (`bindings/python`, PyPI package `ruffle`): a fully typed,
  documented pure-Python surface over the compiled engine. Behaviour and persisted
  state are identical to the crate down to the serialized bytes, enforced by a golden
  parity suite (`tests/fixtures/parity/`) generated from the Rust engine and replayed
  by the binding's tests. Wheels build for Linux (x86_64, aarch64), macOS (x86_64,
  arm64), and Windows (x64, arm64) on CPython 3.10 and later, and publish to PyPI
  from the release workflow via Trusted Publishing, version-locked to the crate.
- In-tree TypeScript bindings (`bindings/wasm`, npm package `@lathrys-at/ruffle`): the
  engine compiled to WebAssembly under a hand-written, fully typed TypeScript
  surface, one ESM artifact for Node 20+, browsers, and edge runtimes. WebAssembly's
  exact IEEE-754 semantics and the `libm` transcendentals make rankings and state
  bytes identical to the native crate, enforced by the same parity suite. Publishes
  to npm from the release workflow via Trusted Publishing with provenance,
  version-locked to the crate.

### Changed

- The copula map's sine and the logistic squash's exponential come from the pure-Rust
  `libm` crate rather than the platform libm. The two std functions can differ in the
  last ulp between platform libms, and both values reach persistent state and the fused
  weights; with `libm` identical inputs produce bit-identical state bytes and rankings
  on every target.

## [0.1.1] - 2026-07-02

Documentation release; no code changes.

### Changed

- The docs.rs front page mirrors the README quick start (score newtypes, a declared
  good-score prior, a rank-only channel) and is compile-tested as a doctest.
- API documentation rewritten throughout: descriptive prose on every public item, with
  design-document section references replaced by links to the relevant types and
  methods.
- The README quick start imports `GoodScore`, which it uses.

## [0.1.0] - 2026-07-02

Initial release: calibration-free fusion of heterogeneous retrieval channels into one
ranking. Ruffle needs no relevance labels and never compares one channel's raw scores
against another's; with the default configuration it stays close to plain
reciprocal-rank fusion and tilts weights only when the channels' own outputs support
it. Section references (§4, §5, …) point into the design document,
`docs/derivation.md`.

### Added

- Per-channel discrimination (§4): a scale-free separation statistic reads how far a
  channel's top results stand above its bulk, and an absolute-goodness statistic reads
  the query's top against a declared, evidence-refined good-score reference. The two
  combine into one bounded, conservative weight, with each factor shrunk toward
  neutral by the evidence backing it.
- Channel coupling (§5): pairwise redundancy is estimated on a full-scored anchor of
  representative queries, away from the live pool's selection bias, as Spearman's rho
  mapped to the Gaussian-copula correlation, invariant under monotone rescaling of the
  scores. The discount is off by default, capped, shrunk toward independence, and
  gated on reliability, on stability across query strata, and on at least
  `min_refreshes` anchor refreshes. Confidence and conflict diagnostics are computed
  from top-set overlap.
- Weighted reciprocal-rank fusion (§6): RRF's scale-freedom and bounded per-channel
  contribution, with per-channel weights added. Absent items are omitted rather than
  charged a worst-rank penalty, tied scores share their midrank, and the output is
  deterministic, independent of any hash seed.
- Persistent state and merge (§8): one confidence-weighted summary per channel and per
  pair. A single associative, commutative merge serves as streaming update, operator
  prior, and cross-deployment reconciliation, gated on format and statistic versions
  and a required per-channel model-version tag, with an advisory divergence reported
  alongside. `rekey` renames a channel; `decay` bounds the state's memory.
- Stateful fuser: `Fuser::new` validates registrations and configuration at
  construction, and `Fuser::resume` and `Fuser::fuse_stateless` run the same
  compatibility gate as a state merge. Fuse results carry the ranking, the weights
  used, per-channel flags, the discrimination reads, and the diagnostics. The
  per-stage estimators are also exposed as pure functions under `components`.
- `ruffle` command-line tool behind the `cli` feature: `reconcile` merges state files,
  refusing on incompatibility, and `rekey` renames a channel's key. Both write
  canonical JSON.
- Dual-licensed under MIT OR Apache-2.0. See [`LICENSE-MIT`](LICENSE-MIT) and
  [`LICENSE-APACHE`](LICENSE-APACHE).

[Unreleased]: https://github.com/lathrys-at/ruffle/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/lathrys-at/ruffle/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/lathrys-at/ruffle/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/lathrys-at/ruffle/releases/tag/v0.1.0
