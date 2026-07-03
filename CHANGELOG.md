# Changelog

All notable changes to this project are recorded in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- In-tree Python bindings (`bindings/python`, PyPI package `ruffle`): a fully typed,
  documented pure-Python surface over the compiled engine. Behaviour and persisted
  state are identical to the crate down to the serialized bytes, enforced by a golden
  parity suite (`tests/fixtures/parity/`) generated from the Rust engine and replayed
  by the binding's tests. Wheels build for Linux (x86_64, aarch64), macOS (x86_64,
  arm64), and Windows (x64, arm64) on CPython 3.10 and later, and publish to PyPI
  from the release workflow via Trusted Publishing, version-locked to the crate.

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

[Unreleased]: https://github.com/lathrys-at/ruffle/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/lathrys-at/ruffle/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/lathrys-at/ruffle/releases/tag/v0.1.0
