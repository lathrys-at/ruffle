# ruffle

[![CI](https://github.com/lathrys-at/ruffle/actions/workflows/test.yaml/badge.svg)](https://github.com/lathrys-at/ruffle/actions/workflows/test.yaml)
[![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/lathrys-at/ruffle/badges/coverage.json)](https://github.com/lathrys-at/ruffle/actions/workflows/coverage.yaml)
[![crates.io](https://img.shields.io/crates/v/ruffle.svg)](https://crates.io/crates/ruffle)
[![docs.rs](https://img.shields.io/docsrs/ruffle)](https://docs.rs/ruffle)
[![MSRV](https://img.shields.io/crates/msrv/ruffle)](https://crates.io/crates/ruffle)

Ruffle is a weighted, adaptive, and calibration-free Reciprocal Rank Fusion (RRF)
engine that fuses the output of several retrieval channels into one ranking. It
does this without per-channel score calibration and without comparing one
channel's raw scores against another's.

It is built for the setting where calibration is either inconvenient, undesirable,
or not possible. It requires no relevance labels, no representative query set, and
natively handles channels whose scores live on different scales. Ruffle maintains
the scale-freedom of RRF but stops treating every channel as equally trustworthy
and every pair of channels as independent. For each query, and still without labels,
it estimates how well each channel separates its top results from its bulk, how good
those top results are against a declared reference, and how redundant the channels
are with each other. It then weights the fusion step for each query from those
estimates.

## Quick start

```toml
[dependencies]
ruffle = "0.1"
```

```rust
use ruffle::{ChannelConfig, ChannelId, ChannelInput, Direction, FuseConfig, Fuser, Score};

// A channel's native score becomes a `Score` only through a newtype that declares what
// the number means.
struct Cos(f64);
impl Score for Cos { fn value(&self) -> f64 { self.0 } }

// Channels represent different retrieval methods for the same query. The set of
// channels and their semantic meaning depend on your application. Each channel's
// id represents a stable key and a semantic+version tag.
let semantic = ChannelConfig::new(
    ChannelId::new("semantic", "text-embedding-v1"),
    Direction::HigherIsBetter, // higher cosine-similarity scores are better
    None,
);

// SQLite FTS5 `bm25()` is negated BM25, so lower (more negative) is better.
struct Bm25(f64);
impl Score for Bm25 { fn value(&self) -> f64 { self.0 } }

// Channel configurations may describe what "good" scores look like, but ruffle
// can also learn this on its own if you do not provide it.
let lexical = ChannelConfig::new(
    ChannelId::new("lexical", "sqlite-fts5-trigram-bm25"),
    Direction::LowerIsBetter,
    // typical top ≈ -4.0, good match ≈ -12.0 (native units), and the value 8.0
    // is a pseudo-count that tells ruffle how strongly to anchor on this prior
    // when observing new traffic.
    Some(GoodScore::new(-4.0, -12.0, 8.0)),
);

// Channels can be rank-only (without score magnitudes), like a recency metric.
let recency = ChannelConfig::new(
    ChannelId::new("recency", "recency-v1"),
    Direction::HigherIsBetter,
    None,
);

// `Fuser::new` validates the registrations and configuration, and builds the channel
// lookup and the empty starting state internally. To continue from a persisted state,
// use `Fuser::resume`, which also checks the state is compatible (same format, tags,
// and orientations) before accepting it.
let mut fuser = Fuser::new(
    &[semantic.clone(), lexical.clone(), recency.clone()],
    FuseConfig::default(),
)
.expect("valid registrations");

// One query's results, per channel.
//
// `scored` lists need no particular order, since ruffle ranks each channel by its
// own oriented scores. Only a `ranked` channel's list order carries meaning, and must
// be sorted best-to-worst.
//
// Ids are opaque to ruffle, any Hash + Eq + Clone type your system keys candidates by can
// be used. In this example strings are used.
let inputs = vec![
    ChannelInput::scored(&semantic, vec![
        ("kelp-forest",  Cos(0.55)),
        ("whale-sketch", Cos(0.91)),
        ("tide-chart",   Cos(0.42)),
    ]),
    ChannelInput::scored(&lexical, vec![
        ("whale-sketch", Bm25(-3.7)),
        ("field-notes",  Bm25(-1.4)),
        ("kelp-forest",  Bm25(-6.4)),
    ]),
    ChannelInput::ranked(&recency, vec!["field-notes", "whale-sketch", "kelp-forest"]),
];

let fused = fuser.fuse(&inputs);
for (id, score) in &fused.ranking {
    println!("{id}: {score:.4}");
}
```

## Design

Ruffle is calibration-free and recall-safe by construction. Every estimate is read from the
channels' own outputs, so none of it needs ground truth. The weighting is conservative:
weights are non-negative, redundancy is shrunk toward independence, and a channel whose
signal is unclear is left near the neutral RRF weight rather than zeroed. The redundancy
discount (coupling) is off by default, because assuming independence is the only setting
that never costs recall. With the defaults, ruffle stays close to plain RRF and tilts
weights only when the evidence supports it.

Persistent state is one confidence-weighted summary per channel and per channel pair. A
single merge operation serves as the streaming update, the operator-supplied prior, and
cross-deployment reconciliation. Every merge is gated by a required per-channel semantic
tag, so a model swap under a kept name is refused rather than silently blended into the
baseline.

Every fuse result carries with it an explanation: the weights actually used, a per-channel
flag for any non-standard weighting, the discrimination readings behind each weight, and
two agreement diagnostics, so "why did this channel get this weight" is answerable from
telemetry alone. That is also the basis for tuning: the defaults are deliberately
conservative, and the [tuning guide](docs/tuning.md) describes exactly what observation
justifies changing each one.

## Features

The library builds with no required features. The `cli` feature adds the `ruffle` binary,
which reconciles and renames persistent state files (`ruffle reconcile`, `ruffle rekey`).

## Performance and behavior

Fusion sits between a retrieval fan-out and a reranker, so it only has to be negligible
next to the retrieval call itself. A full stateful fuse (discrimination, weighting,
fusion, and the baseline update) over four channels runs in a few 10s of microseconds
at 100 candidates per channel and a few 100s of microseconds at 1,000 candidates.
`cargo bench` reproduces these numbers from `benches/fusion.rs`.

Every fuse Ruffle produces is deterministic: the same inputs, state, and configuration
produce the identical ranking, independent of any hash seed.

## Documentation

- API reference: [docs.rs/ruffle](https://docs.rs/ruffle).
- Operating and tuning: [`docs/tuning.md`](docs/tuning.md) describes what to log,
  how to read the state, and, for each default, when and why to change it.
- Design doc: [`docs/derivation.md`](docs/derivation.md) contains Ruffle's design
  covering discrimination and coupling statistics, the weighted RRF, the state
  model, the validity boundaries, and the simulations behind the defaults.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in ruffle by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.
