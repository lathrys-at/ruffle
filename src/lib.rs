//! Ruffle is a weighted, adaptive, and calibration-free Reciprocal Rank Fusion (RRF)
//! engine that fuses the output of several retrieval channels into one ranking. It does
//! this without per-channel score calibration and without comparing one channel's raw
//! scores against another's.
//!
//! It is built for the setting where calibration is either inconvenient, undesirable, or
//! not possible. It requires no relevance labels, no representative query set, and
//! natively handles channels whose scores live on different scales. Ruffle maintains the
//! scale-freedom of RRF but stops treating every channel as equally trustworthy and
//! every pair of channels as independent. For each query, and still without labels, it
//! estimates two properties from the channels' own outputs:
//!
//! - Per-channel discrimination: how far a channel's top results stand above its bulk,
//!   and how good those top results are against a declared, evidence-refined good-score
//!   reference.
//! - Pairwise redundancy: a correlation between two channels, measured on a shared
//!   anchor of queries that both score in full, away from the live pool's selection
//!   bias.
//!
//! The estimates weight a rank-based RRF. Every estimate is conservative: weights stay
//! non-negative, redundancy shrinks toward independence, and independence is assumed
//! whenever the evidence is too thin to say otherwise, since assuming independence never
//! costs recall. With the default configuration Ruffle stays close to plain RRF and
//! tilts weights only when the channels' own outputs support it.
//!
//! # State and reconciliation
//!
//! All persistent state is one confidence-weighted summary, a [`RuffleState`]. Its
//! single merge operation serves three roles: streaming update as new queries arrive,
//! operator prior seeded before any traffic, and cross-deployment reconciliation of
//! states accumulated on separate machines. Every merge is gated on a required
//! per-channel model-version tag: two states must agree on it before their statistics
//! for that channel combine, so a model swapped in under a kept channel name is refused
//! rather than silently blended.
//!
//! Ruffle treats candidate ids as opaque and knows a channel only as a key with a
//! history, so it stays independent of the retrieval systems that feed it.
//!
//! # Example
//!
//! The following fuses three channels for one query: semantic and lexical channels
//! scored on their own native scales, and a rank-only recency channel.
//!
//! ```
//! use ruffle::{
//!     ChannelConfig, ChannelId, ChannelInput, Direction, FuseConfig, Fuser, GoodScore,
//!     Score,
//! };
//!
//! // A channel's native score becomes a `Score` only through a newtype that declares
//! // what the number means. There is no blanket `impl Score for f64`.
//! struct Cos(f64);
//! impl Score for Cos { fn value(&self) -> f64 { self.0 } }
//!
//! // Channels represent different retrieval methods for the same query. The set of
//! // channels and their semantic meaning depend on your application. Each channel's
//! // id represents a stable key and a semantic+version tag.
//! let semantic = ChannelConfig::new(
//!     ChannelId::new("semantic", "text-embedding-v1"),
//!     Direction::HigherIsBetter, // higher cosine-similarity scores are better
//!     None,
//! );
//!
//! // SQLite FTS5 `bm25()` is negated BM25, so lower (more negative) is better.
//! struct Bm25(f64);
//! impl Score for Bm25 { fn value(&self) -> f64 { self.0 } }
//!
//! // Channel configurations may describe what "good" scores look like, but ruffle
//! // can also learn this on its own if you do not provide it.
//! let lexical = ChannelConfig::new(
//!     ChannelId::new("lexical", "sqlite-fts5-trigram-bm25"),
//!     Direction::LowerIsBetter,
//!     // typical top ≈ -4.0, good match ≈ -12.0 (native units), and the value 8.0
//!     // is a pseudo-count that tells ruffle how strongly to anchor on this prior
//!     // when observing new traffic.
//!     Some(GoodScore::new(-4.0, -12.0, 8.0)),
//! );
//!
//! // Channels can be rank-only (without score magnitudes), like a recency metric.
//! let recency = ChannelConfig::new(
//!     ChannelId::new("recency", "recency-v1"),
//!     Direction::HigherIsBetter,
//!     None,
//! );
//!
//! // `Fuser::new` validates the registrations and configuration, and builds the channel
//! // lookup and the empty starting state internally. To continue from a persisted
//! // state, use `Fuser::resume`, which also checks the state is compatible (same
//! // format, tags, and orientations) before accepting it.
//! let mut fuser = Fuser::new(
//!     &[semantic.clone(), lexical.clone(), recency.clone()],
//!     FuseConfig::default(),
//! )
//! .expect("valid registrations");
//!
//! // One query's results, per channel.
//! //
//! // `scored` lists need no particular order, since ruffle ranks each channel by its
//! // own oriented scores. Only a `ranked` channel's list order carries meaning, and
//! // must be sorted best-to-worst.
//! //
//! // Ids are opaque to ruffle, any Hash + Eq + Clone type your system keys candidates
//! // by can be used. In this example strings are used.
//! let inputs = vec![
//!     ChannelInput::scored(&semantic, vec![
//!         ("kelp-forest",  Cos(0.55)),
//!         ("whale-sketch", Cos(0.91)),
//!         ("tide-chart",   Cos(0.42)),
//!     ]),
//!     ChannelInput::scored(&lexical, vec![
//!         ("whale-sketch", Bm25(-3.7)),
//!         ("field-notes",  Bm25(-1.4)),
//!         ("kelp-forest",  Bm25(-6.4)),
//!     ]),
//!     ChannelInput::ranked(&recency, vec!["field-notes", "whale-sketch", "kelp-forest"]),
//! ];
//!
//! // `fused.ranking` is the merged order, best first, each id with its fused score.
//! let fused = fuser.fuse(&inputs);
//! for (id, score) in &fused.ranking {
//!     println!("{id}: {score:.4}");
//! }
//! ```
//!
//! # Further documentation
//!
//! The [tuning guide](https://github.com/lathrys-at/ruffle/blob/main/docs/tuning.md)
//! describes what to log, how to read the persisted state, and, for each configuration
//! default, when and why to change it. The design document,
//! [`docs/derivation.md`](https://github.com/lathrys-at/ruffle/blob/main/docs/derivation.md),
//! contains the full derivation. Both ship inside the published crate under `docs/`.
//!
//! # Module map
//!
//! The everyday entry point is [`Fuser`]; the types it works with are all re-exported at
//! the crate root, grouped by role:
//!
//! - Entry point: [`Fuser`], its [`Fused`] result, and the per-channel [`ChannelFlag`]
//!   that explains any non-standard weighting.
//! - Channel registration and ingest: a [`ChannelConfig`] keyed by [`ChannelId`], the
//!   [`Score`] trait, channel [`Direction`], the good-score reference [`GoodScore`], one
//!   query's [`ChannelInput`] / [`Items`], and the coupling [`Anchor`].
//! - Configuration: the fusion knobs [`FuseConfig`] and its sub-configs
//!   ([`DiscriminationConfig`], [`CouplingConfig`], [`RrfConfig`], [`DecayConfig`]).
//! - Persistent state and reconciliation: [`RuffleState`] and its summaries
//!   ([`ChannelSummary`], [`PairSummary`], [`MeanVar`]), reconciled through [`MergePolicy`]
//!   with [`Mismatch`] on refusal and an advisory [`Divergence`] alongside; the state's
//!   identity plumbing is [`StatFingerprint`], [`BaselineMode`], and the canonical
//!   [`UnorderedPair`] pair key.
//! - Advanced building blocks: the per-stage functions [`Fuser`] composes internally are
//!   re-exported under [`components`].

mod config;
mod error;
mod fuser;
mod ingest;
mod keys;
mod score;
mod state;
mod summary;
mod weighting;

pub mod components {
    //! Advanced building blocks: the per-stage functions that [`Fuser`](crate::Fuser)
    //! composes internally. The stable entry point is [`Fuser`](crate::Fuser); these
    //! lower-level functions are exposed for advanced composition and for exercising the
    //! discrimination, coupling, and fusion stages in isolation.
    pub use crate::weighting::coupling::{
        CoupledWeights, Diagnostics, PairBaseline, PairObservation, anchor_correlations,
        coupled_weights, diagnostics,
    };
    pub use crate::weighting::discrimination::{ChannelDiscrimination, discriminate};
    pub use crate::weighting::fusion::weighted_rrf;
}

// Tier-1 everyday surface, kept at the crate root, grouped by role.

// Entry point: build a fuser, fuse a query, read the result and its per-channel flags.
pub use fuser::{ChannelFlag, Fused, Fuser};

// Channel registration and ingest: declare channels and feed one query's inputs.
pub use config::ChannelConfig;
pub use ingest::anchor::Anchor;
pub use ingest::input::{ChannelInput, Items};
pub use keys::ChannelId;
pub use score::{Direction, GoodScore, Score};

// Configuration: the fusion knobs and their sub-configs.
pub use config::{CouplingConfig, DecayConfig, DiscriminationConfig, FuseConfig, RrfConfig};

// Persistent state and reconciliation: persist, merge, reconcile. The fingerprint and
// pair-key types (`StatFingerprint`, `BaselineMode`, `UnorderedPair`) and the `MeanVar`
// summary are state plumbing, not everyday fusion.
pub use error::{ConfigError, Mismatch, ResumeError};
pub use keys::{BaselineMode, StatFingerprint, UnorderedPair};
pub use state::{ChannelSummary, Divergence, MergePolicy, PairSummary, RuffleState};
pub use summary::MeanVar;
