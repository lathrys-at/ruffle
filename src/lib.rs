//! `ruffle` fuses the ranked outputs of several retrieval channels into one ranking.
//! It never calibrates a channel's scores and never compares one channel's raw scores
//! against another's.
//!
//! It targets the regime where calibration is impossible: no relevance labels, no
//! representative queries, and channels on wildly different, individually meaningless
//! scales. Reciprocal-rank fusion (RRF) is the standard answer there because it discards
//! magnitudes and votes on ranks. `ruffle` keeps that scale-freedom but stops treating
//! the channels as equally trustworthy and mutually independent. Per query and without
//! labels, it estimates how discriminating each channel is and how redundant the
//! channels are with each other, then weights the fusion accordingly.
//!
//! Both estimands are dimensionless properties of the channels' own outputs, readable
//! without ground truth:
//!
//! - Per-channel discrimination: a scale-free separation of the top of a channel's
//!   result pool from its bulk, plus an absolute goodness measured against a declared,
//!   evidence-refined reference score.
//! - Pairwise redundancy: a correlation between two channels measured on a shared anchor
//!   of queries that both score in full, away from the live pool's selection bias.
//!
//! The weights feed a rank-based weighted RRF. Every estimate is conservative: weights
//! stay non-negative, redundancy shrinks toward independence, and independence is assumed
//! whenever the data is too thin to say otherwise, which is both the correct statistical
//! default and the recall-safe one.
//!
//! # State and reconciliation
//!
//! All persistent state is one confidence-weighted summary, a [`RuffleState`]. Its single
//! merge operation serves three roles: streaming update as new queries arrive, operator
//! prior seeded before any traffic, and cross-deployment reconciliation of states
//! accumulated on separate machines. A merge is gated. Two states must agree on a
//! per-channel model-version tag before their statistics for that channel combine, so a
//! model swapped in under a kept channel name is refused rather than silently blended.
//!
//! `ruffle` never interprets a candidate id and never knows which channel is which. It
//! knows only that the channel under a given key has a given history, so it stays
//! independent of the channels that feed it.
//!
//! # Example
//!
//! Fuse three retrieval channels into one ranking. Two carry scores on their own native
//! scales; the third contributes ranks only. No labels and no calibration: a channel's
//! scores are never compared against another's.
//!
//! ```
//! use ruffle::{ChannelConfig, ChannelId, ChannelInput, Direction, FuseConfig, Fuser, Score};
//!
//! // A channel's native score becomes a `Score` only through a caller newtype, which
//! // declares what the number means. There is no blanket `impl Score for f64`.
//! struct Sim(f64);
//! impl Score for Sim {
//!     fn value(&self) -> f64 {
//!         self.0
//!     }
//! }
//!
//! // Register three channels: two scored (higher is better), one ranks-only.
//! let dense = ChannelConfig::new(
//!     ChannelId::new("dense", "clip-v1"),
//!     Direction::HigherIsBetter,
//!     None,
//! );
//! let lexical = ChannelConfig::new(
//!     ChannelId::new("lexical", "bm25-v1"),
//!     Direction::HigherIsBetter,
//!     None,
//! );
//! let recency = ChannelConfig::new(
//!     ChannelId::new("recency", "recency-v1"),
//!     Direction::HigherIsBetter,
//!     None,
//! );
//!
//! // A fresh fuser over the three channels at the conservative default (plain RRF).
//! // `Fuser::new` validates the registrations and configuration, and builds the
//! // channel lookup and the empty starting state internally.
//! let mut fuser = Fuser::new(&[dense.clone(), lexical.clone(), recency.clone()], FuseConfig::default())?;
//!
//! // One query's results per channel. `scored` orients and sanitizes the scores;
//! // `ranked` takes an already-ranked list, best first.
//! let inputs = vec![
//!     ChannelInput::scored(&dense, vec![(1u32, Sim(0.91)), (2, Sim(0.55)), (3, Sim(0.42))]),
//!     ChannelInput::scored(&lexical, vec![(2u32, Sim(7.3)), (1, Sim(4.1)), (4, Sim(2.0))]),
//!     ChannelInput::ranked(&recency, vec![4u32, 1, 2]),
//! ];
//!
//! let fused = fuser.fuse(&inputs);
//!
//! // `fused.ranking` is the merged order, best first, each id with its fused score.
//! assert!(!fused.ranking.is_empty());
//! let (best_id, best_score) = &fused.ranking[0];
//! println!("top result: {best_id} at {best_score:.4}");
//! # Ok::<(), ruffle::ConfigError>(())
//! ```
//!
//! # Further documentation
//!
//! Operational guidance lives in the repository's
//! [tuning guide](https://github.com/lathrys-at/ruffle/blob/main/docs/tuning.md):
//! what to log, how to read the persisted state, and, for each configuration default,
//! the symptom that motivates changing it, the evaluation that confirms the diagnosis,
//! and the signal that justifies the move. The full derivation, including the section
//! numbers (§4, §5, …) cited throughout these docs, is the design document,
//! [`docs/derivation.md`](https://github.com/lathrys-at/ruffle/blob/main/docs/derivation.md).
//! Both ship inside the published crate under `docs/`.
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
