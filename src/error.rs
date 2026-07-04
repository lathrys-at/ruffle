//! Error types for state reconciliation and fuser construction.
//!
//! The merge of two persistent states refuses on any incompatibility rather than
//! silently blending distributions that were never measuring the same thing, and
//! building a [`Fuser`](crate::Fuser) refuses an invalid configuration or an
//! incompatible resumed state rather than fusing on top of it.

use thiserror::Error;

/// A reason two [`RuffleState`](crate::state::RuffleState)s cannot be merged.
///
/// Every variant is a hard refusal. Merging incompatible states would produce a baseline
/// that fits neither and keeps no record of the conflict.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Mismatch {
    /// The two states use different serialization schema versions.
    #[error("format version mismatch: {left} vs {right}")]
    FormatVersion {
        /// Format version of the left-hand state.
        left: u32,
        /// Format version of the right-hand state.
        right: u32,
    },

    /// The two states were built with different statistic definitions or baseline modes
    /// (different `stat_version` or `baseline_mode`), so their summaries are numerically
    /// incompatible.
    #[error("statistic fingerprint mismatch")]
    Fingerprint,

    /// A channel present in more than one state has a different orientation in the
    /// fingerprint. A direction flip negates the channel's scores, so its persisted
    /// baseline measures the opposite quantity, and pooling the two would corrupt the
    /// baseline.
    #[error("direction conflict for channel {channel}")]
    DirectionConflict {
        /// The channel key whose orientations disagree across states.
        channel: String,
    },

    /// A channel present in both states has a different model-version tag, the signature
    /// of a model swapped in under a kept name. Merging across it would blend statistics
    /// from different models.
    #[error("semantic tag mismatch for channel {channel}: {left} vs {right}")]
    Tag {
        /// The channel key whose tags disagree.
        channel: String,
        /// The tag on the left-hand state.
        left: String,
        /// The tag on the right-hand state.
        right: String,
    },

    /// Merge was called with no input states.
    #[error("cannot merge an empty set of states")]
    Empty,
}

/// A reason a [`Fuser`](crate::Fuser) cannot be built from the given registrations and
/// configuration.
///
/// Every variant is a hard refusal at construction. Fusing on top of an invalid
/// configuration would panic mid-query or silently degrade, so the problems are
/// surfaced before any query runs.
#[derive(Error, Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ConfigError {
    /// A fusion knob holds a value outside its documented range (non-finite, negative
    /// where a non-negative is required, or an inverted bound pair).
    #[error("invalid fuse configuration: {field}: {reason}")]
    InvalidFuseConfig {
        /// The offending field, as `sub_config.field`.
        field: &'static str,
        /// Why the value is rejected.
        reason: &'static str,
    },

    /// A channel's declared [`GoodScore`](crate::score::GoodScore) does not orient to a
    /// usable reference: an anchor is non-finite, or `good` does not exceed `typical`
    /// after orientation, so the reference scale would be zero or negative. Accepting it
    /// would silently cold-start the channel as if nothing had been declared.
    #[error("channel {channel}: declared good score is unusable: {reason}")]
    InvalidGoodScore {
        /// The channel key whose declaration is rejected.
        channel: String,
        /// Why the declaration is rejected.
        reason: &'static str,
    },

    /// Two channel registrations share one join-handle key. Both would write to the same
    /// baseline, so the duplication is refused.
    #[error("duplicate channel key {key} in the registrations")]
    DuplicateChannelKey {
        /// The key that appears more than once.
        key: String,
    },

    /// A channel's declared base weight is non-finite or negative. A negative multiplier
    /// would invert the channel's votes and a non-finite one would poison every fused
    /// score, so both are refused at construction.
    #[error("channel {channel}: base weight must be finite and non-negative")]
    InvalidBaseWeight {
        /// The channel key whose declaration is rejected.
        channel: String,
    },
}

/// A reason a [`Fuser`](crate::Fuser) cannot resume from a persisted state.
///
/// A model swap happens across a restart, so resume runs the same compatibility gate
/// as a state merge: the registrations must agree with the persisted state on format,
/// statistic definitions, per-channel orientation, and the per-channel model-version
/// tag.
#[derive(Error, Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ResumeError {
    /// The registrations or configuration are invalid on their own, before the state is
    /// even considered.
    #[error(transparent)]
    Config(#[from] ConfigError),

    /// The persisted state is incompatible with the registrations or with this build:
    /// a format or statistic-version mismatch, a channel whose configured direction
    /// contradicts the state fingerprint, or a channel whose configured tag differs from
    /// the tag its accumulated statistics were measured under (the signature of a model
    /// swap; the statistics must not be blended across it).
    #[error(transparent)]
    State(#[from] Mismatch),
}
