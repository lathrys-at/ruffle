//! Weighting: the per-stage estimators the fuser composes.
//!
//! This groups the three weighting stages and the small solver they share:
//!
//! - [`discrimination`]: per-channel discrimination, how well each channel ranks on
//!   this query.
//! - [`coupling`]: the pairwise redundancy discount.
//! - [`fusion`]: weighted reciprocal-rank fusion.
//! - [`linalg`]: the small SPD solver used by coupling; crate-internal, with coupling
//!   as its sole consumer.

pub mod coupling;
pub mod discrimination;
pub mod fusion;
pub(crate) mod linalg;

/// The neutral channel weight: the fusion invariant that the average channel has weight
/// `1`. Discrimination shrinks a thin read toward it, coupling falls back to it for
/// a missing or degenerate `g` entry and for the all-zero clamp, and fusion uses
/// it for a channel present in the inputs but absent from the weight map.
pub(crate) const NEUTRAL_WEIGHT: f64 = 1.0;
