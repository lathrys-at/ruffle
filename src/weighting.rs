//! Weighting: the per-stage estimators the fuser composes (§4–§6).
//!
//! This groups the three weighting stages and the small solver they share:
//!
//! - [`discrimination`]: per-channel discrimination, how well each channel ranks on
//!   this query (§4).
//! - [`coupling`]: the pairwise redundancy discount (§5).
//! - [`fusion`]: weighted reciprocal-rank fusion (§6).
//! - [`linalg`]: the tiny SPD solver coupling needs; crate-internal, coupling's sole
//!   consumer.

pub mod coupling;
pub mod discrimination;
pub mod fusion;
pub(crate) mod linalg;

/// The neutral channel weight: the §6 invariant that the average channel carries weight
/// `1`. Discrimination shrinks a thin read toward it (§4), coupling falls back to it for
/// a missing or degenerate `g` entry and for the all-zero clamp (§5.4), and fusion uses
/// it for a channel present in the inputs but absent from the weight map (§6).
pub(crate) const NEUTRAL_WEIGHT: f64 = 1.0;
