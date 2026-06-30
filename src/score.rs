//! Channel scores, orientation, and sanitization (§7).
//!
//! A channel's native score is whatever scalar it produces. `ruffle` requires only
//! that it convert to a canonical [`f64`] through the [`Score`] trait. There are no
//! blanket implementations for numeric types: a bare `f32`/`f64`/integer is not a
//! `Score` until the caller newtypes it and thereby declares what the number means.
//!
//! Everything downstream works in canonical higher-is-better units. Orientation is
//! applied once at ingest by [`orient`], and non-finite values are dropped by
//! [`sanitize`] before they can corrupt a streaming mean.

use serde::{Deserialize, Serialize};

/// A channel's native score, convertible to a canonical [`f64`].
///
/// There are deliberately no blanket implementations for numeric types. A bare
/// `f32`, `f64`, or integer is not a `Score`; the caller must wrap it in a newtype
/// that declares what the score means. Newtypes are cheap in Rust, and requiring one
/// forces that declaration instead of accepting an unlabeled float.
///
/// ```
/// use ruffle::Score;
///
/// // A bare f32 is NOT a Score. The caller newtypes it:
/// struct Cosine(f32);
/// impl Score for Cosine {
///     fn value(&self) -> f64 {
///         self.0 as f64
///     }
/// }
/// assert_eq!(Cosine(0.5).value(), 0.5);
/// ```
pub trait Score {
    /// This score as a canonical [`f64`], in the channel's native units and
    /// native orientation (orientation is applied at ingest, not here).
    fn value(&self) -> f64;
}

/// Whether a higher native score means a better match, or a lower one.
///
/// Declared once per channel at configuration, never inferred from data or supplied per
/// query. `ruffle` orients every score to higher-is-better at ingest. Orientation cannot
/// be recovered from a score distribution, and a channel registered with the wrong
/// direction ranks anti-relevantly and corrupts its own persistent baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    /// A higher native score is a better match (already canonical).
    HigherIsBetter,
    /// A lower native score is a better match (negated at ingest).
    LowerIsBetter,
}

/// An operator-declared reference for how good a channel's scores are in absolute terms,
/// in the channel's NATIVE units (before orientation).
///
/// The discrimination stage rewards a channel whose top results score well against this
/// reference, and not only one whose top separates from its own bulk. The operator states
/// two interpretable anchors plus a pseudo-count:
///
/// - `typical`: the top score a typical, unremarkable query produces. Sets the reference
///   location.
/// - `good`: the score a genuinely good match reaches. The gap from `typical` to `good`
///   sets the reference scale.
/// - `weight`: a pseudo-count for how firmly the declaration holds before observed top
///   scores refine it.
///
/// Both anchors are oriented with the scores at ingest, so for a `LowerIsBetter` channel a
/// good match is a smaller native value, and `typical` and `good` are negated together
/// with the scores. After orientation `good` must exceed `typical`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GoodScore {
    /// The top score a typical, unremarkable query produces, in native units. Sets the
    /// reference location.
    pub typical: f64,
    /// The score a genuinely good match reaches, in native units. The gap from `typical`
    /// to `good` sets the reference scale, so `good` must exceed `typical` after
    /// orientation.
    pub good: f64,
    /// A pseudo-count for how firmly the declaration holds before observed top scores
    /// refine it. Its influence after `n` observed top scores is `weight / (weight + n)`.
    pub weight: f64,
}

/// The oriented reference location and scale returned by [`GoodScore::oriented`] (§4).
///
/// Both are in canonical higher-is-better units. The fields are named so the location
/// and scale cannot be swapped where two bare `f64`s would have passed unnoticed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct OrientedReference {
    /// The oriented reference location: the oriented `typical` anchor.
    pub mu_ref: f64,
    /// The oriented reference scale: `(oriented good - oriented typical) / 2`, always
    /// positive and finite.
    pub sigma_ref: f64,
}

impl GoodScore {
    /// Build a reference from native-unit anchors and a pseudo-count.
    ///
    /// The anchors are not checked here. `good` need only exceed `typical` after
    /// orientation, and `ruffle` enforces that when it orients the reference at ingest.
    pub fn new(typical: f64, good: f64, weight: f64) -> Self {
        Self {
            typical,
            good,
            weight,
        }
    }

    /// The oriented reference location and scale under `dir`, as an [`OrientedReference`].
    ///
    /// `mu_ref` is the oriented typical anchor and
    /// `sigma_ref = (oriented good - oriented typical) / 2`. Returns `None` when the
    /// anchors are non-finite or when `sigma_ref <= 0` (i.e. `good` does not exceed
    /// `typical` after orientation), which a `LowerIsBetter` channel requires to hold
    /// only post-negation (§4, §7).
    #[must_use]
    pub(crate) fn oriented(&self, dir: Direction) -> Option<OrientedReference> {
        let mu = orient(dir, sanitize(self.typical)?);
        let g = orient(dir, sanitize(self.good)?);
        let sigma = (g - mu) / 2.0;
        if sigma > 0.0 && sigma.is_finite() {
            Some(OrientedReference {
                mu_ref: mu,
                sigma_ref: sigma,
            })
        } else {
            None
        }
    }
}

/// Orient a raw native score to canonical higher-is-better.
///
/// Passes a `HigherIsBetter` score through unchanged and negates a `LowerIsBetter`
/// score. Negation preserves relative order and gaps, which is all the rank and
/// separation statistics need; the good-score reference is oriented alongside (§7).
#[inline]
pub(crate) fn orient(dir: Direction, raw: f64) -> f64 {
    match dir {
        Direction::HigherIsBetter => raw,
        Direction::LowerIsBetter => -raw,
    }
}

/// Drop a non-finite value at ingest.
///
/// Returns `Some(x)` when `x` is finite and `None` for `NaN` or infinity. A single
/// `NaN` would otherwise permanently corrupt the streaming mean it feeds (§7, §8).
#[inline]
pub(crate) fn sanitize(x: f64) -> Option<f64> {
    if x.is_finite() { Some(x) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A caller-side newtype: the only way a bare number becomes a Score (§7).
    struct Raw(f64);
    impl Score for Raw {
        fn value(&self) -> f64 {
            self.0
        }
    }

    #[test]
    fn newtype_is_required_for_score() {
        // This compiles only because Raw declares what its number means. There is
        // intentionally no `impl Score for f64`, so `(0.5f64).value()` would not
        // compile. The newtype is the contract (§7).
        assert_eq!(Raw(0.5).value(), 0.5);
    }

    #[test]
    fn orient_passes_higher_through() {
        assert_eq!(orient(Direction::HigherIsBetter, 3.0), 3.0);
        assert_eq!(orient(Direction::HigherIsBetter, -2.0), -2.0);
    }

    #[test]
    fn orient_negates_lower() {
        assert_eq!(orient(Direction::LowerIsBetter, 3.0), -3.0);
        assert_eq!(orient(Direction::LowerIsBetter, -2.0), 2.0);
    }

    #[test]
    fn sanitize_drops_non_finite() {
        assert_eq!(sanitize(1.5), Some(1.5));
        assert_eq!(sanitize(0.0), Some(0.0));
        assert_eq!(sanitize(f64::NAN), None);
        assert_eq!(sanitize(f64::INFINITY), None);
        assert_eq!(sanitize(f64::NEG_INFINITY), None);
    }

    #[test]
    fn good_score_oriented_higher_is_better() {
        let gs = GoodScore::new(0.3, 0.5, 4.0);
        let OrientedReference { mu_ref, sigma_ref } =
            gs.oriented(Direction::HigherIsBetter).unwrap();
        assert_eq!(mu_ref, 0.3);
        assert!((sigma_ref - 0.1).abs() < 1e-12);
    }

    #[test]
    fn good_score_oriented_lower_is_better_negates_anchors() {
        // For LowerIsBetter, a good match is a SMALLER native score: good < typical
        // natively, but after negation good > typical and sigma > 0 (§7).
        let gs = GoodScore::new(0.5, 0.3, 4.0);
        let OrientedReference { mu_ref, sigma_ref } =
            gs.oriented(Direction::LowerIsBetter).unwrap();
        assert_eq!(mu_ref, -0.5);
        assert!((sigma_ref - 0.1).abs() < 1e-12);
    }

    #[test]
    fn good_score_rejects_non_positive_sigma() {
        // good == typical after orientation -> sigma == 0 -> None.
        let flat = GoodScore::new(0.4, 0.4, 1.0);
        assert_eq!(flat.oriented(Direction::HigherIsBetter), None);

        // good < typical after orientation -> sigma < 0 -> None.
        let inverted = GoodScore::new(0.5, 0.3, 1.0);
        assert_eq!(inverted.oriented(Direction::HigherIsBetter), None);

        // A LowerIsBetter channel whose anchors are stated as if HigherIsBetter
        // (good > typical natively) negates to sigma < 0 -> None, catching the
        // misdeclaration.
        let mis = GoodScore::new(0.3, 0.5, 1.0);
        assert_eq!(mis.oriented(Direction::LowerIsBetter), None);
    }

    #[test]
    fn good_score_rejects_non_finite_anchors() {
        let bad = GoodScore::new(f64::NAN, 0.5, 1.0);
        assert_eq!(bad.oriented(Direction::HigherIsBetter), None);
    }
}
