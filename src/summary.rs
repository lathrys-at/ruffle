//! The single mergeable summary: confidence-weighted streaming mean and variance (§8).
//!
//! Everything `ruffle` persists is one of these. Given a count-weighted, associative
//! merge, three operations collapse into one (§8):
//!
//! ```text
//! streaming update = prior = cross-deployment reconciliation = merge(.)
//! ```
//!
//! A streaming update is a merge with a count-1 summary; an operator prior is a
//! hand-written summary with a pseudo-count; reconciliation is an N-argument merge.
//!
//! The merge uses Chan's parallel formulas (`ChanGolubLeVeque1983`; `Welford1962`), which
//! are associative and commutative exactly up to f64 rounding, so update order never
//! matters. The count is [`f64`] to support pseudo-counts and decay. Accumulation is
//! f64 throughout because f32 running moments drift measurably over millions of
//! updates.

use serde::{Deserialize, Serialize};

/// Confidence-weighted streaming mean and variance.
///
/// Tracks an effective `count`, a running `mean`, and `m2` (the sum of squared deviations
/// from the mean); the population variance is `m2 / count`. Merging two of these is
/// associative and commutative and, with decay off, exact up to f64 rounding, so the same
/// type serves as a prior, a streaming accumulator, and a reconciliation target.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeanVar {
    /// Effective observation count. Fractional to support pseudo-counts and decay.
    count: f64,
    /// Running mean of the observations.
    mean: f64,
    /// Sum of squared deviations from the mean (`Σ (x - mean)²`). Population
    /// variance is `m2 / count`.
    m2: f64,
}

impl MeanVar {
    /// An empty summary: zero count, zero mean, zero `m2`.
    pub fn new() -> Self {
        Self {
            count: 0.0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Seed a prior summary from a mean, a population variance, and a pseudo-count.
    ///
    /// The pseudo-count sets how much evidence the prior stands in for, and so how firmly
    /// it holds: after `n` real observations the prior's pull has shrunk to
    /// `1 / (pseudo_count + n)`. A non-positive or non-finite pseudo-count, or a
    /// non-finite mean or variance, yields an empty summary; a negative variance is
    /// clamped to zero.
    pub fn from_prior(mean: f64, variance: f64, pseudo_count: f64) -> Self {
        if !pseudo_count.is_finite()
            || pseudo_count <= 0.0
            || !mean.is_finite()
            || !variance.is_finite()
        {
            return Self::new();
        }
        Self {
            count: pseudo_count,
            mean,
            m2: variance.max(0.0) * pseudo_count,
        }
    }

    /// Fold one observation in, equivalent to merging a count-1 summary.
    ///
    /// A non-finite `x` is ignored, so a single stray value cannot corrupt the mean. This
    /// is a backstop; scores are normally already sanitized when they are ingested.
    pub fn push(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        *self = Self::merge(
            self,
            &Self {
                count: 1.0,
                mean: x,
                m2: 0.0,
            },
        );
    }

    /// Merge two summaries with Chan's parallel formulas.
    ///
    /// Associative and commutative, exact up to f64 rounding. With either operand
    /// empty the result equals the other operand. A combined count of zero or less
    /// yields an empty summary.
    #[must_use]
    pub fn merge(a: &Self, b: &Self) -> Self {
        let n = a.count + b.count;
        if n <= 0.0 {
            return Self::new();
        }
        let delta = b.mean - a.mean;
        let mean = a.mean + delta * (b.count / n);
        let m2 = a.m2 + b.m2 + delta * delta * (a.count * b.count / n);
        Self { count: n, mean, m2 }
    }

    /// Fold another summary into this one in place (`self = merge(self, other)`).
    pub fn merge_in(&mut self, other: &Self) {
        *self = Self::merge(self, other);
    }

    /// The effective observation count.
    #[must_use]
    pub fn count(&self) -> f64 {
        self.count
    }

    /// The running mean. Zero for an empty summary.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// The population variance `m2 / count`, guarded.
    ///
    /// Returns `0.0` when the count is non-positive, and clamps tiny negative values
    /// from rounding up to zero, so the result is never `NaN` or negative.
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count <= 0.0 {
            0.0
        } else {
            (self.m2 / self.count).max(0.0)
        }
    }

    /// The population standard deviation `sqrt(variance)`, guarded against `NaN`.
    #[must_use]
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Standardize `x` against this baseline: `(x - mean) / std`.
    ///
    /// Returns `None` when the count is non-positive, when the standard deviation is not
    /// finite and strictly positive, or when the result itself is not finite, so a
    /// degenerate baseline never yields a `NaN` or infinite z-score. This is only the
    /// numerical guard; whether the count is high enough to trust the score is a separate
    /// decision left to the caller. Because a summary seeded by [`MeanVar::from_prior`]
    /// with a positive variance already has a usable spread, it standardizes from the very
    /// first query, which lets a declared reference produce a z-score before any observed
    /// data has arrived.
    #[must_use]
    pub fn zscore(&self, x: f64) -> Option<f64> {
        if self.count <= 0.0 {
            return None;
        }
        let s = self.std();
        if !s.is_finite() || s <= 0.0 {
            return None;
        }
        let z = (x - self.mean) / s;
        if z.is_finite() { Some(z) } else { None }
    }

    /// Reduce confidence by scaling the count and `m2` by `factor`, leaving the mean and
    /// variance unchanged.
    ///
    /// `factor` is clamped to `[0, 1]`. Because the variance is `m2 / count` and both are
    /// scaled by the same factor, the mean and variance are preserved while the effective
    /// count backing them shrinks. A non-finite factor is treated as `0`.
    pub fn decay(&mut self, factor: f64) {
        let f = if factor.is_finite() {
            factor.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.count *= f;
        self.m2 *= f;
    }
}

impl Default for MeanVar {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::{Distribution, Normal};

    /// Population mean and variance computed directly (two-pass, f64) for reference.
    fn batch_mean_var(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        (mean, var)
    }

    fn from_slice(xs: &[f64]) -> MeanVar {
        let mut mv = MeanVar::new();
        for &x in xs {
            mv.push(x);
        }
        mv
    }

    #[test]
    fn merge_is_commutative() {
        let a = from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let b = from_slice(&[10.0, 11.0, 9.0]);
        let ab = MeanVar::merge(&a, &b);
        let ba = MeanVar::merge(&b, &a);
        assert_abs_diff_eq!(ab.count(), ba.count(), epsilon = 1e-9);
        assert_abs_diff_eq!(ab.mean(), ba.mean(), epsilon = 1e-9);
        assert_abs_diff_eq!(ab.variance(), ba.variance(), epsilon = 1e-9);
    }

    #[test]
    fn merge_is_associative() {
        let a = from_slice(&[1.0, 2.0, 3.0]);
        let b = from_slice(&[10.0, 12.0]);
        let c = from_slice(&[100.0, 101.0, 102.0, 103.0]);
        let left = MeanVar::merge(&MeanVar::merge(&a, &b), &c);
        let right = MeanVar::merge(&a, &MeanVar::merge(&b, &c));
        assert_abs_diff_eq!(left.count(), right.count(), epsilon = 1e-9);
        assert_abs_diff_eq!(left.mean(), right.mean(), epsilon = 1e-9);
        assert_abs_diff_eq!(left.variance(), right.variance(), epsilon = 1e-9);
    }

    #[test]
    fn streaming_equals_batch() {
        let xs = [3.1, 4.1, 5.9, 2.6, 5.3, 5.8, 9.7, 9.3];
        let mv = from_slice(&xs);
        let (mean, var) = batch_mean_var(&xs);
        assert_abs_diff_eq!(mv.count(), xs.len() as f64, epsilon = 1e-9);
        assert_abs_diff_eq!(mv.mean(), mean, epsilon = 1e-9);
        assert_abs_diff_eq!(mv.variance(), var, epsilon = 1e-9);
    }

    #[test]
    fn merging_count_one_summaries_equals_pushing() {
        let xs = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let pushed = from_slice(&xs);

        // Merge a bag of count-1 summaries in a different order.
        let singles: Vec<MeanVar> = xs
            .iter()
            .map(|&x| MeanVar {
                count: 1.0,
                mean: x,
                m2: 0.0,
            })
            .collect();
        let mut merged = MeanVar::new();
        for s in singles.iter().rev() {
            merged = MeanVar::merge(&merged, s);
        }

        assert_abs_diff_eq!(merged.count(), pushed.count(), epsilon = 1e-9);
        assert_abs_diff_eq!(merged.mean(), pushed.mean(), epsilon = 1e-9);
        assert_abs_diff_eq!(merged.variance(), pushed.variance(), epsilon = 1e-9);
    }

    #[test]
    fn from_prior_shifts_at_one_over_n0_plus_n() {
        // A prior centred at m0 with pseudo-count n0; one observation x should move
        // the mean by exactly (x - m0) / (n0 + 1).
        let m0 = 0.35;
        let n0 = 9.0;
        let x = 0.95;
        let mut mv = MeanVar::from_prior(m0, 0.01, n0);
        assert_abs_diff_eq!(mv.count(), n0, epsilon = 1e-12);
        assert_abs_diff_eq!(mv.mean(), m0, epsilon = 1e-12);
        mv.push(x);
        let expected = m0 + (x - m0) / (n0 + 1.0);
        assert_abs_diff_eq!(mv.mean(), expected, epsilon = 1e-12);
        assert_abs_diff_eq!(mv.count(), n0 + 1.0, epsilon = 1e-12);
    }

    #[test]
    fn from_prior_recovers_declared_variance() {
        let mv = MeanVar::from_prior(0.35, 0.0049, 9.0);
        assert_abs_diff_eq!(mv.variance(), 0.0049, epsilon = 1e-12);
        assert_abs_diff_eq!(mv.std(), 0.07, epsilon = 1e-9);
    }

    #[test]
    fn from_prior_rejects_degenerate_inputs() {
        assert_eq!(MeanVar::from_prior(1.0, 1.0, 0.0), MeanVar::new());
        assert_eq!(MeanVar::from_prior(1.0, 1.0, -3.0), MeanVar::new());
        assert_eq!(MeanVar::from_prior(f64::NAN, 1.0, 1.0), MeanVar::new());
        assert_eq!(MeanVar::from_prior(1.0, f64::INFINITY, 1.0), MeanVar::new());
    }

    #[test]
    fn decay_halves_count_and_preserves_mean_and_variance() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut mv = from_slice(&xs);
        let (mean0, var0, count0) = (mv.mean(), mv.variance(), mv.count());
        mv.decay(0.5);
        assert_abs_diff_eq!(mv.count(), count0 * 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(mv.mean(), mean0, epsilon = 1e-12);
        assert_abs_diff_eq!(mv.variance(), var0, epsilon = 1e-12);
    }

    #[test]
    fn decay_clamps_factor() {
        let mut mv = from_slice(&[1.0, 2.0, 3.0]);
        let c = mv.count();
        mv.decay(2.0); // clamps to 1.0
        assert_abs_diff_eq!(mv.count(), c, epsilon = 1e-12);
        mv.decay(-1.0); // clamps to 0.0
        assert_abs_diff_eq!(mv.count(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn decay_treats_non_finite_factor_as_zero() {
        let mut mv = from_slice(&[1.0, 2.0, 3.0]);
        mv.decay(f64::NAN);
        assert_abs_diff_eq!(mv.count(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn default_equals_new() {
        assert_eq!(MeanVar::default(), MeanVar::new());
    }

    #[test]
    fn numerically_stable_at_large_offset() {
        // One million values at a large offset with unit-scale noise. The naive
        // single-pass formula E[x²] - E[x]² catastrophically cancels here; f64 + Chan
        // does not. This is why the summary uses Chan's parallel formulas, not the
        // naive sum of squares (§8).
        let offset = 1.0e9_f64;
        let n = 1_000_000usize;
        let mut rng = ChaCha8Rng::seed_from_u64(0x0FFE_u64);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let mut mv = MeanVar::new();
        let mut data_mean_acc = 0.0f64;
        let mut samples: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            let noise: f64 = normal.sample(&mut rng);
            let x = offset + noise;
            mv.push(x);
            data_mean_acc += x;
            samples.push(x);
        }

        // Two-pass reference variance (accurate in f64).
        let ref_mean = data_mean_acc / n as f64;
        let ref_var = samples.iter().map(|x| (x - ref_mean).powi(2)).sum::<f64>() / n as f64;

        // Chan matches the accurate two-pass variance closely, and the variance is
        // near the noise variance of 1.0 (not collapsed to 0 or blown up).
        assert_abs_diff_eq!(mv.variance(), ref_var, epsilon = 1e-3);
        assert!(
            (mv.variance() - 1.0).abs() < 0.02,
            "variance should be ~1.0, got {}",
            mv.variance()
        );

        // Demonstrate the naive formula fails: accumulate Σx² and form E[x²]-E[x]².
        let sum: f64 = samples.iter().sum();
        let sum_sq: f64 = samples.iter().map(|x| x * x).sum();
        let naive_var = sum_sq / n as f64 - (sum / n as f64).powi(2);
        assert!(
            (naive_var - 1.0).abs() > 0.1,
            "naive single-pass variance should be badly wrong at this offset, got {naive_var}"
        );
    }

    #[test]
    fn zscore_guards_against_degenerate_baselines() {
        // Empty: count 0.
        assert_eq!(MeanVar::new().zscore(1.0), None);

        // One streaming observation: count 1, zero variance.
        let mut one = MeanVar::new();
        one.push(5.0);
        assert_eq!(one.zscore(7.0), None);

        // Many identical values: positive count, zero variance.
        let flat = from_slice(&[3.0, 3.0, 3.0, 3.0]);
        assert_eq!(flat.zscore(3.0), None);
        assert_eq!(flat.zscore(9.0), None);
    }

    #[test]
    fn zscore_returns_none_for_non_finite_std() {
        // A declared prior with an enormous variance overflows m2 (`variance *
        // pseudo_count`) to infinity, so the std is non-finite. A non-finite std has no
        // usable scale, so zscore must return None. This pins the `!s.is_finite()` arm of
        // the guard as load-bearing: it is NOT redundant with the `s <= 0.0` arm. Were the
        // `||` an `&&`, an infinite std would slip through and `(x - mean) / inf = 0.0`
        // would be returned as a spurious finite z-score.
        let mv = MeanVar::from_prior(0.0, f64::MAX, 2.0);
        assert!(
            !mv.std().is_finite(),
            "an overflowed variance should give a non-finite std"
        );
        assert_eq!(mv.zscore(0.5), None);
    }

    #[test]
    fn zscore_is_finite_and_correct_when_well_posed() {
        let mv = from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        // mean = 3, population variance = 2, std = sqrt(2).
        let z = mv.zscore(3.0 + 2.0f64.sqrt()).unwrap();
        assert_abs_diff_eq!(z, 1.0, epsilon = 1e-9);
        assert!(mv.zscore(100.0).unwrap().is_finite());
    }

    #[test]
    fn zscore_works_from_declared_prior_at_low_count() {
        // A declared good-score reference (positive variance) standardizes from the
        // first query even at count 1, powering D^abs at cold start (§4).
        let mv = MeanVar::from_prior(0.35, 0.0049, 1.0);
        let z = mv.zscore(0.49).unwrap(); // (0.49 - 0.35) / 0.07 = 2.0
        assert_abs_diff_eq!(z, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn empty_merge_is_identity() {
        let a = from_slice(&[2.0, 4.0, 6.0]);
        let e = MeanVar::new();
        let left = MeanVar::merge(&e, &a);
        let right = MeanVar::merge(&a, &e);
        assert_abs_diff_eq!(left.mean(), a.mean(), epsilon = 1e-12);
        assert_abs_diff_eq!(left.variance(), a.variance(), epsilon = 1e-12);
        assert_abs_diff_eq!(right.mean(), a.mean(), epsilon = 1e-12);
        assert_abs_diff_eq!(right.variance(), a.variance(), epsilon = 1e-12);
    }
}
