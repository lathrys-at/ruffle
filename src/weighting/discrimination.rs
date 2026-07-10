//! Per-channel discrimination: how well each channel ranks on this query.
//!
//! Two statistics combine into one weight. Separation measures whether the channel can
//! rank at all: a scale-free, shift-free ratio of the extreme top's elevation to the
//! bulk's own scale. Absolute goodness measures whether the channel found anything good:
//! this query's top against the channel's own good-score reference. They enter a single
//! AND-like map, so the weight is high only when both hold. The read is conservative
//! throughout: one noisy query cannot zero a channel, and a thin pool is pulled back
//! toward the channel's own running baseline.

use crate::config::DiscriminationConfig;
use crate::ingest::input::Items;
use crate::score::sanitize;
use crate::summary::MeanVar;
use crate::weighting::NEUTRAL_WEIGHT;
use serde::{Deserialize, Serialize};

/// Relative threshold below which the separation denominator counts as collapsed.
///
/// A denominator this small next to the bulk's own span (`q0.75 − q0.10`) carries no
/// usable shape, so the ratio is undefined and the read is dropped. The comparison
/// scale is the bulk's span, not the pool's full range: the full range includes the
/// top's elevation, so an extremely well-separated pool would otherwise be misread as
/// degenerate exactly when it is most informative.
const SEPARATION_SCALE_EPS: f64 = 1e-9;

/// The scale of the combined weight: `1 / (0.5 · 0.5)`, so that both logistic factors at
/// their neutral midpoint give `g = 1.0` exactly, the neutral weight of an average
/// channel in the fusion. This is a fixed property of the map, deliberately independent of
/// [`DiscriminationConfig::g_upper_bound`]: the bound is a pure cap, and tightening it
/// must not silently move the neutral point.
const NEUTRAL_G_SCALE: f64 = 4.0;

/// The neutral value of one logistic factor: `squash(0) = 0.5`, a statistic at its norm.
const NEUTRAL_FACTOR: f64 = 0.5;

/// How well one channel ranked on one query: a combined discrimination weight plus the
/// raw statistics behind it.
///
/// Marked `#[non_exhaustive]`: a result type that callers read but never construct, and
/// it may gain fields over time. Produced by [`discriminate`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ChannelDiscrimination {
    /// The channel's combined discrimination weight: higher when the channel both ranks
    /// its results cleanly and surfaces genuinely good ones. Bounded, and near `1.0` when
    /// the channel performs at its own norm.
    pub g: f64,
    /// The raw top-vs-bulk separation statistic, or `None` when the score pool is too
    /// degenerate to measure it (ranks-only, empty, or a collapsed bulk scale).
    pub raw_separation: Option<f64>,
    /// The fixed-count top-`m` average exported for good-score reference refinement.
    /// `None` for a ranks-only or empty channel, when the average is not finite, or when
    /// the pool is shallower than `top_m`: the statistic is defined as a *fixed-count*
    /// average, and a shallow pool's average over fewer items is not the same quantity,
    /// so it must not refine the reference (the statistic is only comparable at a fixed
    /// depth). The absolute-goodness term for the current query is still read from the
    /// available top, shrunk by how thin the pool is.
    pub top_m_average: Option<f64>,
    /// Whether the bulk had too little scale to measure separation, so the read was
    /// floored.
    pub degenerate_separation: bool,
    /// Whether the good-score reference was too sparse to standardize against, leaving
    /// the absolute-goodness term unavailable this query.
    pub reference_cold: bool,
}

/// Scores how well a channel ranked on one query, from its scores and two running
/// baselines: a `separation` baseline it standardizes its top-vs-bulk read against, and a
/// good-score `reference` for absolute goodness.
///
/// This is one of the building blocks [`Fuser`](crate::Fuser) composes; the `Fuser` reads
/// discrimination for every channel each query, then folds the raw reads back into the
/// baselines afterward. The function is pure and never mutates either baseline.
///
/// The two statistics combine into the single weight [`g`](ChannelDiscrimination::g).
/// Separation measures whether the channel can rank at all: how far its extreme top
/// stands above the bulk, in units of the bulk's own scale, so it is invariant to any
/// rescale or shift of the scores. Absolute goodness measures whether the channel found
/// anything good: this query's top standardized against the reference. They enter an
/// AND-like map, so the weight is high only when both hold.
///
/// Each factor is shrunk toward its own neutral midpoint in proportion to the evidence
/// backing *that* factor. Both are shrunk by the pool size. The separation read is
/// additionally protected by a hard gate: below
/// [`min_count_for_z`](DiscriminationConfig::min_count_for_z) baseline observations the
/// standardized read is neutralized outright. The absolute factor is instead graded by
/// the reference's count, because a declared prior arrives mid-ramp with its own
/// pseudo-count. The shrinkage is per factor rather than global, so a declared good-score
/// reference takes effect from the very first query even while the separation baseline is
/// still cold.
///
/// A ranks-only channel has no scores to read, so it returns the neutral weight `1.0`
/// with no statistics. A scored channel with an empty pool is treated the same way.
#[must_use]
pub fn discriminate<Id>(
    items: &Items<Id>,
    separation: &MeanVar,
    reference: &MeanVar,
    cfg: &DiscriminationConfig,
) -> ChannelDiscrimination {
    let pool = match items {
        Items::Ranks(_) => return neutral(),
        Items::Scored(pool) => pool,
    };
    let n = pool.len();
    if n == 0 {
        return neutral();
    }

    let mut sorted: Vec<f64> = pool.iter().map(|(_, s)| *s).collect();
    sorted.sort_unstable_by(f64::total_cmp);

    // Separation D^sep: the extreme top's elevation in units of the bulk's scale (§4).
    let (raw_separation, degenerate_separation) = separation_statistic(&sorted, cfg);

    // Absolute goodness D^abs: the pool's top, standardized against the channel's
    // good-score reference (§4). A reference too cold to read leaves D^abs unavailable,
    // and the channel is carried by separation alone.
    let m = cfg.top_m.max(1).min(n);
    let s_top = sorted[n - m..].iter().sum::<f64>() / m as f64;
    // The refinement export is the FIXED-count top-m average, so a pool shallower than
    // top_m exports nothing: its average over fewer items is a different statistic and
    // would drift the reference off the depth-comparable quantity D^abs is standardized
    // against (§4). `sanitize` also degrades an overflowed (non-finite) sum to `None`
    // rather than leaking `Some(±inf)`.
    let top_m_average = if n >= cfg.top_m {
        sanitize(s_top)
    } else {
        None
    };
    let d_abs = reference.zscore(s_top);
    let reference_cold = d_abs.is_none();

    // Standardize the separation read within this channel, against its own running
    // baseline (§4). A baseline with too little backing, or one that cannot give a
    // finite z-score, leaves the read neutral and lets shrinkage carry the weight.
    let z_sep = match raw_separation {
        Some(raw) if separation.count() >= cfg.min_count_for_z => {
            separation.zscore(raw).unwrap_or(0.0)
        }
        _ => 0.0,
    };

    // Shrink each factor toward its neutral midpoint by the evidence backing THAT
    // factor (§4). Both share the pool-size factor (a thin pool makes both statistics
    // noisy). The separation factor's baseline-thinness protection is the hard z-gate
    // above (below `min_count_for_z` the read is already neutralized, so a graded
    // backing term would be unobservable); the absolute factor is graded by the
    // reference's count instead, because a declared prior arrives mid-ramp with its own
    // pseudo-count. A brand-new channel with nothing declared ends exactly neutral,
    // which is recall-safe.
    let pool_factor = (n as f64 / cfg.shrink_pool_size as f64).min(1.0);
    let ref_backing = (reference.count() / cfg.min_count_for_z).min(1.0);

    let sep_factor = shrunk_factor(z_sep, pool_factor, cfg.g_slope);
    let abs_factor = match d_abs {
        Some(z) => shrunk_factor(z, pool_factor * ref_backing, cfg.g_slope),
        None => NEUTRAL_FACTOR,
    };
    let g = combine(sep_factor, abs_factor, cfg);

    ChannelDiscrimination {
        g,
        raw_separation,
        top_m_average,
        degenerate_separation,
        reference_cold,
    }
}

/// The neutral read: average weight and no usable statistics, used for a ranks-only
/// channel and for an empty scored pool.
fn neutral() -> ChannelDiscrimination {
    ChannelDiscrimination {
        g: NEUTRAL_WEIGHT,
        raw_separation: None,
        top_m_average: None,
        degenerate_separation: false,
        reference_cold: false,
    }
}

/// Computes the raw separation statistic D^sep over a sorted (ascending) pool.
///
/// Returns `(Some(d_sep), floored)` when the statistic is defined, where `floored` marks
/// that the denominator was widened by the degeneracy floor. Returns `(None, true)` when
/// the bulk is too degenerate to read: fewer than `min_distinct_values` distinct values,
/// or a denominator that collapses even after the floor.
///
/// The numerator is the mean of the top `ceil(top_eps * n)` scores (at least one) minus
/// the median. The denominator is the bulk scale `q0.5 - q0.1`, floored toward a
/// fraction of the inter-quartile range `q0.75 - q0.25` so a near-tied lower bulk cannot
/// inflate the ratio. The whole statistic is a ratio of score differences, so it is
/// invariant to a rescale and to a shift of every score.
fn separation_statistic(sorted: &[f64], cfg: &DiscriminationConfig) -> (Option<f64>, bool) {
    let n = sorted.len();
    if count_distinct(sorted) < cfg.min_distinct_values {
        return (None, true);
    }

    let q10 = quantile_sorted(sorted, 0.10);
    let q25 = quantile_sorted(sorted, 0.25);
    let q50 = quantile_sorted(sorted, 0.50);
    let q75 = quantile_sorted(sorted, 0.75);

    let top_k = ((cfg.top_eps * n as f64).ceil() as usize).clamp(1, n);
    let top_mean = sorted[n - top_k..].iter().sum::<f64>() / top_k as f64;

    let bulk_gap = q50 - q10;
    let floor = cfg.denom_floor_frac * (q75 - q25);
    let floored = floor > bulk_gap;
    let denom = bulk_gap.max(floor);

    // The bulk's own span is the scale to test the denominator against. The pool's full
    // range would include the top's elevation, so a pool whose top stands enormously
    // above a perfectly healthy bulk (the most informative shape there is) would be
    // misread as degenerate. When the bulk span is itself zero, the denominator is zero
    // too and the read is degenerate either way.
    let bulk_span = q75 - q10;
    if denom <= SEPARATION_SCALE_EPS * bulk_span {
        return (None, true);
    }

    // A bulk saturated at extreme scale overflows `top_mean` to a non-finite value while
    // the spread-relative floor check above cannot fire (the spread is at the same MAX
    // scale). Guard the ratio's finiteness so the read degrades to degenerate rather than
    // leaking `Some(±inf)` (§4).
    let ratio = (top_mean - q50) / denom;
    if !ratio.is_finite() {
        return (None, true);
    }

    (Some(ratio), floored)
}

/// Counts distinct values in an ascending-sorted slice. The slice is never empty at any
/// call site, so the count starts at one for the first value.
fn count_distinct(sorted: &[f64]) -> usize {
    let mut distinct = 1usize;
    for pair in sorted.windows(2) {
        if pair[1] != pair[0] {
            distinct += 1;
        }
    }
    distinct
}

/// The `p`-quantile of an ascending-sorted slice by linear interpolation between order
/// statistics (the type-7 method: `h = (n-1) p`, interpolate between `floor(h)` and the
/// next index). The slice must be non-empty.
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n as f64 - 1.0) * p;
    let lo = h.floor() as usize;
    let frac = h - lo as f64;
    if lo + 1 < n {
        sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
    } else {
        sorted[n - 1]
    }
}

/// The logistic squash with slope `k`, mapping a z-score to `(0, 1)` with `f(0) = 0.5`.
/// It is finite for every finite input: a large negative argument drives the exponential
/// to infinity and the result to zero without producing a `NaN`.
///
/// The exponential comes from `libm` rather than `std`: `std::f64::exp` resolves to the
/// platform libm, which can differ in the last ulp between platforms, and this factor
/// reaches the fused weights. `libm` is pure Rust and bit-identical on every target, so
/// identical inputs produce identical weights and rankings everywhere.
fn squash(z: f64, k: f64) -> f64 {
    1.0 / (1.0 + libm::exp(-k * z))
}

/// One statistic's logistic factor, shrunk toward the neutral midpoint `0.5` by how much
/// evidence backs the read.
///
/// `lambda` in `[0, 1]` is the evidence weight: at `1` the factor is the full logistic
/// read `squash(z, k)`, at `0` it is exactly neutral, and in between it interpolates.
/// The shrinkage is per statistic, so a well-backed factor is never dragged neutral by
/// the *other* statistic's thin baseline. `lambda` is bounded without `clamp` so an
/// out-of-range value can never panic.
fn shrunk_factor(z: f64, lambda: f64, k: f64) -> f64 {
    #[allow(clippy::manual_clamp)] // min/max never panics, even on a NaN lambda
    let l = lambda.min(1.0).max(0.0);
    l * squash(z, k) + (1.0 - l) * NEUTRAL_FACTOR
}

/// Combines the two evidence-shrunk factors into the channel weight.
///
/// The two factors multiply, so the weight is small whenever either input is small: a
/// high separation cannot cover for an absolutely poor pool. The product is scaled by
/// the fixed [`NEUTRAL_G_SCALE`] (`4 = 1 / (0.5 · 0.5)`), so both factors at their norm
/// give exactly `1.0` and the fusion stays near plain RRF; the neutral point is a
/// property of the map and does not move when [`DiscriminationConfig::g_upper_bound`]
/// is tuned. The result is bounded into `[g_floor, g_upper_bound]`: the floor keeps an
/// uncertain channel contributing, and the bound keeps any one channel from dominating the
/// order. The bounding uses `min`/`max` rather than `clamp`, so even an inverted
/// floor/bound pair (rejected at construction, but defended here too) resolves
/// deterministically instead of panicking.
fn combine(sep_factor: f64, abs_factor: f64, cfg: &DiscriminationConfig) -> f64 {
    (NEUTRAL_G_SCALE * sep_factor * abs_factor)
        .min(cfg.g_upper_bound)
        .max(cfg.g_floor)
}

/// Normalizes a discrimination weight by the channel's own running `g` level, shrinks
/// the remaining deviation from neutral by `g_deviation_keep`, and re-applies the
/// floor and cap.
///
/// The map behind `g` is neutral at the norm (both statistics at `z = 0` give exactly
/// `1`) but nonlinear, so the expectation of `g` depends on the shape of the channel's
/// read distribution: a right-skewed scorer's occasional large reads saturate toward
/// the cap and drag its mean `g` above neutral while its median sits below,
/// independent of retrieval quality. Under the sum-to-`N` weight normalization that
/// surplus becomes a persistent tax on the other channels. Dividing by the channel's
/// own running mean of raw `g` removes the level without importing any cross-channel
/// fact: both statistics behind `g` are already standardized against the channel's own
/// baselines, so the level is a property of the channel's score shape through the map
/// and nothing else. Persistent cross-channel preference remains the operator's
/// `base_weight`.
///
/// The remaining per-query deviation is then scaled by
/// [`g_deviation_keep`](DiscriminationConfig::g_deviation_keep): how informative the
/// deviation is varies by corpus and scorer family, and the measured do-no-harm
/// optimum across regimes keeps less than all of it. At `0` the weighting reduces to
/// plain RRF.
///
/// The level baseline accumulates RAW `g` reads, never this function's output:
/// feeding the normalized value back would drive the baseline mean toward `1` and the
/// normalization would cancel itself. A level baseline below
/// [`min_count_for_z`](DiscriminationConfig::min_count_for_z) observations, or one
/// whose mean is non-positive or non-finite (only reachable through a hand-authored
/// state), is not trusted: the raw `g` passes through and only the deviation shrink
/// applies, which is the conservative direction while the baseline warms.
pub(crate) fn level_shrunk(g: f64, level: &MeanVar, cfg: &DiscriminationConfig) -> f64 {
    let mean = level.mean();
    let normalized = if level.count() >= cfg.min_count_for_z && mean.is_finite() && mean > 0.0 {
        g / mean
    } else {
        g
    };
    let kept = NEUTRAL_WEIGHT + cfg.g_deviation_keep * (normalized - NEUTRAL_WEIGHT);
    // min/max rather than clamp: even a NaN from a degenerate division resolves into
    // the bounded range instead of panicking or leaking.
    kept.min(cfg.g_upper_bound).max(cfg.g_floor)
}

/// Clamps a raw separation read to the baseline mean plus or minus `winsor_z` standard
/// deviations before it is merged into the separation baseline.
///
/// A single extreme read (a saturated cone gives a large ratio) would otherwise pull the
/// streaming mean the standardization depends on. The clamp applies only once the
/// baseline is well-posed: with too few observations, or a zero or non-finite standard
/// deviation, the read passes through unchanged, since there is no trustworthy band to
/// clamp it to yet.
pub(crate) fn winsorize_separation(
    raw_sep: f64,
    baseline: &MeanVar,
    cfg: &DiscriminationConfig,
) -> f64 {
    if baseline.count() < cfg.min_count_for_z {
        return raw_sep;
    }
    let std = baseline.std();
    if !std.is_finite() || std <= 0.0 {
        return raw_sep;
    }
    let mean = baseline.mean();
    raw_sep.clamp(mean - cfg.winsor_z * std, mean + cfg.winsor_z * std)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::{Direction, GoodScore, OrientedReference};
    use approx::assert_abs_diff_eq;

    fn scored(xs: &[f64]) -> Items<u32> {
        Items::Scored(xs.iter().enumerate().map(|(i, &x)| (i as u32, x)).collect())
    }

    /// A good-score reference baseline seeded to a chosen mean, std, and pseudo-count.
    fn ref_baseline(mu: f64, sigma: f64, n0: f64) -> MeanVar {
        MeanVar::from_prior(mu, sigma * sigma, n0)
    }

    /// A separation baseline filled from the given reads.
    fn sep_baseline(reads: &[f64]) -> MeanVar {
        let mut sep = MeanVar::new();
        for &r in reads {
            sep.push(r);
        }
        sep
    }

    /// A bulk packed into `[0, 10]` with two items spiked far above it.
    fn spiked_pool() -> Vec<f64> {
        let mut v: Vec<f64> = (0..48).map(|i| 10.0 * (i as f64) / 47.0).collect();
        v.push(100.0);
        v.push(100.0);
        v
    }

    fn raw_sep_of(xs: &[f64]) -> f64 {
        discriminate(
            &scored(xs),
            &MeanVar::new(),
            &MeanVar::new(),
            &DiscriminationConfig::default(),
        )
        .raw_separation
        .expect("separation should be defined for this pool")
    }

    // --- ranks-only and empty pools ------------------------------------------------

    #[test]
    fn ranks_only_is_neutral() {
        let items: Items<u32> = Items::Ranks(vec![3, 1, 2]);
        let d = discriminate(
            &items,
            &MeanVar::new(),
            &MeanVar::new(),
            &DiscriminationConfig::default(),
        );
        assert_eq!(d.g, 1.0);
        assert_eq!(d.raw_separation, None);
        assert_eq!(d.top_m_average, None);
        assert!(!d.degenerate_separation);
        assert!(!d.reference_cold);
    }

    #[test]
    fn empty_pool_is_neutral() {
        let items: Items<u32> = Items::Scored(vec![]);
        let d = discriminate(
            &items,
            &MeanVar::new(),
            &ref_baseline(0.3, 0.07, 8.0),
            &DiscriminationConfig::default(),
        );
        assert_eq!(d.g, 1.0);
        assert_eq!(d.raw_separation, None);
        assert_eq!(d.top_m_average, None);
    }

    // --- D^sep scale and shift invariance ------------------------------------------

    #[test]
    fn separation_is_scale_invariant() {
        let base = spiked_pool();
        let up: Vec<f64> = base.iter().map(|x| x * 100.0).collect();
        let down: Vec<f64> = base.iter().map(|x| x * 0.01).collect();
        // An extreme downscale pins the degeneracy guard's scale-invariance too: the
        // guard compares the denominator against the bulk span at a RELATIVE epsilon
        // (`eps * span`), so shrinking every score by 1e-7 changes nothing. Dividing by
        // the span instead would make the threshold grow as the units shrink and
        // misread a perfectly-shaped tiny-units pool as degenerate.
        let tiny: Vec<f64> = base.iter().map(|x| x * 1.0e-7).collect();

        let d0 = raw_sep_of(&base);
        assert_abs_diff_eq!(raw_sep_of(&up), d0, epsilon = 1e-9);
        assert_abs_diff_eq!(raw_sep_of(&down), d0, epsilon = 1e-9);
        assert_abs_diff_eq!(raw_sep_of(&tiny), d0, epsilon = 1e-9);
    }

    #[test]
    fn separation_is_shift_invariant() {
        let base = spiked_pool();
        let shifted: Vec<f64> = base.iter().map(|x| x + 1000.0).collect();
        // A large shift over an order-one bulk forfeits a few digits to f64
        // cancellation, so the tolerance is looser than the rescale test above.
        assert_abs_diff_eq!(raw_sep_of(&shifted), raw_sep_of(&base), epsilon = 1e-7);
    }

    #[test]
    fn separation_tracks_informativeness() {
        // A tight bulk with no standout against the same bulk with a few items lifted
        // far above it: the second is the more informative channel.
        let flat: Vec<f64> = (0..50).map(|i| 0.40 + 0.20 * (i as f64) / 49.0).collect();
        let mut spiked: Vec<f64> = (0..48).map(|i| 0.40 + 0.20 * (i as f64) / 47.0).collect();
        spiked.push(2.0);
        spiked.push(2.0);

        let flat_sep = raw_sep_of(&flat);
        let spiked_sep = raw_sep_of(&spiked);
        assert!(flat_sep > 0.0);
        assert!(
            spiked_sep > flat_sep,
            "spiked {spiked_sep} should exceed flat {flat_sep}"
        );
    }

    // --- degeneracy ----------------------------------------------------------------

    #[test]
    fn too_few_distinct_values_is_degenerate_none() {
        // Five distinct values, below the default minimum of eight.
        let d = discriminate(
            &scored(&[1.0, 2.0, 3.0, 4.0, 5.0]),
            &MeanVar::new(),
            &MeanVar::new(),
            &DiscriminationConfig::default(),
        );
        assert_eq!(d.raw_separation, None);
        assert!(d.degenerate_separation);
        assert!(d.g.is_finite());
    }

    #[test]
    fn collapsed_bulk_is_degenerate_none() {
        // Eight distinct values, but the lower three quarters are tied at zero, so the
        // bulk scale collapses even after the floor.
        let mut pool = vec![0.0; 33];
        pool.extend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let d = discriminate(
            &scored(&pool),
            &MeanVar::new(),
            &MeanVar::new(),
            &DiscriminationConfig::default(),
        );
        assert_eq!(d.raw_separation, None);
        assert!(d.degenerate_separation);
        assert!(d.g.is_finite());
    }

    #[test]
    fn tied_lower_half_is_floored_but_bounded() {
        // An integer-count channel: the lower half is tied at zero (so q0.5 - q0.1 is
        // zero) but the upper half spreads, so the inter-quartile floor keeps the
        // denominator positive and the statistic stays a bounded number.
        let mut pool = vec![0.0; 11];
        pool.extend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let d = discriminate(
            &scored(&pool),
            &MeanVar::new(),
            &MeanVar::new(),
            &DiscriminationConfig::default(),
        );
        let sep = d.raw_separation.expect("floored read is still defined");
        assert!(sep.is_finite() && sep > 0.0);
        assert!(d.degenerate_separation, "the floor was applied");
    }

    #[test]
    fn no_nan_on_degenerate_inputs() {
        let cfg = DiscriminationConfig::default();
        let pools: Vec<Vec<f64>> = vec![
            vec![3.0; 20],            // all equal
            vec![1.0],                // single value
            vec![0.0, 0.0, 0.0, 5.0], // tiny, mostly tied
        ];
        for pool in pools {
            for (sep, refb) in [
                (MeanVar::new(), MeanVar::new()),
                (MeanVar::new(), ref_baseline(0.3, 0.07, 8.0)),
            ] {
                let d = discriminate(&scored(&pool), &sep, &refb, &cfg);
                assert!(d.g.is_finite(), "g must be finite for pool {pool:?}");
                assert!(d.g >= cfg.g_floor && d.g <= cfg.g_upper_bound);
            }
        }
    }

    // --- D^abs closes the separation blind spot ------------------------------------

    #[test]
    fn absolute_goodness_separates_flat_low_from_flat_high() {
        // The same tight pool shifted up by a constant: separation cannot tell the two
        // apart (it is shift-invariant), while absolute goodness reads the second as
        // clearly above the channel's typical top and the first as clearly below it.
        let low: Vec<f64> = (0..12).map(|i| 0.10 + 0.06 * (i as f64) / 11.0).collect();
        let high: Vec<f64> = low.iter().map(|x| x + 0.40).collect();

        // A declared good-score reference: typical top near 0.30, a good match near 0.44.
        let OrientedReference {
            mu_ref: mu,
            sigma_ref: sigma,
        } = GoodScore::new(0.30, 0.44, 8.0)
            .oriented(Direction::HigherIsBetter)
            .unwrap();
        // A populated separation baseline so the weights are not shrunk to neutral; the
        // two pools share the same separation read, so any weight gap is from D^abs.
        let separation = sep_baseline(&[1.5, 2.0, 2.5, 1.8, 2.2]);
        let reference = MeanVar::from_prior(mu, sigma * sigma, 8.0);
        let cfg = DiscriminationConfig::default();

        let d_low = discriminate(&scored(&low), &separation, &reference, &cfg);
        let d_high = discriminate(&scored(&high), &separation, &reference, &cfg);

        // Separation cannot distinguish them.
        assert_abs_diff_eq!(
            d_low.raw_separation.unwrap(),
            d_high.raw_separation.unwrap(),
            epsilon = 1e-7
        );

        // Absolute goodness does, cleanly and with opposite sign.
        let z_low = reference.zscore(d_low.top_m_average.unwrap()).unwrap();
        let z_high = reference.zscore(d_high.top_m_average.unwrap()).unwrap();
        assert!(
            z_low < -1.0,
            "nothing-matches should read below the reference, got {z_low}"
        );
        assert!(
            z_high > 1.0,
            "everything-matches should read above the reference, got {z_high}"
        );
        // The high pool earns more weight than the low one.
        assert!(d_high.g > d_low.g);
    }

    // --- the g map: AND-like, monotone, bounded, neutral ---------------------------

    /// The full (unshrunk) logistic factor for a standardized read at the default slope.
    fn factor(z: f64) -> f64 {
        squash(z, DiscriminationConfig::default().g_slope)
    }

    #[test]
    fn combine_is_neutral_at_the_norm() {
        // Both statistics at the channel's norm map to average weight.
        let g = combine(factor(0.0), factor(0.0), &DiscriminationConfig::default());
        assert_abs_diff_eq!(g, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn neutral_point_is_independent_of_the_upper_bound() {
        // The neutral scale is the fixed 4 = 1/(0.5*0.5), NOT g_upper_bound: tightening
        // the cap must not silently down-weight an at-norm channel relative to the
        // neutral 1.0 a ranks-only channel carries. (Under the old
        // `g_upper_bound * f * f` map, a bound of 2.0 made the norm read 0.5.)
        let cfg = DiscriminationConfig {
            g_upper_bound: 2.0,
            ..Default::default()
        };
        let g = combine(factor(0.0), factor(0.0), &cfg);
        assert_abs_diff_eq!(g, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn combine_is_and_like() {
        // An open floor so the product is visible rather than clamped.
        let cfg = DiscriminationConfig {
            g_floor: 0.0,
            ..Default::default()
        };
        let high_sep_low_abs = combine(factor(5.0), factor(-5.0), &cfg);
        let high_both = combine(factor(5.0), factor(5.0), &cfg);
        let neutral = combine(factor(0.0), factor(0.0), &cfg);

        // A strong separation cannot rescue an absolutely poor pool: the product stays
        // near zero. A sum of the two factors would instead sit near its midpoint.
        assert!(high_sep_low_abs < 0.1, "got {high_sep_low_abs}");
        assert!(high_sep_low_abs < neutral);
        assert!(high_both > 3.5, "got {high_both}");
    }

    #[test]
    fn combine_is_monotone_in_both() {
        let cfg = DiscriminationConfig {
            g_floor: 0.0,
            ..Default::default()
        };
        // Monotone in separation, with absolute goodness held fixed.
        assert!(combine(factor(-2.0), factor(0.0), &cfg) < combine(factor(0.0), factor(0.0), &cfg));
        assert!(combine(factor(0.0), factor(0.0), &cfg) < combine(factor(2.0), factor(0.0), &cfg));
        // Monotone in absolute goodness, with separation held fixed.
        assert!(combine(factor(0.0), factor(-2.0), &cfg) < combine(factor(0.0), factor(0.0), &cfg));
        assert!(combine(factor(0.0), factor(0.0), &cfg) < combine(factor(0.0), factor(2.0), &cfg));
    }

    #[test]
    fn combine_is_bounded() {
        let cfg = DiscriminationConfig::default();
        let hi = combine(factor(100.0), factor(100.0), &cfg);
        let lo = combine(factor(-100.0), factor(-100.0), &cfg);
        assert!(hi.is_finite() && lo.is_finite());
        assert!(hi <= cfg.g_upper_bound && hi >= cfg.g_floor);
        assert!(lo <= cfg.g_upper_bound && lo >= cfg.g_floor);
        assert_abs_diff_eq!(hi, cfg.g_upper_bound, epsilon = 1e-6);
        assert_abs_diff_eq!(lo, cfg.g_floor, epsilon = 1e-12);
    }

    #[test]
    fn combine_never_panics_on_an_inverted_floor_and_bound() {
        // Construction-time validation rejects this pair, but the map itself must also
        // resolve deterministically (min-then-max) rather than panic the way a bare
        // `clamp` would.
        let cfg = DiscriminationConfig {
            g_floor: 5.0,
            g_upper_bound: 4.0,
            ..Default::default()
        };
        let g = combine(factor(0.0), factor(0.0), &cfg);
        assert!(g.is_finite());
    }

    #[test]
    fn shrunk_factor_interpolates_toward_neutral() {
        let k = 1.0;
        // Full evidence: the raw logistic read. No evidence: exactly neutral.
        assert_abs_diff_eq!(shrunk_factor(2.0, 1.0, k), squash(2.0, k), epsilon = 1e-12);
        assert_abs_diff_eq!(shrunk_factor(2.0, 0.0, k), 0.5, epsilon = 1e-12);
        // Halfway evidence: the midpoint of the two.
        let mid = 0.5 * squash(2.0, k) + 0.25;
        assert_abs_diff_eq!(shrunk_factor(2.0, 0.5, k), mid, epsilon = 1e-12);
        // Out-of-range lambda is bounded, never a panic or an extrapolation.
        assert_abs_diff_eq!(shrunk_factor(2.0, 7.0, k), squash(2.0, k), epsilon = 1e-12);
        assert_abs_diff_eq!(shrunk_factor(2.0, -3.0, k), 0.5, epsilon = 1e-12);
    }

    // --- cold start and shrinkage --------------------------------------------------

    #[test]
    fn cold_reference_is_carried_by_separation() {
        // A populated separation baseline but no reference: D^abs is unavailable, and
        // the channel still earns weight from a strongly separated pool.
        let separation = sep_baseline(&[4.0, 4.5, 5.0, 3.5, 4.2, 4.8]);
        let d = discriminate(
            &scored(&spiked_pool()),
            &separation,
            &MeanVar::new(),
            &DiscriminationConfig::default(),
        );
        assert!(d.reference_cold);
        assert!(d.raw_separation.is_some());
        assert!(
            d.g > 1.5,
            "a well-separated query should lift the weight, got {}",
            d.g
        );
    }

    #[test]
    fn tiny_pool_shrinks_toward_neutral() {
        // A tied pool fixes the separation factor to its neutral 0.5 and the absolute
        // read to a known z-score, so the shrunk weight is fully determined and can be
        // checked.
        let cfg = DiscriminationConfig::default();
        let reference = MeanVar::from_prior(0.0, 1.0, 10.0); // mu 0, sigma 1
        let separation = sep_baseline(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // s_top = 2.0 -> d_abs = 2.0. Tiny pool of four (all 2.0): pool_factor = 4/20 =
        // 0.2 and the reference is fully backed, so the absolute factor is the logistic
        // read shrunk by lambda = 0.2 toward 0.5; the separation factor is neutral.
        let f2 = 1.0 / (1.0 + (-2.0f64).exp());
        let tiny = discriminate(&scored(&[2.0; 4]), &separation, &reference, &cfg);
        let expected = 4.0 * 0.5 * (0.2 * f2 + 0.8 * 0.5);
        assert_abs_diff_eq!(tiny.g, expected, epsilon = 1e-9);

        // A large pool of the same value keeps the full (unshrunk) absolute factor.
        let large = discriminate(&scored(&[2.0; 40]), &separation, &reference, &cfg);
        assert_abs_diff_eq!(large.g, 4.0 * 0.5 * f2, epsilon = 1e-9);

        // Shrinkage pulled the tiny-pool weight toward neutral, below the large pool.
        assert!(tiny.g < large.g);
        assert!((tiny.g - 1.0).abs() < (large.g - 1.0).abs());
    }

    #[test]
    fn brand_new_channel_is_neutral() {
        // No separation baseline at all: base_factor is zero, so the weight collapses to
        // neutral regardless of the pool.
        let cfg = DiscriminationConfig::default();
        let d = discriminate(
            &scored(&spiked_pool()),
            &MeanVar::new(),
            &MeanVar::new(),
            &cfg,
        );
        assert_abs_diff_eq!(d.g, 1.0, epsilon = 1e-12);
    }

    // --- winsorization of the separation read --------------------------------------

    #[test]
    fn winsorize_clamps_against_a_well_posed_baseline() {
        let cfg = DiscriminationConfig::default();
        let mut baseline = MeanVar::new();
        for r in [1.0, 2.0, 3.0, 1.5, 2.5, 2.0] {
            baseline.push(r);
        }
        let hi = baseline.mean() + cfg.winsor_z * baseline.std();
        let lo = baseline.mean() - cfg.winsor_z * baseline.std();

        assert_abs_diff_eq!(
            winsorize_separation(100.0, &baseline, &cfg),
            hi,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            winsorize_separation(-100.0, &baseline, &cfg),
            lo,
            epsilon = 1e-9
        );
        // A read already inside the band passes through.
        assert_abs_diff_eq!(
            winsorize_separation(2.0, &baseline, &cfg),
            2.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn winsorize_passes_through_an_unposed_baseline() {
        let cfg = DiscriminationConfig::default();
        // Empty baseline.
        assert_eq!(winsorize_separation(100.0, &MeanVar::new(), &cfg), 100.0);
        // Too few observations to trust the band.
        let mut thin = MeanVar::new();
        thin.push(1.0);
        thin.push(3.0);
        assert_eq!(winsorize_separation(100.0, &thin, &cfg), 100.0);
        // Enough count but zero spread: no band to clamp to.
        let mut flat = MeanVar::new();
        for _ in 0..6 {
            flat.push(2.0);
        }
        assert_eq!(winsorize_separation(100.0, &flat, &cfg), 100.0);
    }

    // --- quantile helper -----------------------------------------------------------

    #[test]
    fn quantile_interpolates_between_order_statistics() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_abs_diff_eq!(quantile_sorted(&xs, 0.5), 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(quantile_sorted(&[0.0, 10.0], 0.1), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(quantile_sorted(&[7.0], 0.9), 7.0, epsilon = 1e-12);
    }

    #[test]
    fn quantile_with_nonzero_fraction_interpolates_exactly() {
        // A non-integer `h` exercises the interpolation arithmetic (the existing 0.5 case
        // lands on an order statistic with frac = 0, hiding it). h = 4*0.3 = 1.2 -> lo=1,
        // frac=0.2 -> sorted[1] + 0.2*(sorted[2]-sorted[1]) = 2 + 0.2*(4-2) = 2.4.
        let xs = [1.0, 2.0, 4.0, 8.0, 16.0];
        assert_abs_diff_eq!(quantile_sorted(&xs, 0.3), 2.4, epsilon = 1e-12);
    }

    #[test]
    fn quantile_at_p_one_returns_the_maximum() {
        // p = 1.0 is the one input that drives `lo` to `n-1`, so `lo + 1 == n`. The guard
        // `lo + 1 < n` must then take the else branch and return the last order statistic;
        // it must NOT enter the interpolation branch and index `sorted[lo + 1]` (out of
        // bounds). This pins the `<` and the `+` in `lo + 1 < n` and the `n - 1` index of
        // the else branch -- each of those mutations either flips into an out-of-bounds
        // index (panic) or returns the wrong element.
        let xs = [1.0, 2.0, 4.0, 8.0, 16.0];
        assert_abs_diff_eq!(quantile_sorted(&xs, 1.0), 16.0, epsilon = 1e-12);
        // The smallest non-trivial slice hits the same edge.
        assert_abs_diff_eq!(quantile_sorted(&[3.0, 9.0], 1.0), 9.0, epsilon = 1e-12);
    }

    // --- separation z-gate, base factor, floor flag, and scale guard ----------------

    #[test]
    fn separation_zscore_gated_by_min_count() {
        // The within-channel z-standardization of the separation read is applied only
        // once the baseline has at least `min_count_for_z` observations; below that the
        // read is neutralized to zero and shrinkage carries the weight (§4). A baseline
        // with count 4 (< default 5) must NOT standardize, even though its mean sits far
        // from the raw read -- so the gate, not the z-score, decides `z_sep`.
        let cfg = DiscriminationConfig::default();
        // Four reads: count 4 < min_count_for_z (5), with spread so a z-score WOULD be
        // defined and large were the gate bypassed.
        let separation = sep_baseline(&[-10.0, -9.0, -11.0, -10.0]);
        assert!(separation.count() < cfg.min_count_for_z);
        let d = discriminate(&scored(&spiked_pool()), &separation, &MeanVar::new(), &cfg);
        // Gate holds: z_sep = 0, so the separation factor sits at its neutral 0.5; the
        // reference is cold so the absolute factor is neutral too, and g = 4*0.5*0.5 =
        // 1.0. Bypassing the gate would standardize a far read and lift g well above 1.0.
        assert_abs_diff_eq!(d.g, 1.0, epsilon = 1e-9);
    }

    #[test]
    fn thin_separation_baseline_does_not_suppress_a_backed_reference() {
        // Shrinkage is PER FACTOR (§4): the separation baseline's thinness shrinks only
        // the separation factor, never the reference-backed absolute factor. Under the
        // old global shrink, a separation count of 2 dragged the whole weight toward
        // neutral and a well-backed D^abs read was mostly discarded.
        let cfg = DiscriminationConfig::default();
        // separation count 2 (< 5): the separation factor is unbacked...
        let separation = MeanVar::from_prior(5.0, 1.0, 2.0);
        // ...but the reference carries count 8 (>= 5): fully backed, zscore(2.0) = 2.0.
        let reference = MeanVar::from_prior(0.0, 1.0, 8.0);
        // Pool of 20 identical 2.0s: pool_factor = 1, separation degenerate (z_sep = 0,
        // a neutral 0.5 factor regardless of lambda), s_top = 2.0 -> d_abs = 2.0.
        let d = discriminate(&scored(&[2.0; 20]), &separation, &reference, &cfg);
        let f2 = 1.0 / (1.0 + (-2.0f64).exp());
        // The absolute factor is the FULL logistic read; only separation sits neutral.
        let expected = 4.0 * 0.5 * f2;
        assert_abs_diff_eq!(d.g, expected, epsilon = 1e-9);
        // Sanity: the read is well above neutral, i.e. not suppressed.
        assert!(d.g > 1.5, "got {}", d.g);
    }

    #[test]
    fn sub_threshold_reference_count_grades_the_absolute_factor() {
        // ref_backing = min(count / min_count_for_z, 1): a declared prior with
        // pseudo-count 2 (< 5) trusts the absolute read at exactly 2/5. The pool of 20
        // identical values fixes the separation factor to neutral and d_abs = 2, so the
        // weight is fully determined: g = 4 * 0.5 * (0.4*f(2) + 0.6*0.5). A backing of
        // `count * min_count` or `pool_factor / ref_backing` saturates the cap at 1 and
        // lands on 2*f(2) instead.
        let cfg = DiscriminationConfig::default();
        let separation = sep_baseline(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let reference = MeanVar::from_prior(0.0, 1.0, 2.0); // pseudo-count 2 < 5
        let d = discriminate(&scored(&[2.0; 20]), &separation, &reference, &cfg);
        let f2 = 1.0 / (1.0 + (-2.0f64).exp());
        let expected = 4.0 * 0.5 * (0.4 * f2 + 0.6 * 0.5);
        assert_abs_diff_eq!(d.g, expected, epsilon = 1e-9);
        // Sanity: strictly between neutral and the fully-backed read.
        assert!(d.g > 1.0 && d.g < 4.0 * 0.5 * f2);
    }

    #[test]
    fn declared_reference_carries_the_channel_at_cold_start() {
        // §4/§8: a declared good-score reference is "the one quantity that can carry the
        // channel before any data arrives". With a stone-cold separation baseline, a
        // flat-low nothing-matches pool must be down-weighted (D^abs strongly negative)
        // and a flat-high everything-matches pool up-weighted, from the very first query.
        let cfg = DiscriminationConfig::default();
        let reference = ref_baseline(0.30, 0.07, 8.0); // declared: mu 0.30, sigma 0.07
        let cold_separation = MeanVar::new();

        // 30 distinct values so separation is defined; far below the reference.
        let low: Vec<f64> = (0..30).map(|i| 0.02 + 0.001 * i as f64).collect();
        let d_low = discriminate(&scored(&low), &cold_separation, &reference, &cfg);
        assert!(
            d_low.g < 0.5,
            "nothing-matches must be down-weighted at cold start, got {}",
            d_low.g
        );

        // The same shape shifted far above the reference.
        let high: Vec<f64> = low.iter().map(|x| x + 0.60).collect();
        let d_high = discriminate(&scored(&high), &cold_separation, &reference, &cfg);
        assert!(
            d_high.g > 1.5,
            "everything-matches must be up-weighted at cold start, got {}",
            d_high.g
        );
    }

    #[test]
    fn shallow_pool_exports_no_reference_read() {
        // A pool shallower than top_m cannot produce the fixed-count top-m average, so
        // nothing is exported for reference refinement (§4's comparable-depth condition);
        // the absolute term for the current query is still read from the available top.
        let cfg = DiscriminationConfig::default(); // top_m = 10
        let reference = ref_baseline(0.0, 1.0, 8.0);
        let d = discriminate(&scored(&[1.0, 2.0, 3.0]), &MeanVar::new(), &reference, &cfg);
        assert_eq!(d.top_m_average, None, "no fixed-count read from 3 items");
        assert!(!d.reference_cold, "the reference itself is warm");
        // At exactly top_m items the fixed-count read is defined again.
        let pool: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let d10 = discriminate(&scored(&pool), &MeanVar::new(), &reference, &cfg);
        assert!(d10.top_m_average.is_some());
    }

    #[test]
    fn floor_equal_to_bulk_gap_is_not_flagged_floored() {
        // The degeneracy floor is "applied" only when it is STRICTLY greater than the bulk
        // gap (`floor > bulk_gap`): at exact equality the bulk gap already supplies the
        // denominator, so nothing was widened and the read is not flagged floored. This
        // pool is built so q50-q10 == 0.5*(q75-q25) exactly. (With `>=` the read would be
        // wrongly flagged degenerate at the boundary.)
        //
        // n = 21 makes the 10/25/50/75th percentiles land on integer indices 2/5/10/15
        // (frac = 0), so the quantiles are exact order statistics: q10=0, q25=5, q50=10,
        // q75=25 -> bulk_gap = 10, floor = 0.5*(25-5) = 10. The fraction is pinned to
        // 0.5 because the pool is constructed for that exact-equality boundary.
        let pool = [
            0.0, 0.0, 0.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0,
            26.0, 27.0, 28.0, 29.0, 30.0,
        ];
        let cfg = DiscriminationConfig {
            denom_floor_frac: 0.5,
            ..Default::default()
        };
        let d = discriminate(&scored(&pool), &MeanVar::new(), &MeanVar::new(), &cfg);
        assert!(d.raw_separation.is_some());
        assert!(
            !d.degenerate_separation,
            "floor == bulk_gap means nothing was widened; not floored"
        );
    }

    #[test]
    fn scale_collapse_guard_uses_the_bulk_span() {
        // The degeneracy guard compares the floored denominator against the BULK's own
        // span `q0.75 - q0.10`, a difference of quantiles. A pool tightly clustered far
        // above zero has a real (if small) bulk shape, so the read is defined; were the
        // guard to use a sum of quantiles, the huge offset would dwarf the denominator
        // and the pool would be wrongly declared degenerate. Large-offset scores are
        // ordinary (unnormalized BM25, log-odds), so this pins the subtraction.
        let pool: Vec<f64> = (0..40).map(|i| 1.0e9 + 0.01 * i as f64).collect();
        let d = discriminate(
            &scored(&pool),
            &MeanVar::new(),
            &MeanVar::new(),
            &DiscriminationConfig::default(),
        );
        assert!(
            d.raw_separation.is_some(),
            "a clustered-but-shaped pool far from zero must read, got None"
        );
        assert!(!d.degenerate_separation);
    }

    #[test]
    fn extreme_top_elevation_is_not_read_as_degenerate() {
        // The guard's comparison scale is the bulk span, NOT the pool's full range: a
        // top standing astronomically above a perfectly healthy bulk is the most
        // informative shape there is, and must read as a (huge) separation, not as
        // degeneracy. Under the old full-range basis, `1e-9 * (max - min)` exceeded the
        // healthy bulk denominator here and the read was dropped.
        let mut pool: Vec<f64> = (0..40).map(|i| i as f64 * 0.025).collect(); // bulk [0, 1)
        pool.push(1.0e10);
        pool.push(1.0e10);
        let d = discriminate(
            &scored(&pool),
            &MeanVar::new(),
            &MeanVar::new(),
            &DiscriminationConfig::default(),
        );
        let sep = d
            .raw_separation
            .expect("an enormously separated pool must read");
        assert!(sep > 1.0e6, "got {sep}");
        assert!(!d.degenerate_separation);
    }

    #[test]
    fn winsorize_clamps_at_exactly_min_count_for_z() {
        // The pass-through applies only BELOW min_count_for_z; a baseline with count
        // exactly at the threshold is well-posed and must clamp (boundary `count <
        // min_count_for_z`, not `<=`).
        let cfg = DiscriminationConfig::default(); // min_count_for_z = 5.0
        let mut baseline = MeanVar::new();
        for r in [1.0, 2.0, 3.0, 4.0, 5.0] {
            baseline.push(r);
        }
        assert_abs_diff_eq!(baseline.count(), cfg.min_count_for_z, epsilon = 1e-12);
        let hi = baseline.mean() + cfg.winsor_z * baseline.std();
        // A read far above the band is clamped to `hi`, not passed through unchanged.
        let clamped = winsorize_separation(100.0, &baseline, &cfg);
        assert_abs_diff_eq!(clamped, hi, epsilon = 1e-9);
        assert!(clamped < 100.0);
    }

    // --- level_shrunk: own-level normalization and deviation shrink ---

    /// A level baseline seeded to a chosen mean with enough backing to pass the gate.
    fn level_at(mean: f64) -> MeanVar {
        MeanVar::from_prior(mean, 0.01, 10.0)
    }

    #[test]
    fn level_shrunk_cold_baseline_shrinks_the_raw_deviation() {
        // Below the evidence gate the raw g passes through unnormalized, and only the
        // deviation shrink applies: the conservative direction while the level warms.
        let cfg = DiscriminationConfig {
            g_deviation_keep: 0.6,
            ..DiscriminationConfig::default()
        };
        let cold = MeanVar::new();
        assert_abs_diff_eq!(level_shrunk(1.5, &cold, &cfg), 1.3, epsilon = 1e-12);
        assert_abs_diff_eq!(level_shrunk(1.0, &cold, &cfg), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn level_shrunk_warm_baseline_divides_then_shrinks() {
        let cfg = DiscriminationConfig {
            g_deviation_keep: 0.6,
            ..DiscriminationConfig::default()
        };
        // 1.5 / 1.25 = 1.2, then 1 + 0.6 * 0.2 = 1.12.
        assert_abs_diff_eq!(
            level_shrunk(1.5, &level_at(1.25), &cfg),
            1.12,
            epsilon = 1e-12
        );
        // A channel reading exactly at its own level is exactly neutral: the persistent
        // shape tilt is gone by construction.
        assert_abs_diff_eq!(
            level_shrunk(1.25, &level_at(1.25), &cfg),
            1.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn level_shrunk_keep_endpoints() {
        // keep = 0 reduces the weighting to plain RRF; keep = 1 uses the normalized
        // deviation as is.
        let zero = DiscriminationConfig {
            g_deviation_keep: 0.0,
            ..DiscriminationConfig::default()
        };
        assert_abs_diff_eq!(
            level_shrunk(3.7, &level_at(0.9), &zero),
            1.0,
            epsilon = 1e-12
        );
        let one = DiscriminationConfig {
            g_deviation_keep: 1.0,
            ..DiscriminationConfig::default()
        };
        assert_abs_diff_eq!(
            level_shrunk(1.8, &level_at(1.2), &one),
            1.5,
            epsilon = 1e-12
        );
    }

    #[test]
    fn level_shrunk_reapplies_floor_and_cap() {
        let cfg = DiscriminationConfig {
            g_deviation_keep: 1.0,
            ..DiscriminationConfig::default()
        };
        // A tiny hand-authored level blows the ratio up; the cap holds.
        assert_abs_diff_eq!(
            level_shrunk(4.0, &level_at(0.05), &cfg),
            cfg.g_upper_bound,
            epsilon = 1e-12
        );
        // A huge one collapses it; the floor holds.
        assert_abs_diff_eq!(
            level_shrunk(0.25, &level_at(100.0), &cfg),
            cfg.g_floor,
            epsilon = 1e-12
        );
    }

    #[test]
    fn level_shrunk_nonpositive_or_nonfinite_mean_passes_raw_through() {
        // Only reachable through a hand-authored state; the read degrades to the cold
        // path (raw g, deviation shrink) rather than dividing by a junk level.
        let cfg = DiscriminationConfig {
            g_deviation_keep: 0.5,
            ..DiscriminationConfig::default()
        };
        assert_abs_diff_eq!(
            level_shrunk(1.4, &level_at(0.0), &cfg),
            1.2,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            level_shrunk(1.4, &level_at(-2.0), &cfg),
            1.2,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            level_shrunk(1.4, &level_at(f64::NAN), &cfg),
            1.2,
            epsilon = 1e-12
        );
    }
}
