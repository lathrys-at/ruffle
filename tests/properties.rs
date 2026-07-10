//! Property-based tests for `ruffle`'s invariants (proptest), over randomized inputs.
//!
//! Each property targets an invariant the derivation states (§2, §4, §5, §6, §7, §8). The
//! goal is the coverage example tests miss: arbitrary finite sequences, arbitrary pools,
//! arbitrary compatible states. Where a property legitimately does not apply to a
//! degenerate input we `prop_assume!` past it and document why. Input sizes are bounded
//! and case counts are modest so the suite stays fast.
//!
//! This is an integration test, so it sees only the public surface; `linalg` and
//! `winsorize_separation` are private and are exercised indirectly.

use proptest::prelude::*;
use ruffle::components::{PairBaseline, coupled_weights, discriminate, weighted_rrf};
use ruffle::{
    BaselineMode, ChannelConfig, ChannelId, ChannelInput, ChannelSummary, CouplingConfig,
    Direction, DiscriminationConfig, Items, MeanVar, MergePolicy, Mismatch, PairSummary, RrfConfig,
    RuffleState, Score, StatFingerprint, UnorderedPair,
};
use std::collections::{BTreeMap, BTreeSet, HashMap};

// ---------------------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------------------

/// A caller-side newtype: the only way a bare number becomes a [`Score`] (§7).
struct S(f64);
impl Score for S {
    fn value(&self) -> f64 {
        self.0
    }
}

/// Build a [`MeanVar`] by streaming the given finite values in.
fn mv_from(xs: &[f64]) -> MeanVar {
    let mut m = MeanVar::new();
    for &x in xs {
        m.push(x);
    }
    m
}

/// Relative-or-absolute closeness: `|a - b| <= tol * max(|a|, |b|, 1)`. The `max(_, 1)`
/// floor gives near-zero values an absolute tolerance and avoids a divide-by-zero.
fn rel_close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * a.abs().max(b.abs()).max(1.0)
}

/// A deterministic Fisher–Yates shuffle (fixed-seed LCG) so "reordering" properties are
/// reproducible without pulling a live RNG into the test.
fn lcg_shuffle<T>(mut v: Vec<T>, mut seed: u64) -> Vec<T> {
    for i in (1..v.len()).rev() {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (seed >> 33) as usize % (i + 1);
        v.swap(i, j);
    }
    v
}

/// A bounded finite f64, kept small so accumulated rounding stays tight in
/// tolerance-sensitive properties.
fn small_f64() -> impl Strategy<Value = f64> {
    -1.0e3f64..1.0e3f64
}

/// A wild-but-overflow-safe finite f64. The `1e150` cap is deliberate: `(1e150)²` is
/// `1e300`, so a few hundred of them sum without overflowing to `inf`. The numerical
/// guards target NaN/degeneracy, not f64 overflow, so probing past the overflow wall
/// would test something the design never promised.
fn wild_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        -1.0e3f64..1.0e3f64,
        -1.0e6f64..1.0e6f64,
        -1.0e150f64..1.0e150f64,
        Just(0.0f64),
    ]
}

fn key(i: usize) -> String {
    format!("c{i}")
}

/// An [`RrfConfig`] with a chosen rank constant `η`, for the fusion calls below.
fn rrf(eta: f64) -> RrfConfig {
    let mut c = RrfConfig::default();
    c.rrf_eta = eta;
    c
}

// =======================================================================================
// MeanVar (§8)
// =======================================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// merge is COMMUTATIVE and ASSOCIATIVE over random finite sequences: mean, variance,
    /// and count match to a tolerance regardless of grouping or order.
    #[test]
    fn meanvar_merge_commutative_associative(
        xs in prop::collection::vec(small_f64(), 0..60),
        ys in prop::collection::vec(small_f64(), 0..60),
        zs in prop::collection::vec(small_f64(), 0..60),
    ) {
        let (a, b, c) = (mv_from(&xs), mv_from(&ys), mv_from(&zs));

        // Commutative.
        let ab = MeanVar::merge(&a, &b);
        let ba = MeanVar::merge(&b, &a);
        prop_assert!(rel_close(ab.count(), ba.count(), 1e-9));
        prop_assert!(rel_close(ab.mean(), ba.mean(), 1e-9));
        prop_assert!(rel_close(ab.variance(), ba.variance(), 1e-7));

        // Associative.
        let left = MeanVar::merge(&MeanVar::merge(&a, &b), &c);
        let right = MeanVar::merge(&a, &MeanVar::merge(&b, &c));
        prop_assert!(rel_close(left.count(), right.count(), 1e-9));
        prop_assert!(rel_close(left.mean(), right.mean(), 1e-9));
        prop_assert!(rel_close(left.variance(), right.variance(), 1e-7));
    }

    /// Streaming push of a sequence equals the batch (two-pass) mean and population
    /// variance.
    #[test]
    fn meanvar_streaming_equals_batch(xs in prop::collection::vec(small_f64(), 1..200)) {
        let mv = mv_from(&xs);
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        prop_assert!(rel_close(mv.count(), n, 1e-12));
        prop_assert!(rel_close(mv.mean(), mean, 1e-9));
        prop_assert!(rel_close(mv.variance(), var, 1e-7));
    }

    /// `from_prior(m, v, n0)` then one push moves the mean toward the datum at exactly the
    /// documented rate `1 / (n0 + 1)`; the result stays between prior and datum, and the
    /// variance stays finite and non-negative.
    #[test]
    fn meanvar_from_prior_shifts_at_documented_rate(
        m in small_f64(),
        x in small_f64(),
        v in 0.0f64..1.0e3f64,
        n0 in 1.0e-3f64..1.0e4f64,
    ) {
        let mut mv = MeanVar::from_prior(m, v, n0);
        // from_prior with a positive finite pseudo-count seeds exactly this.
        prop_assert!(rel_close(mv.count(), n0, 1e-9));
        prop_assert!(rel_close(mv.mean(), m, 1e-9));

        mv.push(x);
        let expected = m + (x - m) / (n0 + 1.0);
        prop_assert!(rel_close(mv.mean(), expected, 1e-9));
        prop_assert!(rel_close(mv.count(), n0 + 1.0, 1e-9));

        // The new mean lies on the segment [m, x] (moves toward the datum, never past it).
        let lo = m.min(x);
        let hi = m.max(x);
        prop_assert!(mv.mean() >= lo - 1e-6 && mv.mean() <= hi + 1e-6);

        prop_assert!(mv.variance().is_finite() && mv.variance() >= 0.0);
    }

    /// `decay(f)` for f in [0, 1] preserves mean and variance and scales the count by f.
    /// At f = 0 the count is 0, where variance is definitionally guarded to 0 (no spread
    /// to report), so that single point is excluded from the variance check only.
    #[test]
    fn meanvar_decay_preserves_mean_variance_scales_count(
        xs in prop::collection::vec(small_f64(), 0..60),
        f in 0.0f64..=1.0f64,
    ) {
        let mv0 = mv_from(&xs);
        let (mean0, var0, count0) = (mv0.mean(), mv0.variance(), mv0.count());
        let mut mv = mv0;
        mv.decay(f);

        prop_assert!(rel_close(mv.count(), count0 * f, 1e-9));
        // The mean field is never touched by decay.
        prop_assert!(rel_close(mv.mean(), mean0, 1e-12));
        // Variance is preserved wherever the decayed count is still positive.
        if mv.count() > 0.0 {
            prop_assert!(rel_close(mv.variance(), var0, 1e-9));
        }
    }

    /// For ANY sequence of finite pushes, variance() is finite and >= 0, std() is finite,
    /// and zscore() is either None or finite — never NaN/inf. Stressed with wild-but-
    /// overflow-safe magnitudes.
    #[test]
    fn meanvar_is_never_nan_or_inf(
        xs in prop::collection::vec(wild_f64(), 0..200),
        q in wild_f64(),
    ) {
        let mv = mv_from(&xs);
        prop_assert!(mv.mean().is_finite());
        prop_assert!(mv.variance().is_finite() && mv.variance() >= 0.0);
        prop_assert!(mv.std().is_finite());
        if let Some(z) = mv.zscore(q) {
            prop_assert!(z.is_finite());
        }
    }
}

// =======================================================================================
// Orientation / sign-flip invariance (§7)
// =======================================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// A HigherIsBetter channel observing native scores `s` and a LowerIsBetter channel
    /// observing native scores `-s` orient to the SAME canonical pool, so
    /// `ChannelInput::scored` produces identical `Items` and `discriminate` returns an
    /// identical read (the rank/separation stats are sign-flip invariant, §7). Equality
    /// is exact because `-(-x) == x` for every finite f64.
    #[test]
    fn discriminate_is_sign_flip_invariant(
        scores in prop::collection::vec(-1.0e6f64..1.0e6f64, 0..40),
        sep in prop::collection::vec(-50.0f64..50.0f64, 0..10),
    ) {
        let tag = "t".to_string();
        let hi = ChannelConfig::new(ChannelId::new(key(0), tag.clone()), Direction::HigherIsBetter, None);
        let lo = ChannelConfig::new(ChannelId::new(key(0), tag.clone()), Direction::LowerIsBetter, None);

        let pos: Vec<(u32, S)> =
            scores.iter().enumerate().map(|(i, &s)| (i as u32, S(s))).collect();
        let neg: Vec<(u32, S)> =
            scores.iter().enumerate().map(|(i, &s)| (i as u32, S(-s))).collect();

        let obs_hi = ChannelInput::scored(&hi, pos);
        let obs_lo = ChannelInput::scored(&lo, neg);

        // Orientation negates the LowerIsBetter scores back to the HigherIsBetter pool.
        prop_assert_eq!(&obs_hi.items, &obs_lo.items);

        let separation = mv_from(&sep);
        let cfg = DiscriminationConfig::default();
        let d_hi = discriminate(&obs_hi.items, &separation, &MeanVar::new(), &cfg);
        let d_lo = discriminate(&obs_lo.items, &separation, &MeanVar::new(), &cfg);
        prop_assert_eq!(d_hi, d_lo);
    }

    /// A channel's RRF contribution is identical whether its scores are fed as
    /// HigherIsBetter `s` or LowerIsBetter `-s` through `ChannelInput::scored` (same fused
    /// order and scores), since both orient to the same canonical pool.
    #[test]
    fn weighted_rrf_is_sign_flip_invariant(
        scores in prop::collection::vec(-1.0e6f64..1.0e6f64, 0..40),
        eta in 0.0f64..200.0f64,
    ) {
        let tag = "t".to_string();
        let hi = ChannelConfig::new(ChannelId::new(key(0), tag.clone()), Direction::HigherIsBetter, None);
        let lo = ChannelConfig::new(ChannelId::new(key(0), tag), Direction::LowerIsBetter, None);

        let pos: Vec<(u32, S)> =
            scores.iter().enumerate().map(|(i, &s)| (i as u32, S(s))).collect();
        let neg: Vec<(u32, S)> =
            scores.iter().enumerate().map(|(i, &s)| (i as u32, S(-s))).collect();

        let out_hi = weighted_rrf(&[ChannelInput::scored(&hi, pos)], &BTreeMap::new(), &rrf(eta));
        let out_lo = weighted_rrf(&[ChannelInput::scored(&lo, neg)], &BTreeMap::new(), &rrf(eta));
        prop_assert_eq!(out_hi, out_lo);
    }
}

// =======================================================================================
// D^sep scale + shift invariance (§4)
// =======================================================================================

/// The raw separation D^sep of a pool of scores (computed against a cold baseline, which
/// does not enter the raw statistic).
fn raw_sep(scores: &[f64]) -> Option<f64> {
    let items: Items<u32> = Items::Scored(
        scores
            .iter()
            .enumerate()
            .map(|(i, &s)| (i as u32, s))
            .collect(),
    );
    discriminate(
        &items,
        &MeanVar::new(),
        &MeanVar::new(),
        &DiscriminationConfig::default(),
    )
    .raw_separation
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// For a pool whose separation is defined, `raw_separation` is invariant (to ~1e-6
    /// relative) under multiplying every score by c > 0 and adding a constant d — it is a
    /// ratio of score differences, so both the scale and the shift cancel.
    #[test]
    fn separation_is_scale_and_shift_invariant(
        base in prop::collection::vec(-50.0f64..50.0f64, 8..40),
        c in 0.01f64..100.0f64,
        d in -50.0f64..50.0f64,
    ) {
        // Skip pools too degenerate to carry a separation read (the property is about an
        // affine transform of a *defined* statistic).
        let Some(base_sep) = raw_sep(&base) else { return Ok(()); };
        prop_assume!(base_sep.is_finite());

        let transformed: Vec<f64> = base.iter().map(|x| c * x + d).collect();
        // An affine image of a non-degenerate pool is non-degenerate too; the rare f64
        // collapse is a precision artifact, not a violation, so skip it.
        let Some(t_sep) = raw_sep(&transformed) else { return Ok(()); };

        prop_assert!(
            rel_close(t_sep, base_sep, 1e-6),
            "base {base_sep} vs transformed {t_sep} (c={c}, d={d})"
        );
    }
}

// =======================================================================================
// Weights (§5.4, §6, §9)
// =======================================================================================

/// A `g` map over keys `c0..cN` from a vector of positive discriminations.
fn gmap(gs: &[f64]) -> BTreeMap<String, f64> {
    gs.iter().enumerate().map(|(i, &g)| (key(i), g)).collect()
}

/// Build a pair-redundancy map over keys `c0..cN` from `(i, j, mean, variance, count)`
/// tuples, keeping only valid distinct in-range pairs.
fn pairmap(
    n: usize,
    specs: &[(usize, usize, f64, f64, f64)],
) -> BTreeMap<UnorderedPair<String>, PairBaseline> {
    let mut pairs = BTreeMap::new();
    for &(i, j, mean, var, count) in specs {
        if i < n && j < n && i != j {
            pairs.insert(
                UnorderedPair::new(key(i), key(j)),
                PairBaseline {
                    redundancy: MeanVar::from_prior(mean, var.max(0.0), count),
                    refreshes: 2.0,
                },
            );
        }
    }
    pairs
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Over random positive `g`, random pairs, and an arbitrary (valid) coupling config:
    /// every weight is non-negative and finite, the weights sum to N, and the effective
    /// channel count is finite and strictly positive.
    #[test]
    fn coupled_weights_are_nonneg_finite_and_sum_to_n(
        gs in prop::collection::vec(0.01f64..10.0f64, 1..=5),
        pairs in prop::collection::vec(
            (0usize..5, 0usize..5, -0.95f64..0.95f64, 0.0f64..0.3f64, 0.0f64..60.0f64),
            0..8,
        ),
        cfgp in (any::<bool>(), 0.0f64..1.0f64, 0.0f64..1.0f64, 0.0f64..50.0f64, 0.0f64..1.0f64),
    ) {
        let n = gs.len();
        let keys: Vec<String> = (0..n).map(key).collect();
        let g = gmap(&gs);
        let p = pairmap(n, &pairs);
        let (enabled, discount_cap, shrink_to_identity, min_reliability, stratum_var) = cfgp;
        let mut cfg = CouplingConfig::default();
        cfg.enabled = enabled;
        cfg.discount_cap = discount_cap;
        cfg.shrink_to_identity = shrink_to_identity;
        cfg.min_overlap = 1;
        cfg.min_reliability = min_reliability;
        cfg.stratum_stability_max_var = stratum_var;

        let cw = coupled_weights(&g, &p, &keys, &cfg);

        prop_assert_eq!(cw.weights.len(), n);
        for &w in cw.weights.values() {
            prop_assert!(w.is_finite(), "weight not finite: {w}");
            prop_assert!(w >= 0.0, "weight negative: {w}");
        }
        let total: f64 = cw.weights.values().sum();
        prop_assert!(rel_close(total, n as f64, 1e-9), "weights sum {total}, expected {n}");

        // effective_channels is always a finite, strictly-positive count.
        prop_assert!(cw.effective_channels.is_finite());
        prop_assert!(cw.effective_channels > 0.0);
    }

    /// With coupling disabled (the default), the weights are exactly proportional to `g`:
    /// `w_c / w_c' == g_c / g_c'` (checked by cross-multiplication to avoid division). A
    /// reliable redundant pair present in the map must be ignored while disabled.
    #[test]
    fn disabled_coupling_weights_are_proportional_to_g(
        gs in prop::collection::vec(0.01f64..10.0f64, 1..=5),
        pairs in prop::collection::vec(
            (0usize..5, 0usize..5, -0.95f64..0.95f64, 0.0f64..0.3f64, 0.0f64..60.0f64),
            0..8,
        ),
    ) {
        let n = gs.len();
        let keys: Vec<String> = (0..n).map(key).collect();
        let g = gmap(&gs);
        let p = pairmap(n, &pairs);
        // Default config has coupling off.
        let cw = coupled_weights(&g, &p, &keys, &CouplingConfig::default());

        for i in 0..n {
            for j in (i + 1)..n {
                let (wi, wj) = (cw.weights[&keys[i]], cw.weights[&keys[j]]);
                let (gi, gj) = (gs[i], gs[j]);
                // w_i / w_j == g_i / g_j  <=>  w_i * g_j == w_j * g_i.
                prop_assert!(
                    rel_close(wi * gj, wj * gi, 1e-9),
                    "w {wi}/{wj} not proportional to g {gi}/{gj}"
                );
            }
        }
        // Disabled coupling ignores the off-diagonal entirely: N independent channels.
        prop_assert!(rel_close(cw.effective_channels, n as f64, 1e-9));
    }

    /// effective_channels lies in (0, N] for the realistic redundancy regime where the
    /// off-diagonal is a NON-NEGATIVE shared-nuisance correlation (§5.1: redundancy
    /// "becomes positive when the channels share a nuisance factor"). Negative anchor
    /// correlations are not redundancy and can inflate the Kish count above N (a property
    /// of `1ᵀR⁻¹1`), so they are excluded here; finiteness/positivity for arbitrary
    /// correlations is covered by the sum-to-N property above.
    #[test]
    fn effective_channels_in_zero_to_n_for_nonneg_redundancy(
        gs in prop::collection::vec(0.01f64..10.0f64, 1..=5),
        pairs in prop::collection::vec(
            // Non-negative redundancy means, reliable (count high), stable (variance 0).
            (0usize..5, 0usize..5, 0.0f64..0.95f64),
            0..8,
        ),
    ) {
        let n = gs.len();
        let keys: Vec<String> = (0..n).map(key).collect();
        let g = gmap(&gs);
        let specs: Vec<(usize, usize, f64, f64, f64)> =
            pairs.iter().map(|&(i, j, m)| (i, j, m, 0.0, 40.0)).collect();
        let p = pairmap(n, &specs);
        let mut cfg = CouplingConfig::default();
        cfg.enabled = true;
        cfg.min_reliability = 1.0;
        let cw = coupled_weights(&g, &p, &keys, &cfg);
        prop_assert!(cw.effective_channels > 0.0);
        prop_assert!(
            cw.effective_channels <= n as f64 + 1e-9,
            "effective {} exceeds N={n}",
            cw.effective_channels
        );
    }

    /// The clamp never lets a weight go negative, for adversarial inputs: tiny/huge `g`,
    /// redundancy pinned near the cap, and zero shrinkage (which permits a non-PD or
    /// shorting solve). Weights stay non-negative and finite and still sum to N.
    #[test]
    fn weights_never_negative_under_adversarial_coupling(
        gs in prop::collection::vec(prop_oneof![1.0e-6f64..1.0e-3f64, 1.0f64..1.0e3f64], 2..=5),
        pairs in prop::collection::vec(
            (0usize..5, 0usize..5, prop_oneof![-0.95f64..-0.5f64, 0.5f64..0.95f64]),
            0..10,
        ),
        shrink in 0.0f64..0.2f64,
    ) {
        let n = gs.len();
        let keys: Vec<String> = (0..n).map(key).collect();
        let g = gmap(&gs);
        let specs: Vec<(usize, usize, f64, f64, f64)> =
            pairs.iter().map(|&(i, j, m)| (i, j, m, 0.0, 50.0)).collect();
        let p = pairmap(n, &specs);
        let mut cfg = CouplingConfig::default();
        cfg.enabled = true;
        cfg.discount_cap = 0.95;
        cfg.shrink_to_identity = shrink;
        cfg.min_overlap = 1;
        cfg.min_reliability = 1.0;
        cfg.stratum_stability_max_var = 1.0;
        let cw = coupled_weights(&g, &p, &keys, &cfg);
        for &w in cw.weights.values() {
            prop_assert!(w.is_finite() && w >= 0.0, "weight {w} not finite/non-negative");
        }
        let total: f64 = cw.weights.values().sum();
        prop_assert!(rel_close(total, n as f64, 1e-9));
    }
}

// =======================================================================================
// discrimination g bounds (§4)
// =======================================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(192))]

    /// For ANY pool (random, including degenerate) and ANY summary, `discriminate(...).g`
    /// is finite and within `[g_floor, g_upper_bound]`. The numerical guards must keep a
    /// NaN from ever reaching the clamp (a NaN would pass through `f64::clamp`).
    #[test]
    fn discrimination_g_is_finite_and_bounded(
        pool in prop::collection::vec(wild_f64(), 0..50),
        sep in prop::collection::vec(small_f64(), 0..20),
        refp in (small_f64(), 0.0f64..1.0e3f64, -5.0f64..50.0f64),
        ranks_only in any::<bool>(),
    ) {
        let (rmean, rvar, rcount) = refp;
        let separation = mv_from(&sep);
        let reference = MeanVar::from_prior(rmean, rvar, rcount);
        let items: Items<u32> = if ranks_only {
            Items::Ranks((0..pool.len() as u32).collect())
        } else {
            Items::Scored(pool.iter().enumerate().map(|(i, &s)| (i as u32, s)).collect())
        };
        let cfg = DiscriminationConfig::default();
        let d = discriminate(&items, &separation, &reference, &cfg);
        prop_assert!(d.g.is_finite(), "g not finite: {}", d.g);
        prop_assert!(
            d.g >= cfg.g_floor && d.g <= cfg.g_upper_bound,
            "g {} out of [{}, {}]",
            d.g, cfg.g_floor, cfg.g_upper_bound
        );
    }
}

// =======================================================================================
// weighted_rrf determinism + structure (§6)
// =======================================================================================

/// One channel's observation spec: `(key index, items as (id, score), ranks-only?)`. Ids
/// are drawn from a small pool so channels overlap.
type ObsSpec = (u8, Vec<(u32, f64)>, bool);

fn build_obs(spec: &[ObsSpec]) -> Vec<ChannelInput<u32>> {
    spec.iter()
        .map(|(k, items, ranks)| {
            let key = format!("c{k}");
            let it = if *ranks {
                Items::Ranks(items.iter().map(|(id, _)| *id).collect())
            } else {
                Items::Scored(items.clone())
            };
            ChannelInput { key, items: it }
        })
        .collect()
}

fn weights_map(specs: &[(u8, f64)]) -> BTreeMap<String, f64> {
    specs.iter().map(|(k, w)| (format!("c{k}"), *w)).collect()
}

/// A strategy for a list of observation specs over a small shared id space.
fn obs_spec_strategy() -> impl Strategy<Value = Vec<ObsSpec>> {
    prop::collection::vec(
        (
            0u8..4,
            prop::collection::vec((0u32..8, -1.0e3f64..1.0e3f64), 0..8),
            any::<bool>(),
        ),
        0..5,
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    /// Determinism: running `weighted_rrf` twice on the same input yields a byte-identical
    /// Vec (no HashMap seed leaks into the output).
    #[test]
    fn weighted_rrf_is_deterministic_across_runs(
        spec in obs_spec_strategy(),
        wspec in prop::collection::vec((0u8..4, 0.0f64..5.0f64), 0..4),
        eta in 0.0f64..200.0f64,
    ) {
        let obs = build_obs(&spec);
        let w = weights_map(&wspec);
        let first = weighted_rrf(&obs, &w, &rrf(eta));
        let second = weighted_rrf(&obs, &w, &rrf(eta));
        prop_assert_eq!(first, second);
    }

    /// Scores are non-negative and finite for eta >= 0 and non-negative weights, and every
    /// surfaced id appears exactly once in the output.
    #[test]
    fn weighted_rrf_scores_nonneg_and_each_id_once(
        spec in obs_spec_strategy(),
        wspec in prop::collection::vec((0u8..4, 0.0f64..5.0f64), 0..4),
        eta in 0.0f64..200.0f64,
    ) {
        let obs = build_obs(&spec);
        let w = weights_map(&wspec);
        let out = weighted_rrf(&obs, &w, &rrf(eta));

        for (_, score) in &out {
            prop_assert!(score.is_finite(), "score not finite: {score}");
            prop_assert!(*score >= 0.0, "score negative: {score}");
        }

        // The set of surfaced ids (union over all channels) appears exactly once each.
        let mut surfaced: BTreeSet<u32> = BTreeSet::new();
        for (_, items, _) in &spec {
            for (id, _) in items {
                surfaced.insert(*id);
            }
        }
        let out_ids: Vec<u32> = out.iter().map(|(id, _)| *id).collect();
        let out_set: BTreeSet<u32> = out_ids.iter().copied().collect();
        prop_assert_eq!(out_ids.len(), out_set.len(), "an id appeared more than once");
        prop_assert_eq!(out_set, surfaced);
    }

    /// A rank-equivalent reordering of one channel's distinctly-scored items yields an
    /// identical fused result: rank is read from score, so item order does not matter, and
    /// distinct scores leave no tie for the first-seen tiebreak to disturb.
    #[test]
    fn weighted_rrf_invariant_under_within_channel_reorder(
        scores in prop::collection::vec(-1.0e3f64..1.0e3f64, 1..30),
        seed in any::<u64>(),
        eta in 0.0f64..200.0f64,
    ) {
        // Distinct scores: no within-channel tie, so the rank order is unambiguous.
        let mut seen = BTreeSet::new();
        for s in &scores {
            prop_assume!(seen.insert(s.to_bits()));
        }
        let items: Vec<(u32, f64)> =
            scores.iter().enumerate().map(|(i, &s)| (i as u32, s)).collect();

        let canonical = weighted_rrf(
            &[ChannelInput { key: key(0), items: Items::Scored(items.clone()) }],
            &BTreeMap::new(),
            &rrf(eta)
        );
        let shuffled = weighted_rrf(
            &[ChannelInput { key: key(0), items: Items::Scored(lcg_shuffle(items, seed)) }],
            &BTreeMap::new(),
            &rrf(eta)
        );
        prop_assert_eq!(canonical, shuffled);
    }

    /// Permuting channel order preserves every surfaced id's fused score (RRF sums each
    /// channel's contribution, and addition is order-independent up to f64 rounding). The
    /// id set is identical; per-id scores match to a tight relative tolerance.
    #[test]
    fn weighted_rrf_invariant_under_channel_reorder(
        spec in obs_spec_strategy(),
        wspec in prop::collection::vec((0u8..4, 0.0f64..5.0f64), 0..4),
        seed in any::<u64>(),
        eta in 0.0f64..200.0f64,
    ) {
        let obs = build_obs(&spec);
        let w = weights_map(&wspec);
        let out1 = weighted_rrf(&obs, &w, &rrf(eta));
        let out2 = weighted_rrf(&lcg_shuffle(obs, seed), &w, &rrf(eta));

        let m1: HashMap<u32, f64> = out1.into_iter().collect();
        let m2: HashMap<u32, f64> = out2.into_iter().collect();
        prop_assert_eq!(
            m1.keys().copied().collect::<BTreeSet<_>>(),
            m2.keys().copied().collect::<BTreeSet<_>>()
        );
        for (id, s1) in &m1 {
            prop_assert!(
                rel_close(*s1, m2[id], 1e-9),
                "id {id}: {s1} vs {} differ under channel reorder", m2[id]
            );
        }
    }
}

// =======================================================================================
// State serde + content-addressing (§8)
// =======================================================================================

/// A per-channel content spec: `(index, separation pushes, reference pushes, tag selector)`.
type ChanContent = (usize, Vec<f64>, Vec<f64>, u8);
/// A per-pair content spec: `(index i, index j, redundancy pushes)`.
type PairContent = (usize, usize, Vec<f64>);

fn chan_summary(sep: &[f64], refs: &[f64], tagsel: u8) -> ChannelSummary {
    ChannelSummary {
        level: MeanVar::new(),
        separation: mv_from(sep),
        reference: mv_from(refs),
        tag: format!("tag{tagsel}"),
    }
}

fn dir_of(i: usize) -> Direction {
    if i % 2 == 0 {
        Direction::HigherIsBetter
    } else {
        Direction::LowerIsBetter
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// A random `RuffleState` round-trips through `serde_json` exactly, for ARBITRARY
    /// accumulated f64 values (not just "nice" ones). All MeanVar contents are finite
    /// (built from finite pushes), so JSON loses nothing.
    ///
    /// Regression guard for a real defect: serde_json's DEFAULT float parser is not
    /// round-trip-correct — it can drop 1 ULP on parse, so `from_str(to_string(s)) != s`.
    /// Minimized counterexample: a separation baseline accumulating
    /// `[834.8005097722045, 983.5434158837168]` gives `mean = 909.1719628279607`;
    /// `to_string` writes the correct shortest string `"909.1719628279607"` (ryu) but the
    /// default parser returns `909.1719628279608`. The fix is serde_json's
    /// `float_roundtrip` feature, enabled on both the runtime (cli) and dev deps in
    /// Cargo.toml, which makes the round-trip bit-exact and keeps this property green.
    #[test]
    fn ruffle_state_serde_round_trips(
        chans in prop::collection::vec(
            (0usize..8, prop::collection::vec(small_f64(), 0..5),
             prop::collection::vec(small_f64(), 0..5), 0u8..4),
            0..6,
        ),
        pairs in prop::collection::vec(
            (0usize..8, 0usize..8, prop::collection::vec(small_f64(), 0..4)),
            0..4,
        ),
        format_version in any::<u32>(),
    ) {
        let mut dirs = BTreeMap::new();
        for (i, _sep, _refs, _t) in &chans {
            dirs.insert(key(*i), dir_of(*i));
        }
        let mut state = RuffleState::new(StatFingerprint::new(BaselineMode::ZScore, dirs));
        for (i, sep, refs, t) in &chans {
            state.channels.insert(key(*i), chan_summary(sep, refs, *t));
        }
        for (i, j, red) in &pairs {
            if i != j {
                state.pairs.insert(
                    UnorderedPair::new(key(*i), key(*j)),
                    PairSummary { redundancy: mv_from(red), refreshes: 2.0 },
                );
            }
        }
        // `format_version` is library-managed with no setter; inject the arbitrary value
        // the way a loaded file carries it — through the serialized JSON, then deserialize.
        let mut value = serde_json::to_value(&state).unwrap();
        value["format_version"] = serde_json::Value::from(format_version);
        let state: RuffleState = serde_json::from_value(value).unwrap();

        let json = serde_json::to_string(&state).unwrap();
        let back: RuffleState = serde_json::from_str(&json).unwrap();
        prop_assert_eq!(state, back);
    }

    /// Two states with identical CONTENT inserted in different orders serialize to
    /// BYTE-IDENTICAL JSON (BTreeMap canonicalization gives content-addressing, §8).
    #[test]
    fn identical_content_serializes_byte_identically(
        chans in prop::collection::vec(
            (0usize..16, prop::collection::vec(small_f64(), 0..4),
             prop::collection::vec(small_f64(), 0..4), 0u8..4),
            1..6,
        ),
        pairs in prop::collection::vec(
            (0usize..16, 0usize..16, prop::collection::vec(small_f64(), 0..4)),
            0..4,
        ),
    ) {
        // Dedup to one canonical content per index/pair, so the two states differ ONLY in
        // insertion order, not in content.
        let mut chan_content: BTreeMap<usize, ChanContent> = BTreeMap::new();
        for c in &chans {
            chan_content.entry(c.0).or_insert_with(|| c.clone());
        }
        let mut pair_content: BTreeMap<(usize, usize), PairContent> = BTreeMap::new();
        for (i, j, red) in &pairs {
            if i != j {
                let (a, b) = if i < j { (*i, *j) } else { (*j, *i) };
                pair_content.entry((a, b)).or_insert((a, b, red.clone()));
            }
        }
        let chan_vec: Vec<&ChanContent> = chan_content.values().collect();
        let pair_vec: Vec<&PairContent> = pair_content.values().collect();

        let build = |forward: bool| -> RuffleState {
            let chan_iter: Vec<&ChanContent> = if forward {
                chan_vec.clone()
            } else {
                chan_vec.iter().rev().copied().collect()
            };
            let mut dirs = BTreeMap::new();
            for (i, _sep, _refs, _t) in &chan_iter {
                dirs.insert(key(*i), dir_of(*i));
            }
            let mut s = RuffleState::new(StatFingerprint::new(BaselineMode::ZScore, dirs));
            for (i, sep, refs, t) in &chan_iter {
                s.channels.insert(key(*i), chan_summary(sep, refs, *t));
            }
            let pair_iter: Vec<&PairContent> = if forward {
                pair_vec.clone()
            } else {
                pair_vec.iter().rev().copied().collect()
            };
            for (i, j, red) in pair_iter {
                s.pairs.insert(
                    UnorderedPair::new(key(*i), key(*j)),
                    PairSummary { redundancy: mv_from(red), refreshes: 2.0 },
                );
            }
            s
        };

        let forward = serde_json::to_string(&build(true)).unwrap();
        let reverse = serde_json::to_string(&build(false)).unwrap();
        prop_assert_eq!(forward, reverse);
    }
}

// =======================================================================================
// RuffleState merge (§8)
// =======================================================================================

const K: usize = 4;

fn chan_tag(i: usize) -> String {
    format!("t{i}")
}

/// A state spec: channels `(index, sep pushes, ref pushes)` and pairs `(i, j, red pushes)`.
/// Tag and direction are deterministic functions of the channel index, so any two states
/// sharing a channel agree on both and are therefore mergeable.
type StateSpec = (
    Vec<(usize, Vec<f64>, Vec<f64>)>,
    Vec<(usize, usize, Vec<f64>)>,
);

/// The direction fingerprint a spec implies (deterministic in the channel indices).
fn spec_fingerprint(spec: &StateSpec) -> StatFingerprint {
    let mut directions = BTreeMap::new();
    for (i, _sep, _refs) in &spec.0 {
        directions.insert(key(*i), dir_of(*i));
    }
    StatFingerprint::new(BaselineMode::ZScore, directions)
}

fn build_state(spec: &StateSpec) -> RuffleState {
    build_state_with(spec, spec_fingerprint(spec))
}

/// Build a state from `spec` but with an explicit fingerprint, so a test can inject a
/// mismatching `stat_version`/`baseline_mode`/direction through `StatFingerprint`'s own
/// (still public) fields rather than mutating the now-crate-private `RuffleState`
/// fingerprint field.
fn build_state_with(spec: &StateSpec, fingerprint: StatFingerprint) -> RuffleState {
    let mut s = RuffleState::new(fingerprint);
    for (i, sep, refs) in &spec.0 {
        s.channels.insert(
            key(*i),
            ChannelSummary {
                level: MeanVar::new(),
                separation: mv_from(sep),
                reference: mv_from(refs),
                tag: chan_tag(*i),
            },
        );
    }
    for (i, j, red) in &spec.1 {
        if i != j {
            s.pairs.insert(
                UnorderedPair::new(key(*i), key(*j)),
                PairSummary {
                    redundancy: mv_from(red),
                    refreshes: 2.0,
                },
            );
        }
    }
    s
}

fn state_spec_strategy() -> impl Strategy<Value = StateSpec> {
    (
        prop::collection::vec(
            (
                0usize..K,
                prop::collection::vec(small_f64(), 0..5),
                prop::collection::vec(small_f64(), 0..5),
            ),
            0..=K + 1,
        ),
        prop::collection::vec(
            (
                0usize..K,
                0usize..K,
                prop::collection::vec(small_f64(), 0..5),
            ),
            0..4,
        ),
    )
}

fn mv_close(a: &MeanVar, b: &MeanVar) -> bool {
    rel_close(a.count(), b.count(), 1e-9)
        && rel_close(a.mean(), b.mean(), 1e-9)
        && rel_close(a.variance(), b.variance(), 1e-7)
}

/// Two merged states agree on key sets and on every summary's mean/variance/count.
fn states_close(a: &RuffleState, b: &RuffleState) -> bool {
    if a.channels.keys().ne(b.channels.keys()) || a.pairs.keys().ne(b.pairs.keys()) {
        return false;
    }
    for (k, ca) in &a.channels {
        let cb = &b.channels[k];
        if ca.tag != cb.tag
            || !mv_close(&ca.separation, &cb.separation)
            || !mv_close(&ca.reference, &cb.reference)
        {
            return false;
        }
    }
    for (k, pa) in &a.pairs {
        if !mv_close(&pa.redundancy, &b.pairs[k].redundancy) {
            return false;
        }
    }
    true
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// merge of compatible states is COMMUTATIVE and ASSOCIATIVE: reordering the parts, or
    /// regrouping a nested merge, gives the same per-channel and per-pair means, variances,
    /// and counts (up to f64 rounding).
    #[test]
    fn merge_is_commutative_and_associative(
        a in state_spec_strategy(),
        b in state_spec_strategy(),
        c in state_spec_strategy(),
    ) {
        let (sa, sb, sc) = (build_state(&a), build_state(&b), build_state(&c));

        let (m1, _) = RuffleState::merge(&[&sa, &sb, &sc], MergePolicy::Strict).unwrap();
        let (m2, _) = RuffleState::merge(&[&sc, &sb, &sa], MergePolicy::Strict).unwrap();
        let (m3, _) = RuffleState::merge(&[&sb, &sa, &sc], MergePolicy::Strict).unwrap();
        prop_assert!(states_close(&m1, &m2));
        prop_assert!(states_close(&m1, &m3));

        // Associative: (a·b)·c == a·(b·c).
        let (ab, _) = RuffleState::merge(&[&sa, &sb], MergePolicy::Strict).unwrap();
        let (ab_c, _) = RuffleState::merge(&[&ab, &sc], MergePolicy::Strict).unwrap();
        let (bc, _) = RuffleState::merge(&[&sb, &sc], MergePolicy::Strict).unwrap();
        let (a_bc, _) = RuffleState::merge(&[&sa, &bc], MergePolicy::Strict).unwrap();
        prop_assert!(states_close(&ab_c, &a_bc));
    }

    /// merge is the UNION of channels and pairs; a shared channel's separation count equals
    /// the SUM of the parts' counts.
    #[test]
    fn merge_unions_keys_and_sums_shared_counts(
        specs in prop::collection::vec(state_spec_strategy(), 2..=3),
    ) {
        let states: Vec<RuffleState> = specs.iter().map(build_state).collect();
        let refs: Vec<&RuffleState> = states.iter().collect();
        let (m, _) = RuffleState::merge(&refs, MergePolicy::Strict).unwrap();

        // Channel keys are the union across all parts.
        let mut union_chan: BTreeSet<String> = BTreeSet::new();
        let mut union_pair: BTreeSet<UnorderedPair<String>> = BTreeSet::new();
        for s in &states {
            union_chan.extend(s.channels.keys().cloned());
            union_pair.extend(s.pairs.keys().cloned());
        }
        prop_assert_eq!(m.channels.keys().cloned().collect::<BTreeSet<_>>(), union_chan);
        prop_assert_eq!(m.pairs.keys().cloned().collect::<BTreeSet<_>>(), union_pair);

        // Each merged channel's separation count is the sum of the parts that carry it.
        for k in m.channels.keys() {
            let sum: f64 = states
                .iter()
                .filter_map(|s| s.channels.get(k))
                .map(|c| c.separation.count())
                .sum();
            prop_assert!(
                rel_close(m.channels[k].separation.count(), sum, 1e-9),
                "channel {k}: merged count {} vs sum {sum}",
                m.channels[k].separation.count()
            );
        }
        // Likewise each merged pair's redundancy count.
        for pr in m.pairs.keys() {
            let sum: f64 = states
                .iter()
                .filter_map(|s| s.pairs.get(pr))
                .map(|p| p.redundancy.count())
                .sum();
            prop_assert!(rel_close(m.pairs[pr].redundancy.count(), sum, 1e-9));
        }
    }

    /// merge REFUSES on a format_version disagreement.
    #[test]
    fn merge_refuses_on_format_version_mismatch(
        a in state_spec_strategy(),
        b in state_spec_strategy(),
        bump in 1u32..1000,
    ) {
        let sa = build_state(&a);
        let sb = build_state(&b);
        // `format_version` is library-managed with no setter: produce the mismatched
        // state the way a loaded file would carry it — edit the serialized value and
        // deserialize back (the realistic "loaded an incompatible file" path).
        let target = sa.format_version().wrapping_add(bump);
        let mut value = serde_json::to_value(&sb).unwrap();
        value["format_version"] = serde_json::Value::from(target);
        let sb: RuffleState = serde_json::from_value(value).unwrap();
        let err = RuffleState::merge(&[&sa, &sb], MergePolicy::Strict).unwrap_err();
        prop_assert!(matches!(err, Mismatch::FormatVersion { .. }), "got {err:?}");
    }

    /// merge REFUSES on a stat_version disagreement (the fingerprint gate).
    #[test]
    fn merge_refuses_on_stat_version_mismatch(
        a in state_spec_strategy(),
        b in state_spec_strategy(),
        bump in 1u32..1000,
    ) {
        let sa = build_state(&a);
        // Inject the mismatching `stat_version` through `StatFingerprint`'s own public
        // field before building the state, so no crate-private field is touched.
        let mut fp = spec_fingerprint(&b);
        fp.stat_version = sa.fingerprint().stat_version.wrapping_add(bump);
        let sb = build_state_with(&b, fp);
        let err = RuffleState::merge(&[&sa, &sb], MergePolicy::Strict).unwrap_err();
        prop_assert_eq!(err, Mismatch::Fingerprint);
    }

    /// merge REFUSES when a shared channel carries a different semantic tag (a model swap
    /// under a kept key).
    #[test]
    fn merge_refuses_on_shared_channel_tag_conflict(
        sep_a in prop::collection::vec(small_f64(), 0..5),
        sep_b in prop::collection::vec(small_f64(), 0..5),
    ) {
        let mut dirs_a = BTreeMap::new();
        dirs_a.insert(key(0), Direction::HigherIsBetter);
        let mut sa = RuffleState::new(StatFingerprint::new(BaselineMode::ZScore, dirs_a));
        sa.channels.insert(
            key(0),
            ChannelSummary { level: MeanVar::new(), separation: mv_from(&sep_a), reference: MeanVar::new(), tag: "model-a".to_string() },
        );

        let mut dirs_b = BTreeMap::new();
        dirs_b.insert(key(0), Direction::HigherIsBetter);
        let mut sb = RuffleState::new(StatFingerprint::new(BaselineMode::ZScore, dirs_b));
        sb.channels.insert(
            key(0),
            ChannelSummary { level: MeanVar::new(), separation: mv_from(&sep_b), reference: MeanVar::new(), tag: "model-b".to_string() },
        );

        let err = RuffleState::merge(&[&sa, &sb], MergePolicy::Strict).unwrap_err();
        prop_assert!(matches!(err, Mismatch::Tag { .. }), "got {err:?}");
    }

    /// merge REFUSES when a shared channel carries a conflicting direction (a fingerprint
    /// orientation flip). Tags are kept equal so the direction gate, which runs first, is
    /// the one that fires.
    #[test]
    fn merge_refuses_on_shared_channel_direction_conflict(
        sep_a in prop::collection::vec(small_f64(), 0..5),
        sep_b in prop::collection::vec(small_f64(), 0..5),
    ) {
        let mk = |sep: &[f64], dir: Direction| {
            let mut dirs = BTreeMap::new();
            dirs.insert(key(0), dir);
            let mut s = RuffleState::new(StatFingerprint::new(BaselineMode::ZScore, dirs));
            s.channels.insert(
                key(0),
                ChannelSummary { level: MeanVar::new(), separation: mv_from(sep), reference: MeanVar::new(), tag: "same".to_string() },
            );
            s
        };
        let sa = mk(&sep_a, Direction::HigherIsBetter);
        let sb = mk(&sep_b, Direction::LowerIsBetter);
        let err = RuffleState::merge(&[&sa, &sb], MergePolicy::Strict).unwrap_err();
        prop_assert!(matches!(err, Mismatch::DirectionConflict { .. }), "got {err:?}");
    }
}
