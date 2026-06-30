//! Adversarial and robustness suite.
//!
//! The goal is sqlite-level robustness: throw pathological, degenerate, and extreme
//! inputs at every public entry point and prove that none of them panics, leaks a
//! `NaN`/`inf` into an output, hangs, or returns a clearly-wrong result. Inputs that the
//! crate handles correctly stay here as passing regression tests; the boundaries the
//! derivation promises (§4 degeneracy guards, §5 coupling gates, §6 absence-omit, §7
//! sanitization, §8 merge gates, §11 fuser lifecycle) are pinned as executable contracts.
//!
//! Only the public API is exercised (integration tests see `pub` items only). Where a
//! constructor documents a precondition (e.g. building `Items::Scored` by hand bypasses
//! the `ChannelInput::scored` sanitization, §7), the supported path is tested through
//! that path; a handful of probes additionally feed the unsupported path to confirm the
//! second line of defence (`MeanVar::push` dropping non-finite values, §8) still holds.

#![allow(clippy::needless_range_loop)]

use approx::assert_abs_diff_eq;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruffle::components::{
    ChannelDiscrimination, Diagnostics, PairBaseline, anchor_correlations, coupled_weights,
    diagnostics, discriminate, weighted_rrf,
};
use ruffle::{
    Anchor, BaselineMode, ChannelConfig, ChannelFlag, ChannelId, ChannelInput, ChannelSummary,
    CouplingConfig, DecayConfig, Direction, DiscriminationConfig, FuseConfig, Fuser, GoodScore,
    Items, MeanVar, MergePolicy, Mismatch, PairSummary, RrfConfig, RuffleState, Score,
    StatFingerprint, UnorderedPair,
};
use std::collections::BTreeMap;

// --- shared helpers ----------------------------------------------------------------

/// A caller-side newtype: the only way a bare number becomes a [`Score`] (§7).
struct Raw(f64);
impl Score for Raw {
    fn value(&self) -> f64 {
        self.0
    }
}

fn key(s: &str) -> String {
    s.to_string()
}

fn tag() -> String {
    "model-1".to_string()
}

fn dcfg() -> DiscriminationConfig {
    DiscriminationConfig::default()
}

/// An [`RrfConfig`] with a chosen rank constant `η`, for the fusion calls below.
fn rrf(eta: f64) -> RrfConfig {
    let mut c = RrfConfig::default();
    c.rrf_eta = eta;
    c
}

/// A higher-is-better channel with no declared reference.
fn chan(name: &str) -> ChannelConfig {
    ChannelConfig::new(
        ChannelId::new(name, "model-1"),
        Direction::HigherIsBetter,
        None,
    )
}

fn chan_with(name: &str, dir: Direction, good: Option<GoodScore>) -> ChannelConfig {
    ChannelConfig::new(ChannelId::new(name, "model-1"), dir, good)
}

/// Build a `Scored` `Items` through the supported `ChannelInput::scored` path (orients
/// + sanitizes).
fn observed(vals: &[f64]) -> Items<u32> {
    let c = chan("x");
    ChannelInput::scored(
        &c,
        vals.iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, Raw(v)))
            .collect(),
    )
    .items
}

/// A cold summary: empty separation baseline, empty reference, just the tag.
fn cold() -> ChannelSummary {
    ChannelSummary::new(tag())
}

/// A summary with a warm separation baseline and a seeded reference.
fn warm() -> ChannelSummary {
    let mut s = ChannelSummary::new(tag());
    for r in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0] {
        s.separation.push(r);
    }
    s.reference = MeanVar::from_prior(3.0, 1.0, 10.0);
    s
}

/// Read `discriminate` against a summary's two baselines — the same projection the Fuser
/// does, kept here so the tests can keep describing their fixtures as a `ChannelSummary`.
fn discriminate_summary<Id>(
    items: &Items<Id>,
    summary: &ChannelSummary,
    cfg: &DiscriminationConfig,
) -> ChannelDiscrimination {
    discriminate(items, &summary.separation, &summary.reference, cfg)
}

fn assert_disc_sane(d: &ChannelDiscrimination, cfg: &DiscriminationConfig, ctx: &str) {
    assert!(d.g.is_finite(), "g not finite [{ctx}]");
    assert!(
        d.g >= cfg.g_floor - 1e-12 && d.g <= cfg.g_upper_bound + 1e-12,
        "g={} out of [{}, {}] [{ctx}]",
        d.g,
        cfg.g_floor,
        cfg.g_upper_bound
    );
}

fn assert_mv_finite(mv: &MeanVar, ctx: &str) {
    assert!(mv.count().is_finite(), "count not finite [{ctx}]");
    assert!(mv.mean().is_finite(), "mean not finite [{ctx}]");
    assert!(mv.variance().is_finite(), "variance not finite [{ctx}]");
    assert!(mv.variance() >= 0.0, "variance negative [{ctx}]");
    assert!(mv.std().is_finite(), "std not finite [{ctx}]");
}

fn assert_state_finite(s: &RuffleState, ctx: &str) {
    for (k, c) in &s.channels {
        assert_mv_finite(&c.separation, &format!("{ctx}/{k}/sep"));
        assert_mv_finite(&c.reference, &format!("{ctx}/{k}/ref"));
    }
    for (p, ps) in &s.pairs {
        assert_mv_finite(&ps.redundancy, &format!("{ctx}/{:?}/red", p.as_tuple()));
    }
}

fn assert_ranking_finite<Id>(r: &[(Id, f64)], ctx: &str) {
    assert!(
        r.iter().all(|(_, s)| s.is_finite()),
        "ranking has non-finite score [{ctx}]"
    );
}

/// Weights are finite, non-negative, and (when any channel is present) sum to `N`.
fn assert_weights_sum_to_n(weights: &BTreeMap<String, f64>, n: usize, ctx: &str) {
    assert!(
        weights.values().all(|w| w.is_finite() && *w >= 0.0),
        "weights not all finite-nonneg [{ctx}]: {weights:?}"
    );
    if !weights.is_empty() {
        let total: f64 = weights.values().sum();
        assert_abs_diff_eq!(total, n as f64, epsilon = 1e-6);
    }
}

fn empty_state() -> RuffleState {
    RuffleState::new(StatFingerprint::new(BaselineMode::ZScore, BTreeMap::new()))
}

// ===================================================================================
// §4 — pools / discriminate
// ===================================================================================

mod discrimination_pools {
    use super::*;

    /// Run discriminate against both a cold and a warm summary and assert sanity.
    fn probe(items: &Items<u32>, ctx: &str) -> ChannelDiscrimination {
        let cfg = dcfg();
        let d_cold = discriminate_summary(items, &cold(), &cfg);
        let d_warm = discriminate_summary(items, &warm(), &cfg);
        assert_disc_sane(&d_cold, &cfg, &format!("{ctx}/cold"));
        assert_disc_sane(&d_warm, &cfg, &format!("{ctx}/warm"));
        d_warm
    }

    #[test]
    fn empty_pool_is_neutral() {
        let d = probe(&observed(&[]), "empty");
        assert_abs_diff_eq!(d.g, 1.0, epsilon = 1e-12);
        assert_eq!(d.raw_separation, None);
        assert_eq!(d.top_m_average, None);
        assert!(!d.degenerate_separation);
    }

    #[test]
    fn single_item() {
        let d = probe(&observed(&[0.7]), "single");
        // One value: too few distinct, separation undefined. The pool is also far
        // shallower than top_m, so no fixed-count reference read is exported either
        // (§4's comparable-depth condition): a depth-1 "top-10 average" is a different
        // statistic and must not refine the reference.
        assert_eq!(d.raw_separation, None);
        assert!(d.degenerate_separation);
        assert_eq!(d.top_m_average, None);
    }

    #[test]
    fn two_items() {
        let d = probe(&observed(&[0.1, 0.9]), "two");
        assert_eq!(d.raw_separation, None);
        assert!(d.degenerate_separation);
    }

    #[test]
    fn all_equal() {
        probe(&observed(&[3.0; 64]), "all-equal");
        let d = discriminate_summary(&observed(&[3.0; 64]), &cold(), &dcfg());
        assert_eq!(d.raw_separation, None);
        assert!(d.degenerate_separation);
    }

    #[test]
    fn all_equal_except_one() {
        let mut v = vec![5.0; 63];
        v.push(9.0);
        let d = probe(&observed(&v), "all-equal-except-one");
        // Two distinct values < min_distinct_values (8): degenerate.
        assert_eq!(d.raw_separation, None);
        assert!(d.degenerate_separation);
    }

    #[test]
    fn exactly_below_and_at_min_distinct_values() {
        let cfg = dcfg();
        let min = cfg.min_distinct_values; // 8 by default
        // min-1 distinct values -> degenerate, no separation.
        let below: Vec<f64> = (0..(min - 1)).map(|i| i as f64).collect();
        let d_below = probe(&observed(&below), "min-1-distinct");
        assert_eq!(d_below.raw_separation, None);
        assert!(d_below.degenerate_separation);
        // exactly min distinct values -> separation is defined and finite.
        let at: Vec<f64> = (0..min).map(|i| i as f64).collect();
        let d_at = probe(&observed(&at), "min-distinct");
        let s = d_at
            .raw_separation
            .expect("separation defined at min_distinct_values");
        assert!(s.is_finite());
        assert!(!d_at.degenerate_separation);
    }

    #[test]
    fn tied_lower_half_spread_upper_integer_counts() {
        // Lower half tied at 0 (q0.5 - q0.1 == 0) but the upper half spreads, so the
        // inter-quartile floor keeps the denominator positive and the read bounded (§4).
        let mut pool = vec![0.0; 11];
        pool.extend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let d = probe(&observed(&pool), "tied-lower-spread-upper");
        let s = d.raw_separation.expect("floored read still defined");
        assert!(s.is_finite() && s > 0.0);
        assert!(d.degenerate_separation, "floor was applied");
    }

    #[test]
    fn collapsed_bulk_is_degenerate() {
        // Eight distinct values but the lower three-quarters tied at zero: the bulk scale
        // collapses even after the floor.
        let mut pool = vec![0.0; 33];
        pool.extend([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let d = probe(&observed(&pool), "collapsed-bulk");
        assert_eq!(d.raw_separation, None);
        assert!(d.degenerate_separation);
    }

    #[test]
    fn saturated_near_constant_cone() {
        // A high-dim contrastive cone: everything packed near 0.9999, a few digits apart.
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let pool: Vec<f64> = (0..200)
            .map(|_| 0.9999 + rng.gen_range(-1e-7..1e-7))
            .collect();
        // Whatever the read, g must be finite and bounded and no NaN escapes.
        probe(&observed(&pool), "saturated-cone");
    }

    #[test]
    fn extreme_f64_values() {
        let cfg = dcfg();
        let pools: Vec<(&str, Vec<f64>)> = vec![
            (
                "max-min",
                vec![f64::MAX, f64::MIN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ),
            (
                "tiny",
                vec![
                    1e-300, 2e-300, 3e-300, 4e-300, 5e-300, 6e-300, 7e-300, 8e-300,
                ],
            ),
            (
                "denormal",
                vec![
                    5e-324,
                    1e-323,
                    1.5e-323,
                    2e-323,
                    f64::MIN_POSITIVE,
                    2.0 * f64::MIN_POSITIVE,
                    3.0 * f64::MIN_POSITIVE,
                    4.0 * f64::MIN_POSITIVE,
                ],
            ),
            (
                "neg-zero",
                vec![-0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            ),
            (
                "all-negative",
                vec![-9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0],
            ),
            ("single-huge-outlier", {
                let mut v: Vec<f64> = (0..50).map(|i| i as f64 * 0.01).collect();
                v.push(1e12);
                v
            }),
        ];
        for (name, pool) in pools {
            let items = observed(&pool);
            let d_cold = discriminate_summary(&items, &cold(), &cfg);
            let d_warm = discriminate_summary(&items, &warm(), &cfg);
            assert_disc_sane(&d_cold, &cfg, name);
            assert_disc_sane(&d_warm, &cfg, name);
            if let Some(s) = d_warm.raw_separation {
                assert!(s.is_finite(), "raw_sep not finite [{name}]");
            }
            if let Some(t) = d_warm.top_m_average {
                assert!(t.is_finite(), "top_m not finite [{name}] (observe path)");
            }
        }
    }

    #[test]
    fn single_huge_outlier_reads_as_extreme_but_finite_separation() {
        // A top standing enormously above a healthy bulk IS separation, the most
        // informative shape there is: the degeneracy guard scales by the bulk's own
        // span, not the full range, so the read is defined, huge, and finite (§4).
        // Baseline protection is winsorization's job at the update step, not the
        // guard's.
        let mut v: Vec<f64> = (0..50).map(|i| i as f64 * 0.01).collect();
        v.push(1e12);
        let d = discriminate_summary(&observed(&v), &warm(), &dcfg());
        let sep = d.raw_separation.expect("extreme elevation must read");
        assert!(sep.is_finite() && sep > 1.0e6, "got {sep}");
        assert!(!d.degenerate_separation);
    }

    #[test]
    fn nan_and_inf_are_dropped_by_observe() {
        // The supported §7 path: observe drops every non-finite score so nothing downstream
        // ever sees one. The resulting pool is clean and discriminate stays finite.
        let c = chan("x");
        let raw = vec![
            (1u32, Raw(0.5)),
            (2, Raw(f64::NAN)),
            (3, Raw(f64::INFINITY)),
            (4, Raw(f64::NEG_INFINITY)),
            (5, Raw(0.25)),
            (6, Raw(f64::NAN)),
            (7, Raw(0.1)),
        ];
        let obs = ChannelInput::scored(&c, raw);
        match &obs.items {
            Items::Scored(v) => {
                assert!(
                    v.iter().all(|(_, s)| s.is_finite()),
                    "non-finite survived observe"
                );
                assert_eq!(v.len(), 3); // only the three finite scores remain
            }
            Items::Ranks(_) => panic!("expected scored"),
        }
        let d = discriminate_summary(&obs.items, &warm(), &dcfg());
        assert_disc_sane(&d, &dcfg(), "post-observe");
    }

    #[test]
    fn hand_built_nonfinite_items_never_nan_g_and_never_corrupt_state() {
        // PRECONDITION-VIOLATING probe (§7 says hand-built Items::Scored must be finite).
        // Even so, g stays finite for every case (the zscore guards neutralize it), and
        // feeding such an observation to a Fuser leaves the persistent state finite because
        // MeanVar::push drops non-finite values (§8). The read struct itself MAY carry a
        // non-finite raw_separation/top_m_average; that is the documented cost of bypassing
        // observe, not a state corruption.
        let cfg = dcfg();
        let pools = [
            vec![f64::INFINITY, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![f64::NEG_INFINITY, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![f64::NAN, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            vec![
                f64::INFINITY,
                f64::NEG_INFINITY,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ],
            vec![f64::NAN; 8],
        ];
        for (i, pool) in pools.iter().enumerate() {
            let items: Items<u32> = Items::Scored(
                pool.iter()
                    .enumerate()
                    .map(|(j, &v)| (j as u32, v))
                    .collect(),
            );
            let d = discriminate_summary(&items, &warm(), &cfg);
            assert!(
                d.g.is_finite(),
                "g leaked non-finite for hand-built pool {i}"
            );
            assert!(d.g >= cfg.g_floor && d.g <= cfg.g_upper_bound);

            // Now drive it through a real Fuser and confirm state stays finite.
            let a = chan("a");
            let mut f = Fuser::new(std::slice::from_ref(&a), FuseConfig::default()).unwrap();
            let obs = vec![ChannelInput {
                key: key("a"),
                items: items.clone(),
            }];
            let fused = f.fuse(&obs);
            assert_ranking_finite(&fused.ranking, "hand-built");
            assert_state_finite(f.state(), &format!("hand-built-{i}"));
        }
    }

    /// The two pools that overflow the discrimination numerators (see the `defect_*` test).
    /// `[f64::MAX, f64::MAX]` overflows the D^abs top-m sum; a MAX-saturated bulk (>= 8
    /// distinct values plus many `f64::MAX`) additionally overflows the separation numerator
    /// while dodging the spread-relative degeneracy guard (the bulk is at MAX scale).
    fn overflow_pools() -> Vec<(&'static str, Vec<f64>)> {
        let mut saturated: Vec<f64> = (1..=8).map(|i| i as f64).collect();
        saturated.extend(vec![f64::MAX; 40]);
        vec![
            ("two-max", vec![f64::MAX, f64::MAX]),
            ("ten-max", vec![f64::MAX; 10]),
            ("two-min", vec![f64::MIN, f64::MIN]),
            ("saturated-bulk", saturated),
        ]
    }

    #[test]
    fn extreme_score_overflow_is_contained_to_the_read_struct() {
        // DEFENCE-IN-DEPTH REGRESSION for the discrimination-numerator overflow defect (the
        // #[ignore]'d `defect_*` test below records the bug itself). When a numerator sum
        // overflows f64, the read struct's `raw_separation` / `top_m_average` go non-finite,
        // but the blast radius is contained and THAT containment is what this test pins:
        //   * `g` stays finite and bounded (the zscore guard turns inf into a neutral read);
        //   * the fused ranking stays finite;
        //   * winsorize clamps and/or `MeanVar::push` drops the inf, so persistent state is
        //     never corrupted (§7, §8).
        let cfg = dcfg();
        for (name, pool) in overflow_pools() {
            let d = discriminate_summary(&observed(&pool), &warm(), &cfg);
            assert!(
                d.g.is_finite() && d.g >= cfg.g_floor && d.g <= cfg.g_upper_bound,
                "g escaped its envelope for {name}: {}",
                d.g
            );
        }

        // Drive an overflowing pool through a Fuser with a declared reference and confirm
        // the persistent state stays finite (the inf reads never reach a baseline).
        let a = chan_with(
            "a",
            Direction::HigherIsBetter,
            Some(GoodScore::new(0.0, 1.0, 5.0)),
        );
        let mut f = Fuser::new(std::slice::from_ref(&a), FuseConfig::default()).unwrap();
        let obs = vec![ChannelInput::scored(
            &a,
            vec![(0u32, Raw(f64::MAX)), (1, Raw(f64::MAX))],
        )];
        let fused = f.fuse(&obs);
        assert_ranking_finite(&fused.ranking, "overflow-contained");
        assert_state_finite(f.state(), "overflow-contained");
        // The inf reference read was dropped: count stayed at the declared prior (5).
        assert_abs_diff_eq!(
            f.state().channels[&key("a")].reference.count(),
            5.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn defect_discrimination_reads_should_stay_finite_under_extreme_scores() {
        let cfg = dcfg();
        for (name, pool) in overflow_pools() {
            let d = discriminate_summary(&observed(&pool), &warm(), &cfg);
            // DESIRED invariant (currently violated): both read fields are finite or None.
            if let Some(t) = d.top_m_average {
                assert!(
                    t.is_finite(),
                    "[{name}] top_m_average leaked non-finite: {t}"
                );
            }
            if let Some(s) = d.raw_separation {
                assert!(
                    s.is_finite(),
                    "[{name}] raw_separation leaked non-finite: {s}"
                );
            }
        }
    }

    #[test]
    fn huge_pool_completes_and_is_finite() {
        // 1e5 items: must sort and read quickly with a finite, bounded g.
        let vals: Vec<f64> = (0..100_000).map(|i| (i as f64) * 1e-3).collect();
        let items = observed(&vals);
        let d = discriminate_summary(&items, &warm(), &dcfg());
        assert_disc_sane(&d, &dcfg(), "huge-pool");
        assert!(d.raw_separation.unwrap().is_finite());
    }

    #[test]
    fn ranks_only_is_neutral() {
        let items: Items<u32> = Items::Ranks(vec![3, 1, 2]);
        let d = discriminate_summary(&items, &warm(), &dcfg());
        assert_eq!(d.g, 1.0);
        assert_eq!(d.raw_separation, None);
        assert_eq!(d.top_m_average, None);
    }

    #[test]
    fn lower_is_better_orientation_is_consistent() {
        // A LowerIsBetter channel: observe negates, so the smallest native score becomes
        // the top. The read must still be finite and bounded.
        let c = chan_with("lo", Direction::LowerIsBetter, None);
        let native: Vec<(u32, Raw)> = (0..40).map(|i| (i, Raw(i as f64))).collect();
        let obs = ChannelInput::scored(&c, native);
        let d = discriminate_summary(&obs.items, &warm(), &dcfg());
        assert_disc_sane(&d, &dcfg(), "lower-is-better");
    }
}

// ===================================================================================
// §6 — fusion / weighted_rrf
// ===================================================================================

mod fusion_rrf {
    use super::*;

    fn scored(k: &str, v: &[(u32, f64)]) -> ChannelInput<u32> {
        ChannelInput {
            key: key(k),
            items: Items::Scored(v.to_vec()),
        }
    }
    fn ranks(k: &str, v: &[u32]) -> ChannelInput<u32> {
        ChannelInput {
            key: key(k),
            items: Items::Ranks(v.to_vec()),
        }
    }
    fn wmap(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
        pairs.iter().map(|(k, w)| (key(k), *w)).collect()
    }
    fn score_of(out: &[(u32, f64)], id: u32) -> Option<f64> {
        out.iter().find(|(i, _)| *i == id).map(|(_, s)| *s)
    }

    #[test]
    fn eta_zero_is_finite() {
        // eta = 0 with ranks starting at 1 keeps every denominator >= 1, so no blow-up.
        let obs = vec![scored("a", &[(1, 0.9), (2, 0.5)]), ranks("b", &[2, 1, 3])];
        let out = weighted_rrf(&obs, &BTreeMap::new(), &rrf(0.0));
        assert_ranking_finite(&out, "eta=0");
        // id 1: rank 1 in a (1/1) + rank 2 in b (1/2) = 1.5.
        assert_abs_diff_eq!(score_of(&out, 1).unwrap(), 1.5, epsilon = 1e-12);
    }

    #[test]
    fn eta_default_and_huge() {
        let obs = vec![ranks("a", &[1, 2, 3]), ranks("b", &[3, 2, 1])];
        for eta in [60.0, 1e6, 1e300, f64::MAX] {
            let out = weighted_rrf(&obs, &BTreeMap::new(), &rrf(eta));
            assert_ranking_finite(&out, &format!("eta={eta}"));
            assert_eq!(out.len(), 3);
            // Every score is strictly between 0 and 1 for these positive etas.
            assert!(out.iter().all(|(_, s)| *s >= 0.0 && *s <= 2.0));
        }
    }

    #[test]
    fn empty_obs_and_all_empty_channels() {
        let empty: Vec<ChannelInput<u32>> = vec![];
        assert!(weighted_rrf(&empty, &BTreeMap::new(), &rrf(60.0)).is_empty());

        let all_empty = vec![scored("a", &[]), ranks("b", &[])];
        assert!(weighted_rrf(&all_empty, &BTreeMap::new(), &rrf(60.0)).is_empty());

        let mixed = vec![scored("a", &[]), ranks("b", &[1, 2])];
        let out = weighted_rrf(&mixed, &BTreeMap::new(), &rrf(1.0));
        assert_eq!(out.len(), 2);
        assert_ranking_finite(&out, "empty+populated");
    }

    #[test]
    fn ranks_only_and_scored_mix_and_all_ranks_only() {
        let mix = vec![scored("a", &[(1, 0.9)]), ranks("b", &[1, 2])];
        assert_eq!(weighted_rrf(&mix, &BTreeMap::new(), &rrf(1.0)).len(), 2);
        let all_ranks = vec![ranks("a", &[1, 2]), ranks("b", &[2, 1])];
        let out = weighted_rrf(&all_ranks, &BTreeMap::new(), &rrf(1.0));
        assert_eq!(out.len(), 2);
        assert_ranking_finite(&out, "all-ranks");
    }

    #[test]
    fn missing_weight_defaults_to_one_zero_mutes() {
        let obs = vec![ranks("a", &[1]), ranks("b", &[1, 2])];
        // a missing -> defaults to 1.0; equals explicit 1.0.
        let out = weighted_rrf(&obs, &wmap(&[("b", 1.0)]), &rrf(1.0));
        let explicit = weighted_rrf(&obs, &wmap(&[("a", 1.0), ("b", 1.0)]), &rrf(1.0));
        assert_eq!(out, explicit);
        // 0.0 mutes a; an id only in the muted channel still appears with score 0.
        let solo_muted = vec![ranks("a", &[1]), ranks("b", &[2])];
        let out2 = weighted_rrf(&solo_muted, &wmap(&[("a", 0.0), ("b", 1.0)]), &rrf(1.0));
        assert_abs_diff_eq!(score_of(&out2, 1).unwrap(), 0.0, epsilon = 1e-12);
        assert_ranking_finite(&out2, "muted");
    }

    #[test]
    fn weights_not_summing_to_n_stay_finite() {
        // Off-spec weights (don't sum to N): still finite and sane, just scaled.
        let obs = vec![ranks("a", &[1, 2]), ranks("b", &[2, 1])];
        for w in [
            wmap(&[("a", 1e9), ("b", 1e-9)]),
            wmap(&[("a", 0.0), ("b", 0.0)]),
            wmap(&[("a", 1e300), ("b", 1e300)]),
        ] {
            let out = weighted_rrf(&obs, &w, &rrf(60.0));
            assert_ranking_finite(&out, "off-spec-weights");
        }
    }

    #[test]
    fn id_in_every_channel_in_one_and_disjoint() {
        // id 7 in every channel; id 8 in only one; ids 1..3 disjoint per channel.
        let obs = vec![
            ranks("a", &[7, 1, 8]),
            ranks("b", &[7, 2]),
            ranks("c", &[7, 3]),
        ];
        let out = weighted_rrf(&obs, &BTreeMap::new(), &rrf(1.0));
        // id 7 rank 1 in three channels: 3 * 1/2.
        assert_abs_diff_eq!(score_of(&out, 7).unwrap(), 1.5, epsilon = 1e-12);
        assert_ranking_finite(&out, "everywhere");
        assert_eq!(out.len(), 5); // distinct ids {1,2,3,7,8}
    }

    #[test]
    fn lopsided_channel_sizes() {
        // One channel with a single item, another with 1e4 items.
        let small = ranks("a", &[999_999]);
        let big_ids: Vec<u32> = (0..10_000).collect();
        let big = ranks("b", &big_ids);
        let out = weighted_rrf(&[small, big], &BTreeMap::new(), &rrf(60.0));
        assert_eq!(out.len(), 10_001);
        assert_ranking_finite(&out, "lopsided");
    }

    // -- the duplicate-id CHARACTERIZATION (flagged for a lead judgment call) --------

    #[test]
    fn duplicate_id_within_one_ranks_list_is_double_counted() {
        // A duplicate id within ONE channel's list violates the §0/§6 precondition that a
        // channel's list carries distinct ids. The engine does NOT dedupe within a channel:
        // each occurrence is charged its own rank, so the id is DOUBLE-COUNTED and every
        // later item in the same list is pushed down one rank by the phantom slot.
        //
        // [7, 7, 8]: id 7 takes ranks 1 AND 2 -> 1/2 + 1/3; id 8 takes rank 3 -> 1/4.
        // No panic, fully finite, single output row per id. JUDGMENT CALL: dedupe within a
        // channel, or document distinct-ids as a hard caller precondition?
        let obs = vec![ChannelInput {
            key: key("a"),
            items: Items::Ranks(vec![7u32, 7, 8]),
        }];
        let out = weighted_rrf(&obs, &BTreeMap::new(), &rrf(1.0));
        assert_eq!(
            out.len(),
            2,
            "output id list is still deduped to one row per id"
        );
        assert_abs_diff_eq!(score_of(&out, 7).unwrap(), 0.5 + 1.0 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(score_of(&out, 8).unwrap(), 0.25, epsilon = 1e-12);
        assert_ranking_finite(&out, "dup-ranks");
    }

    #[test]
    fn duplicate_id_within_one_scored_list_is_double_counted() {
        // Same characterization for a Scored channel with a repeated (tied) id: the two
        // occurrences tie in score, share the midrank 1.5, and each contributes
        // 1/(eta + 1.5) -- still double-counted (garbage in), but order-invariant.
        let obs = vec![ChannelInput {
            key: key("a"),
            items: Items::Scored(vec![(7u32, 0.9), (7, 0.9)]),
        }];
        let out = weighted_rrf(&obs, &BTreeMap::new(), &rrf(1.0));
        assert_eq!(out.len(), 1);
        assert_abs_diff_eq!(score_of(&out, 7).unwrap(), 2.0 / 2.5, epsilon = 1e-12);
        assert_ranking_finite(&out, "dup-scored");
    }
}

// ===================================================================================
// §5 — coupling
// ===================================================================================

mod coupling {
    use super::*;

    fn gmap(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
        entries.iter().map(|(k, v)| (key(k), *v)).collect()
    }
    fn pair_redundancy(mean: f64, variance: f64, count: f64) -> PairBaseline {
        PairBaseline {
            redundancy: MeanVar::from_prior(mean, variance, count),
            refreshes: 2.0,
        }
    }
    fn assert_coupled_sane(
        g: &BTreeMap<String, f64>,
        pairs: &BTreeMap<UnorderedPair<String>, PairBaseline>,
        keys: &[String],
        cfg: &CouplingConfig,
        ctx: &str,
    ) {
        let cw = coupled_weights(g, pairs, keys, cfg);
        assert_weights_sum_to_n(&cw.weights, keys.len(), ctx);
        assert!(
            cw.effective_channels.is_finite(),
            "effective_channels not finite [{ctx}]: {}",
            cw.effective_channels
        );
        if !keys.is_empty() {
            assert!(
                cw.effective_channels >= 0.0,
                "effective_channels negative [{ctx}]"
            );
        }
    }

    #[test]
    fn channel_counts_one_two_three_sixteen() {
        for n in [1usize, 2, 3, 16] {
            let names: Vec<String> = (0..n).map(|i| format!("c{i}")).collect();
            let keys: Vec<String> = names.iter().map(|s| key(s)).collect();
            let g: BTreeMap<String, f64> = keys.iter().cloned().map(|k| (k, 1.0 + 0.1)).collect();
            assert_coupled_sane(
                &g,
                &BTreeMap::new(),
                &keys,
                &CouplingConfig::default(),
                &format!("n={n}"),
            );
        }
    }

    #[test]
    fn no_pairs_is_decoupled_limit() {
        let g = gmap(&[("a", 1.0), ("b", 2.0), ("c", 3.0)]);
        let keys = [key("a"), key("b"), key("c")];
        let cw = coupled_weights(&g, &BTreeMap::new(), &keys, &CouplingConfig::default());
        // w_c proportional to g_c, normalized to sum N.
        assert_abs_diff_eq!(cw.weights[&key("a")], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(cw.weights[&key("c")], 1.5, epsilon = 1e-12);
        assert_abs_diff_eq!(cw.effective_channels, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn pairs_referencing_unknown_channels_are_ignored() {
        let g = gmap(&[("a", 1.0), ("b", 1.0)]);
        let keys = [key("a"), key("b")];
        let mut pairs = BTreeMap::new();
        // Pairs that mention channels not in `keys` must not affect or crash the solve.
        pairs.insert(
            UnorderedPair::new(key("ghost1"), key("ghost2")),
            pair_redundancy(0.9, 0.0, 99.0),
        );
        pairs.insert(
            UnorderedPair::new(key("a"), key("ghost")),
            pair_redundancy(0.9, 0.0, 99.0),
        );
        let mut cfg = CouplingConfig::default();
        cfg.enabled = true;
        cfg.min_reliability = 1.0;
        let cw = coupled_weights(&g, &pairs, &keys, &cfg);
        // No usable in-set pair: weights stay at the decoupled limit (equal here).
        assert_abs_diff_eq!(cw.weights[&key("a")], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cw.weights[&key("b")], 1.0, epsilon = 1e-12);
        assert_coupled_sane(&g, &pairs, &keys, &cfg, "unknown-pairs");
    }

    #[test]
    fn near_singular_reliable_redundancy_stays_pd_and_finite() {
        // Redundancy ~0.999 on every pair. With the mandatory default shrink (0.5) the
        // assembled Sigma stays PD and the weights stay finite, non-negative, sum-N.
        let g = gmap(&[("a", 1.0), ("b", 1.0), ("c", 1.0)]);
        let keys = [key("a"), key("b"), key("c")];
        let mut pairs = BTreeMap::new();
        for (x, y) in [("a", "b"), ("a", "c"), ("b", "c")] {
            pairs.insert(
                UnorderedPair::new(key(x), key(y)),
                pair_redundancy(0.999, 0.0, 50.0),
            );
        }
        let mut cfg = CouplingConfig::default();
        cfg.enabled = true;
        cfg.min_reliability = 1.0;
        assert_coupled_sane(&g, &pairs, &keys, &cfg, "near-singular");
        let cw = coupled_weights(&g, &pairs, &keys, &cfg);
        // Three mutually-redundant channels collapse toward ~1 effective channel.
        assert!(cw.effective_channels > 0.0 && cw.effective_channels < 3.0);
    }

    #[test]
    fn pre_shrink_indefinite_r_falls_back_without_panic() {
        // shrink = 0, high cap, redundancies chosen so the raw R is indefinite
        // ([[1,.9,.9],[.9,1,-.9],[.9,-.9,1]] has a negative eigenvalue). The solve and the
        // inverse both fail gracefully and fall back to independence; nothing panics.
        let g = gmap(&[("a", 1.0), ("b", 1.0), ("c", 1.0)]);
        let keys = [key("a"), key("b"), key("c")];
        let mut pairs = BTreeMap::new();
        pairs.insert(
            UnorderedPair::new(key("a"), key("b")),
            pair_redundancy(0.9, 0.0, 50.0),
        );
        pairs.insert(
            UnorderedPair::new(key("a"), key("c")),
            pair_redundancy(0.9, 0.0, 50.0),
        );
        pairs.insert(
            UnorderedPair::new(key("b"), key("c")),
            pair_redundancy(-0.9, 0.0, 50.0),
        );
        let mut cfg = CouplingConfig::default();
        cfg.enabled = true;
        cfg.shrink_to_identity = 0.0;
        cfg.discount_cap = 0.95;
        cfg.min_reliability = 1.0;
        assert_coupled_sane(&g, &pairs, &keys, &cfg, "indefinite-R");
    }

    #[test]
    fn pathological_config_knobs() {
        let g = gmap(&[("a", 1.0), ("b", 1.0)]);
        let keys = [key("a"), key("b")];
        let mut pairs = BTreeMap::new();
        // A redundancy "mean" of 5.0 is not even a valid correlation (garbage-in).
        pairs.insert(
            UnorderedPair::new(key("a"), key("b")),
            pair_redundancy(5.0, 0.0, 50.0),
        );
        let variants = [
            ("cap-negative", {
                let mut c = CouplingConfig::default();
                c.enabled = true;
                c.discount_cap = -1.0;
                c
            }),
            ("cap-zero", {
                let mut c = CouplingConfig::default();
                c.enabled = true;
                c.discount_cap = 0.0;
                c
            }),
            ("cap-gt-one", {
                let mut c = CouplingConfig::default();
                c.enabled = true;
                c.discount_cap = 2.0;
                c.min_reliability = 1.0;
                c
            }),
            ("min-reliability-zero", {
                let mut c = CouplingConfig::default();
                c.enabled = true;
                c.min_reliability = 0.0;
                c
            }),
            ("var-gate-zero", {
                let mut c = CouplingConfig::default();
                c.enabled = true;
                c.stratum_stability_max_var = 0.0;
                c.min_reliability = 1.0;
                c
            }),
            ("var-gate-huge", {
                let mut c = CouplingConfig::default();
                c.enabled = true;
                c.stratum_stability_max_var = 1e300;
                c.min_reliability = 1.0;
                c
            }),
            ("shrink-negative", {
                let mut c = CouplingConfig::default();
                c.enabled = true;
                c.shrink_to_identity = -5.0;
                c.min_reliability = 1.0;
                c
            }),
            ("shrink-gt-one", {
                let mut c = CouplingConfig::default();
                c.enabled = true;
                c.shrink_to_identity = 5.0;
                c.min_reliability = 1.0;
                c
            }),
        ];
        for (name, cfg) in variants {
            assert_coupled_sane(&g, &pairs, &keys, &cfg, name);
        }
    }

    #[test]
    fn empty_channel_set() {
        let cw = coupled_weights(
            &BTreeMap::new(),
            &BTreeMap::new(),
            &[],
            &CouplingConfig::default(),
        );
        assert!(cw.weights.is_empty());
        assert_eq!(cw.effective_channels, 0.0);
    }

    #[test]
    fn missing_or_nonpositive_g_falls_back_to_neutral() {
        // g entries missing, zero, negative, or non-finite must not break the matrix.
        let g = gmap(&[("a", 0.0), ("b", -3.0), ("c", f64::NAN)]);
        let keys = [key("a"), key("b"), key("c"), key("d")]; // d absent from g
        assert_coupled_sane(
            &g,
            &BTreeMap::new(),
            &keys,
            &CouplingConfig::default(),
            "bad-g",
        );
    }

    // -- anchor_correlations ---------------------------------------------------------

    fn anchor_n(n: u32, scorer: impl Fn(u32, &str) -> Option<f64>) -> Anchor {
        let cands: Vec<u32> = (0..n).collect();
        let ca = chan("a");
        let cb = chan("b");
        let cc = chan("c");
        Anchor::build(&cands, &[&ca, &cb, &cc], |id, k| scorer(*id, k).map(Raw))
    }

    #[test]
    fn anchor_channel_all_none_is_omitted() {
        let anchor = anchor_n(200, |id, k| if k == "b" { None } else { Some(id as f64) });
        let corr = anchor_correlations(&anchor, &CouplingConfig::default());
        // No pair involving b can form (b is fully absent); only a-c may appear.
        assert!(corr.keys().all(|p| {
            let (x, y) = p.as_tuple();
            x.as_str() != "b" && y.as_str() != "b"
        }));
        for obs in corr.values() {
            assert!(obs.correlation.is_finite() && obs.correlation.abs() <= 1.0);
        }
    }

    #[test]
    fn anchor_all_constant_is_omitted() {
        let anchor = anchor_n(200, |_, _| Some(1.0));
        let corr = anchor_correlations(&anchor, &CouplingConfig::default());
        assert!(
            corr.is_empty(),
            "zero-variance channels yield no correlation"
        );
    }

    #[test]
    fn anchor_single_and_two_candidates() {
        for n in [1u32, 2] {
            let anchor = anchor_n(n, |id, _| Some(id as f64));
            let corr = anchor_correlations(&anchor, &CouplingConfig::default());
            assert!(corr.is_empty(), "n={n} below min_overlap");
        }
        // With min_overlap lowered to 2, two candidates still can't form a stable corr
        // (n=2 has nonzero variance, so it may appear, but must be finite and in range).
        let anchor = anchor_n(2, |id, _| Some(id as f64));
        let mut cfg = CouplingConfig::default();
        cfg.min_overlap = 2;
        let corr = anchor_correlations(&anchor, &cfg);
        for obs in corr.values() {
            assert!(obs.correlation.is_finite() && obs.correlation.abs() <= 1.0);
        }
    }

    #[test]
    fn anchor_one_channel_yields_no_pairs() {
        let cands: Vec<u32> = (0..100).collect();
        let ca = chan("a");
        let anchor = Anchor::build(&cands, &[&ca], |id, _| Some(Raw(*id as f64)));
        assert!(anchor_correlations(&anchor, &CouplingConfig::default()).is_empty());
    }

    #[test]
    fn anchor_facet_absent_random_subset() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        // b is absent on a random ~30% subset; correlation reads only the both-scored rest.
        let absent: std::collections::HashSet<u32> =
            (0..300u32).filter(|_| rng.gen_bool(0.3)).collect();
        let anchor = anchor_n(300, |id, k| {
            if k == "b" && absent.contains(&id) {
                None
            } else {
                Some(id as f64)
            }
        });
        let corr = anchor_correlations(&anchor, &CouplingConfig::default());
        for obs in corr.values() {
            assert!(obs.correlation.is_finite() && obs.correlation.abs() <= 1.0);
            assert!(obs.n_both >= CouplingConfig::default().min_overlap);
        }
    }

    // -- diagnostics -----------------------------------------------------------------

    /// Assert a diagnostics read against expected (confidence, conflict). Field access
    /// rather than struct literals: `Diagnostics` is `#[non_exhaustive]` outside the
    /// crate, which is exactly the consumer's view this integration test exercises.
    fn assert_diag(d: Diagnostics, confidence: f64, conflict: f64) {
        assert_abs_diff_eq!(d.confidence, confidence, epsilon = 1e-12);
        assert_abs_diff_eq!(d.conflict, conflict, epsilon = 1e-12);
    }

    #[test]
    fn diagnostics_edge_cases() {
        // Empty and single-set -> (0, 0).
        let none: Vec<(String, Vec<u32>)> = vec![];
        assert_diag(diagnostics(&none), 0.0, 0.0);
        assert_diag(diagnostics(&[(key("a"), vec![1u32, 2, 3])]), 0.0, 0.0);
        // All-identical -> (1, 0); all-disjoint -> (0, 1).
        let identical = vec![(key("a"), vec![1u32, 2, 3]), (key("b"), vec![1, 2, 3])];
        assert_diag(diagnostics(&identical), 1.0, 0.0);
        let disjoint = vec![(key("a"), vec![1u32, 2, 3]), (key("b"), vec![4, 5, 6])];
        assert_diag(diagnostics(&disjoint), 0.0, 1.0);
        // Duplicate ids inside a set collapse via the set; empty sets -> empty union -> (0,0).
        let dup = vec![
            (key("a"), vec![1u32, 1, 2, 2]),
            (key("b"), vec![2u32, 2, 2]),
        ];
        let d = diagnostics(&dup);
        assert_abs_diff_eq!(d.confidence, 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(d.confidence + d.conflict, 1.0, epsilon = 1e-12);
        let empties = vec![(key("a"), Vec::<u32>::new()), (key("b"), Vec::<u32>::new())];
        assert_diag(diagnostics(&empties), 0.0, 0.0);
    }

    #[test]
    fn diagnostics_many_sets() {
        let sets: Vec<(String, Vec<u32>)> = (0..16u32)
            .map(|i| (key(&format!("c{i}")), vec![0u32, i + 1]))
            .collect();
        let d = diagnostics(&sets);
        // Shared id 0 in every set; the rest disjoint. Jaccard = 1/17.
        assert!(d.confidence.is_finite() && (0.0..=1.0).contains(&d.confidence));
        assert_abs_diff_eq!(d.confidence + d.conflict, 1.0, epsilon = 1e-12);
    }
}

// ===================================================================================
// §8 — MeanVar
// ===================================================================================

mod meanvar {
    use super::*;

    #[test]
    fn decay_zero_then_push_recovers() {
        let mut mv = MeanVar::new();
        for x in [1.0, 2.0, 3.0, 4.0] {
            mv.push(x);
        }
        mv.decay(0.0);
        assert_abs_diff_eq!(mv.count(), 0.0, epsilon = 1e-12);
        assert_eq!(mv.zscore(1.0), None); // count 0 -> no z
        mv.push(5.0);
        assert_abs_diff_eq!(mv.count(), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mv.mean(), 5.0, epsilon = 1e-12);
        assert_mv_finite(&mv, "decay0-then-push");
    }

    #[test]
    fn decay_one_is_identity() {
        let mut mv = MeanVar::new();
        for x in [1.0, 2.0, 3.0] {
            mv.push(x);
        }
        let (c, m, v) = (mv.count(), mv.mean(), mv.variance());
        mv.decay(1.0);
        assert_abs_diff_eq!(mv.count(), c, epsilon = 1e-12);
        assert_abs_diff_eq!(mv.mean(), m, epsilon = 1e-12);
        assert_abs_diff_eq!(mv.variance(), v, epsilon = 1e-12);
    }

    #[test]
    fn repeated_decay_to_zero_is_graceful() {
        let mut mv = MeanVar::new();
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
            mv.push(x);
        }
        for _ in 0..200 {
            mv.decay(0.5);
            assert_mv_finite(&mv, "repeated-decay");
        }
        assert!(mv.count() < 1e-40); // driven essentially to zero
        // Still recovers on push.
        mv.push(9.0);
        assert!(mv.mean().is_finite());
    }

    #[test]
    fn from_prior_zero_variance_has_no_zscore() {
        let mv = MeanVar::from_prior(2.0, 0.0, 10.0);
        assert_eq!(mv.zscore(5.0), None);
        assert_mv_finite(&mv, "zero-var-prior");
        // A streamed value still folds in cleanly.
        let mut mv2 = mv;
        mv2.push(4.0);
        assert!(mv2.mean().is_finite() && mv2.mean() > 2.0);
    }

    #[test]
    fn from_prior_rejects_degenerate_inputs() {
        for mv in [
            MeanVar::from_prior(1.0, 1.0, 0.0),
            MeanVar::from_prior(1.0, 1.0, -3.0),
            MeanVar::from_prior(f64::NAN, 1.0, 1.0),
            MeanVar::from_prior(1.0, f64::INFINITY, 1.0),
            MeanVar::from_prior(1.0, 1.0, f64::NAN),
        ] {
            assert_eq!(mv, MeanVar::new());
        }
        // Negative variance is clamped to zero, not rejected.
        let clamped = MeanVar::from_prior(1.0, -5.0, 4.0);
        assert_abs_diff_eq!(clamped.variance(), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(clamped.count(), 4.0, epsilon = 1e-12);
    }

    #[test]
    fn push_after_decay_to_zero() {
        let mut mv = MeanVar::from_prior(3.0, 1.0, 5.0);
        mv.decay(0.0);
        mv.push(10.0);
        assert_abs_diff_eq!(mv.mean(), 10.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mv.count(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn merge_empty_empty_and_prior_with_stream() {
        let e1 = MeanVar::new();
        let e2 = MeanVar::new();
        let merged = MeanVar::merge(&e1, &e2);
        assert_eq!(merged, MeanVar::new());

        // Merge a from_prior summary with a streamed one: order-independent and finite.
        let prior = MeanVar::from_prior(0.5, 0.04, 4.0);
        let mut streamed = MeanVar::new();
        for x in [0.1, 0.9, 0.5] {
            streamed.push(x);
        }
        let ab = MeanVar::merge(&prior, &streamed);
        let ba = MeanVar::merge(&streamed, &prior);
        assert_abs_diff_eq!(ab.mean(), ba.mean(), epsilon = 1e-12);
        assert_abs_diff_eq!(ab.variance(), ba.variance(), epsilon = 1e-12);
        assert_abs_diff_eq!(ab.count(), 7.0, epsilon = 1e-12);
        assert_mv_finite(&ab, "prior+stream");
    }

    #[test]
    fn push_non_finite_is_dropped() {
        // The §7/§8 second line of defence: a stray non-finite never reaches the mean.
        let mut mv = MeanVar::new();
        mv.push(2.0);
        mv.push(f64::NAN);
        mv.push(f64::INFINITY);
        mv.push(f64::NEG_INFINITY);
        mv.push(4.0);
        assert_abs_diff_eq!(mv.count(), 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(mv.mean(), 3.0, epsilon = 1e-12);
    }

    #[test]
    fn extreme_magnitude_variance_never_nans_zscore() {
        // Huge declared variance with a tiny pseudo-count: variance/std may be large but
        // zscore must stay finite or return None, never NaN.
        let mv = MeanVar::from_prior(0.0, 1e200, 1e-3);
        let z = mv.zscore(1e100);
        if let Some(z) = z {
            assert!(z.is_finite());
        }
        assert_mv_finite(&mv, "huge-var");
    }
}

// ===================================================================================
// §8 — RuffleState merge / divergence / rekey / serde
// ===================================================================================

mod state_ops {
    use super::*;

    fn fp(dirs: &[(&str, Direction)]) -> StatFingerprint {
        let mut m = BTreeMap::new();
        for (k, d) in dirs {
            m.insert(key(k), *d);
        }
        StatFingerprint::new(BaselineMode::ZScore, m)
    }
    fn chan_sum(t: &str, sep: &[f64]) -> ChannelSummary {
        let mut c = ChannelSummary::new(t.to_string());
        for &x in sep {
            c.separation.push(x);
        }
        c
    }
    fn state(dirs: &[(&str, Direction)], chans: &[(&str, ChannelSummary)]) -> RuffleState {
        let mut s = RuffleState::new(fp(dirs));
        for (k, c) in chans {
            s.channels.insert(key(k), c.clone());
        }
        s
    }

    #[test]
    fn merge_single_and_ten_parts() {
        let dirs = &[("x", Direction::HigherIsBetter)];
        let one = state(dirs, &[("x", chan_sum("m", &[1.0, 2.0]))]);
        let (m1, d1) = RuffleState::merge(&[&one], MergePolicy::Strict).unwrap();
        assert_eq!(m1, one);
        assert_eq!(d1.max, 0.0);

        let parts: Vec<RuffleState> = (0..10)
            .map(|i| state(dirs, &[("x", chan_sum("m", &[i as f64]))]))
            .collect();
        let refs: Vec<&RuffleState> = parts.iter().collect();
        let (m10, _) = RuffleState::merge(&refs, MergePolicy::Strict).unwrap();
        assert_abs_diff_eq!(
            m10.channels[&key("x")].separation.count(),
            10.0,
            epsilon = 1e-9
        );
        assert_state_finite(&m10, "ten-parts");
    }

    #[test]
    fn merge_empty_channels_and_pairs_and_union_many() {
        // Parts with disjoint channels union to many; empty parts contribute nothing.
        let a = state(
            &[("a", Direction::HigherIsBetter)],
            &[("a", chan_sum("m", &[1.0]))],
        );
        let b = state(
            &[("b", Direction::HigherIsBetter)],
            &[("b", chan_sum("m", &[2.0]))],
        );
        let empty = RuffleState::new(fp(&[]));
        let (m, _) = RuffleState::merge(&[&a, &empty, &b], MergePolicy::Strict).unwrap();
        assert_eq!(m.channels.len(), 2);
        assert_state_finite(&m, "union-many");
    }

    #[test]
    fn merge_refusals_each_mismatch() {
        // Empty parts.
        let none: [&RuffleState; 0] = [];
        assert_eq!(
            RuffleState::merge(&none, MergePolicy::Strict).unwrap_err(),
            Mismatch::Empty
        );

        // Format version. `format_version` is library-managed with no setter, so the
        // mismatched state is produced the way a loaded file would carry it: edit the
        // serialized value and deserialize back.
        let a = state(&[], &[]);
        let mut bad_fmt_value = serde_json::to_value(state(&[], &[])).unwrap();
        bad_fmt_value["format_version"] = serde_json::Value::from(99u32);
        let bad_fmt: RuffleState = serde_json::from_value(bad_fmt_value).unwrap();
        assert!(matches!(
            RuffleState::merge(&[&a, &bad_fmt], MergePolicy::Strict).unwrap_err(),
            Mismatch::FormatVersion { .. }
        ));

        // Fingerprint stat_version. Set it on the `StatFingerprint` (its own fields stay
        // public) before building the state.
        let mut bad_fp_print = fp(&[]);
        bad_fp_print.stat_version = 42;
        let bad_fp = RuffleState::new(bad_fp_print);
        assert_eq!(
            RuffleState::merge(&[&a, &bad_fp], MergePolicy::Strict).unwrap_err(),
            Mismatch::Fingerprint
        );

        // Shared-tag conflict.
        let t1 = state(
            &[("x", Direction::HigherIsBetter)],
            &[("x", chan_sum("model-1", &[1.0]))],
        );
        let t2 = state(
            &[("x", Direction::HigherIsBetter)],
            &[("x", chan_sum("model-2", &[2.0]))],
        );
        assert!(matches!(
            RuffleState::merge(&[&t1, &t2], MergePolicy::Strict).unwrap_err(),
            Mismatch::Tag { .. }
        ));

        // Shared-direction conflict.
        let d1 = state(&[("x", Direction::HigherIsBetter)], &[]);
        let d2 = state(&[("x", Direction::LowerIsBetter)], &[]);
        assert!(matches!(
            RuffleState::merge(&[&d1, &d2], MergePolicy::Strict).unwrap_err(),
            Mismatch::DirectionConflict { .. }
        ));
    }

    #[test]
    fn rekey_self_existing_absent_roundtrip_and_pairs() {
        // to-self: no-op.
        let mut s = state(
            &[("a", Direction::HigherIsBetter)],
            &[("a", chan_sum("m", &[1.0, 2.0]))],
        );
        let before = s.clone();
        s.rekey(&key("a"), key("a"));
        assert_eq!(s, before);

        // absent key: no-op-ish, no panic.
        s.rekey(&key("ghost"), key("z"));
        assert!(!s.channels.contains_key(&key("z")));

        // round-trip a -> b -> a preserves the mean.
        let mean0 = s.channels[&key("a")].separation.mean();
        s.rekey(&key("a"), key("b"));
        s.rekey(&key("b"), key("a"));
        assert_abs_diff_eq!(
            s.channels[&key("a")].separation.mean(),
            mean0,
            epsilon = 1e-12
        );

        // to-existing merges and keeps destination tag; a pair mentioning `from` is rebuilt.
        let mut s2 = state(
            &[
                ("old", Direction::HigherIsBetter),
                ("new", Direction::HigherIsBetter),
                ("p", Direction::HigherIsBetter),
            ],
            &[
                ("old", chan_sum("m", &[1.0, 2.0, 3.0])),
                ("new", chan_sum("keep", &[10.0, 11.0])),
                ("p", chan_sum("m", &[4.0])),
            ],
        );
        let mut pr = PairSummary::new();
        pr.redundancy.push(0.5);
        s2.pairs
            .insert(UnorderedPair::new(key("old"), key("p")), pr);
        s2.rekey(&key("old"), key("new"));
        assert!(!s2.channels.contains_key(&key("old")));
        assert_abs_diff_eq!(
            s2.channels[&key("new")].separation.count(),
            5.0,
            epsilon = 1e-12
        );
        assert_eq!(s2.channels[&key("new")].tag.as_str(), "keep");
        assert!(
            s2.pairs
                .contains_key(&UnorderedPair::new(key("new"), key("p")))
        );
        assert!(
            !s2.pairs
                .contains_key(&UnorderedPair::new(key("old"), key("p")))
        );
        assert_state_finite(&s2, "rekey-merge");
    }

    #[test]
    fn divergence_identical_disjoint_and_collapsed() {
        // Identical -> 0.
        let a = state(
            &[("x", Direction::HigherIsBetter)],
            &[("x", chan_sum("m", &[1.0, 2.0, 3.0]))],
        );
        let d_id = a.divergence(&a.clone());
        assert_eq!(d_id.max, 0.0);

        // Disjoint channels -> empty per-channel map, max 0.
        let b = state(
            &[("y", Direction::HigherIsBetter)],
            &[("y", chan_sum("m", &[4.0, 5.0]))],
        );
        let d_dis = a.divergence(&b);
        assert!(d_dis.per_channel.is_empty());
        assert_eq!(d_dis.max, 0.0);

        // Zero-variance vs shifted-zero-variance -> capped, finite (an infinite z-distance).
        let z1 = state(
            &[("x", Direction::HigherIsBetter)],
            &[("x", chan_sum("m", &[5.0, 5.0, 5.0]))],
        );
        let z2 = state(
            &[("x", Direction::HigherIsBetter)],
            &[("x", chan_sum("m", &[8.0, 8.0, 8.0]))],
        );
        let d_cap = z1.divergence(&z2);
        assert!(d_cap.per_channel[&key("x")].is_finite());
        assert!(d_cap.per_channel[&key("x")] >= 1e6 - 1.0);
        assert!(d_cap.max.is_finite());
    }

    #[test]
    fn serde_json_round_trips_with_pairs_decay_and_many_channels() {
        let mut s = RuffleState::new(fp(&[
            ("x", Direction::HigherIsBetter),
            ("y", Direction::LowerIsBetter),
            ("z", Direction::HigherIsBetter),
        ]));
        for (k, t) in [("x", "model-x"), ("y", "model-y"), ("z", "model-z")] {
            let mut c = chan_sum(t, &[1.0, 2.0, 3.0, 4.0]);
            c.reference = MeanVar::from_prior(0.4, 0.02, 3.0);
            s.channels.insert(key(k), c);
        }
        let mut p1 = PairSummary::new();
        p1.redundancy.push(0.25);
        p1.redundancy.push(0.35);
        s.pairs.insert(UnorderedPair::new(key("x"), key("y")), p1);
        let mut p2 = PairSummary::new();
        p2.redundancy.push(0.5);
        s.pairs.insert(UnorderedPair::new(key("y"), key("z")), p2);
        // Decay so the counts are fractional and exercise the f64 round-trip.
        s.decay(0.97);

        let json = serde_json::to_string(&s).unwrap();
        let back: RuffleState = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back, "serde round-trip not exact");

        // Identical content serializes byte-identically (content-addressing, §8).
        let json2 = serde_json::to_string(&back).unwrap();
        assert_eq!(json, json2);
    }
}

// ===================================================================================
// §11 — Fuser lifecycle
// ===================================================================================

mod fuser_lifecycle {
    use super::*;

    /// A spiked pool with a clear top elevation (>= 8 distinct bulk values + 2 spikes).
    fn spiked(base: u32) -> Vec<(u32, Raw)> {
        let mut v: Vec<(u32, Raw)> = (0..30).map(|i| (base + i, Raw(i as f64 * 0.1))).collect();
        v.push((base + 100, Raw(10.0)));
        v.push((base + 101, Raw(10.5)));
        v
    }
    fn scored_obs(cfg: &ChannelConfig, base: u32) -> ChannelInput<u32> {
        ChannelInput::scored(cfg, spiked(base))
    }
    fn ranks_obs(cfg: &ChannelConfig, ids: &[u32]) -> ChannelInput<u32> {
        ChannelInput::ranked(cfg, ids.to_vec())
    }

    #[test]
    fn fuse_empty_obs_is_empty_and_safe() {
        let a = chan("a");
        let mut f = Fuser::new(&[a], FuseConfig::default()).unwrap();
        let fused = f.fuse(&[] as &[ChannelInput<u32>]);
        assert!(fused.ranking.is_empty());
        assert!(fused.weights.is_empty());
        assert_eq!((fused.confidence, fused.conflict), (0.0, 0.0));
        assert_state_finite(f.state(), "fuse-empty");
    }

    #[test]
    fn only_unregistered_channels_are_skipped() {
        let a = chan("a");
        let mut f = Fuser::new(&[a], FuseConfig::default()).unwrap();
        // ChannelInput for "ghost", which is not registered: skipped entirely.
        let ghost = ChannelInput {
            key: key("ghost"),
            items: Items::Scored(vec![(1u32, 0.9), (2, 0.5)]),
        };
        let fused = f.fuse(std::slice::from_ref(&ghost));
        assert!(
            fused.ranking.is_empty(),
            "unregistered channel must not fuse"
        );
        assert!(fused.weights.is_empty());
        assert!(!f.state().channels.contains_key(&key("ghost")));
    }

    #[test]
    fn same_query_one_thousand_times_stays_finite_and_sum_n() {
        let a = chan("a");
        let b = chan("b");
        let mut f = Fuser::new(&[a.clone(), b.clone()], FuseConfig::default()).unwrap();
        let obs = vec![scored_obs(&a, 0), scored_obs(&b, 0)];
        for i in 0..1000 {
            let fused = f.fuse(&obs);
            assert_weights_sum_to_n(&fused.weights, 2, &format!("iter-{i}"));
            assert_ranking_finite(&fused.ranking, &format!("iter-{i}"));
            assert!(fused.confidence.is_finite() && fused.conflict.is_finite());
        }
        // Baselines grew (one separation push per fuse) and stayed finite.
        assert_abs_diff_eq!(
            f.state().channels[&key("a")].separation.count(),
            1000.0,
            epsilon = 1e-6
        );
        assert_state_finite(f.state(), "1000x");
    }

    #[test]
    fn fuse_stateless_empty_prior_is_unweighted_rrf_and_does_not_mutate() {
        let a = chan("a");
        let b = chan("b");
        let cfgs = [a.clone(), b.clone()];
        let prior = empty_state();
        let before = prior.clone();
        let obs = vec![scored_obs(&a, 0), scored_obs(&b, 0)];
        let fused = Fuser::fuse_stateless(&obs, &cfgs, &prior, &FuseConfig::default()).unwrap();
        for w in fused.weights.values() {
            assert_abs_diff_eq!(*w, 1.0, epsilon = 1e-9);
        }
        let direct = weighted_rrf(
            &obs,
            &fused.weights,
            &rrf(FuseConfig::default().fusion.rrf_eta),
        );
        assert_eq!(fused.ranking, direct);
        assert_eq!(prior, before, "fuse_stateless must not mutate the prior");
    }

    #[test]
    fn refresh_coupling_empty_and_single_candidate_anchor() {
        let mut fc = FuseConfig::default();
        fc.coupling.enabled = true;
        let mut f = Fuser::new(&[chan("a"), chan("b")], fc).unwrap();
        // Empty anchor: no candidates, nothing accumulates.
        let ca = chan("a");
        let cb = chan("b");
        let empty_anchor = Anchor::build(&[] as &[u32], &[&ca, &cb], |id, _| Some(Raw(*id as f64)));
        f.refresh_coupling(&empty_anchor);
        assert!(f.state().pairs.is_empty());

        // Single candidate: overlap 1 < min_overlap, still nothing.
        let one = Anchor::build(&[0u32], &[&ca, &cb], |id, _| Some(Raw(*id as f64)));
        f.refresh_coupling(&one);
        assert!(f.state().pairs.is_empty());
        assert_state_finite(f.state(), "refresh-empty");
    }

    #[test]
    fn all_three_flags_exercised_weights_still_sum_n() {
        // Three channels: one ranks-only (RanksOnly flag), one scored-degenerate (Degenerate
        // flag), one scored-well-separated-but-cold-reference (NoReference flag). Every flag
        // path fires in one fuse and the weights still sum to N = 3.
        let ranks_ch = chan("ranks");
        let degen_ch = chan("degen");
        let cold_ref_ch = chan("coldref");
        let mut f = Fuser::new(
            &[ranks_ch.clone(), degen_ch.clone(), cold_ref_ch.clone()],
            FuseConfig::default(),
        )
        .unwrap();
        let obs = vec![
            ranks_obs(&ranks_ch, &[1, 2, 3]),
            // < 8 distinct values -> degenerate separation.
            ChannelInput::scored(
                &degen_ch,
                vec![(10u32, Raw(1.0)), (11, Raw(1.0)), (12, Raw(2.0))],
            ),
            // well-separated, but no declared reference and a cold baseline -> NoReference.
            scored_obs(&cold_ref_ch, 100),
        ];
        let fused = f.fuse(&obs);
        assert_eq!(
            fused.flags.get(&key("ranks")),
            Some(&ChannelFlag::RanksOnlyDefaultWeighted)
        );
        assert_eq!(
            fused.flags.get(&key("degen")),
            Some(&ChannelFlag::DegenerateSeparation)
        );
        assert_eq!(
            fused.flags.get(&key("coldref")),
            Some(&ChannelFlag::NoReference)
        );
        assert_weights_sum_to_n(&fused.weights, 3, "three-flags");
        assert_ranking_finite(&fused.ranking, "three-flags");
    }

    #[test]
    fn coupling_enabled_end_to_end_stays_finite() {
        // Turn coupling ON, accumulate a reliable redundancy via the anchor, then fuse and
        // confirm the discounted weights are finite, non-negative, and sum to N.
        let a = chan("a");
        let b = chan("b");
        let c = chan("c");
        let mut fc = FuseConfig::default();
        fc.coupling.enabled = true;
        fc.coupling.min_reliability = 1.0;
        let mut f = Fuser::new(&[a.clone(), b.clone(), c.clone()], fc).unwrap();
        // a and b strongly redundant on the anchor; c independent-ish.
        let cands: Vec<u32> = (0..200).collect();
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let noise: Vec<f64> = (0..200).map(|_| rng.gen_range(-0.01..0.01)).collect();
        let anchor = Anchor::build(&cands, &[&a, &b, &c], |id, k| {
            let v = match k {
                "a" => *id as f64,
                "b" => *id as f64 + noise[*id as usize],
                _ => (200 - *id) as f64,
            };
            Some(Raw(v))
        });
        f.refresh_coupling(&anchor);
        let obs = vec![scored_obs(&a, 0), scored_obs(&b, 0), scored_obs(&c, 500)];
        for _ in 0..10 {
            let fused = f.fuse(&obs);
            assert_weights_sum_to_n(&fused.weights, 3, "coupled-e2e");
            assert_ranking_finite(&fused.ranking, "coupled-e2e");
        }
        assert_state_finite(f.state(), "coupled-e2e");
    }
}

// ===================================================================================
// Randomized fuzz across every surface (deterministic seed).
// ===================================================================================

mod fuzz {
    use super::*;

    /// One random finite f64, occasionally an extreme-but-finite magnitude.
    fn rand_val(rng: &mut ChaCha8Rng) -> f64 {
        match rng.gen_range(0u8..10) {
            0 => f64::MAX,
            1 => f64::MIN,
            2 => 1e-300,
            3 => -0.0,
            4 => f64::MIN_POSITIVE,
            5 => rng.gen_range(-1e6..1e6),
            _ => rng.gen_range(-3.0..3.0),
        }
    }

    #[test]
    fn fuzz_discriminate_via_observe_stays_within_envelope() {
        let cfg = dcfg();
        let mut rng = ChaCha8Rng::seed_from_u64(0xD15C);
        for it in 0..3000 {
            let n = rng.gen_range(0usize..60);
            // Build through observe, occasionally injecting NaN/inf that observe must drop.
            let scored: Vec<(u32, Raw)> = (0..n)
                .map(|i| {
                    let v = if rng.gen_bool(0.1) {
                        let bad = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
                        bad[rng.gen_range(0..3)]
                    } else {
                        rand_val(&mut rng)
                    };
                    (i as u32, Raw(v))
                })
                .collect();
            let dir = if rng.gen_bool(0.5) {
                Direction::HigherIsBetter
            } else {
                Direction::LowerIsBetter
            };
            let c = chan_with("x", dir, None);
            let obs = ChannelInput::scored(&c, scored);
            // No non-finite survived observe.
            if let Items::Scored(v) = &obs.items {
                assert!(
                    v.iter().all(|(_, s)| s.is_finite()),
                    "observe leaked non-finite [it={it}]"
                );
            }
            // Random summary: cold, warm, or seeded.
            let summary = match rng.gen_range(0u8..3) {
                0 => cold(),
                1 => warm(),
                _ => {
                    let mut s = ChannelSummary::new(tag());
                    s.separation = MeanVar::from_prior(
                        rng.gen_range(-2.0..2.0),
                        rng.gen_range(0.0..3.0),
                        rng.gen_range(0.0..30.0),
                    );
                    s.reference = MeanVar::from_prior(
                        rng.gen_range(-2.0..2.0),
                        rng.gen_range(0.0..3.0),
                        rng.gen_range(0.0..30.0),
                    );
                    s
                }
            };
            let d = discriminate_summary(&obs.items, &summary, &cfg);
            // g is ALWAYS finite and bounded, for every pool, every summary (the headline
            // robustness guarantee: the zscore guards neutralize any non-finite read).
            assert_disc_sane(&d, &cfg, &format!("fuzz-disc it={it}"));

            // The discrimination READ fields (raw_separation, top_m_average) are finite
            // EXCEPT for the documented numerator-overflow defect: a top dominated by
            // ~f64::MAX scores sums past f64 and reads +/-inf (NEVER NaN). See
            // `defect_discrimination_reads_should_stay_finite_under_extreme_scores`. Anything
            // else non-finite (a NaN, or an inf with no extreme score present) is a NEW defect.
            let maxabs = match &obs.items {
                Items::Scored(v) => v.iter().map(|(_, s)| s.abs()).fold(0.0, f64::max),
                Items::Ranks(_) => 0.0,
            };
            let overflow_possible = maxabs > 1e307;
            for (field, val) in [("raw_sep", d.raw_separation), ("top_m", d.top_m_average)] {
                if let Some(x) = val {
                    if !x.is_finite() {
                        assert!(
                            x.is_infinite() && overflow_possible,
                            "{field} non-finite outside the known overflow envelope \
                             [it={it}]: val={x} maxabs={maxabs}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fuzz_weighted_rrf_never_panics_or_nans() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xF00D);
        for it in 0..3000 {
            let n_ch = rng.gen_range(0usize..6);
            let mut obs: Vec<ChannelInput<u32>> = Vec::new();
            let mut weights: BTreeMap<String, f64> = BTreeMap::new();
            for c in 0..n_ch {
                let kname = format!("c{c}");
                let len = rng.gen_range(0usize..40);
                let items = if rng.gen_bool(0.5) {
                    let v: Vec<(u32, f64)> = (0..len)
                        .map(|_| (rng.gen_range(0u32..50), rng.gen_range(-1e3..1e3)))
                        .collect();
                    Items::Scored(v)
                } else {
                    let v: Vec<u32> = (0..len).map(|_| rng.gen_range(0u32..50)).collect();
                    Items::Ranks(v)
                };
                obs.push(ChannelInput {
                    key: key(&kname),
                    items,
                });
                // Sometimes give a weight (incl. 0 / huge), sometimes leave it to default.
                if rng.gen_bool(0.7) {
                    let w = match rng.gen_range(0u8..4) {
                        0 => 0.0,
                        1 => 1e9,
                        2 => rng.gen_range(0.0..5.0),
                        _ => 1.0,
                    };
                    weights.insert(key(&kname), w);
                }
            }
            let etas = [0.0, 1.0, 60.0, 1e6, f64::MAX];
            let eta = etas[rng.gen_range(0..5)];
            let out = weighted_rrf(&obs, &weights, &rrf(eta));
            assert_ranking_finite(&out, &format!("fuzz-rrf it={it}"));
            // Output is sorted by score descending.
            for w in out.windows(2) {
                assert!(w[0].1 >= w[1].1 - 1e-9, "not sorted desc [it={it}]");
            }
        }
    }

    #[test]
    fn fuzz_coupled_weights_invariants_hold() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xC0DE);
        for it in 0..3000 {
            let n = rng.gen_range(1usize..8);
            let keys: Vec<String> = (0..n).map(|i| key(&format!("c{i}"))).collect();
            let g: BTreeMap<String, f64> = keys
                .iter()
                .cloned()
                .map(|k| (k, rng.gen_range(0.25..4.0)))
                .collect();
            // Random pair redundancy baselines over random in-set (and a few out-of-set) pairs.
            let mut pairs: BTreeMap<UnorderedPair<String>, PairBaseline> = BTreeMap::new();
            for _ in 0..rng.gen_range(0..(n * 2)) {
                let i = rng.gen_range(0..n);
                let j = rng.gen_range(0..n);
                if i == j {
                    continue;
                }
                let mean = rng.gen_range(-1.5..1.5);
                let var = rng.gen_range(0.0..0.5);
                let count = rng.gen_range(0.0..40.0);
                let refreshes = rng.gen_range(0.0..5.0);
                pairs.insert(
                    UnorderedPair::new(keys[i].clone(), keys[j].clone()),
                    PairBaseline {
                        redundancy: MeanVar::from_prior(mean, var, count),
                        refreshes,
                    },
                );
            }
            let mut cfg = CouplingConfig::default();
            cfg.enabled = rng.gen_bool(0.7);
            cfg.discount_cap = rng.gen_range(-0.2..1.5);
            cfg.shrink_to_identity = rng.gen_range(0.0..1.0);
            cfg.min_overlap = rng.gen_range(1..40);
            cfg.min_reliability = rng.gen_range(0.0..20.0);
            cfg.stratum_stability_max_var = rng.gen_range(0.0..1.0);
            let cw = coupled_weights(&g, &pairs, &keys, &cfg);
            assert_weights_sum_to_n(&cw.weights, n, &format!("fuzz-coupled it={it}"));
            assert!(
                cw.effective_channels.is_finite() && cw.effective_channels >= 0.0,
                "effective_channels bad [it={it}]: {}",
                cw.effective_channels
            );
        }
    }

    #[test]
    fn fuzz_fuser_lifecycle_keeps_state_finite() {
        let mut rng = ChaCha8Rng::seed_from_u64(0xBEEF);
        let names = ["a", "b", "c"];
        let cfgs: Vec<ChannelConfig> = names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let good = if i == 0 {
                    Some(GoodScore::new(0.0, 1.0, 5.0))
                } else {
                    None
                };
                chan_with(n, Direction::HigherIsBetter, good)
            })
            .collect();
        let mut coupling = CouplingConfig::default();
        coupling.enabled = true;
        coupling.min_reliability = 1.0;
        let mut decay = DecayConfig::default();
        decay.enabled = rng.gen_bool(0.5);
        decay.factor = 0.95;
        let mut fc = FuseConfig::default();
        fc.coupling = coupling;
        fc.decay = decay;
        let mut f = Fuser::new(&cfgs, fc).unwrap();
        for it in 0..400 {
            // Random observations over a random subset of channels, mixing scored & ranks,
            // empty pools, and degenerate pools.
            let mut obs: Vec<ChannelInput<u32>> = Vec::new();
            for cfg in &cfgs {
                if rng.gen_bool(0.3) {
                    continue; // channel absent this query
                }
                if rng.gen_bool(0.5) {
                    let len = rng.gen_range(0usize..40);
                    let scored: Vec<(u32, Raw)> = (0..len)
                        .map(|i| (i as u32, Raw(rng.gen_range(-5.0..5.0))))
                        .collect();
                    obs.push(ChannelInput::scored(cfg, scored));
                } else {
                    let len = rng.gen_range(0usize..40);
                    let ids: Vec<u32> = (0..len).map(|_| rng.gen_range(0u32..60)).collect();
                    obs.push(ChannelInput::ranked(cfg, ids));
                }
            }
            let present = obs.len();
            let fused = f.fuse(&obs);
            assert_weights_sum_to_n(&fused.weights, present, &format!("fuzz-fuse it={it}"));
            assert_ranking_finite(&fused.ranking, &format!("fuzz-fuse it={it}"));
            assert!(fused.confidence.is_finite() && fused.conflict.is_finite());
            assert_state_finite(f.state(), &format!("fuzz-fuse it={it}"));

            // Occasionally refresh coupling from a random anchor.
            if rng.gen_bool(0.1) {
                let refs: Vec<&ChannelConfig> = cfgs.iter().collect();
                let cands: Vec<u32> = (0..rng.gen_range(0u32..120)).collect();
                let anchor = Anchor::build(&cands, &refs, |id, _| {
                    if rng_bool_from(*id) {
                        None
                    } else {
                        Some(Raw(*id as f64))
                    }
                });
                f.refresh_coupling(&anchor);
                assert_state_finite(f.state(), &format!("fuzz-refresh it={it}"));
            }
        }
    }

    /// A deterministic "facet absent" predicate (no RNG capture needed inside the closure).
    fn rng_bool_from(id: u32) -> bool {
        id % 7 == 0
    }
}
