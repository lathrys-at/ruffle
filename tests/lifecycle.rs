//! End-to-end lifecycle tests for `ruffle` as a unit (§8, §11).
//!
//! These drive the public library surface the way a deployment would: warm baselines by
//! fusing real queries, persist the state through `serde_json`, reload it, and continue;
//! reconcile state accumulated by independent deployments; and exercise the corruption
//! guards (the required tag) and the safe rename (`rekey`). The focus is the whole
//! system across persistence, reconciliation, and continued fusion, not any one layer.
//!
//! Every float assertion that demands a *bit-exact* reload depends on
//! `serde_json`'s `float_roundtrip` feature (enabled in `Cargo.toml`); without it a
//! reloaded baseline would differ in the last ULP and the round-trip equality below
//! would be a defect, not a pass.

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use ruffle::{
    Anchor, BaselineMode, ChannelConfig, ChannelId, ChannelInput, ChannelSummary, Direction,
    FuseConfig, Fuser, GoodScore, MeanVar, MergePolicy, Mismatch, RuffleState, Score,
    StatFingerprint, UnorderedPair,
};
use std::collections::BTreeMap;

// --- the caller-side score newtype: the only way a bare number becomes a Score (§7) ---

struct Sim(f64);
impl Score for Sim {
    fn value(&self) -> f64 {
        self.0
    }
}

fn key(s: &str) -> String {
    s.to_string()
}

/// A scored channel config: higher-is-better, its own tag, an optional declared prior.
fn cfg(name: &str, tag: &str, good: Option<GoodScore>) -> ChannelConfig {
    ChannelConfig::new(ChannelId::new(name, tag), Direction::HigherIsBetter, good)
}

fn empty_state() -> RuffleState {
    RuffleState::new(StatFingerprint::new(BaselineMode::ZScore, BTreeMap::new()))
}

/// A spiked pool: `bulk` distinct random values in `[0, 1)` plus two spikes `high`
/// units up, so the separation statistic is well defined (≥ 8 distinct values and a
/// clear top elevation, §4). `base` offsets the ids so channels can disjoin their pools.
fn spiked(rng: &mut ChaCha8Rng, base: u32, bulk: u32, high: f64) -> Vec<(u32, Sim)> {
    let mut v: Vec<(u32, Sim)> = (0..bulk)
        .map(|i| (base + i, Sim(rng.r#gen::<f64>())))
        .collect();
    v.push((base + 1_000, Sim(high)));
    v.push((base + 1_001, Sim(high + 0.5)));
    v
}

/// One query of scored observations, one per config, each on its own id range and at the
/// given spike height. Deterministic for a fixed rng stream.
fn query(rng: &mut ChaCha8Rng, cfgs: &[ChannelConfig], high: f64) -> Vec<ChannelInput<u32>> {
    cfgs.iter()
        .enumerate()
        .map(|(i, c)| ChannelInput::scored(c, spiked(rng, (i as u32) * 10_000, 30, high)))
        .collect()
}

/// A full-scored anchor over every channel, with deterministic per-channel score rows
/// that carry variance and pairwise co-movement, so `refresh_coupling` stores real pair
/// summaries. 60 candidates clears the default `min_overlap` of 30 (§5.3).
fn anchor(cfgs: &[ChannelConfig]) -> Anchor {
    let cands: Vec<u32> = (0..60).collect();
    let refs: Vec<&ChannelConfig> = cfgs.iter().collect();
    Anchor::build(&cands, &refs, |id, k| {
        let x = *id as f64;
        let v = match k {
            // a and b co-move (redundant); c is on a different axis.
            "a" => x,
            "b" => x + (id % 5) as f64,
            _ => ((id * 7) % 13) as f64,
        };
        Some(Sim(v))
    })
}

// --- 1. warm -> persist -> reload -> continue: the round-trip is bit-exact ------------

#[test]
fn warm_persist_reload_then_continue_is_bit_exact_and_consistent() {
    let cfgs = [
        cfg("a", "ta", None),
        cfg("b", "tb", None),
        cfg("c", "tc", None),
    ];
    let mut warm = Fuser::new(&cfgs, FuseConfig::default()).unwrap();

    // Burn-in: 40 seeded queries grow each channel's separation and reference baselines.
    let mut rng = ChaCha8Rng::seed_from_u64(0xA11CE);
    for _ in 0..40 {
        let obs = query(&mut rng, &cfgs, 5.0);
        warm.fuse(&obs);
    }
    // Refresh coupling on an unselected anchor so the persisted state carries pairs too.
    warm.refresh_coupling(&anchor(&cfgs));

    // The state is non-trivial: every channel warmed, and the pair map is populated.
    for name in ["a", "b", "c"] {
        let ch = &warm.state().channels[&key(name)];
        assert_eq!(
            ch.separation.count(),
            40.0,
            "{name} separation grew once per query"
        );
        assert_eq!(
            ch.reference.count(),
            40.0,
            "{name} reference learned a top per query"
        );
    }
    assert!(
        !warm.state().pairs.is_empty(),
        "refresh_coupling stored pair summaries"
    );

    // Persist -> tempfile -> reload.
    let json = serde_json::to_string(warm.state()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.json");
    std::fs::write(&path, &json).unwrap();
    let reloaded_state: RuffleState =
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();

    // Bit-exact: the reloaded state equals the in-memory one, and re-serializes to the
    // identical bytes (content-addressing, §8). This is the float_roundtrip guarantee.
    assert_eq!(reloaded_state, *warm.state(), "reload must be bit-exact");
    assert_eq!(
        serde_json::to_string(&reloaded_state).unwrap(),
        json,
        "re-serialization is byte-identical"
    );

    // A fresh Fuser over the reloaded state continues exactly as the un-persisted one
    // would: same query, same weights, same ranking, same diagnostics, same next state.
    let mut reloaded = Fuser::resume(&cfgs, reloaded_state, FuseConfig::default()).unwrap();
    let cont = query(&mut rng, &cfgs, 5.0);
    let from_warm = warm.fuse(&cont);
    let from_reload = reloaded.fuse(&cont);

    assert_eq!(
        from_warm, from_reload,
        "continued fusion matches across the persist boundary"
    );
    assert_eq!(
        warm.state(),
        reloaded.state(),
        "the two states stay in lockstep after continuing"
    );
    assert!(!from_reload.ranking.is_empty());
    let total: f64 = from_reload.weights.values().sum();
    assert!((total - 3.0).abs() < 1e-9, "weights sum to N = 3");
}

// --- 2. cross-deployment reconcile: two streams pool into one usable state ------------

#[test]
fn cross_deployment_reconcile_pools_counts_and_reports_drift() {
    // Same channel configs and tags, two independent deployments, different query
    // streams (different seeds AND different spike regimes, so the baselines drift apart).
    let cfgs = [cfg("a", "ta", None), cfg("b", "tb", None)];

    let mut left = Fuser::new(&cfgs, FuseConfig::default()).unwrap();
    let mut rng_l = ChaCha8Rng::seed_from_u64(1);
    for _ in 0..25 {
        let obs = query(&mut rng_l, &cfgs, 5.0);
        left.fuse(&obs);
    }

    let mut right = Fuser::new(&cfgs, FuseConfig::default()).unwrap();
    let mut rng_r = ChaCha8Rng::seed_from_u64(2);
    for _ in 0..18 {
        let obs = query(&mut rng_r, &cfgs, 9.0); // a different separation regime
        right.fuse(&obs);
    }

    // Serialize both, then reconcile the deserialized states (the real persistence path).
    let a: RuffleState =
        serde_json::from_str(&serde_json::to_string(left.state()).unwrap()).unwrap();
    let b: RuffleState =
        serde_json::from_str(&serde_json::to_string(right.state()).unwrap()).unwrap();

    let (merged, divergence) = RuffleState::merge(&[&a, &b], MergePolicy::Strict).unwrap();

    // The pooled separation count is exactly the sum of the two deployments' counts.
    for name in ["a", "b"] {
        let want =
            a.channels[&key(name)].separation.count() + b.channels[&key(name)].separation.count();
        assert!(
            (merged.channels[&key(name)].separation.count() - want).abs() < 1e-9,
            "{name}: merged count {} != {} + {}",
            merged.channels[&key(name)].separation.count(),
            a.channels[&key(name)].separation.count(),
            b.channels[&key(name)].separation.count()
        );
    }
    assert!((merged.channels[&key("a")].separation.count() - 43.0).abs() < 1e-9); // 25 + 18

    // The advisory divergence is finite and reflects the drift between the two streams.
    assert!(divergence.max.is_finite(), "divergence stays finite");
    assert!(
        divergence.max > 0.0,
        "differing streams drift, so divergence is positive"
    );
    for d in divergence.per_channel.values() {
        assert!(d.is_finite() && *d >= 0.0);
    }

    // The merged state drops straight into a fresh Fuser and keeps fusing.
    let mut resumed = Fuser::resume(&cfgs, merged, FuseConfig::default()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let fused = resumed.fuse(&query(&mut rng, &cfgs, 5.0));
    assert!(
        !fused.ranking.is_empty(),
        "merged state is immediately usable"
    );
}

// --- 3. operator prior cold-start: a declared good-score powers D^abs immediately -----

#[test]
fn operator_prior_powers_d_abs_from_the_first_query() {
    // x carries a declared good-score reference (so D^abs is available at cold start),
    // y carries none (so it cold-starts on separation alone and D^abs is unavailable).
    // The reference is declared low against the pool's tops, so a top reads as good.
    let x = cfg("x", "tx", Some(GoodScore::new(0.0, 0.5, 50.0)));
    let y = cfg("y", "ty", None);
    let cfgs = [x.clone(), y.clone()];

    // First query: the prior channel already has a usable reference (no NoReference
    // flag), while the no-prior channel is flagged NoReference (§4, §8).
    let mut f = Fuser::new(&cfgs, FuseConfig::default()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(11);
    let first = f.fuse(&query(&mut rng, &cfgs, 5.0));
    assert_ne!(
        first.flags.get(&key("x")),
        Some(&ruffle::ChannelFlag::NoReference),
        "the declared prior makes D^abs available from the first query"
    );
    assert_eq!(
        first.flags.get(&key("y")),
        Some(&ruffle::ChannelFlag::NoReference),
        "the no-prior channel cold-starts without D^abs"
    );
    // The prior reference was seeded with its pseudo-count, not learned from traffic.
    assert!(f.state().channels[&key("x")].reference.count() >= 50.0);
    assert_eq!(f.state().channels[&key("y")].reference.count(), 1.0); // learned its first top only

    // Once the separation baseline warms (so the per-channel shrinkage relaxes), the
    // good-score channel's D^abs tilts weight toward it on an identical pool. Feed both
    // channels the SAME pool each query so only the reference distinguishes them.
    let mut f = Fuser::new(&cfgs, FuseConfig::default()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(12);
    let mut last_weights = BTreeMap::new();
    for _ in 0..8 {
        let shared = spiked(&mut rng, 0, 30, 5.0);
        // Reuse the identical scored list for both channels (same ids, same scores).
        let items: Vec<(u32, f64)> = shared.iter().map(|(id, s)| (*id, s.value())).collect();
        let obs = vec![
            ChannelInput {
                key: key("x"),
                items: ruffle::Items::Scored(items.clone()),
            },
            ChannelInput {
                key: key("y"),
                items: ruffle::Items::Scored(items),
            },
        ];
        last_weights = f.fuse(&obs).weights;
    }
    assert!(
        last_weights[&key("x")] > last_weights[&key("y")],
        "D^abs-driven weighting favours the good-score channel: {last_weights:?}"
    );
}

// --- 4. tag-swap refusal: a model swap under a kept key is a loud refusal (§8) --------

#[test]
fn tag_swap_under_a_kept_key_is_refused_but_matching_tags_merge() {
    // Two deployments under the SAME channel key but DIFFERENT semantic tags: a model
    // swapped under a kept name. Merging would silently blend two distributions, so it
    // must refuse (§8).
    let swap_a = cfg("shared", "model-1", None);
    let swap_b = cfg("shared", "model-2", None);

    let warm = |c: &ChannelConfig, seed: u64| {
        let cs = [c.clone()];
        let mut f = Fuser::new(&cs, FuseConfig::default()).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for _ in 0..10 {
            f.fuse(&query(&mut rng, &cs, 5.0));
        }
        // Round-trip through serde, the way a reload-on-restart would.
        serde_json::from_str::<RuffleState>(&serde_json::to_string(f.state()).unwrap()).unwrap()
    };

    let a = warm(&swap_a, 100);
    let b = warm(&swap_b, 101);
    let err = RuffleState::merge(&[&a, &b], MergePolicy::Strict).unwrap_err();
    match err {
        Mismatch::Tag {
            channel,
            left,
            right,
        } => {
            assert_eq!(channel, "shared", "the refusal names the corrupt channel");
            assert_eq!(left, "model-1");
            assert_eq!(right, "model-2");
        }
        other => panic!("expected a tag refusal, got {other:?}"),
    }

    // Same key, SAME tag, two streams: compatible, so it merges and pools the counts.
    let c = warm(&swap_a, 102);
    let d = warm(&swap_a, 103);
    let (merged, _) = RuffleState::merge(&[&c, &d], MergePolicy::Strict).unwrap();
    assert!((merged.channels[&key("shared")].separation.count() - 20.0).abs() < 1e-9);
}

// --- 5. rekey lifecycle: the renamed channel carries its history and keeps fusing -----

#[test]
fn rekey_preserves_history_and_continues_under_the_new_key() {
    let cfgs = [cfg("old", "to", None), cfg("keep", "tk", None)];
    let mut f = Fuser::new(&cfgs, FuseConfig::default()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(21);
    for _ in 0..15 {
        f.fuse(&query(&mut rng, &cfgs, 5.0));
    }
    f.refresh_coupling(&anchor(&[cfg("old", "to", None), cfg("keep", "tk", None)]));

    // Snapshot the pre-rename history of `old` and confirm the pair exists.
    let old = &f.state().channels[&key("old")];
    let (sep_count, ref_count, sep_mean) = (
        old.separation.count(),
        old.reference.count(),
        old.separation.mean(),
    );
    let pair_old_keep = UnorderedPair::new(key("old"), key("keep"));
    assert!(
        f.state().pairs.contains_key(&pair_old_keep),
        "the old pair exists pre-rename"
    );

    // Rename old -> new on the persistent state (the safe rename path, §8). Take the
    // state out of the fuser, rekey it, and resume under the renamed configs — the
    // supported path now that the fuser owns its state privately.
    let new_cfgs = [cfg("new", "to", None), cfg("keep", "tk", None)];
    let mut renamed_state = f.state().clone();
    renamed_state.rekey(&key("old"), key("new"));
    let mut f = Fuser::resume(&new_cfgs, renamed_state, FuseConfig::default()).unwrap();

    // The renamed channel carries its full history; the old key is gone.
    assert!(!f.state().channels.contains_key(&key("old")));
    let new = &f.state().channels[&key("new")];
    assert_eq!(
        new.separation.count(),
        sep_count,
        "separation history preserved"
    );
    assert_eq!(
        new.reference.count(),
        ref_count,
        "reference history preserved"
    );
    assert!((new.separation.mean() - sep_mean).abs() < 1e-12);
    // Pairs and orientation followed the rename.
    assert!(
        f.state()
            .pairs
            .contains_key(&UnorderedPair::new(key("new"), key("keep")))
    );
    assert!(!f.state().pairs.contains_key(&pair_old_keep));
    assert_eq!(
        f.state().fingerprint().directions.get(&key("new")),
        Some(&Direction::HigherIsBetter)
    );

    // Continue fusing under the new key: the carried baseline keeps growing from its
    // preserved count.
    let mut rng = ChaCha8Rng::seed_from_u64(22);
    let fused = f.fuse(&query(&mut rng, &new_cfgs, 5.0));
    assert!(fused.weights.contains_key(&key("new")), "the new key fuses");
    assert_eq!(
        f.state().channels[&key("new")].separation.count(),
        sep_count + 1.0,
        "the carried baseline continued from its preserved count"
    );
}

// --- 6. decay tracks recency: a mid-run regime shift moves a decayed baseline faster --

#[test]
fn decay_tracks_a_mid_run_regime_shift_faster_than_no_decay() {
    let cfgs = [cfg("a", "ta", None)];

    let run = |decay_on: bool| -> MeanVar {
        let mut cfgv = FuseConfig::default();
        cfgv.decay.enabled = decay_on;
        cfgv.decay.factor = 0.8;
        let mut f = Fuser::new(&cfgs, cfgv).unwrap();
        // First regime: modest top elevation. Second regime: a much higher elevation,
        // so the raw separation read jumps. A decayed baseline, weighting recent reads
        // more, should sit closer to the second-regime reads than an undecayed one.
        let mut rng = ChaCha8Rng::seed_from_u64(31);
        for _ in 0..40 {
            f.fuse(&query(&mut rng, &cfgs, 3.0));
        }
        for _ in 0..40 {
            f.fuse(&query(&mut rng, &cfgs, 30.0));
        }
        f.state().channels[&key("a")].separation
    };

    let decayed = run(true);
    let plain = run(false);

    // Both are finite and well-posed.
    assert!(decayed.mean().is_finite() && plain.mean().is_finite());
    // Decay shrinks the effective count (it forgets), so it carries less history.
    assert!(
        decayed.count() < plain.count(),
        "decay reduces the effective count: {} vs {}",
        decayed.count(),
        plain.count()
    );
    // The recent regime reads higher, and the decayed baseline tracks it more closely.
    assert!(
        decayed.mean() > plain.mean(),
        "decayed baseline moved toward the recent (higher) regime: {} vs {}",
        decayed.mean(),
        plain.mean()
    );
}

// --- 7. adversarial lifecycle: decayed/undecayed merge, 3-way, disjoint, corrupt JSON -

#[test]
fn decayed_and_undecayed_states_still_merge() {
    // Decay softens the exact merge identity but never blocks a merge: tags and
    // fingerprint still match, so reconciliation succeeds with an advisory divergence.
    let cfgs = [cfg("a", "ta", None)];
    let mut warm = Fuser::new(&cfgs, FuseConfig::default()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(41);
    for _ in 0..20 {
        warm.fuse(&query(&mut rng, &cfgs, 5.0));
    }
    let undecayed = warm.state().clone();
    let mut decayed = warm.state().clone();
    decayed.decay(0.5);

    let (merged, divergence) =
        RuffleState::merge(&[&undecayed, &decayed], MergePolicy::Strict).unwrap();
    assert!(divergence.max.is_finite());
    // The decayed half contributes a smaller effective count, so the pooled count sits
    // between one and two copies of the original.
    let pooled = merged.channels[&key("a")].separation.count();
    let one = undecayed.channels[&key("a")].separation.count();
    assert!(
        pooled > one && pooled < 2.0 * one,
        "pooled count reflects the decayed weight"
    );
}

#[test]
fn three_way_reconcile_sums_all_counts() {
    let cfgs = [cfg("a", "ta", None)];
    let warm = |seed: u64, n: usize| {
        let mut f = Fuser::new(&cfgs, FuseConfig::default()).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for _ in 0..n {
            f.fuse(&query(&mut rng, &cfgs, 5.0));
        }
        f.state().clone()
    };
    let a = warm(51, 10);
    let b = warm(52, 7);
    let c = warm(53, 5);
    let (merged, _) = RuffleState::merge(&[&a, &b, &c], MergePolicy::Strict).unwrap();
    assert!((merged.channels[&key("a")].separation.count() - 22.0).abs() < 1e-9); // 10+7+5
}

#[test]
fn disjoint_channel_sets_reconcile_as_a_union_with_no_loss() {
    // One deployment knows only channel p, another only channel q. Reconciliation is a
    // union: both channels survive, neither is dropped.
    let p_cfgs = [cfg("p", "tp", None)];
    let q_cfgs = [cfg("q", "tq", None)];

    let mut fp = Fuser::new(&p_cfgs, FuseConfig::default()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(61);
    for _ in 0..12 {
        fp.fuse(&query(&mut rng, &p_cfgs, 5.0));
    }
    let mut fq = Fuser::new(&q_cfgs, FuseConfig::default()).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(62);
    for _ in 0..9 {
        fq.fuse(&query(&mut rng, &q_cfgs, 5.0));
    }

    let (merged, divergence) =
        RuffleState::merge(&[fp.state(), fq.state()], MergePolicy::Strict).unwrap();
    assert!(
        merged.channels.contains_key(&key("p")),
        "p survives the union"
    );
    assert!(
        merged.channels.contains_key(&key("q")),
        "q survives the union"
    );
    assert_eq!(merged.channels[&key("p")].separation.count(), 12.0);
    assert_eq!(merged.channels[&key("q")].separation.count(), 9.0);
    // No channel is shared, so divergence covers nothing and reads zero.
    assert_eq!(divergence.max, 0.0);
    assert!(divergence.per_channel.is_empty());
    // Both directions carried into the merged fingerprint.
    assert_eq!(
        merged.fingerprint().directions.get(&key("p")),
        Some(&Direction::HigherIsBetter)
    );
    assert_eq!(
        merged.fingerprint().directions.get(&key("q")),
        Some(&Direction::HigherIsBetter)
    );
}

#[test]
fn corrupt_or_truncated_json_errors_cleanly_without_panicking() {
    // A truncated / malformed state must fail to parse as an Err, never panic.
    for bad in [
        "",
        "{",
        "{ not valid ruffle state ]",
        r#"{"format_version":1,"#,
        r#"{"format_version":1,"fingerprint":"#,
    ] {
        let parsed: Result<RuffleState, _> = serde_json::from_str(bad);
        assert!(
            parsed.is_err(),
            "malformed input {bad:?} must Err, not parse"
        );
    }

    // A structurally valid state still round-trips, as a control on the negatives above.
    let mut s = empty_state();
    s.channels
        .insert(key("a"), ChannelSummary::new("ta".to_string()));
    let round: RuffleState = serde_json::from_str(&serde_json::to_string(&s).unwrap()).unwrap();
    assert_eq!(round, s);
}
