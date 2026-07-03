//! Golden parity fixtures shared by every language binding.
//!
//! Each fixture is a JSON file under `tests/fixtures/parity/` describing a scenario
//! against the public API: the channel registrations, the configuration, the inputs in
//! native units, and the expected outputs (rankings, weights, flags, diagnostics,
//! posterior state bytes, and merge or refusal outcomes). A binding's test suite
//! replays the fixtures through its own public surface and asserts equality; the engine
//! is bit-deterministic, so the comparison is exact, including the serialized state
//! bytes.
//!
//! `parity_fixtures_are_current` rebuilds every fixture in memory and compares it with
//! the committed file, so the fixtures cannot drift from the engine. After an
//! intentional behaviour change, regenerate with:
//!
//! ```text
//! RUFFLE_REGEN_FIXTURES=1 cargo test --test parity_fixtures
//! ```
//!
//! Schema notes for binding authors:
//!
//! - `channels[*].direction` is `"higher_is_better"` or `"lower_is_better"`;
//!   `good_score` is `null` or `{typical, good, weight}` in native units.
//! - `config` mirrors the configuration tree field by field; `baseline_mode` is
//!   `"z_score"`.
//! - Scored inputs and anchor score matrices hold native, pre-orientation scores, so
//!   the replay exercises the binding's own ingest path.
//! - States travel as canonical JSON strings and are compared byte for byte.
//! - Refusal kinds are snake_case: `format_version`, `fingerprint`,
//!   `direction_conflict`, `tag`, `empty`, `invalid_fuse_config`, `invalid_good_score`,
//!   `duplicate_channel_key`.

use ruffle::{
    Anchor, BaselineMode, ChannelConfig, ChannelFlag, ChannelId, ChannelInput, ConfigError,
    Direction, FuseConfig, Fused, Fuser, GoodScore, MergePolicy, Mismatch, ResumeError,
    RuffleState, Score,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

struct Val(f64);
impl Score for Val {
    fn value(&self) -> f64 {
        self.0
    }
}

/// A tiny deterministic generator (a PCG-style LCG step), so pools are reproducible
/// on every platform without a dev-dependency in the fixture path.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

// --- fixture-side input specs: one source of truth for the engine and the JSON -------

enum Spec {
    Scored(&'static str, Vec<(String, f64)>),
    Ranked(&'static str, Vec<String>),
}

fn spec_json(spec: &Spec) -> Value {
    match spec {
        Spec::Scored(key, items) => json!({
            "key": key,
            "scored": items.iter().map(|(id, s)| json!([id, s])).collect::<Vec<_>>(),
        }),
        Spec::Ranked(key, ids) => json!({ "key": key, "ranked": ids }),
    }
}

fn build_input(spec: &Spec, configs: &BTreeMap<String, ChannelConfig>) -> ChannelInput<String> {
    match spec {
        Spec::Scored(key, items) => ChannelInput::scored(
            &configs[*key],
            items.iter().map(|(id, s)| (id.clone(), Val(*s))).collect(),
        ),
        Spec::Ranked(key, ids) => ChannelInput::ranked(&configs[*key], ids.clone()),
    }
}

fn config_map(channels: &[ChannelConfig]) -> BTreeMap<String, ChannelConfig> {
    channels
        .iter()
        .map(|c| (c.id.key.clone(), c.clone()))
        .collect()
}

// --- JSON encoders --------------------------------------------------------------------

fn dir_str(d: Direction) -> &'static str {
    match d {
        Direction::HigherIsBetter => "higher_is_better",
        Direction::LowerIsBetter => "lower_is_better",
    }
}

fn channels_json(channels: &[ChannelConfig]) -> Value {
    Value::Array(
        channels
            .iter()
            .map(|c| {
                json!({
                    "key": c.id.key,
                    "tag": c.id.tag,
                    "direction": dir_str(c.direction),
                    "good_score": c.good_score.map(|g| json!({
                        "typical": g.typical, "good": g.good, "weight": g.weight,
                    })),
                })
            })
            .collect(),
    )
}

fn config_json(cfg: &FuseConfig) -> Value {
    let d = &cfg.discrimination;
    let c = &cfg.coupling;
    let baseline_mode = match cfg.baseline_mode {
        BaselineMode::ZScore => "z_score",
        _ => panic!("unencoded baseline mode"),
    };
    json!({
        "discrimination": {
            "top_eps": d.top_eps,
            "top_m": d.top_m,
            "min_distinct_values": d.min_distinct_values,
            "denom_floor_frac": d.denom_floor_frac,
            "winsor_z": d.winsor_z,
            "min_count_for_z": d.min_count_for_z,
            "shrink_pool_size": d.shrink_pool_size,
            "g_upper_bound": d.g_upper_bound,
            "g_floor": d.g_floor,
            "g_slope": d.g_slope,
        },
        "coupling": {
            "enabled": c.enabled,
            "discount_cap": c.discount_cap,
            "shrink_to_identity": c.shrink_to_identity,
            "min_overlap": c.min_overlap,
            "min_reliability": c.min_reliability,
            "min_refreshes": c.min_refreshes,
            "stratum_stability_max_var": c.stratum_stability_max_var,
        },
        "fusion": { "rrf_eta": cfg.fusion.rrf_eta },
        "decay": { "enabled": cfg.decay.enabled, "factor": cfg.decay.factor },
        "baseline_mode": baseline_mode,
    })
}

fn flag_str(flag: ChannelFlag) -> &'static str {
    match flag {
        ChannelFlag::RanksOnlyDefaultWeighted => "ranks_only_default_weighted",
        ChannelFlag::DegenerateSeparation => "degenerate_separation",
        ChannelFlag::NoReference => "no_reference",
        _ => panic!("unencoded channel flag"),
    }
}

fn fused_json(f: &Fused<String>) -> Value {
    json!({
        "ranking": f.ranking.iter().map(|(id, s)| json!([id, s])).collect::<Vec<_>>(),
        "weights": f.weights,
        "flags": f.flags.iter()
            .map(|(k, v)| (k.clone(), Value::from(flag_str(*v))))
            .collect::<serde_json::Map<_, _>>(),
        "discrimination": f.discrimination.iter().map(|(k, d)| (k.clone(), json!({
            "g": d.g,
            "raw_separation": d.raw_separation,
            "top_m_average": d.top_m_average,
            "degenerate_separation": d.degenerate_separation,
            "reference_cold": d.reference_cold,
        }))).collect::<serde_json::Map<_, _>>(),
        "confidence": f.confidence,
        "conflict": f.conflict,
    })
}

fn state_str(state: &RuffleState) -> String {
    serde_json::to_string(state).expect("state serializes")
}

fn mismatch_kind(m: &Mismatch) -> &'static str {
    match m {
        Mismatch::FormatVersion { .. } => "format_version",
        Mismatch::Fingerprint => "fingerprint",
        Mismatch::DirectionConflict { .. } => "direction_conflict",
        Mismatch::Tag { .. } => "tag",
        Mismatch::Empty => "empty",
        _ => panic!("unencoded mismatch"),
    }
}

fn config_error_kind(e: &ConfigError) -> &'static str {
    match e {
        ConfigError::InvalidFuseConfig { .. } => "invalid_fuse_config",
        ConfigError::InvalidGoodScore { .. } => "invalid_good_score",
        ConfigError::DuplicateChannelKey { .. } => "duplicate_channel_key",
        _ => panic!("unencoded config error"),
    }
}

// --- shared channel registrations ------------------------------------------------------

fn semantic() -> ChannelConfig {
    ChannelConfig::new(
        ChannelId::new("semantic", "text-embedding-v1"),
        Direction::HigherIsBetter,
        None,
    )
}

fn lexical() -> ChannelConfig {
    ChannelConfig::new(
        ChannelId::new("lexical", "sqlite-fts5-trigram-bm25"),
        Direction::LowerIsBetter,
        Some(GoodScore::new(-4.0, -12.0, 8.0)),
    )
}

fn recency() -> ChannelConfig {
    ChannelConfig::new(
        ChannelId::new("recency", "recency-v1"),
        Direction::HigherIsBetter,
        None,
    )
}

/// Bulk scores for ids `doc{lo:03}..doc{hi:03}` in `[base, base + scale)`.
fn bulk(seed: u64, lo: usize, hi: usize, base: f64, scale: f64) -> Vec<(String, f64)> {
    let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
    (lo..hi)
        .map(|i| (format!("doc{i:03}"), base + scale * lcg(&mut s)))
        .collect()
}

fn quickstart_specs(q: u64) -> Vec<Spec> {
    // Semantic: higher is better, bulk cosine scores plus two strong hits.
    let mut sem = bulk(1000 + q, 0, 30, 0.10, 0.30);
    sem.push(("hit0".into(), 0.90 + 0.01 * q as f64));
    sem.push(("hit1".into(), 0.95));
    // Lexical: native lower-is-better (negated BM25), bulk near-typical plus two strong
    // hits toward the declared good anchor.
    let mut lex = bulk(2000 + q, 10, 40, -3.0, 2.0);
    lex.push(("hit0".into(), -9.5));
    lex.push(("hit1".into(), -11.0 - 0.05 * q as f64));
    // Recency: rank-only, best first.
    let rec = vec![
        "hit1".to_string(),
        format!("doc{:03}", 12 + q),
        "doc033".to_string(),
        "hit0".to_string(),
        "doc002".to_string(),
    ];
    vec![
        Spec::Scored("semantic", sem),
        Spec::Scored("lexical", lex),
        Spec::Ranked("recency", rec),
    ]
}

// --- a generic session runner -----------------------------------------------------------

enum Step {
    Fuse(Vec<Spec>),
    Refresh {
        candidates: Vec<String>,
        channel_keys: Vec<&'static str>,
        scores: Vec<Vec<Option<f64>>>,
    },
}

/// Runs a stateful session and encodes it: each fuse step records its inputs and the
/// engine's expected output, each refresh records the native anchor matrix, and the
/// final state is embedded as canonical bytes.
fn session(
    name: &str,
    description: &str,
    channels: &[ChannelConfig],
    cfg: FuseConfig,
    steps: Vec<Step>,
    unregistered: &[ChannelConfig],
) -> Value {
    let mut fuser = Fuser::new(channels, cfg).expect("valid fixture registrations");
    let configs: BTreeMap<String, ChannelConfig> = config_map(channels)
        .into_iter()
        .chain(config_map(unregistered))
        .collect();

    let mut steps_json = Vec::new();
    for step in &steps {
        match step {
            Step::Fuse(specs) => {
                let inputs: Vec<ChannelInput<String>> =
                    specs.iter().map(|s| build_input(s, &configs)).collect();
                let fused = fuser.fuse(&inputs);
                steps_json.push(json!({
                    "op": "fuse",
                    "inputs": specs.iter().map(spec_json).collect::<Vec<_>>(),
                    "expected": fused_json(&fused),
                }));
            }
            Step::Refresh {
                candidates,
                channel_keys,
                scores,
            } => {
                let index: BTreeMap<&String, usize> =
                    candidates.iter().enumerate().map(|(i, c)| (c, i)).collect();
                let refs: Vec<&ChannelConfig> = channel_keys.iter().map(|k| &configs[*k]).collect();
                let anchor = Anchor::build(candidates, &refs, |id: &String, key: &str| {
                    let row = channel_keys.iter().position(|k| *k == key)?;
                    scores[row][index[id]].map(Val)
                });
                fuser.refresh_coupling(&anchor);
                steps_json.push(json!({
                    "op": "refresh_coupling",
                    "anchor": {
                        "candidates": candidates,
                        "channels": channel_keys,
                        "scores": scores,
                    },
                }));
            }
        }
    }

    json!({
        "name": name,
        "description": description,
        "kind": "session",
        "channels": channels_json(channels),
        "unregistered_channels": channels_json(unregistered),
        "config": config_json(&fuser.config().clone()),
        "steps": steps_json,
        "expected_state": state_str(fuser.state()),
    })
}

// --- scenarios --------------------------------------------------------------------------

fn quickstart() -> Value {
    session(
        "quickstart_three_channels",
        "The README shape: a higher-is-better scored channel, a lower-is-better scored \
         channel with a declared good-score reference, and a rank-only channel, fused \
         over three sequential queries at the default configuration.",
        &[semantic(), lexical(), recency()],
        FuseConfig::default(),
        (0..3).map(|q| Step::Fuse(quickstart_specs(q))).collect(),
        &[],
    )
}

fn ties_and_edges() -> Value {
    // Quantized scores produce heavy midrank ties; the lexical channel surfaces
    // nothing; a rogue unregistered channel and a duplicate semantic input are both
    // skipped by the engine.
    let mut sem: Vec<(String, f64)> = (0..30)
        .map(|i| (format!("doc{i:03}"), 0.1 * f64::from(i % 12)))
        .collect();
    sem.push(("hit0".into(), 3.0));
    sem.push(("hit1".into(), 3.0)); // an exact tie at the top
    let dup: Vec<(String, f64)> = (0..5).map(|i| (format!("doc{i:03}"), 0.5)).collect();
    let rogue: Vec<(String, f64)> = (0..4)
        .map(|i| (format!("doc{i:03}"), 9.0 + i as f64))
        .collect();

    let rogue_cfg = ChannelConfig::new(
        ChannelId::new("rogue", "unregistered-v1"),
        Direction::HigherIsBetter,
        None,
    );
    session(
        "ties_and_edge_inputs",
        "Exact score ties sharing a midrank, an empty scored channel, an unregistered \
         input skipped entirely, and a duplicate channel input of which only the first \
         is fused.",
        &[semantic(), lexical(), recency()],
        FuseConfig::default(),
        vec![Step::Fuse(vec![
            Spec::Scored("semantic", sem),
            Spec::Scored("lexical", vec![]),
            Spec::Ranked(
                "recency",
                vec!["hit1".into(), "doc003".into(), "hit0".into()],
            ),
            Spec::Scored("rogue", rogue),
            Spec::Scored("semantic", dup),
        ])],
        &[rogue_cfg],
    )
}

fn coupling_channels() -> Vec<ChannelConfig> {
    vec![
        ChannelConfig::new(
            ChannelId::new("alpha", "dense-v1"),
            Direction::HigherIsBetter,
            None,
        ),
        ChannelConfig::new(
            ChannelId::new("beta", "dense-v2-distilled"),
            Direction::HigherIsBetter,
            None,
        ),
        ChannelConfig::new(
            ChannelId::new("gamma", "recount-v1"),
            Direction::LowerIsBetter,
            None,
        ),
    ]
}

fn coupling_session() -> Value {
    let mut cfg = FuseConfig::default();
    cfg.coupling.enabled = true;

    let candidates: Vec<String> = (0..40).map(|i| format!("c{i:03}")).collect();
    // Alpha and beta are strongly rank-correlated (beta is alpha with small
    // inversions); gamma is native lower-is-better and near-independent, with the
    // facet absent on every seventh candidate.
    let alpha_row: Vec<Option<f64>> = (0..40).map(|i| Some(f64::from(i))).collect();
    let beta_row: Vec<Option<f64>> = (0..40)
        .map(|i| Some(f64::from(i) + 3.0 * f64::from(i % 5)))
        .collect();
    let gamma_row: Vec<Option<f64>> = (0..40u32)
        .map(|i| {
            if i % 7 == 0 {
                None
            } else {
                Some(f64::from((i * 17) % 40))
            }
        })
        .collect();

    let refresh = || Step::Refresh {
        candidates: candidates.clone(),
        channel_keys: vec!["alpha", "beta", "gamma"],
        scores: vec![alpha_row.clone(), beta_row.clone(), gamma_row.clone()],
    };

    let pool = |seed: u64, base: f64| -> Vec<Spec> {
        let mut a = bulk(seed, 0, 30, base, 0.5);
        a.push(("hit0".into(), base + 4.0));
        a.push(("hit1".into(), base + 4.5));
        let mut b = bulk(seed + 50, 0, 30, base, 0.5);
        b.push(("hit0".into(), base + 4.2));
        b.push(("hit1".into(), base + 4.4));
        // Gamma is lower-is-better natively: strong hits are the most negative.
        let mut g = bulk(seed + 100, 5, 35, 1.0, 2.0);
        g.push(("hit0".into(), -6.0));
        g.push(("hit2".into(), -5.5));
        vec![
            Spec::Scored("alpha", a),
            Spec::Scored("beta", b),
            Spec::Scored("gamma", g),
        ]
    };

    session(
        "coupling_redundancy_discount",
        "Coupling enabled: two anchor refreshes accumulate a strongly correlated \
         alpha-beta pair and a near-independent gamma, clearing the reliability, \
         refresh, and stability gates, so the discount moves weight at the following \
         fuses.",
        &coupling_channels(),
        cfg,
        vec![
            refresh(),
            refresh(),
            Step::Fuse(pool(3000, 1.0)),
            Step::Fuse(pool(4000, 1.2)),
        ],
        &[],
    )
}

fn decay_session() -> Value {
    let mut cfg = FuseConfig::default();
    cfg.decay.enabled = true;
    cfg.decay.factor = 0.9;
    session(
        "decay_per_update",
        "Per-update decay at factor 0.9: each fuse decays the appearing channels' \
         baselines before pushing the new reads, so counts grow sublinearly.",
        &[semantic(), lexical(), recency()],
        cfg,
        (0..4).map(|q| Step::Fuse(quickstart_specs(q))).collect(),
        &[],
    )
}

fn stateless_with_prior() -> Value {
    let channels = [semantic(), lexical()];
    let cfg = FuseConfig::default();
    let mut fuser = Fuser::new(&channels, cfg).unwrap();
    let configs = config_map(&channels);
    for q in 0..2 {
        let specs: Vec<Spec> = quickstart_specs(q)
            .into_iter()
            .filter(|s| !matches!(s, Spec::Ranked(..)))
            .collect();
        let inputs: Vec<ChannelInput<String>> =
            specs.iter().map(|s| build_input(s, &configs)).collect();
        fuser.fuse(&inputs);
    }
    let prior = fuser.state().clone();

    let specs: Vec<Spec> = quickstart_specs(2)
        .into_iter()
        .filter(|s| !matches!(s, Spec::Ranked(..)))
        .collect();
    let inputs: Vec<ChannelInput<String>> =
        specs.iter().map(|s| build_input(s, &configs)).collect();
    let fused = Fuser::fuse_stateless(&inputs, &channels, &prior, &FuseConfig::default())
        .expect("compatible prior");

    json!({
        "name": "stateless_with_prior",
        "description": "fuse_stateless reads a warm prior without mutating it; the \
                        prior bytes are the pre- and post-call state.",
        "kind": "stateless",
        "channels": channels_json(&channels),
        "config": config_json(&FuseConfig::default()),
        "prior_state": state_str(&prior),
        "inputs": specs.iter().map(spec_json).collect::<Vec<_>>(),
        "expected": fused_json(&fused),
    })
}

/// A short session's posterior state, for the merge and state-op fixtures.
fn accumulated_state(channels: &[ChannelConfig], queries: std::ops::Range<u64>) -> RuffleState {
    let mut fuser = Fuser::new(channels, FuseConfig::default()).unwrap();
    let configs = config_map(channels);
    for q in queries {
        let specs: Vec<Spec> = quickstart_specs(q)
            .into_iter()
            .filter(|s| match s {
                Spec::Scored(k, _) | Spec::Ranked(k, _) => configs.contains_key(*k),
            })
            .collect();
        let inputs: Vec<ChannelInput<String>> =
            specs.iter().map(|s| build_input(s, &configs)).collect();
        fuser.fuse(&inputs);
    }
    fuser.state().clone()
}

fn merge_reconcile() -> Value {
    // Two deployments accumulate separately: A fuses semantic+lexical, B fuses
    // semantic+recency over different queries. The merge unions recency in and pools
    // the shared channels; the divergence is advisory.
    let a = accumulated_state(&[semantic(), lexical()], 0..2);
    let b = accumulated_state(&[semantic(), recency()], 2..5);
    let (merged, divergence) =
        RuffleState::merge(&[&a, &b], MergePolicy::Strict).expect("compatible parts");
    json!({
        "name": "merge_reconcile",
        "description": "Cross-deployment reconciliation of two compatible states; the \
                        merge is commutative, so replays should also check the reversed \
                        order produces identical bytes.",
        "kind": "merge",
        "parts": [state_str(&a), state_str(&b)],
        "expected_state": state_str(&merged),
        "expected_divergence": {
            "per_channel": divergence.per_channel,
            "max": divergence.max,
        },
    })
}

fn state_ops() -> Value {
    // Start from a state that carries pair summaries, so rekey moves them too.
    let channels = coupling_channels();
    let mut cfg = FuseConfig::default();
    cfg.coupling.enabled = true;
    let mut fuser = Fuser::new(&channels, cfg).unwrap();
    let configs = config_map(&channels);
    let candidates: Vec<String> = (0..40).map(|i| format!("c{i:03}")).collect();
    let index: BTreeMap<&String, usize> =
        candidates.iter().enumerate().map(|(i, c)| (c, i)).collect();
    let rows: Vec<Vec<Option<f64>>> = vec![
        (0..40).map(|i| Some(f64::from(i))).collect(),
        (0..40)
            .map(|i| Some(f64::from(i) + 3.0 * f64::from(i % 5)))
            .collect(),
        (0..40u32).map(|i| Some(f64::from((i * 17) % 40))).collect(),
    ];
    let keys = ["alpha", "beta", "gamma"];
    let refs: Vec<&ChannelConfig> = keys.iter().map(|k| &configs[*k]).collect();
    let anchor = Anchor::build(&candidates, &refs, |id: &String, key: &str| {
        let row = keys.iter().position(|k| *k == key)?;
        rows[row][index[id]].map(Val)
    });
    fuser.refresh_coupling(&anchor);

    let start = fuser.state().clone();
    let mut state = start.clone();
    state.rekey("alpha", "dense".to_string());
    state.decay(0.5);

    json!({
        "name": "state_ops_rekey_decay",
        "description": "rekey moves a channel's summaries, its pair summaries, and its \
                        fingerprint orientation to the new key; decay halves every \
                        effective count while preserving means and variances.",
        "kind": "state_ops",
        "start_state": state_str(&start),
        "ops": [
            { "op": "rekey", "from": "alpha", "to": "dense" },
            { "op": "decay", "factor": 0.5 },
        ],
        "expected_state": state_str(&state),
    })
}

fn refusals() -> Value {
    let mut cases = Vec::new();

    // Resume under a bumped tag: the model-swap signature.
    let state_v1 = accumulated_state(&[semantic(), lexical()], 0..1);
    let semantic_v2 = ChannelConfig::new(
        ChannelId::new("semantic", "text-embedding-v2"),
        Direction::HigherIsBetter,
        None,
    );
    let err = Fuser::resume(
        &[semantic_v2.clone(), lexical()],
        state_v1.clone(),
        FuseConfig::default(),
    )
    .expect_err("bumped tag must refuse");
    let ResumeError::State(m) = &err else {
        panic!("expected a state mismatch");
    };
    cases.push(json!({
        "name": "resume_tag_bump",
        "kind": "resume",
        "channels": channels_json(&[semantic_v2, lexical()]),
        "config": config_json(&FuseConfig::default()),
        "state": state_str(&state_v1),
        "error": mismatch_kind(m),
    }));

    // Resume under a flipped direction.
    let semantic_flipped = ChannelConfig::new(
        ChannelId::new("semantic", "text-embedding-v1"),
        Direction::LowerIsBetter,
        None,
    );
    let err = Fuser::resume(
        std::slice::from_ref(&semantic_flipped),
        state_v1.clone(),
        FuseConfig::default(),
    )
    .expect_err("flipped direction must refuse");
    let ResumeError::State(m) = &err else {
        panic!("expected a state mismatch");
    };
    cases.push(json!({
        "name": "resume_direction_flip",
        "kind": "resume",
        "channels": channels_json(&[semantic_flipped]),
        "config": config_json(&FuseConfig::default()),
        "state": state_str(&state_v1),
        "error": mismatch_kind(m),
    }));

    // Merging across a model swap.
    let left = accumulated_state(&[semantic(), lexical()], 0..1);
    let mut right = accumulated_state(&[semantic(), lexical()], 1..2);
    right.channels.get_mut("semantic").unwrap().tag = "text-embedding-v2".to_string();
    let err = RuffleState::merge(&[&left, &right], MergePolicy::Strict)
        .expect_err("tag mismatch must refuse");
    cases.push(json!({
        "name": "merge_tag_mismatch",
        "kind": "merge",
        "parts": [state_str(&left), state_str(&right)],
        "error": mismatch_kind(&err),
    }));

    // Merging across a stale statistic version (both parts agree; still refused).
    let stale = |src: &RuffleState| -> RuffleState {
        let mut v = serde_json::to_value(src).unwrap();
        v["fingerprint"]["stat_version"] = Value::from(1u32);
        serde_json::from_value(v).unwrap()
    };
    let (sa, sb) = (stale(&left), stale(&right)); // right's tag edit is irrelevant here
    let sb = {
        let mut v = sb;
        v.channels.get_mut("semantic").unwrap().tag = "text-embedding-v1".to_string();
        v
    };
    let err = RuffleState::merge(&[&sa, &sb], MergePolicy::Strict)
        .expect_err("stale stat version must refuse");
    cases.push(json!({
        "name": "merge_stale_stat_version",
        "kind": "merge",
        "parts": [state_str(&sa), state_str(&sb)],
        "error": mismatch_kind(&err),
    }));

    // Merging across a foreign format version.
    let foreign = |src: &RuffleState| -> RuffleState {
        let mut v = serde_json::to_value(src).unwrap();
        v["format_version"] = Value::from(99u32);
        serde_json::from_value(v).unwrap()
    };
    let fa = foreign(&left);
    let err = RuffleState::merge(&[&fa, &left], MergePolicy::Strict)
        .expect_err("foreign format version must refuse");
    cases.push(json!({
        "name": "merge_foreign_format_version",
        "kind": "merge",
        "parts": [state_str(&fa), state_str(&left)],
        "error": mismatch_kind(&err),
    }));

    // Construction refusals: duplicate key, unusable good score, out-of-range knob.
    let err = Fuser::new(&[semantic(), semantic()], FuseConfig::default())
        .expect_err("duplicate key must refuse");
    cases.push(json!({
        "name": "config_duplicate_key",
        "kind": "config",
        "channels": channels_json(&[semantic(), semantic()]),
        "config": config_json(&FuseConfig::default()),
        "error": config_error_kind(&err),
    }));

    let bad_ref = ChannelConfig::new(
        ChannelId::new("semantic", "text-embedding-v1"),
        Direction::HigherIsBetter,
        Some(GoodScore::new(0.5, 0.3, 4.0)), // good below typical after orientation
    );
    let err = Fuser::new(std::slice::from_ref(&bad_ref), FuseConfig::default())
        .expect_err("unusable good score must refuse");
    cases.push(json!({
        "name": "config_invalid_good_score",
        "kind": "config",
        "channels": channels_json(std::slice::from_ref(&bad_ref)),
        "config": config_json(&FuseConfig::default()),
        "error": config_error_kind(&err),
    }));

    let mut bad_cfg = FuseConfig::default();
    bad_cfg.discrimination.g_floor = 5.0;
    bad_cfg.discrimination.g_upper_bound = 4.0;
    let err = Fuser::new(&[semantic()], bad_cfg).expect_err("inverted bounds must refuse");
    cases.push(json!({
        "name": "config_inverted_g_bounds",
        "kind": "config",
        "channels": channels_json(&[semantic()]),
        "config": config_json(&bad_cfg),
        "error": config_error_kind(&err),
    }));

    json!({
        "name": "refusals",
        "description": "Every gate a binding must surface as a raised error: resume and \
                        merge incompatibilities and construction-time configuration \
                        refusals.",
        "kind": "refusals",
        "cases": cases,
    })
}

// --- the generator test ------------------------------------------------------------------

fn fixtures() -> Vec<(&'static str, Value)> {
    vec![
        ("quickstart_three_channels.json", quickstart()),
        ("ties_and_edge_inputs.json", ties_and_edges()),
        ("coupling_redundancy_discount.json", coupling_session()),
        ("decay_per_update.json", decay_session()),
        ("stateless_with_prior.json", stateless_with_prior()),
        ("merge_reconcile.json", merge_reconcile()),
        ("state_ops_rekey_decay.json", state_ops()),
        ("refusals.json", refusals()),
    ]
}

#[test]
fn parity_fixtures_are_current() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/parity");
    let regen = std::env::var_os("RUFFLE_REGEN_FIXTURES").is_some();
    if regen {
        fs::create_dir_all(&dir).expect("create fixture dir");
    }
    for (name, value) in fixtures() {
        let path = dir.join(name);
        let mut rendered = serde_json::to_string_pretty(&value).expect("fixture serializes");
        rendered.push('\n');
        if regen {
            fs::write(&path, &rendered).expect("write fixture");
        } else {
            let committed = fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!("missing fixture {name} ({e}); run RUFFLE_REGEN_FIXTURES=1 cargo test")
            });
            assert_eq!(
                committed, rendered,
                "fixture {name} is stale; run RUFFLE_REGEN_FIXTURES=1 cargo test"
            );
        }
    }
}
