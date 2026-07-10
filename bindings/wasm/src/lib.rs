//! The compiled core of the `@lathrys-at/ruffle` npm package.
//!
//! The public TypeScript surface lives in `ts/`; this module is private plumbing
//! behind it. It links the Rust engine directly and reimplements no statistic, so the
//! binding's behaviour is the crate's behaviour by construction. WebAssembly specifies
//! IEEE-754 `f64` arithmetic exactly and the engine's transcendentals come from the
//! pure-Rust `libm`, so wasm and native builds produce bit-identical rankings and
//! state bytes.
//!
//! Boundary conventions, typed on the TypeScript side by `ts/boundary.ts` and
//! exercised by the parity fixtures under `tests/fixtures/parity/` in the repository
//! root:
//!
//! - Channel registrations, the fusion configuration, per-query inputs, and anchors
//!   cross as structured values via `serde-wasm-bindgen` (camelCase fields;
//!   `direction` as `"higher_is_better"`/`"lower_is_better"`, `baselineMode` as
//!   `"z_score"`). Nothing on the hot path is stringified.
//! - Results come back as structured values: maps as ES `Map`s, tuples as arrays.
//! - States cross as canonical JSON strings, byte-identical to the core crate's
//!   serialization; the string is the persistence format itself.
//! - Errors are thrown as `{ kind, message }` values; the TypeScript layer maps the
//!   kind to its exception classes.

use ruffle as rf;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use wasm_bindgen::prelude::*;

/// The one score newtype the binding needs: a native JS number, declared as-is. The
/// meaning a Rust caller declares through a newtype is carried on the TypeScript side
/// by the channel configuration (direction, tag, good-score reference).
struct Native(f64);
impl rf::Score for Native {
    fn value(&self) -> f64 {
        self.0
    }
}

// --- errors: thrown as `{ kind, message }` ------------------------------------------

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ErrorOut<'a> {
    kind: &'a str,
    message: String,
}

fn throw(kind: &str, message: String) -> JsValue {
    let fallback = JsValue::from_str(&message);
    serde_wasm_bindgen::to_value(&ErrorOut { kind, message }).unwrap_or(fallback)
}

fn value_err(message: String) -> JsValue {
    throw("value", message)
}

fn config_err(e: rf::ConfigError) -> JsValue {
    throw("config", e.to_string())
}

fn resume_err(e: rf::ResumeError) -> JsValue {
    match e {
        rf::ResumeError::Config(c) => config_err(c),
        other => throw("resume", other.to_string()),
    }
}

fn merge_err(e: rf::Mismatch) -> JsValue {
    throw("merge", e.to_string())
}

fn state_err(message: String) -> JsValue {
    throw("state", message)
}

fn internal_err(e: impl std::fmt::Display) -> JsValue {
    throw("internal", format!("internal serialization failure: {e}"))
}

// --- the typed boundary DTOs (mirrored by `ts/boundary.ts`) --------------------------
//
// `deny_unknown_fields` is inert under serde-wasm-bindgen, which reads known field
// names off the JS object rather than iterating its keys, so an unknown key is
// silently ignored here. The attribute stays for any future serde_json path, but the
// check that catches a typo'd configuration knob lives in the TypeScript layer
// (`resolveConfig` in `ts/config.ts`), validated against the engine's own defaults.

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct GoodScoreDto {
    typical: f64,
    good: f64,
    weight: f64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ChannelDto {
    key: String,
    tag: String,
    direction: String,
    #[serde(default)]
    good_score: Option<GoodScoreDto>,
    #[serde(default = "default_base_weight")]
    base_weight: f64,
}

fn default_base_weight() -> f64 {
    1.0
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DiscriminationDto {
    top_eps: f64,
    top_m: usize,
    min_distinct_values: usize,
    denom_floor_frac: f64,
    winsor_z: f64,
    min_count_for_z: f64,
    shrink_pool_size: usize,
    g_upper_bound: f64,
    g_floor: f64,
    g_slope: f64,
    g_deviation_keep: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CouplingDto {
    enabled: bool,
    discount_cap: f64,
    shrink_to_identity: f64,
    min_overlap: usize,
    min_reliability: f64,
    min_refreshes: f64,
    stratum_stability_max_var: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FusionDto {
    rrf_eta: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DecayDto {
    enabled: bool,
    factor: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ConfigDto {
    discrimination: DiscriminationDto,
    coupling: CouplingDto,
    fusion: FusionDto,
    decay: DecayDto,
    baseline_mode: String,
}

/// One channel's per-query input. Exactly one of `scored`/`ranked` is present; a
/// scored input carries the direction its scores orient under, injected by the
/// TypeScript layer from the channel's registration.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct InputDto {
    key: String,
    #[serde(default)]
    direction: Option<String>,
    #[serde(default)]
    scored: Option<Vec<(String, f64)>>,
    #[serde(default)]
    ranked: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AnchorDto {
    /// `(key, direction)` per channel, in row order.
    channels: Vec<(String, String)>,
    /// `rows[channel][candidate]`: the native score, or `null` where the channel's
    /// facet does not apply.
    rows: Vec<Vec<Option<f64>>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DiscriminationOut {
    g: f64,
    raw_separation: Option<f64>,
    top_m_average: Option<f64>,
    degenerate_separation: bool,
    reference_cold: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct FusedOut {
    ranking: Vec<(String, f64)>,
    weights: BTreeMap<String, f64>,
    flags: BTreeMap<String, &'static str>,
    discrimination: BTreeMap<String, DiscriminationOut>,
    confidence: f64,
    conflict: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DivergenceOut {
    per_channel: BTreeMap<String, f64>,
    max: f64,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct MergeOut {
    merged: String,
    divergence: DivergenceOut,
}

// --- conversions ------------------------------------------------------------------

fn parse_direction(s: &str) -> Result<rf::Direction, JsValue> {
    match s {
        "higher_is_better" => Ok(rf::Direction::HigherIsBetter),
        "lower_is_better" => Ok(rf::Direction::LowerIsBetter),
        other => Err(value_err(format!("unknown direction {other:?}"))),
    }
}

fn from_js<T: serde::de::DeserializeOwned>(value: JsValue, what: &str) -> Result<T, JsValue> {
    serde_wasm_bindgen::from_value(value).map_err(|e| value_err(format!("invalid {what}: {e}")))
}

fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(value).map_err(internal_err)
}

fn to_channel_configs(channels: JsValue) -> Result<Vec<rf::ChannelConfig>, JsValue> {
    let dtos: Vec<ChannelDto> = from_js(channels, "channel registrations")?;
    dtos.into_iter()
        .map(|c| {
            Ok(rf::ChannelConfig::new(
                rf::ChannelId::new(c.key, c.tag),
                parse_direction(&c.direction)?,
                c.good_score
                    .map(|g| rf::GoodScore::new(g.typical, g.good, g.weight)),
            )
            .with_base_weight(c.base_weight))
        })
        .collect()
}

fn to_fuse_config(config: JsValue) -> Result<rf::FuseConfig, JsValue> {
    let dto: ConfigDto = from_js(config, "fuse configuration")?;
    if dto.baseline_mode != "z_score" {
        return Err(value_err(format!(
            "unknown baseline mode {:?}",
            dto.baseline_mode
        )));
    }
    let mut cfg = rf::FuseConfig::default();
    let d = &dto.discrimination;
    cfg.discrimination.top_eps = d.top_eps;
    cfg.discrimination.top_m = d.top_m;
    cfg.discrimination.min_distinct_values = d.min_distinct_values;
    cfg.discrimination.denom_floor_frac = d.denom_floor_frac;
    cfg.discrimination.winsor_z = d.winsor_z;
    cfg.discrimination.min_count_for_z = d.min_count_for_z;
    cfg.discrimination.shrink_pool_size = d.shrink_pool_size;
    cfg.discrimination.g_upper_bound = d.g_upper_bound;
    cfg.discrimination.g_floor = d.g_floor;
    cfg.discrimination.g_slope = d.g_slope;
    cfg.discrimination.g_deviation_keep = d.g_deviation_keep;
    let c = &dto.coupling;
    cfg.coupling.enabled = c.enabled;
    cfg.coupling.discount_cap = c.discount_cap;
    cfg.coupling.shrink_to_identity = c.shrink_to_identity;
    cfg.coupling.min_overlap = c.min_overlap;
    cfg.coupling.min_reliability = c.min_reliability;
    cfg.coupling.min_refreshes = c.min_refreshes;
    cfg.coupling.stratum_stability_max_var = c.stratum_stability_max_var;
    cfg.fusion.rrf_eta = dto.fusion.rrf_eta;
    cfg.decay.enabled = dto.decay.enabled;
    cfg.decay.factor = dto.decay.factor;
    cfg.baseline_mode = rf::BaselineMode::ZScore;
    Ok(cfg)
}

fn to_inputs(inputs: JsValue) -> Result<Vec<rf::ChannelInput<String>>, JsValue> {
    let dtos: Vec<InputDto> = from_js(inputs, "channel inputs")?;
    dtos.into_iter()
        .map(|dto| match (dto.scored, dto.ranked) {
            (Some(items), None) => {
                let direction = dto.direction.ok_or_else(|| {
                    value_err(format!("scored input {:?} carries no direction", dto.key))
                })?;
                let dir = parse_direction(&direction)?;
                // Tag and reference play no role at ingest; the engine's own scored()
                // path performs the orientation and non-finite filtering.
                let cfg =
                    rf::ChannelConfig::new(rf::ChannelId::new(dto.key, String::new()), dir, None);
                Ok(rf::ChannelInput::scored(
                    &cfg,
                    items
                        .into_iter()
                        .map(|(id, s)| (id, Native(s)))
                        .collect::<Vec<_>>(),
                ))
            }
            (None, Some(ids)) => {
                let cfg = rf::ChannelConfig::new(
                    rf::ChannelId::new(dto.key, String::new()),
                    rf::Direction::HigherIsBetter,
                    None,
                );
                Ok(rf::ChannelInput::ranked(&cfg, ids))
            }
            _ => Err(value_err(format!(
                "input {:?} must carry exactly one of scored/ranked",
                dto.key
            ))),
        })
        .collect()
}

fn to_anchor(anchor: JsValue) -> Result<rf::Anchor, JsValue> {
    let dto: AnchorDto = from_js(anchor, "anchor")?;
    if dto.channels.len() != dto.rows.len() {
        return Err(value_err(format!(
            "anchor has {} channels but {} score rows",
            dto.channels.len(),
            dto.rows.len()
        )));
    }
    let n = dto.rows.first().map_or(0, Vec::len);
    if dto.rows.iter().any(|r| r.len() != n) {
        return Err(value_err(
            "anchor score rows must all have one entry per candidate".to_string(),
        ));
    }
    let configs: Vec<rf::ChannelConfig> = dto
        .channels
        .iter()
        .map(|(key, dir)| {
            Ok(rf::ChannelConfig::new(
                rf::ChannelId::new(key.clone(), String::new()),
                parse_direction(dir)?,
                None,
            ))
        })
        .collect::<Result<_, JsValue>>()?;
    let refs: Vec<&rf::ChannelConfig> = configs.iter().collect();
    let candidates: Vec<usize> = (0..n).collect();
    Ok(rf::Anchor::build(&candidates, &refs, |i: &usize, key| {
        let row = dto.channels.iter().position(|(k, _)| k == key)?;
        dto.rows[row][*i].map(Native)
    }))
}

fn flag_str(flag: rf::ChannelFlag) -> &'static str {
    match flag {
        rf::ChannelFlag::RanksOnlyDefaultWeighted => "ranks_only_default_weighted",
        rf::ChannelFlag::DegenerateSeparation => "degenerate_separation",
        rf::ChannelFlag::NoReference => "no_reference",
        _ => "unknown",
    }
}

fn fused_out(f: &rf::Fused<String>) -> FusedOut {
    FusedOut {
        ranking: f.ranking.clone(),
        weights: f.weights.clone(),
        flags: f
            .flags
            .iter()
            .map(|(k, v)| (k.clone(), flag_str(*v)))
            .collect(),
        discrimination: f
            .discrimination
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    DiscriminationOut {
                        g: v.g,
                        raw_separation: v.raw_separation,
                        top_m_average: v.top_m_average,
                        degenerate_separation: v.degenerate_separation,
                        reference_cold: v.reference_cold,
                    },
                )
            })
            .collect(),
        confidence: f.confidence,
        conflict: f.conflict,
    }
}

fn divergence_out(d: &rf::Divergence) -> DivergenceOut {
    DivergenceOut {
        per_channel: d.per_channel.clone(),
        max: d.max,
    }
}

fn parse_state(s: &str) -> Result<rf::RuffleState, JsValue> {
    serde_json::from_str(s).map_err(|e| state_err(format!("invalid ruffle state JSON: {e}")))
}

fn state_json(state: &rf::RuffleState) -> Result<String, JsValue> {
    serde_json::to_string(state).map_err(internal_err)
}

// --- the Fuser handle ------------------------------------------------------------

/// The stateful engine handle behind the TypeScript `Fuser` class.
#[wasm_bindgen]
pub struct Fuser {
    inner: rf::Fuser,
}

#[wasm_bindgen]
impl Fuser {
    #[wasm_bindgen(constructor)]
    pub fn new(channels: JsValue, config: JsValue) -> Result<Fuser, JsValue> {
        let channels = to_channel_configs(channels)?;
        let cfg = to_fuse_config(config)?;
        let inner = rf::Fuser::new(&channels, cfg).map_err(config_err)?;
        Ok(Fuser { inner })
    }

    pub fn resume(channels: JsValue, config: JsValue, state: &str) -> Result<Fuser, JsValue> {
        let channels = to_channel_configs(channels)?;
        let cfg = to_fuse_config(config)?;
        let state = parse_state(state)?;
        let inner = rf::Fuser::resume(&channels, state, cfg).map_err(resume_err)?;
        Ok(Fuser { inner })
    }

    pub fn fuse(&mut self, inputs: JsValue) -> Result<JsValue, JsValue> {
        let inputs = to_inputs(inputs)?;
        to_js(&fused_out(&self.inner.fuse(&inputs)))
    }

    #[wasm_bindgen(js_name = fuseStateless)]
    pub fn fuse_stateless(
        inputs: JsValue,
        channels: JsValue,
        config: JsValue,
        prior: &str,
    ) -> Result<JsValue, JsValue> {
        let inputs = to_inputs(inputs)?;
        let channels = to_channel_configs(channels)?;
        let cfg = to_fuse_config(config)?;
        let prior = parse_state(prior)?;
        let fused =
            rf::Fuser::fuse_stateless(&inputs, &channels, &prior, &cfg).map_err(resume_err)?;
        to_js(&fused_out(&fused))
    }

    #[wasm_bindgen(js_name = refreshCoupling)]
    pub fn refresh_coupling(&mut self, anchor: JsValue) -> Result<(), JsValue> {
        let anchor = to_anchor(anchor)?;
        self.inner.refresh_coupling(&anchor);
        Ok(())
    }

    #[wasm_bindgen(js_name = stateJson)]
    pub fn state_json(&self) -> Result<String, JsValue> {
        state_json(self.inner.state())
    }
}

// --- state operations ----------------------------------------------------------------

/// Parses a state and re-serializes it canonically, validating it in the process.
#[wasm_bindgen(js_name = stateCanonicalize)]
pub fn state_canonicalize(state: &str) -> Result<String, JsValue> {
    state_json(&parse_state(state)?)
}

/// Merges several states, returning the merged canonical bytes and the advisory
/// divergence.
#[wasm_bindgen(js_name = stateMerge)]
pub fn state_merge(parts: Vec<String>) -> Result<JsValue, JsValue> {
    let states: Vec<rf::RuffleState> = parts
        .iter()
        .map(|s| parse_state(s))
        .collect::<Result<_, JsValue>>()?;
    let refs: Vec<&rf::RuffleState> = states.iter().collect();
    let (merged, divergence) =
        rf::RuffleState::merge(&refs, rf::MergePolicy::Strict).map_err(merge_err)?;
    to_js(&MergeOut {
        merged: state_json(&merged)?,
        divergence: divergence_out(&divergence),
    })
}

/// The advisory divergence between two states.
#[wasm_bindgen(js_name = stateDivergence)]
pub fn state_divergence(a: &str, b: &str) -> Result<JsValue, JsValue> {
    let (a, b) = (parse_state(a)?, parse_state(b)?);
    to_js(&divergence_out(&a.divergence(&b)))
}

/// Renames a channel key, returning the updated canonical bytes.
#[wasm_bindgen(js_name = stateRekey)]
pub fn state_rekey(state: &str, from_key: &str, to_key: &str) -> Result<String, JsValue> {
    let mut st = parse_state(state)?;
    st.rekey(from_key, to_key.to_string());
    state_json(&st)
}

/// Scales every summary's confidence down by `factor`, returning the updated
/// canonical bytes.
#[wasm_bindgen(js_name = stateDecay)]
pub fn state_decay(state: &str, factor: f64) -> Result<String, JsValue> {
    let mut st = parse_state(state)?;
    st.decay(factor);
    state_json(&st)
}

/// The default fusion configuration, in the boundary schema. The TypeScript
/// configuration layer reads its defaults from this, so the crate's defaults are
/// never duplicated.
#[wasm_bindgen(js_name = defaultConfig)]
pub fn default_config() -> Result<JsValue, JsValue> {
    let cfg = rf::FuseConfig::default();
    let d = &cfg.discrimination;
    let c = &cfg.coupling;
    to_js(&ConfigDto {
        discrimination: DiscriminationDto {
            top_eps: d.top_eps,
            top_m: d.top_m,
            min_distinct_values: d.min_distinct_values,
            denom_floor_frac: d.denom_floor_frac,
            winsor_z: d.winsor_z,
            min_count_for_z: d.min_count_for_z,
            shrink_pool_size: d.shrink_pool_size,
            g_upper_bound: d.g_upper_bound,
            g_floor: d.g_floor,
            g_slope: d.g_slope,
            g_deviation_keep: d.g_deviation_keep,
        },
        coupling: CouplingDto {
            enabled: c.enabled,
            discount_cap: c.discount_cap,
            shrink_to_identity: c.shrink_to_identity,
            min_overlap: c.min_overlap,
            min_reliability: c.min_reliability,
            min_refreshes: c.min_refreshes,
            stratum_stability_max_var: c.stratum_stability_max_var,
        },
        fusion: FusionDto {
            rrf_eta: cfg.fusion.rrf_eta,
        },
        decay: DecayDto {
            enabled: cfg.decay.enabled,
            factor: cfg.decay.factor,
        },
        baseline_mode: "z_score".to_string(),
    })
}

/// The engine version this artifact was built from (the ruffle-wasm crate version,
/// kept in lockstep with the core crate and the npm package).
#[wasm_bindgen(js_name = engineVersion)]
pub fn engine_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// The state schema version this build writes.
#[wasm_bindgen(js_name = formatVersion)]
pub fn format_version() -> u32 {
    rf::RuffleState::FORMAT_VERSION
}

/// The statistic-definition version this build writes.
#[wasm_bindgen(js_name = statVersion)]
pub fn stat_version() -> u32 {
    rf::StatFingerprint::STAT_VERSION
}
