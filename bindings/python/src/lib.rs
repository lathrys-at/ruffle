//! The compiled core of the `ruffle` Python package, exposed as `ruffle._core`.
//!
//! The public Python surface lives in `python/ruffle/`; this module is private plumbing
//! behind it. It links the Rust engine directly and reimplements no statistic, so the
//! binding's behaviour is the crate's behaviour by construction.
//!
//! Boundary conventions, typed on the Python side by `ruffle._types` and exercised by
//! the parity fixtures under `tests/fixtures/parity/` in the repository root:
//!
//! - Channel registrations and the fusion configuration cross as typed dictionaries
//!   (`direction` as `"higher_is_better"`/`"lower_is_better"`, `baseline_mode` as
//!   `"z_score"`, `good_score` as `None` or `{typical, good, weight}`).
//! - Per-query inputs cross as typed tuples, `("scored", key, direction, items)` or
//!   `("ranked", key, ids)`, with native pre-orientation scores; orientation and
//!   sanitization run in the engine's own ingest path.
//! - States cross as canonical JSON strings, byte-identical to the core crate's
//!   serialization; the string is the persistence format itself, not an encoding
//!   convenience.

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use ruffle as rf;
use std::collections::BTreeMap;

create_exception!(
    ruffle,
    RuffleError,
    PyException,
    "The base class for every error Ruffle raises."
);
create_exception!(
    ruffle,
    ConfigError,
    RuffleError,
    "The channel registrations or the fusion configuration are invalid on their own: \
     an out-of-range knob, a duplicate channel key, or a declared good score that does \
     not orient to a usable reference."
);
create_exception!(
    ruffle,
    ResumeError,
    RuffleError,
    "A persisted state is incompatible with the registrations or with this build: a \
     format or statistic-version mismatch, a flipped direction, or a changed \
     model-version tag (the signature of a model swap)."
);
create_exception!(
    ruffle,
    MergeError,
    RuffleError,
    "Two states cannot be merged: they disagree on format, statistic definitions, a \
     channel's orientation, or a channel's model-version tag, or the merge received no \
     states at all."
);
create_exception!(
    ruffle,
    StateError,
    RuffleError,
    "A serialized state document could not be parsed: it is not JSON, or not the \
     state schema."
);

/// The one score newtype the binding needs: a native Python float, declared as-is.
/// The meaning a Rust caller declares through a newtype is carried on the Python side
/// by the channel configuration (direction, tag, good-score reference).
struct Native(f64);
impl rf::Score for Native {
    fn value(&self) -> f64 {
        self.0
    }
}

// --- the typed-dict boundary (mirrored by `ruffle._types`) --------------------------

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct GoodScoreArg {
    typical: f64,
    good: f64,
    weight: f64,
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct ChannelArg {
    key: String,
    tag: String,
    direction: String,
    good_score: Option<GoodScoreArg>,
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct DiscriminationArg {
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
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct CouplingArg {
    enabled: bool,
    discount_cap: f64,
    shrink_to_identity: f64,
    min_overlap: usize,
    min_reliability: f64,
    min_refreshes: f64,
    stratum_stability_max_var: f64,
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct FusionArg {
    rrf_eta: f64,
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct DecayArg {
    enabled: bool,
    factor: f64,
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct ConfigArg {
    discrimination: DiscriminationArg,
    coupling: CouplingArg,
    fusion: FusionArg,
    decay: DecayArg,
    baseline_mode: String,
}

fn parse_direction(s: &str) -> PyResult<rf::Direction> {
    match s {
        "higher_is_better" => Ok(rf::Direction::HigherIsBetter),
        "lower_is_better" => Ok(rf::Direction::LowerIsBetter),
        other => Err(PyValueError::new_err(format!(
            "unknown direction {other:?}"
        ))),
    }
}

fn to_channel_configs(channels: Vec<ChannelArg>) -> PyResult<Vec<rf::ChannelConfig>> {
    channels
        .into_iter()
        .map(|c| {
            Ok(rf::ChannelConfig::new(
                rf::ChannelId::new(c.key, c.tag),
                parse_direction(&c.direction)?,
                c.good_score
                    .map(|g| rf::GoodScore::new(g.typical, g.good, g.weight)),
            ))
        })
        .collect()
}

fn to_fuse_config(arg: ConfigArg) -> PyResult<rf::FuseConfig> {
    if arg.baseline_mode != "z_score" {
        return Err(PyValueError::new_err(format!(
            "unknown baseline mode {:?}",
            arg.baseline_mode
        )));
    }
    let mut cfg = rf::FuseConfig::default();
    let d = &arg.discrimination;
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
    let c = &arg.coupling;
    cfg.coupling.enabled = c.enabled;
    cfg.coupling.discount_cap = c.discount_cap;
    cfg.coupling.shrink_to_identity = c.shrink_to_identity;
    cfg.coupling.min_overlap = c.min_overlap;
    cfg.coupling.min_reliability = c.min_reliability;
    cfg.coupling.min_refreshes = c.min_refreshes;
    cfg.coupling.stratum_stability_max_var = c.stratum_stability_max_var;
    cfg.fusion.rrf_eta = arg.fusion.rrf_eta;
    cfg.decay.enabled = arg.decay.enabled;
    cfg.decay.factor = arg.decay.factor;
    cfg.baseline_mode = rf::BaselineMode::ZScore;
    Ok(cfg)
}

// --- error mapping -------------------------------------------------------------------

fn config_err(e: rf::ConfigError) -> PyErr {
    ConfigError::new_err(e.to_string())
}

fn resume_err(e: rf::ResumeError) -> PyErr {
    match e {
        rf::ResumeError::Config(c) => config_err(c),
        other => ResumeError::new_err(other.to_string()),
    }
}

fn merge_err(e: rf::Mismatch) -> PyErr {
    MergeError::new_err(e.to_string())
}

fn internal_err(e: impl std::fmt::Display) -> PyErr {
    RuffleError::new_err(format!("internal serialization failure: {e}"))
}

// --- the state boundary ----------------------------------------------------------------

fn parse_state(s: &str) -> PyResult<rf::RuffleState> {
    serde_json::from_str(s)
        .map_err(|e| StateError::new_err(format!("invalid ruffle state JSON: {e}")))
}

fn state_json(state: &rf::RuffleState) -> PyResult<String> {
    serde_json::to_string(state).map_err(internal_err)
}

// --- per-query inputs ---------------------------------------------------------------

fn extract_inputs(inputs: &Bound<'_, PyAny>) -> PyResult<Vec<rf::ChannelInput<String>>> {
    let mut out = Vec::new();
    for item in inputs.try_iter()? {
        let item = item?;
        let kind: String = item.get_item(0)?.extract()?;
        match kind.as_str() {
            "scored" => {
                let (_, key, direction, items): (String, String, String, Vec<(String, f64)>) =
                    item.extract()?;
                let dir = parse_direction(&direction)?;
                // Tag and reference play no role at ingest; the engine's own scored()
                // path performs the orientation and non-finite filtering.
                let cfg = rf::ChannelConfig::new(rf::ChannelId::new(key, String::new()), dir, None);
                out.push(rf::ChannelInput::scored(
                    &cfg,
                    items
                        .into_iter()
                        .map(|(id, s)| (id, Native(s)))
                        .collect::<Vec<_>>(),
                ));
            }
            "ranked" => {
                let (_, key, ids): (String, String, Vec<String>) = item.extract()?;
                let cfg = rf::ChannelConfig::new(
                    rf::ChannelId::new(key, String::new()),
                    rf::Direction::HigherIsBetter,
                    None,
                );
                out.push(rf::ChannelInput::ranked(&cfg, ids));
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown input kind {other:?}"
                )));
            }
        }
    }
    Ok(out)
}

fn flag_str(flag: rf::ChannelFlag) -> PyResult<&'static str> {
    match flag {
        rf::ChannelFlag::RanksOnlyDefaultWeighted => Ok("ranks_only_default_weighted"),
        rf::ChannelFlag::DegenerateSeparation => Ok("degenerate_separation"),
        rf::ChannelFlag::NoReference => Ok("no_reference"),
        // ChannelFlag is #[non_exhaustive]; a variant this binding predates raises
        // rather than leaking an out-of-contract string through the typed surface.
        other => Err(RuffleError::new_err(format!(
            "the engine reported a channel flag this binding does not know: {other:?}"
        ))),
    }
}

fn fused_to_py<'py>(py: Python<'py>, f: &rf::Fused<String>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ranking", f.ranking.clone())?;
    d.set_item("weights", f.weights.clone())?;
    let flags: BTreeMap<String, &'static str> = f
        .flags
        .iter()
        .map(|(k, v)| Ok((k.clone(), flag_str(*v)?)))
        .collect::<PyResult<_>>()?;
    d.set_item("flags", flags)?;
    let disc = PyDict::new(py);
    for (k, v) in &f.discrimination {
        let e = PyDict::new(py);
        e.set_item("g", v.g)?;
        e.set_item("raw_separation", v.raw_separation)?;
        e.set_item("top_m_average", v.top_m_average)?;
        e.set_item("degenerate_separation", v.degenerate_separation)?;
        e.set_item("reference_cold", v.reference_cold)?;
        disc.set_item(k, e)?;
    }
    d.set_item("discrimination", disc)?;
    d.set_item("confidence", f.confidence)?;
    d.set_item("conflict", f.conflict)?;
    Ok(d)
}

fn divergence_to_py<'py>(
    py: Python<'py>,
    divergence: &rf::Divergence,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("per_channel", divergence.per_channel.clone())?;
    d.set_item("max", divergence.max)?;
    Ok(d)
}

fn build_anchor(
    channels: Vec<(String, String)>,
    rows: Vec<Vec<Option<f64>>>,
) -> PyResult<rf::Anchor> {
    if channels.len() != rows.len() {
        return Err(PyValueError::new_err(format!(
            "anchor has {} channels but {} score rows",
            channels.len(),
            rows.len()
        )));
    }
    let n = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|r| r.len() != n) {
        return Err(PyValueError::new_err(
            "anchor score rows must all have one entry per candidate",
        ));
    }
    let configs: Vec<rf::ChannelConfig> = channels
        .iter()
        .map(|(key, dir)| {
            Ok(rf::ChannelConfig::new(
                rf::ChannelId::new(key.clone(), String::new()),
                parse_direction(dir)?,
                None,
            ))
        })
        .collect::<PyResult<_>>()?;
    let refs: Vec<&rf::ChannelConfig> = configs.iter().collect();
    let candidates: Vec<usize> = (0..n).collect();
    Ok(rf::Anchor::build(&candidates, &refs, |i: &usize, key| {
        let row = channels.iter().position(|(k, _)| k == key)?;
        rows[row][*i].map(Native)
    }))
}

// --- the Fuser class -----------------------------------------------------------------

/// The stateful engine handle behind `ruffle.Fuser`.
#[pyclass(module = "ruffle._core")]
struct Fuser {
    inner: rf::Fuser,
}

#[pymethods]
impl Fuser {
    #[new]
    fn new(channels: Vec<ChannelArg>, config: ConfigArg) -> PyResult<Self> {
        let channels = to_channel_configs(channels)?;
        let cfg = to_fuse_config(config)?;
        let inner = rf::Fuser::new(&channels, cfg).map_err(config_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn resume(channels: Vec<ChannelArg>, config: ConfigArg, state: &str) -> PyResult<Self> {
        let channels = to_channel_configs(channels)?;
        let cfg = to_fuse_config(config)?;
        let state = parse_state(state)?;
        let inner = rf::Fuser::resume(&channels, state, cfg).map_err(resume_err)?;
        Ok(Self { inner })
    }

    fn fuse<'py>(
        &mut self,
        py: Python<'py>,
        inputs: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let inputs = extract_inputs(inputs)?;
        let inner = &mut self.inner;
        let fused = py.detach(|| inner.fuse(&inputs));
        fused_to_py(py, &fused)
    }

    #[staticmethod]
    fn fuse_stateless<'py>(
        py: Python<'py>,
        inputs: &Bound<'py, PyAny>,
        channels: Vec<ChannelArg>,
        config: ConfigArg,
        prior: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let inputs = extract_inputs(inputs)?;
        let channels = to_channel_configs(channels)?;
        let cfg = to_fuse_config(config)?;
        let prior = parse_state(prior)?;
        let fused = py
            .detach(|| rf::Fuser::fuse_stateless(&inputs, &channels, &prior, &cfg))
            .map_err(resume_err)?;
        fused_to_py(py, &fused)
    }

    fn refresh_coupling(
        &mut self,
        py: Python<'_>,
        channels: Vec<(String, String)>,
        rows: Vec<Vec<Option<f64>>>,
    ) -> PyResult<()> {
        let anchor = build_anchor(channels, rows)?;
        let inner = &mut self.inner;
        py.detach(|| inner.refresh_coupling(&anchor));
        Ok(())
    }

    fn state_json(&self) -> PyResult<String> {
        state_json(self.inner.state())
    }
}

// --- state operations ------------------------------------------------------------------

/// Parses a state document and re-serializes it canonically; a document that does
/// not parse as a state is refused with `StateError`. Parsing checks the document's
/// shape against the state schema, not its versions; the format and statistic
/// version gates run at resume and merge.
#[pyfunction]
fn state_canonicalize(state: &str) -> PyResult<String> {
    state_json(&parse_state(state)?)
}

fn parse_policy(policy: &str) -> PyResult<rf::MergePolicy> {
    match policy {
        "strict" => Ok(rf::MergePolicy::Strict),
        other => Err(PyValueError::new_err(format!(
            "unknown merge policy {other:?}"
        ))),
    }
}

/// Merges several states under the named policy, returning the merged canonical
/// bytes and the advisory divergence.
#[pyfunction]
fn state_merge<'py>(
    py: Python<'py>,
    parts: Vec<String>,
    policy: &str,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let policy = parse_policy(policy)?;
    let states: Vec<rf::RuffleState> = parts
        .iter()
        .map(|s| parse_state(s))
        .collect::<PyResult<_>>()?;
    let refs: Vec<&rf::RuffleState> = states.iter().collect();
    let (merged, divergence) = py
        .detach(|| rf::RuffleState::merge(&refs, policy))
        .map_err(merge_err)?;
    Ok((state_json(&merged)?, divergence_to_py(py, &divergence)?))
}

/// The advisory divergence between two states.
#[pyfunction]
fn state_divergence<'py>(py: Python<'py>, a: &str, b: &str) -> PyResult<Bound<'py, PyDict>> {
    let (a, b) = (parse_state(a)?, parse_state(b)?);
    divergence_to_py(py, &a.divergence(&b))
}

/// Renames a channel key, returning the updated canonical bytes.
#[pyfunction]
fn state_rekey(state: &str, from_key: &str, to_key: &str) -> PyResult<String> {
    let mut st = parse_state(state)?;
    st.rekey(from_key, to_key.to_string());
    state_json(&st)
}

/// Scales every summary's confidence down by `factor`, returning the updated
/// canonical bytes.
#[pyfunction]
fn state_decay(state: &str, factor: f64) -> PyResult<String> {
    let mut st = parse_state(state)?;
    st.decay(factor);
    state_json(&st)
}

/// The default fusion configuration, in the boundary schema. The pure-Python
/// configuration classes read their field defaults from this, so the crate's defaults
/// are never duplicated.
#[pyfunction]
fn default_config(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let cfg = rf::FuseConfig::default();
    let out = PyDict::new(py);

    let d = PyDict::new(py);
    d.set_item("top_eps", cfg.discrimination.top_eps)?;
    d.set_item("top_m", cfg.discrimination.top_m)?;
    d.set_item(
        "min_distinct_values",
        cfg.discrimination.min_distinct_values,
    )?;
    d.set_item("denom_floor_frac", cfg.discrimination.denom_floor_frac)?;
    d.set_item("winsor_z", cfg.discrimination.winsor_z)?;
    d.set_item("min_count_for_z", cfg.discrimination.min_count_for_z)?;
    d.set_item("shrink_pool_size", cfg.discrimination.shrink_pool_size)?;
    d.set_item("g_upper_bound", cfg.discrimination.g_upper_bound)?;
    d.set_item("g_floor", cfg.discrimination.g_floor)?;
    d.set_item("g_slope", cfg.discrimination.g_slope)?;
    out.set_item("discrimination", d)?;

    let c = PyDict::new(py);
    c.set_item("enabled", cfg.coupling.enabled)?;
    c.set_item("discount_cap", cfg.coupling.discount_cap)?;
    c.set_item("shrink_to_identity", cfg.coupling.shrink_to_identity)?;
    c.set_item("min_overlap", cfg.coupling.min_overlap)?;
    c.set_item("min_reliability", cfg.coupling.min_reliability)?;
    c.set_item("min_refreshes", cfg.coupling.min_refreshes)?;
    c.set_item(
        "stratum_stability_max_var",
        cfg.coupling.stratum_stability_max_var,
    )?;
    out.set_item("coupling", c)?;

    let f = PyDict::new(py);
    f.set_item("rrf_eta", cfg.fusion.rrf_eta)?;
    out.set_item("fusion", f)?;

    let y = PyDict::new(py);
    y.set_item("enabled", cfg.decay.enabled)?;
    y.set_item("factor", cfg.decay.factor)?;
    out.set_item("decay", y)?;

    out.set_item("baseline_mode", "z_score")?;
    Ok(out)
}

/// The private extension module behind the `ruffle` Python package.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("FORMAT_VERSION", rf::RuffleState::FORMAT_VERSION)?;
    m.add("STAT_VERSION", rf::StatFingerprint::STAT_VERSION)?;
    m.add("RuffleError", py.get_type::<RuffleError>())?;
    m.add("ConfigError", py.get_type::<ConfigError>())?;
    m.add("ResumeError", py.get_type::<ResumeError>())?;
    m.add("MergeError", py.get_type::<MergeError>())?;
    m.add("StateError", py.get_type::<StateError>())?;
    m.add_class::<Fuser>()?;
    m.add_function(wrap_pyfunction!(state_canonicalize, m)?)?;
    m.add_function(wrap_pyfunction!(state_merge, m)?)?;
    m.add_function(wrap_pyfunction!(state_divergence, m)?)?;
    m.add_function(wrap_pyfunction!(state_rekey, m)?)?;
    m.add_function(wrap_pyfunction!(state_decay, m)?)?;
    m.add_function(wrap_pyfunction!(default_config, m)?)?;
    Ok(())
}
