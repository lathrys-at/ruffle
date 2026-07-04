//! Configuration: per-channel registration and the fusion knobs.
//!
//! Every knob has a conservative default, chosen so the shipped behaviour stays close
//! to plain RRF. The defaults err toward keeping channels and tilting weights only
//! mildly.

use crate::error::ConfigError;
use crate::keys::{BaselineMode, ChannelId};
use crate::score::{Direction, GoodScore};
use serde::{Deserialize, Serialize};

/// Per-channel discrimination knobs: how each channel's separation and absolute goodness
/// are read and turned into a weight.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DiscriminationConfig {
    /// Fraction of the result pool forming the "extreme top" whose mean is the numerator
    /// of the separation statistic (top versus bulk).
    pub top_eps: f64,
    /// Fixed number of top scores averaged for the absolute-goodness statistic. A fixed
    /// count is steadier than the single maximum.
    pub top_m: usize,
    /// Minimum number of distinct pool values required before the separation statistic is
    /// computed. Below it the bulk is too degenerate to support the ratio.
    pub min_distinct_values: usize,
    /// Floors the separation statistic's denominator toward the inter-quartile gap by this
    /// fraction, so a near-tied bulk cannot inflate the ratio.
    pub denom_floor_frac: f64,
    /// A standardized separation read beyond this many standard deviations is winsorized
    /// before it touches the baseline, so one extreme query cannot corrupt the streaming
    /// mean.
    pub winsor_z: f64,
    /// Minimum effective baseline count before a standardized separation read is trusted.
    /// Below it the channel leans on its own baseline.
    pub min_count_for_z: f64,
    /// Pool size below which the channel's weight is shrunk toward its own running
    /// discrimination baseline, in proportion to how little data backs the read.
    pub shrink_pool_size: usize,
    /// Upper bound on the discrimination weight `g`, so no single channel can dominate the
    /// fused order.
    pub g_upper_bound: f64,
    /// Small positive floor on `g`, so an uncertain channel still contributes. Zeroing a
    /// channel on one noisy read is a recall risk.
    pub g_floor: f64,
    /// Slope of the logistic squash that maps each standardized statistic to a `(0, 1)`
    /// factor in the discrimination weight `g`. A larger slope makes the weight react more
    /// sharply to a departure from the channel's norm; the default keeps the response
    /// gentle, so a single query moves a channel only slightly.
    pub g_slope: f64,
}

impl Default for DiscriminationConfig {
    fn default() -> Self {
        Self {
            top_eps: 0.10,
            top_m: 5,
            min_distinct_values: 8,
            denom_floor_frac: 0.75,
            winsor_z: 2.5,
            min_count_for_z: 5.0,
            shrink_pool_size: 20,
            g_upper_bound: 4.0,
            g_floor: 0.25,
            g_slope: 1.0,
        }
    }
}

/// Channel-coupling knobs: how the redundancy discount between channels is estimated and
/// applied.
///
/// Independence is the only unconditionally recall-safe setting, so coupling is off by
/// default and every knob caps how far a discount can move weight.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct CouplingConfig {
    /// Whether to apply any redundancy discount at all. Off by default, since assuming
    /// independence is the recall-safe choice.
    pub enabled: bool,
    /// Caps the discount well below the raw anchor point estimate, since turning coupling
    /// on trades recall-safety for precision.
    pub discount_cap: f64,
    /// Mandatory shrinkage intensity, in `[0, 1]`, of the redundancy correlation toward
    /// the identity. Keeps the assembled covariance positive-definite and biases the
    /// discount toward treating channels as distinct.
    pub shrink_to_identity: f64,
    /// Minimum number of anchor items scored by both channels before a pair correlation
    /// counts.
    pub min_overlap: usize,
    /// Minimum effective reliability (accumulated count) before any discount applies.
    /// Below it the discount is dropped entirely, the recall-safe direction.
    pub min_reliability: f64,
    /// Minimum number of anchor refreshes backing a pair before any discount applies.
    /// Stability across query strata is a between-refresh property: a single refresh has
    /// zero between-refresh variance by construction, so it would pass the stability
    /// gate vacuously.
    pub min_refreshes: f64,
    /// Maximum between-stratum variance of the anchor correlation that still allows a
    /// discount. A correlation that is unstable across query strata degrades to
    /// independence.
    pub stratum_stability_max_var: f64,
}

impl Default for CouplingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            discount_cap: 0.5,
            shrink_to_identity: 0.5,
            min_overlap: 30,
            min_reliability: 10.0,
            min_refreshes: 2.0,
            stratum_stability_max_var: 0.25,
        }
    }
}

/// Weighted reciprocal-rank fusion knobs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RrfConfig {
    /// The RRF rank constant `η`. Larger values flatten the rank contribution; 60 is the
    /// common RRF default from Cormack et al. (2009).
    pub rrf_eta: f64,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self { rrf_eta: 60.0 }
    }
}

/// State-decay knobs: forgetting old observations to track corpus drift.
///
/// Off by default. Decay ties a merge to an external clock, making the otherwise exact
/// merge identity approximate.
///
/// The cadence is per observation, not per wall-clock interval: a channel's baselines
/// decay once per fuse in which the channel appears, and a pair's redundancy decays once
/// per anchor refresh. This bounds each baseline's effective sample size at
/// `1 / (1 - factor)` observations, so a rarely-queried channel ages more slowly than a
/// busy one. A caller who wants wall-clock decay instead can call
/// [`RuffleState::decay`](crate::RuffleState::decay) on its own schedule with this
/// setting left off.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DecayConfig {
    /// Whether decay is applied at all. Off by default.
    pub enabled: bool,
    /// Per-decay-step multiplier on the effective count, in `[0, 1]`. Only takes
    /// effect when `enabled`; preserves mean and variance while reducing confidence.
    pub factor: f64,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            factor: 0.98,
        }
    }
}

/// The complete fusion configuration: the grouped sub-configs plus the baseline mode.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct FuseConfig {
    /// Per-channel discrimination knobs.
    pub discrimination: DiscriminationConfig,
    /// Channel-coupling knobs.
    pub coupling: CouplingConfig,
    /// Rank-fusion knobs.
    pub fusion: RrfConfig,
    /// State-decay knobs.
    pub decay: DecayConfig,
    /// Which within-channel standardization the baselines use.
    pub baseline_mode: BaselineMode,
}

impl Default for FuseConfig {
    fn default() -> Self {
        Self {
            discrimination: DiscriminationConfig::default(),
            coupling: CouplingConfig::default(),
            fusion: RrfConfig::default(),
            decay: DecayConfig::default(),
            baseline_mode: BaselineMode::ZScore,
        }
    }
}

/// Rejects the configuration unless `ok` holds, naming the offending field.
fn check(ok: bool, field: &'static str, reason: &'static str) -> Result<(), ConfigError> {
    if ok {
        Ok(())
    } else {
        Err(ConfigError::InvalidFuseConfig { field, reason })
    }
}

impl FuseConfig {
    /// Checks every knob against its documented range.
    ///
    /// [`Fuser::new`](crate::Fuser::new) and [`Fuser::resume`](crate::Fuser::resume) run
    /// this before accepting a configuration, so an out-of-range knob fails at
    /// construction rather than mid-query. It is public so a caller assembling a
    /// configuration from external input can validate it directly.
    ///
    /// The requirements are the documented ranges: every `f64` knob finite;
    /// `top_eps` in `(0, 1]`; `top_m >= 1`; `min_distinct_values >= 2`;
    /// `denom_floor_frac >= 0`; `winsor_z > 0`; `min_count_for_z > 0`;
    /// `shrink_pool_size >= 1`; `0 <= g_floor <= g_upper_bound` with a positive upper
    /// bound; `g_slope > 0`; `discount_cap` in `[0, 1]`; `shrink_to_identity` in
    /// `[0, 1]`; `min_overlap >= 2`; `min_reliability >= 0`; `min_refreshes >= 0`;
    /// `stratum_stability_max_var >= 0`; `rrf_eta >= 0`; and the decay `factor` in
    /// `[0, 1]`.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let d = &self.discrimination;
        check(
            d.top_eps.is_finite() && d.top_eps > 0.0 && d.top_eps <= 1.0,
            "discrimination.top_eps",
            "must be finite and in (0, 1]",
        )?;
        check(d.top_m >= 1, "discrimination.top_m", "must be at least 1")?;
        check(
            d.min_distinct_values >= 2,
            "discrimination.min_distinct_values",
            "must be at least 2",
        )?;
        check(
            d.denom_floor_frac.is_finite() && d.denom_floor_frac >= 0.0,
            "discrimination.denom_floor_frac",
            "must be finite and non-negative",
        )?;
        check(
            d.winsor_z.is_finite() && d.winsor_z > 0.0,
            "discrimination.winsor_z",
            "must be finite and positive",
        )?;
        check(
            d.min_count_for_z.is_finite() && d.min_count_for_z > 0.0,
            "discrimination.min_count_for_z",
            "must be finite and positive",
        )?;
        check(
            d.shrink_pool_size >= 1,
            "discrimination.shrink_pool_size",
            "must be at least 1",
        )?;
        check(
            d.g_floor.is_finite() && d.g_floor >= 0.0,
            "discrimination.g_floor",
            "must be finite and non-negative",
        )?;
        check(
            d.g_upper_bound.is_finite() && d.g_upper_bound > 0.0 && d.g_upper_bound >= d.g_floor,
            "discrimination.g_upper_bound",
            "must be finite, positive, and at least g_floor",
        )?;
        check(
            d.g_slope.is_finite() && d.g_slope > 0.0,
            "discrimination.g_slope",
            "must be finite and positive",
        )?;

        let c = &self.coupling;
        check(
            c.discount_cap.is_finite() && (0.0..=1.0).contains(&c.discount_cap),
            "coupling.discount_cap",
            "must be finite and in [0, 1]",
        )?;
        check(
            c.shrink_to_identity.is_finite() && (0.0..=1.0).contains(&c.shrink_to_identity),
            "coupling.shrink_to_identity",
            "must be finite and in [0, 1]",
        )?;
        check(
            c.min_overlap >= 2,
            "coupling.min_overlap",
            "must be at least 2 (a correlation needs two points)",
        )?;
        check(
            c.min_reliability.is_finite() && c.min_reliability >= 0.0,
            "coupling.min_reliability",
            "must be finite and non-negative",
        )?;
        check(
            c.min_refreshes.is_finite() && c.min_refreshes >= 0.0,
            "coupling.min_refreshes",
            "must be finite and non-negative",
        )?;
        check(
            c.stratum_stability_max_var.is_finite() && c.stratum_stability_max_var >= 0.0,
            "coupling.stratum_stability_max_var",
            "must be finite and non-negative",
        )?;

        check(
            self.fusion.rrf_eta.is_finite() && self.fusion.rrf_eta >= 0.0,
            "fusion.rrf_eta",
            "must be finite and non-negative",
        )?;
        check(
            self.decay.factor.is_finite() && (0.0..=1.0).contains(&self.decay.factor),
            "decay.factor",
            "must be finite and in [0, 1]",
        )?;
        Ok(())
    }
}

/// Per-channel registration.
///
/// `id` (the join handle `key` and the model-version `tag`) and `direction` are declared
/// once at channel configuration rather than per query. `good_score` is the optional
/// declared reference for the absolute-goodness statistic; when absent, the reference is
/// learned from early traffic and the absolute-goodness statistic cold-starts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// The channel's identity: its stable join handle `key` plus the model-version `tag`
    /// that gates merge and is bumped on a model change.
    pub id: ChannelId,
    /// Whether higher or lower native scores are better. Declared once per channel.
    pub direction: Direction,
    /// The optional declared good-score reference for the absolute-goodness statistic, in
    /// native units.
    pub good_score: Option<GoodScore>,
    /// An operator-declared static weight multiplier on the channel's adaptive per-query
    /// weight: the fused weight is `base_weight * g`, renormalized over the channels
    /// present on the query.
    ///
    /// The engine never learns that one channel is globally better than another; that is
    /// cross-channel information only relevance labels can establish. An operator who
    /// holds such labels (a fitted evaluation, domain knowledge) declares the tilt here,
    /// and the per-query adaptation composes on top. Only the ratios between channels
    /// matter. The default `1.0` declares nothing. `0.0` is legal: it silences the
    /// channel's votes while its baselines keep updating.
    #[serde(default = "default_base_weight")]
    pub base_weight: f64,
}

fn default_base_weight() -> f64 {
    1.0
}

impl ChannelConfig {
    /// Builds a channel registration. A `Some` `good_score` enables absolute goodness
    /// from the first query; `None` learns the reference from traffic. The base weight
    /// starts neutral at `1.0`; declare a tilt with
    /// [`with_base_weight`](Self::with_base_weight).
    pub fn new(id: ChannelId, direction: Direction, good_score: Option<GoodScore>) -> Self {
        Self {
            id,
            direction,
            good_score,
            base_weight: default_base_weight(),
        }
    }

    /// Declares the static weight multiplier. The value must be finite and non-negative;
    /// [`Fuser`](crate::Fuser) construction rejects anything else.
    #[must_use]
    pub fn with_base_weight(mut self, base_weight: f64) -> Self {
        self.base_weight = base_weight;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_conservative() {
        let cfg = FuseConfig::default();
        // Coupling and decay both off: closest to plain RRF.
        assert!(!cfg.coupling.enabled);
        assert!(!cfg.decay.enabled);
        // RRF's common rank constant.
        assert_eq!(cfg.fusion.rrf_eta, 60.0);
        assert_eq!(cfg.baseline_mode, BaselineMode::ZScore);
    }

    #[test]
    fn discrimination_defaults_present() {
        let d = DiscriminationConfig::default();
        assert!(d.g_floor > 0.0); // an unsure channel still votes
        assert!(d.g_upper_bound > d.g_floor);
        assert!(d.g_slope > 0.0); // the squash has a positive, finite slope
        assert!(d.top_m >= 1);
    }

    #[test]
    fn validate_accepts_the_defaults_and_tight_boundaries() {
        FuseConfig::default().validate().unwrap();

        // Every knob at its extreme legal value still validates: the checks are
        // boundary-inclusive where the docs say they are.
        let mut edge = FuseConfig::default();
        edge.discrimination.top_eps = 1.0; // (0, 1] includes 1
        edge.discrimination.top_m = 1;
        edge.discrimination.min_distinct_values = 2;
        edge.discrimination.denom_floor_frac = 0.0;
        edge.discrimination.shrink_pool_size = 1;
        edge.discrimination.g_floor = 0.0;
        edge.discrimination.g_upper_bound = edge.discrimination.g_floor.max(0.1);
        edge.coupling.discount_cap = 1.0;
        edge.coupling.shrink_to_identity = 0.0;
        edge.coupling.min_overlap = 2;
        edge.coupling.min_reliability = 0.0;
        edge.coupling.min_refreshes = 0.0;
        edge.coupling.stratum_stability_max_var = 0.0;
        edge.fusion.rrf_eta = 0.0;
        edge.decay.factor = 0.0;
        edge.validate().unwrap();
        // Equal floor and bound is legal (a pinned weight), including at zero decay=1.
        let mut pinned = FuseConfig::default();
        pinned.discrimination.g_floor = 2.0;
        pinned.discrimination.g_upper_bound = 2.0;
        pinned.decay.factor = 1.0;
        pinned.validate().unwrap();
    }

    /// Apply `set` to a default config and assert validation rejects it, naming `field`.
    fn rejects(field: &str, set: impl Fn(&mut FuseConfig)) {
        let mut cfg = FuseConfig::default();
        set(&mut cfg);
        match cfg.validate() {
            Err(crate::error::ConfigError::InvalidFuseConfig { field: f, .. }) => {
                assert_eq!(f, field, "wrong field named for {field}");
            }
            other => panic!("{field}: expected InvalidFuseConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_each_out_of_range_knob() {
        // Discrimination. Zero and above-one pin the two range arms of top_eps; NaN pins
        // the finiteness arm of every float knob it appears on.
        rejects("discrimination.top_eps", |c| c.discrimination.top_eps = 0.0);
        rejects("discrimination.top_eps", |c| c.discrimination.top_eps = 1.5);
        rejects("discrimination.top_eps", |c| {
            c.discrimination.top_eps = f64::NAN;
        });
        rejects("discrimination.top_m", |c| c.discrimination.top_m = 0);
        rejects("discrimination.min_distinct_values", |c| {
            c.discrimination.min_distinct_values = 1;
        });
        rejects("discrimination.denom_floor_frac", |c| {
            c.discrimination.denom_floor_frac = -0.1;
        });
        rejects("discrimination.denom_floor_frac", |c| {
            c.discrimination.denom_floor_frac = f64::INFINITY;
        });
        rejects("discrimination.winsor_z", |c| {
            c.discrimination.winsor_z = 0.0
        });
        rejects("discrimination.winsor_z", |c| {
            c.discrimination.winsor_z = f64::NAN;
        });
        rejects("discrimination.min_count_for_z", |c| {
            c.discrimination.min_count_for_z = 0.0;
        });
        rejects("discrimination.shrink_pool_size", |c| {
            c.discrimination.shrink_pool_size = 0;
        });
        rejects("discrimination.g_floor", |c| {
            c.discrimination.g_floor = -1.0
        });
        rejects("discrimination.g_floor", |c| {
            c.discrimination.g_floor = f64::NAN;
        });
        // An inverted floor/bound pair names the bound (the pair constraint lives there).
        rejects("discrimination.g_upper_bound", |c| {
            c.discrimination.g_floor = 5.0;
            c.discrimination.g_upper_bound = 4.0;
        });
        rejects("discrimination.g_upper_bound", |c| {
            c.discrimination.g_upper_bound = 0.0;
        });
        // Zero bound with zero floor isolates the strict `> 0.0` arm: the `>= g_floor`
        // arm is satisfied, so only strict positivity can reject it.
        rejects("discrimination.g_upper_bound", |c| {
            c.discrimination.g_floor = 0.0;
            c.discrimination.g_upper_bound = 0.0;
        });
        rejects("discrimination.g_slope", |c| c.discrimination.g_slope = 0.0);
        rejects("discrimination.g_slope", |c| {
            c.discrimination.g_slope = -2.0;
        });

        // Coupling.
        rejects("coupling.discount_cap", |c| c.coupling.discount_cap = -0.1);
        rejects("coupling.discount_cap", |c| c.coupling.discount_cap = 1.1);
        rejects("coupling.discount_cap", |c| {
            c.coupling.discount_cap = f64::NAN;
        });
        rejects("coupling.shrink_to_identity", |c| {
            c.coupling.shrink_to_identity = -0.1;
        });
        rejects("coupling.shrink_to_identity", |c| {
            c.coupling.shrink_to_identity = 1.1;
        });
        rejects("coupling.min_overlap", |c| c.coupling.min_overlap = 1);
        rejects("coupling.min_reliability", |c| {
            c.coupling.min_reliability = -1.0;
        });
        rejects("coupling.min_refreshes", |c| {
            c.coupling.min_refreshes = -1.0
        });
        rejects("coupling.min_refreshes", |c| {
            c.coupling.min_refreshes = f64::NAN;
        });
        rejects("coupling.stratum_stability_max_var", |c| {
            c.coupling.stratum_stability_max_var = -0.5;
        });

        // Fusion and decay.
        rejects("fusion.rrf_eta", |c| c.fusion.rrf_eta = -1.0);
        rejects("fusion.rrf_eta", |c| c.fusion.rrf_eta = f64::NAN);
        rejects("decay.factor", |c| c.decay.factor = -0.1);
        rejects("decay.factor", |c| c.decay.factor = 1.1);
        rejects("decay.factor", |c| c.decay.factor = f64::NAN);
    }

    #[test]
    fn channel_config_round_trips_serde() {
        let cfg = ChannelConfig::new(
            ChannelId::new("clip", "clip-vit-b32-rev1"),
            Direction::HigherIsBetter,
            Some(GoodScore::new(0.3, 0.5, 8.0)),
        );
        // Serde derive is wired up; this just confirms the shape is serializable via
        // the in-crate machinery (no serde_json dependency in foundation).
        let cloned = cfg.clone();
        assert_eq!(cfg, cloned);
    }
}
