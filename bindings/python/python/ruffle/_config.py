"""The fusion configuration: grouped knob dataclasses over the engine's defaults.

Every default is read from the compiled engine at import, so the values shown by
``repr`` and used by the dataclasses are the crate's own defaults, never a copy that
could drift. Every knob has a conservative default, chosen so the shipped behaviour
stays close to plain reciprocal-rank fusion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ruffle import _core
from ruffle._types import (
    CouplingConfigDict,
    DecayConfigDict,
    DiscriminationConfigDict,
    FuseConfigDict,
    RrfConfigDict,
)

__all__ = [
    "BaselineMode",
    "CouplingConfig",
    "DecayConfig",
    "DiscriminationConfig",
    "FuseConfig",
    "RrfConfig",
]

_DEFAULTS: FuseConfigDict = _core.default_config()
_D: DiscriminationConfigDict = _DEFAULTS["discrimination"]
_C: CouplingConfigDict = _DEFAULTS["coupling"]
_F: RrfConfigDict = _DEFAULTS["fusion"]
_Y: DecayConfigDict = _DEFAULTS["decay"]


class BaselineMode(Enum):
    """How a channel's scores are standardized within the channel before comparison.

    Only z-score standardization ships today; a mergeable quantile sketch is a
    planned robustness upgrade.
    """

    Z_SCORE = "z_score"
    """Standardizes each score against the channel's running mean and variance."""


@dataclass(frozen=True)
class DiscriminationConfig:
    """Per-channel discrimination knobs: how each channel's separation and absolute
    goodness are read and turned into a weight.

    - ``top_eps``: fraction of the result pool forming the "extreme top" whose mean is
      the numerator of the separation statistic (top versus bulk).
    - ``top_m``: fixed number of top scores averaged for the absolute-goodness
      statistic. A fixed count is steadier than the single maximum.
    - ``min_distinct_values``: minimum number of distinct pool values required before
      the separation statistic is computed. Below it the bulk is too degenerate to
      support the ratio.
    - ``denom_floor_frac``: floors the separation statistic's denominator toward the
      inter-quartile gap by this fraction, so a near-tied bulk cannot inflate the
      ratio.
    - ``winsor_z``: a standardized separation read beyond this many standard
      deviations is winsorized before it touches the baseline, so one extreme query
      cannot corrupt the streaming mean.
    - ``min_count_for_z``: minimum effective baseline count before a standardized
      separation read is trusted. Below it the channel leans on its own baseline.
    - ``shrink_pool_size``: pool size below which the channel's weight is shrunk
      toward its own running discrimination baseline, in proportion to how little
      data backs the read.
    - ``g_upper_bound``: upper bound on the discrimination weight ``g``, so no single
      channel can dominate the fused order.
    - ``g_floor``: small positive floor on ``g``, so an uncertain channel still
      contributes. Zeroing a channel on one noisy read is a recall risk.
    - ``g_slope``: slope of the logistic squash that maps each standardized statistic
      to a ``(0, 1)`` factor in ``g``. A larger slope makes the weight react more
      sharply to a departure from the channel's norm.
    - ``g_deviation_keep``: fraction of the per-query weight deviation from neutral
      kept after ``g`` is normalized by the channel's own running mean. The
      normalization removes the persistent level a channel's score-distribution
      shape leaks into its average ``g``; this factor then scales the remaining
      per-query bet. ``1.0`` uses the normalized deviation as is; ``0.0`` reduces
      the weighting to plain RRF. The default sits below ``1.0`` because the
      per-query signal's informativeness varies by corpus and scorer family (tuned
      on three-channel setups; a text-only deployment can justify up to ``1.0``, a
      channel-dominated multimodal one as low as ``0.5``).
    """

    top_eps: float = _D["top_eps"]
    top_m: int = _D["top_m"]
    min_distinct_values: int = _D["min_distinct_values"]
    denom_floor_frac: float = _D["denom_floor_frac"]
    winsor_z: float = _D["winsor_z"]
    min_count_for_z: float = _D["min_count_for_z"]
    shrink_pool_size: int = _D["shrink_pool_size"]
    g_upper_bound: float = _D["g_upper_bound"]
    g_floor: float = _D["g_floor"]
    g_slope: float = _D["g_slope"]
    g_deviation_keep: float = _D["g_deviation_keep"]


@dataclass(frozen=True)
class CouplingConfig:
    """Channel-coupling knobs: how the redundancy discount between channels is
    estimated and applied.

    Independence is the only unconditionally recall-safe setting, so coupling is off
    by default and every knob caps how far a discount can move weight.

    - ``enabled``: whether to apply any redundancy discount at all.
    - ``discount_cap``: caps the discount well below the raw anchor point estimate.
    - ``shrink_to_identity``: mandatory shrinkage intensity, in ``[0, 1]``, of the
      redundancy correlation toward the identity. Keeps the assembled covariance
      positive-definite and biases the discount toward treating channels as distinct.
    - ``min_overlap``: minimum number of anchor items scored by both channels before
      a pair correlation counts.
    - ``min_reliability``: minimum accumulated overlap count before any discount
      applies. Below it the discount is dropped entirely, the recall-safe direction.
    - ``min_refreshes``: minimum number of anchor refreshes backing a pair before any
      discount applies. Stability across query strata is a between-refresh property;
      a single refresh has zero between-refresh variance by construction.
    - ``stratum_stability_max_var``: maximum between-stratum variance of the anchor
      correlation that still allows a discount. A correlation that is unstable across
      query strata degrades to independence.
    """

    enabled: bool = _C["enabled"]
    discount_cap: float = _C["discount_cap"]
    shrink_to_identity: float = _C["shrink_to_identity"]
    min_overlap: int = _C["min_overlap"]
    min_reliability: float = _C["min_reliability"]
    min_refreshes: float = _C["min_refreshes"]
    stratum_stability_max_var: float = _C["stratum_stability_max_var"]


@dataclass(frozen=True)
class RrfConfig:
    """Weighted reciprocal-rank fusion knobs.

    - ``rrf_eta``: the RRF rank constant. Larger values flatten the rank
      contribution; ``60`` is the Cormack et al. (2009) value, calibrated on
      1000-deep TREC pools. The default ``20.0`` improved every evaluation
      collection measured with Recall@100 unchanged; the supported band is 10
      to 30, and the measurements cover pool depths from 20 to 1000 items
      with the optimum unmoved by depth. For pools deeper than 1000 items, or
      channel mixes very unlike the measured ones, ``60.0`` is the literature
      reference point.
    - ``min_g_dispersion``: the minimum within-query dispersion of the
      channels' level-normalized weights (a sample standard deviation) before
      the per-query weighting acts; ``0.0`` disables the gate. Below the
      threshold the channels' reads sit inside estimation noise of one
      another, so every adaptive weight becomes exactly ``1.0`` and, with
      coupling off and no base-weight tilt, the fusion is plain RRF. The
      default ``0.45`` is the conservative point of the supported 0.40 to
      0.50 band, tuned at two and three channels; a channel whose weight
      level baseline has not warmed yet contributes exact neutral, so a cold
      system fuses at the RRF floor and warms toward weighting.
    """

    rrf_eta: float = _F["rrf_eta"]
    min_g_dispersion: float = _F["min_g_dispersion"]


@dataclass(frozen=True)
class DecayConfig:
    """State-decay knobs: forgetting old observations to track corpus drift.

    Off by default. Decay ties a merge to an external clock, making the otherwise
    exact merge identity approximate. The cadence is per observation rather than per
    wall-clock interval: a channel's baselines decay once per fuse in which the
    channel appears, and a pair's redundancy decays once per anchor refresh. This
    bounds each baseline's effective sample size at ``1 / (1 - factor)``
    observations. A caller who wants wall-clock decay instead can call
    :meth:`ruffle.RuffleState.decay` on its own schedule with this setting left off.

    - ``enabled``: whether decay is applied at all.
    - ``factor``: per-decay-step multiplier on the effective count, in ``[0, 1]``.
      Preserves mean and variance while reducing confidence.
    """

    enabled: bool = _Y["enabled"]
    factor: float = _Y["factor"]


@dataclass(frozen=True)
class FuseConfig:
    """The complete fusion configuration: the grouped sub-configs plus the baseline
    mode.

    Constructed with keyword arguments over the engine's defaults::

        FuseConfig(coupling=CouplingConfig(enabled=True))

    Validation runs when a :class:`ruffle.Fuser` is built, so an out-of-range knob
    fails at construction with :class:`ruffle.ConfigError` rather than mid-query.

    ``dataclasses.asdict`` yields ``baseline_mode`` as the :class:`BaselineMode`
    member itself, which ``json.dumps`` refuses; a JSON-bound dump wants
    ``baseline_mode.value``. The configuration is not a persistence format; the
    persisted object is :class:`ruffle.RuffleState`.
    """

    discrimination: DiscriminationConfig = field(default_factory=DiscriminationConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    fusion: RrfConfig = field(default_factory=RrfConfig)
    decay: DecayConfig = field(default_factory=DecayConfig)
    baseline_mode: BaselineMode = BaselineMode.Z_SCORE

    def _to_dict(self) -> FuseConfigDict:
        """The configuration in the boundary schema the engine consumes."""
        d = self.discrimination
        c = self.coupling
        return {
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
                "g_deviation_keep": d.g_deviation_keep,
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
            "fusion": {
                "rrf_eta": self.fusion.rrf_eta,
                "min_g_dispersion": self.fusion.min_g_dispersion,
            },
            "decay": {"enabled": self.decay.enabled, "factor": self.decay.factor},
            "baseline_mode": self.baseline_mode.value,
        }
