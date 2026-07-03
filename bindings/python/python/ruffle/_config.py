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
      contribution; 60 is the common RRF default from Cormack et al. (2009).
    """

    rrf_eta: float = _F["rrf_eta"]


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
            "fusion": {"rrf_eta": self.fusion.rrf_eta},
            "decay": {"enabled": self.decay.enabled, "factor": self.decay.factor},
            "baseline_mode": self.baseline_mode.value,
        }
