"""Structural types for the boundary between the package and ``ruffle._core``.

These describe, exactly, the shapes that cross the extension boundary: channel
registrations and the fusion configuration go in as typed dictionaries, per-query
inputs go in as typed tuples, and fuse results and divergences come back as typed
dictionaries. The persisted-state schema mirrors the engine's canonical JSON
serialization. Everything here is private plumbing; the public API wraps these in
real classes.
"""

from __future__ import annotations

from typing import Literal, TypedDict

DirectionValue = Literal["higher_is_better", "lower_is_better"]
BaselineModeValue = Literal["z_score"]
FlagValue = Literal[
    "ranks_only_default_weighted",
    "degenerate_separation",
    "no_reference",
]

# --- registrations and configuration, as the engine consumes them -------------------


class GoodScoreDict(TypedDict):
    typical: float
    good: float
    weight: float


class ChannelDict(TypedDict):
    key: str
    tag: str
    direction: DirectionValue
    good_score: GoodScoreDict | None
    base_weight: float


class DiscriminationConfigDict(TypedDict):
    top_eps: float
    top_m: int
    min_distinct_values: int
    denom_floor_frac: float
    winsor_z: float
    min_count_for_z: float
    shrink_pool_size: int
    g_upper_bound: float
    g_floor: float
    g_slope: float
    g_deviation_keep: float


class CouplingConfigDict(TypedDict):
    enabled: bool
    discount_cap: float
    shrink_to_identity: float
    min_overlap: int
    min_reliability: float
    min_refreshes: float
    stratum_stability_max_var: float


class RrfConfigDict(TypedDict):
    rrf_eta: float


class DecayConfigDict(TypedDict):
    enabled: bool
    factor: float


class FuseConfigDict(TypedDict):
    discrimination: DiscriminationConfigDict
    coupling: CouplingConfigDict
    fusion: RrfConfigDict
    decay: DecayConfigDict
    baseline_mode: BaselineModeValue


# --- per-query inputs ------------------------------------------------------------------

ScoredSpec = tuple[Literal["scored"], str, DirectionValue, list[tuple[str, float]]]
RankedSpec = tuple[Literal["ranked"], str, list[str]]
InputSpec = ScoredSpec | RankedSpec

# --- fuse results and divergences, as the engine produces them ------------------------


class DiscriminationReadDict(TypedDict):
    g: float
    raw_separation: float | None
    top_m_average: float | None
    degenerate_separation: bool
    reference_cold: bool


class FusedDict(TypedDict):
    ranking: list[tuple[str, float]]
    weights: dict[str, float]
    flags: dict[str, FlagValue]
    discrimination: dict[str, DiscriminationReadDict]
    confidence: float
    conflict: float


class DivergenceDict(TypedDict):
    per_channel: dict[str, float]
    max: float


# --- the persisted-state schema (the engine's canonical JSON) --------------------------

PersistedDirection = Literal["HigherIsBetter", "LowerIsBetter"]
PersistedBaselineMode = Literal["ZScore"]


class MeanVarDict(TypedDict):
    count: float
    mean: float
    m2: float


class ChannelSummaryDict(TypedDict):
    separation: MeanVarDict
    reference: MeanVarDict
    tag: str


class PairSummaryDict(TypedDict):
    redundancy: MeanVarDict
    refreshes: float


class FingerprintDict(TypedDict):
    stat_version: int
    baseline_mode: PersistedBaselineMode
    directions: dict[str, PersistedDirection]


# A pair entry deserializes as a two-element array: the canonical (sorted) channel
# pair followed by its summary.
PairEntry = tuple[tuple[str, str], PairSummaryDict]


class StateDict(TypedDict):
    format_version: int
    fingerprint: FingerprintDict
    channels: dict[str, ChannelSummaryDict]
    pairs: list[PairEntry]
