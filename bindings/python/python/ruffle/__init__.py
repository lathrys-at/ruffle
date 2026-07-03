"""Weighted, adaptive, calibration-free reciprocal-rank fusion.

Ruffle fuses the output of several retrieval channels into one ranking, without
per-channel score calibration and without comparing one channel's raw scores against
another's. It requires no relevance labels and natively handles channels whose scores
live on different scales. For each query it estimates two properties from the
channels' own outputs: per-channel discrimination (how far a channel's top results
stand above its bulk, and how good they are against a declared, evidence-refined
good-score reference) and pairwise redundancy (a correlation measured on a shared
full-scored anchor, away from the live pool's selection bias). The estimates weight a
rank-based RRF. Every estimate is conservative: with the default configuration Ruffle
stays close to plain RRF and tilts weights only when the channels' own outputs
support it.

The package wraps the Rust crate ``ruffle``; the compiled engine does all the
statistics, so behaviour and persisted state are identical across the two, down to
the serialized bytes. The everyday entry point is :class:`Fuser`::

    from ruffle import ChannelConfig, ChannelId, ChannelInput, Direction, Fuser

    semantic = ChannelConfig(ChannelId("semantic", "text-embedding-v1"),
                             Direction.HIGHER_IS_BETTER)
    fuser = Fuser([semantic])
    fused = fuser.fuse([ChannelInput.scored(semantic, [("doc-1", 0.91)])])

The design document and tuning guide live in the repository under ``docs/``:
https://github.com/lathrys-at/ruffle
"""

from ruffle._channels import (
    Anchor,
    ChannelConfig,
    ChannelId,
    ChannelInput,
    Direction,
    GoodScore,
)
from ruffle._config import (
    BaselineMode,
    CouplingConfig,
    DecayConfig,
    DiscriminationConfig,
    FuseConfig,
    RrfConfig,
)
from ruffle._core import (
    FORMAT_VERSION,
    STAT_VERSION,
    ConfigError,
    MergeError,
    ResumeError,
    RuffleError,
    __version__,
)
from ruffle._fuser import ChannelDiscrimination, ChannelFlag, Fused, Fuser
from ruffle._state import (
    ChannelSummary,
    Divergence,
    MeanVar,
    MergePolicy,
    PairSummary,
    RuffleState,
    StatFingerprint,
)

__all__ = [
    "FORMAT_VERSION",
    "STAT_VERSION",
    "Anchor",
    "BaselineMode",
    "ChannelConfig",
    "ChannelDiscrimination",
    "ChannelFlag",
    "ChannelId",
    "ChannelInput",
    "ChannelSummary",
    "ConfigError",
    "CouplingConfig",
    "DecayConfig",
    "Direction",
    "DiscriminationConfig",
    "Divergence",
    "FuseConfig",
    "Fused",
    "Fuser",
    "GoodScore",
    "MeanVar",
    "MergeError",
    "MergePolicy",
    "PairSummary",
    "ResumeError",
    "RrfConfig",
    "RuffleError",
    "RuffleState",
    "StatFingerprint",
    "__version__",
]
