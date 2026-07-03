"""The entry point: fuses several retrieval channels' ranked outputs into one ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from ruffle import _core
from ruffle._channels import Anchor, ChannelConfig, ChannelInput, _registrations
from ruffle._config import FuseConfig
from ruffle._state import RuffleState
from ruffle._types import FusedDict

__all__ = [
    "ChannelDiscrimination",
    "ChannelFlag",
    "Fused",
    "Fuser",
]


class ChannelFlag(Enum):
    """Why a channel was not weighted by its full discrimination score."""

    RANKS_ONLY_DEFAULT_WEIGHTED = "ranks_only_default_weighted"
    """The channel supplied ranks only, with no scores to compute a discrimination
    statistic from, so it was carried at the neutral default weight."""

    DEGENERATE_SEPARATION = "degenerate_separation"
    """The score pool's bulk had no usable scale to measure the top's elevation
    against, so the separation read was floored rather than trusted."""

    NO_REFERENCE = "no_reference"
    """The channel had no usable good-score reference yet, so its absolute-goodness
    term could not be computed this query and it was weighted on separation alone."""


@dataclass(frozen=True)
class ChannelDiscrimination:
    """One channel's discrimination read for one query: the combined weight and the
    raw statistics behind it.

    ``g`` is the channel's combined discrimination weight, bounded, and near ``1.0``
    when the channel performs at its own norm. ``raw_separation`` is the top-vs-bulk
    separation statistic, ``None`` when the score pool is too degenerate to measure
    it (rank-only, empty, or a collapsed bulk scale). ``top_m_average`` is the
    fixed-count top-m average exported for good-score reference refinement, ``None``
    when the pool is rank-only, empty, or shallower than the fixed count.
    ``degenerate_separation`` and ``reference_cold`` mirror the conditions behind
    :class:`ChannelFlag`.
    """

    g: float
    raw_separation: float | None
    top_m_average: float | None
    degenerate_separation: bool
    reference_cold: bool


@dataclass(frozen=True)
class Fused:
    """The outcome of fusing one query: the merged ranking plus the weights, flags,
    and diagnostics behind it.

    ``ranking`` is the fused order, best first, each id with its fused score.
    ``weights`` holds the per-channel weights actually used. ``flags`` explains any
    non-standard weighting; a channel absent from it was weighted on its full
    discrimination score. ``discrimination`` holds the per-channel reads behind the
    weights, so the reasoning is readable from the result alone. ``confidence`` is
    the top-set agreement of the discriminating channels, in ``[0, 1]``;
    ``conflict`` is its complement, high when confident channels disagree on which
    items are relevant.
    """

    ranking: tuple[tuple[str, float], ...]
    weights: Mapping[str, float]
    flags: Mapping[str, ChannelFlag]
    discrimination: Mapping[str, ChannelDiscrimination]
    confidence: float
    conflict: float

    @classmethod
    def _from_core(cls, raw: FusedDict) -> Fused:
        return cls(
            ranking=tuple((item_id, score) for item_id, score in raw["ranking"]),
            weights=MappingProxyType(dict(raw["weights"])),
            flags=MappingProxyType(
                {key: ChannelFlag(value) for key, value in raw["flags"].items()}
            ),
            discrimination=MappingProxyType(
                {
                    key: ChannelDiscrimination(
                        g=d["g"],
                        raw_separation=d["raw_separation"],
                        top_m_average=d["top_m_average"],
                        degenerate_separation=d["degenerate_separation"],
                        reference_cold=d["reference_cold"],
                    )
                    for key, d in raw["discrimination"].items()
                }
            ),
            confidence=raw["confidence"],
            conflict=raw["conflict"],
        )


class Fuser:
    """The entry point: fuses several retrieval channels' ranked outputs into one
    ranking.

    A fuser holds the channel registrations, the fusion configuration, and the
    persistent baselines it accumulates across queries. Each :meth:`fuse` call
    weights the channels by how well each is discriminating on this query and how
    redundant the channels are with each other, then combines them by weighted
    reciprocal-rank fusion. A fuser is built fresh from registrations or resumed
    from saved state with :meth:`resume`; :attr:`state` exposes the baselines to
    persist.

    :meth:`fuse` releases the GIL while the engine runs, so a multi-threaded host
    can fuse on worker threads; a single fuser is not itself thread-safe, since each
    stateful fuse updates the baselines.
    """

    __slots__ = ("_channels", "_config", "_inner")

    _channels: tuple[ChannelConfig, ...]
    _config: FuseConfig
    _inner: _core.Fuser

    def __init__(
        self,
        channels: Sequence[ChannelConfig],
        config: FuseConfig | None = None,
    ) -> None:
        """Builds a fresh fuser from channel registrations and a configuration, with
        empty starting baselines.

        Raises:
            ruffle.ConfigError: the configuration holds an out-of-range knob, two
                registrations share one join-handle key, or a declared
                :class:`ruffle.GoodScore` does not orient to a usable reference.
        """
        cfg = FuseConfig() if config is None else config
        self._inner = _core.Fuser(_registrations(channels), cfg._to_dict())
        self._channels = tuple(channels)
        self._config = cfg

    @classmethod
    def resume(
        cls,
        channels: Sequence[ChannelConfig],
        state: RuffleState,
        config: FuseConfig | None = None,
    ) -> Fuser:
        """Builds a fuser from channel registrations, a previously persisted state,
        and a configuration, continuing to accumulate from that state.

        Resume is the live boundary a real model change crosses (a swap happens
        across a restart), so it runs the same compatibility gate a state merge
        does before accepting the state. Without the gate, a model swapped in behind
        a bumped tag would silently keep accumulating into the old model's
        baselines, which is the corruption the tag exists to prevent.

        Raises:
            ruffle.ConfigError: the registrations or configuration are invalid on
                their own.
            ruffle.ResumeError: the state is incompatible with the registrations or
                this build: a foreign format or statistic version, a flipped
                direction, or a changed model-version tag.
        """
        cfg = FuseConfig() if config is None else config
        fuser = object.__new__(cls)
        fuser._inner = _core.Fuser.resume(
            _registrations(channels), cfg._to_dict(), state.to_json()
        )
        fuser._channels = tuple(channels)
        fuser._config = cfg
        return fuser

    def fuse(self, inputs: Sequence[ChannelInput]) -> Fused:
        """Fuses one query's per-channel results into a single ranking, and folds
        this query's readings into the running baselines.

        An input whose key is not a registered channel is skipped entirely: without
        a registration the engine has no direction, tag, or reference to interpret
        the channel safely, so it is ignored rather than fused at a guessed weight.
        When one channel key appears more than once, only the first input is fused;
        a later duplicate would double-count the channel's vote under a single
        weight.
        """
        return Fused._from_core(self._inner.fuse([i._to_spec() for i in inputs]))

    @staticmethod
    def fuse_stateless(
        inputs: Sequence[ChannelInput],
        channels: Sequence[ChannelConfig],
        prior: RuffleState,
        config: FuseConfig | None = None,
    ) -> Fused:
        """Fuses one query against the given registrations and a prior state,
        without mutating any baseline.

        This runs the same weighting and fusion as :meth:`fuse` but updates nothing.
        With an empty prior and no declared references, every weight lands at the
        neutral ``1.0`` and the fusion reduces to standard, unweighted
        reciprocal-rank fusion.

        Raises:
            ruffle.ConfigError: the registrations or configuration are invalid.
            ruffle.ResumeError: the prior is incompatible with the registrations,
                which would standardize this query against baselines measured under
                a different model or orientation.
        """
        cfg = FuseConfig() if config is None else config
        raw = _core.Fuser.fuse_stateless(
            [i._to_spec() for i in inputs],
            _registrations(channels),
            cfg._to_dict(),
            prior.to_json(),
        )
        return Fused._from_core(raw)

    def refresh_coupling(self, anchor: Anchor) -> None:
        """Folds a full-scored anchor's pairwise correlations into the persistent
        redundancy baselines.

        Each pair's correlation is accumulated into its persistent summary, giving
        the redundancy estimate its reliability (total both-scored overlap), its
        point estimate (the overlap-weighted pooled correlation), and its stability
        signal (the variability across refreshes and strata). Anchor construction is
        an offline concern; the redundancy discount itself stays off unless
        :attr:`ruffle.CouplingConfig.enabled` is set.
        """
        self._inner.refresh_coupling(anchor._channels, anchor._rows)

    @property
    def state(self) -> RuffleState:
        """A snapshot of the persistent baseline state, for serialization and
        inspection.

        Each access returns an independent snapshot; later fuses do not mutate a
        previously returned state. The snapshot is restored through :meth:`resume`.
        """
        return RuffleState._from_canonical(self._inner.state_json())

    @property
    def config(self) -> FuseConfig:
        """The fusion configuration in force."""
        return self._config

    @property
    def channels(self) -> tuple[ChannelConfig, ...]:
        """The channel registrations this fuser was built with."""
        return self._channels

    def __repr__(self) -> str:
        keys = [c.id.key for c in self._channels]
        return f"Fuser(channels={keys!r})"
