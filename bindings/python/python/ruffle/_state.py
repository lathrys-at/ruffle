"""Persistent state: the single mergeable object, plus its read-only views.

Everything Ruffle persists is a confidence-weighted summary plus the identifiers
needed to merge it safely. The canonical representation is the engine's JSON
serialization: maps are stored ordered, so two states with identical contents
serialize byte-for-byte identically, making a serialized state content-addressable
and its diffs clean. A :class:`RuffleState` holds those canonical bytes and delegates
every operation to the engine.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import cast

from ruffle import _core
from ruffle._channels import Direction
from ruffle._config import BaselineMode
from ruffle._types import (
    DivergenceDict,
    MeanVarDict,
    PersistedBaselineMode,
    PersistedDirection,
    StateDict,
)

__all__ = [
    "ChannelSummary",
    "Divergence",
    "MeanVar",
    "MergePolicy",
    "PairSummary",
    "RuffleState",
    "StatFingerprint",
]


class MergePolicy(Enum):
    """How :meth:`RuffleState.merge` treats incompatible inputs."""

    STRICT = "strict"
    """Refuses on any format, fingerprint, or tag mismatch. The only policy for now."""


@dataclass(frozen=True)
class MeanVar:
    """Confidence-weighted streaming mean and variance, as persisted by the engine.

    ``count`` is the effective observation count, fractional to support
    pseudo-counts and decay; ``mean`` is the running mean; ``m2`` is the sum of
    squared deviations from the mean, so the population variance is ``m2 / count``.
    """

    count: float
    mean: float
    m2: float

    @property
    def variance(self) -> float:
        """The population variance ``m2 / count``, zero for an empty summary and
        clamped so rounding never yields a negative value."""
        if self.count <= 0.0:
            return 0.0
        return max(self.m2 / self.count, 0.0)

    @property
    def std(self) -> float:
        """The population standard deviation."""
        return math.sqrt(self.variance)

    @classmethod
    def _from_dict(cls, d: MeanVarDict) -> MeanVar:
        return cls(count=d["count"], mean=d["mean"], m2=d["m2"])


@dataclass(frozen=True)
class ChannelSummary:
    """The persistent statistics for one channel: the separation baseline, the
    good-score reference, and the model-version tag that gates merging."""

    separation: MeanVar
    reference: MeanVar
    tag: str


@dataclass(frozen=True)
class PairSummary:
    """The persistent statistics for one pair of channels: the accumulated redundancy
    correlation plus how many anchor refreshes back it."""

    redundancy: MeanVar
    refreshes: float


@dataclass(frozen=True)
class StatFingerprint:
    """A fingerprint answering whether two states were measuring the same thing the
    same way: the statistic-definition version, the baseline mode, and the
    per-channel orientation in force when the state was built."""

    stat_version: int
    baseline_mode: BaselineMode
    directions: Mapping[str, Direction]


@dataclass(frozen=True)
class Divergence:
    """An advisory standardized distance between two states' per-channel summaries.

    The number never gates a merge; gating is done by the model-version tag. It flags
    a silent model swap, where two summaries have drifted far apart while their tags
    still match, so a caller can catch it at the reconcile boundary. ``max`` is the
    largest per-channel distance, the single number a caller can threshold on.
    """

    per_channel: Mapping[str, float]
    max: float

    @classmethod
    def _from_dict(cls, d: DivergenceDict) -> Divergence:
        return cls(per_channel=MappingProxyType(dict(d["per_channel"])), max=d["max"])


_DIRECTIONS: dict[PersistedDirection, Direction] = {
    "HigherIsBetter": Direction.HIGHER_IS_BETTER,
    "LowerIsBetter": Direction.LOWER_IS_BETTER,
}
_BASELINE_MODES: dict[PersistedBaselineMode, BaselineMode] = {"ZScore": BaselineMode.Z_SCORE}


class RuffleState:
    """The persistent statistics Ruffle accumulates: a confidence-weighted summary
    per channel and per channel pair, plus the versioning needed to merge two of them
    safely.

    A state comes from :attr:`ruffle.Fuser.state`, from :meth:`from_json`, or from
    :meth:`merge`; there is no public constructor, since an empty state is created by
    building a :class:`ruffle.Fuser`. Its single merge operation serves three roles:
    streaming update as new queries arrive, operator prior seeded before any traffic,
    and cross-deployment reconciliation of states accumulated on separate machines.
    Every merge is gated on a required per-channel model-version tag, so a model
    swapped in under a kept channel name is refused rather than silently blended.

    A state written by this binding loads, merges, and resumes byte-for-byte under
    the Rust crate and its command-line tool, and vice versa; the serialization is
    the same canonical JSON everywhere.
    """

    __slots__ = ("_json",)

    _json: str

    def __init__(self) -> None:
        raise TypeError(
            "RuffleState has no public constructor; a state comes from Fuser.state, "
            "RuffleState.from_json, or RuffleState.merge"
        )

    @classmethod
    def _from_canonical(cls, canonical: str) -> RuffleState:
        state = object.__new__(cls)
        state._json = canonical
        return state

    @classmethod
    def from_json(cls, data: str) -> RuffleState:
        """Loads a state from its JSON serialization, validating it in the process.

        The input is re-serialized canonically, so :meth:`to_json` on the result
        yields the engine's canonical bytes even when the input was formatted
        differently.

        Raises:
            ValueError: the input is not a well-formed serialized state.
        """
        return cls._from_canonical(_core.state_canonicalize(data))

    def to_json(self) -> str:
        """The canonical JSON serialization.

        Byte-identical for equal contents, so the output is content-addressable and
        safe to compare, hash, and diff.
        """
        return self._json

    @staticmethod
    def merge(
        parts: Sequence[RuffleState],
        policy: MergePolicy = MergePolicy.STRICT,
    ) -> tuple[RuffleState, Divergence]:
        """Combines several states into one, returning the merged state and an
        advisory divergence between the inputs.

        The merge is associative and commutative and, with decay off, exact up to
        f64 rounding. Under :attr:`MergePolicy.STRICT` it refuses on the first
        incompatibility: a foreign format or statistic version, a channel present in
        more than one part with a conflicting orientation, or a channel present in
        more than one part with a different model-version tag (the signature of a
        model swap).

        Raises:
            ruffle.MergeError: the parts are incompatible, or ``parts`` is empty.
        """
        if not isinstance(policy, MergePolicy):
            raise TypeError(f"policy must be a MergePolicy, not {type(policy).__name__}")
        merged, divergence = _core.state_merge([p._json for p in parts])
        return RuffleState._from_canonical(merged), Divergence._from_dict(divergence)

    def divergence(self, other: RuffleState) -> Divergence:
        """The advisory divergence between this state and another, callable on its
        own before any merge.

        For every channel present in both states it reports a standardized distance
        over each of the channel's two baselines and keeps the larger. The good-score
        reference lives in the channel's native units, so it is the baseline that
        jumps under a silent model swap; the separation statistic is deliberately
        scale- and shift-invariant, so a swap that rescales scores can leave it
        untouched.
        """
        return Divergence._from_dict(_core.state_divergence(self._json, other._json))

    def rekey(self, from_key: str, to_key: str) -> None:
        """Renames a channel's key, moving all of its statistics with it: the channel
        summary, every pair summary that referenced the old key, and the channel's
        orientation in the fingerprint.

        When the destination already exists, the moved data and the existing data are
        merged, and the destination keeps its own model-version tag and orientation:
        the caller is asserting that the old key's history belongs to the channel
        already living under the new one. A no-op rename leaves the state unchanged.
        Unlike :meth:`merge`, rekey runs no tag gate; it is a deliberate rename and
        cannot fail.
        """
        self._json = _core.state_rekey(self._json, from_key, to_key)

    def decay(self, factor: float) -> None:
        """Scales the confidence of every persisted summary down by ``factor``,
        shrinking effective counts while leaving means and variances unchanged.

        ``factor`` is clamped to ``[0, 1]``. Decay is the one operation that breaks
        the exactness of :meth:`merge`: decaying then merging no longer gives the
        same result as merging then decaying.
        """
        self._json = _core.state_decay(self._json, factor)

    # --- read-only views -------------------------------------------------------------

    @property
    def format_version(self) -> int:
        """The schema version this state was built or loaded at."""
        return self._parsed()["format_version"]

    @property
    def fingerprint(self) -> StatFingerprint:
        """The statistic fingerprint: which statistic definitions, baseline mode, and
        per-channel orientations the state was measured under."""
        raw = self._parsed()["fingerprint"]
        return StatFingerprint(
            stat_version=raw["stat_version"],
            baseline_mode=_BASELINE_MODES[raw["baseline_mode"]],
            directions=MappingProxyType(
                {k: _DIRECTIONS[v] for k, v in raw["directions"].items()}
            ),
        )

    @property
    def channels(self) -> Mapping[str, ChannelSummary]:
        """The per-channel summaries, keyed by join handle. A snapshot: later
        mutations of the state are not reflected in a previously returned mapping."""
        return MappingProxyType(
            {
                key: ChannelSummary(
                    separation=MeanVar._from_dict(raw["separation"]),
                    reference=MeanVar._from_dict(raw["reference"]),
                    tag=raw["tag"],
                )
                for key, raw in self._parsed()["channels"].items()
            }
        )

    @property
    def pairs(self) -> Mapping[tuple[str, str], PairSummary]:
        """The per-pair coupling summaries, keyed by the canonical (sorted) channel
        pair. A snapshot, like :attr:`channels`."""
        return MappingProxyType(
            {
                (pair[0], pair[1]): PairSummary(
                    redundancy=MeanVar._from_dict(raw["redundancy"]),
                    refreshes=raw["refreshes"],
                )
                for pair, raw in self._parsed()["pairs"]
            }
        )

    def _parsed(self) -> StateDict:
        # The bytes come from the engine's canonical serializer (every constructor
        # funnels through it), so the shape assertion holds by construction.
        return cast(StateDict, json.loads(self._json))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuffleState):
            return NotImplemented
        return self._json == other._json

    # Mutable (rekey and decay update the state in place), so not hashable.
    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        parsed = self._parsed()
        return (
            f"RuffleState(format_version={parsed['format_version']}, "
            f"channels={sorted(parsed['channels'])!r}, pairs={len(parsed['pairs'])})"
        )
