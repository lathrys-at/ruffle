"""Channel identity, registration, per-query input, and the coupling anchor."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from ruffle._types import ChannelDict, DirectionValue, InputSpec

__all__ = [
    "Anchor",
    "ChannelConfig",
    "ChannelId",
    "ChannelInput",
    "Direction",
    "GoodScore",
]


class Direction(Enum):
    """Whether a higher native score means a better match, or a lower one.

    Declared once per channel at configuration; Ruffle does not infer it from data.
    Every score is oriented to higher-is-better at ingest. A channel registered with
    the wrong direction ranks anti-relevantly and corrupts its own persistent
    baseline.
    """

    HIGHER_IS_BETTER = "higher_is_better"
    """A higher native score is a better match (already canonical)."""

    LOWER_IS_BETTER = "lower_is_better"
    """A lower native score is a better match (negated at ingest)."""


@dataclass(frozen=True, kw_only=True)
class GoodScore:
    """An operator-declared reference for how good a channel's scores are in absolute
    terms, in the channel's native units (before orientation).

    The discrimination stage rewards a channel whose top results score well against
    this reference, complementing the separation of top from bulk. The declaration is
    two anchors and a pseudo-count:

    - ``typical``: the top score a typical, unremarkable query produces. Sets the
      reference location.
    - ``good``: the score a genuinely good match reaches. The gap from ``typical`` to
      ``good`` sets the reference scale.
    - ``weight``: a pseudo-count for how firmly the declaration holds before observed
      top scores refine it. Its influence after ``n`` observed top scores is
      ``weight / (weight + n)``.

    Both anchors are oriented with the scores at ingest, so for a
    ``Direction.LOWER_IS_BETTER`` channel a good match is a smaller native value, and
    ``typical`` and ``good`` are negated together with the scores. After orientation
    ``good`` must exceed ``typical``; a declaration that cannot orient is refused at
    ``Fuser`` construction with :class:`ruffle.ConfigError`.

    The fields are keyword-only: all three are floats, so a positional call could
    transpose them silently.
    """

    typical: float
    good: float
    weight: float


@dataclass(frozen=True)
class ChannelId:
    """A channel's identity: a stable join handle (``key``) plus a model-version
    ``tag``.

    The two fields serve different roles:

    - ``key`` is the stable join handle. Every persistent map is keyed by it alone, so
      accumulation across time and deployments lands on the right channel. It stays
      fixed across model versions. A changed key mislabels statistics, recoverable by
      rekeying or a cold start.
    - ``tag`` is the model version (for example ``"clip-vit-b32-rev1"``), changed
      whenever the model behind the channel changes. Ruffle never interprets it; it
      only checks it for equality on every merge, so a model swapped in under a kept
      key is refused rather than silently blended. An unnecessary tag change costs a
      cold start; a missed one corrupts the baseline.
    """

    key: str
    tag: str

    def __str__(self) -> str:
        return f"{self.key}@{self.tag}"


@dataclass(frozen=True)
class ChannelConfig:
    """Per-channel registration.

    ``id`` (the join handle ``key`` and the model-version ``tag``) and ``direction``
    are declared once at channel configuration rather than per query. ``good_score``
    is the optional declared reference for the absolute-goodness statistic; when
    absent, the reference is learned from early traffic and the absolute-goodness
    statistic cold-starts.
    """

    id: ChannelId
    direction: Direction
    good_score: GoodScore | None = None


class ChannelInput:
    """One channel's input for one query: the channel's key plus its surfaced items.

    An input is either scored or rank-only, a stable property of how the channel is
    wired rather than something inferred per query. Instances come from
    :meth:`scored` or :meth:`ranked`; there is no public constructor. Two inputs
    compare equal when they carry the same channel, kind, and items.
    """

    __slots__ = ("_key", "_spec")

    _key: str
    _spec: InputSpec

    def __init__(self) -> None:
        raise TypeError(
            "ChannelInput has no public constructor; an input comes from "
            "ChannelInput.scored or ChannelInput.ranked"
        )

    @classmethod
    def _from_spec(cls, key: str, spec: InputSpec) -> ChannelInput:
        value = object.__new__(cls)
        value._key = key
        value._spec = spec
        return value

    @property
    def key(self) -> str:
        """The channel this input belongs to, named by its join-handle key."""
        return self._key

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChannelInput):
            return NotImplemented
        return self._key == other._key and self._spec == other._spec

    # The scored payload holds lists, so an input is equality-bearing but unhashable,
    # like the lists themselves.
    __hash__ = None  # type: ignore[assignment]

    @classmethod
    def scored(cls, config: ChannelConfig, items: Iterable[tuple[str, float]]) -> ChannelInput:
        """Builds a scored input from ``(id, native_score)`` pairs.

        The channel's key and direction are taken from ``config``. Scores are in the
        channel's native units; orientation to higher-is-better and the dropping of
        non-finite values happen inside the engine at fuse time, so a stray NaN never
        reaches a baseline. The item order carries no meaning: the engine ranks each
        scored channel by its own oriented scores.

        Each channel lists each item at most once. A repeated id within one input is
        counted twice by the fusion, so the ids in one input must be distinct.
        """
        payload = [(str(item_id), float(score)) for item_id, score in items]
        return cls._from_spec(
            config.id.key,
            ("scored", config.id.key, config.direction.value, payload),
        )

    @classmethod
    def ranked(cls, config: ChannelConfig, ids: Iterable[str]) -> ChannelInput:
        """Builds a rank-only input for a channel that produces no scores.

        The order is used as given, best first; a rank carries no magnitude, so there
        is nothing to orient or filter. A rank-only channel contributes no
        discrimination estimate and is carried at the neutral default weight, flagged
        as :attr:`ruffle.ChannelFlag.RANKS_ONLY_DEFAULT_WEIGHTED` in the result.

        Each channel lists each item at most once. A repeated id within one input is
        counted twice by the fusion, so the ids in one input must be distinct.
        """
        return cls._from_spec(config.id.key, ("ranked", config.id.key, [str(i) for i in ids]))

    def _to_spec(self) -> InputSpec:
        return self._spec

    def __repr__(self) -> str:
        kind = self._spec[0]
        count = len(self._spec[-1])
        return f"ChannelInput({self._key!r}, {kind}, {count} items)"


class Anchor:
    """A shared evaluation set in which every candidate is scored by every channel,
    used to estimate how redundant the channels are with each other.

    Because every candidate is scored by every channel, a ``None`` entry unambiguously
    means the channel's facet does not apply to that item, rather than that the item
    was ranked below a cutoff and dropped. The candidate set must be an unselected
    sample (a random or whole-corpus draw) rather than any channel's top-k results;
    restricting the candidates to a top-k pool conditions on a selection effect
    (Berkson's paradox) that pushes the channels spuriously anti-correlated and
    destroys the redundancy estimate. Whether a candidate set is unselected cannot be
    checked from the ids alone, so this contract rests with the caller.

    Instances come from :meth:`build`; there is no public constructor. The anchor is
    fed to :meth:`ruffle.Fuser.refresh_coupling`.
    """

    __slots__ = ("_channels", "_rows")

    _channels: list[tuple[str, DirectionValue]]
    _rows: list[list[float | None]]

    def __init__(self) -> None:
        raise TypeError("Anchor has no public constructor; an anchor comes from Anchor.build")

    @classmethod
    def _from_rows(
        cls,
        channels: list[tuple[str, DirectionValue]],
        rows: list[list[float | None]],
    ) -> Anchor:
        anchor = object.__new__(cls)
        anchor._channels = channels
        anchor._rows = rows
        return anchor

    @classmethod
    def build(
        cls,
        candidates: Sequence[str],
        channels: Sequence[ChannelConfig],
        score: Callable[[str, str], float | None],
    ) -> Anchor:
        """Builds an anchor by scoring every ``(candidate, channel)`` pair.

        ``score(candidate_id, channel_key)`` is called once for each pair. A float
        return is the channel's native score for that candidate, oriented to
        higher-is-better by the channel's declared direction inside the engine; a
        non-finite value is treated as absent. A ``None`` return means the channel's
        facet does not apply to that candidate. Coverage is structural: because the
        callable runs for every pair, the anchor is always full-scored, and an absent
        score is never a hidden top-k cutoff.
        """
        rows: list[list[float | None]] = []
        for config in channels:
            key = config.id.key
            row: list[float | None] = []
            for candidate in candidates:
                value = score(candidate, key)
                row.append(None if value is None else float(value))
            rows.append(row)
        return cls._from_rows([(c.id.key, c.direction.value) for c in channels], rows)

    def __repr__(self) -> str:
        keys = [k for k, _ in self._channels]
        count = len(self._rows[0]) if self._rows else 0
        return f"Anchor(channels={keys!r}, candidates={count})"


def _registrations(channels: Sequence[ChannelConfig]) -> list[ChannelDict]:
    """The registrations in the boundary schema the engine consumes."""
    return [
        {
            "key": c.id.key,
            "tag": c.id.tag,
            "direction": c.direction.value,
            "good_score": None
            if c.good_score is None
            else {
                "typical": c.good_score.typical,
                "good": c.good_score.good,
                "weight": c.good_score.weight,
            },
        }
        for c in channels
    ]
