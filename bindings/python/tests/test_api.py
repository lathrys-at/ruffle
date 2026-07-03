"""Idiomatic tests for the Python-only surface: construction, immutability,
exceptions, defaults, and the read-only state views."""

from __future__ import annotations

import dataclasses
import importlib.metadata

import pytest
import ruffle
from ruffle import (
    Anchor,
    ChannelConfig,
    ChannelFlag,
    ChannelId,
    ChannelInput,
    CouplingConfig,
    Direction,
    DiscriminationConfig,
    FuseConfig,
    Fuser,
    GoodScore,
    MergePolicy,
    RuffleState,
)


def spiked_pool(n: int = 30) -> list[tuple[str, float]]:
    pool = [(f"doc{i:03}", 0.01 * i) for i in range(n)]
    pool.append(("hit0", 10.0))
    pool.append(("hit1", 10.5))
    return pool


class TestVersioning:
    def test_version_lockstep_with_distribution(self) -> None:
        assert ruffle.__version__ == importlib.metadata.version("ruffle")

    def test_format_and_stat_versions_are_exposed(self) -> None:
        assert isinstance(ruffle.FORMAT_VERSION, int)
        assert isinstance(ruffle.STAT_VERSION, int)


class TestConfig:
    def test_defaults_come_from_the_engine(self) -> None:
        from ruffle import _core

        assert FuseConfig()._to_dict() == _core.default_config()

    def test_kwargs_over_defaults(self) -> None:
        cfg = FuseConfig(coupling=CouplingConfig(enabled=True))
        assert cfg.coupling.enabled
        assert cfg.coupling.discount_cap == CouplingConfig().discount_cap
        assert not FuseConfig().coupling.enabled

    def test_configs_are_frozen(self) -> None:
        cfg = FuseConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.baseline_mode = None  # type: ignore[misc, assignment]

    def test_invalid_knob_raises_config_error_naming_the_field(self) -> None:
        bad = FuseConfig(discrimination=DiscriminationConfig(g_floor=5.0, g_upper_bound=4.0))
        semantic = ChannelConfig(ChannelId("s", "v1"), Direction.HIGHER_IS_BETTER)
        with pytest.raises(ruffle.ConfigError, match="g_upper_bound"):
            Fuser([semantic], bad)

    def test_invalid_good_score_raises_at_construction(self) -> None:
        bad = ChannelConfig(
            ChannelId("s", "v1"),
            Direction.HIGHER_IS_BETTER,
            GoodScore(typical=0.5, good=0.3, weight=4.0),
        )
        with pytest.raises(ruffle.ConfigError, match="good score is unusable"):
            Fuser([bad])


class TestExceptions:
    def test_hierarchy(self) -> None:
        assert issubclass(ruffle.ConfigError, ruffle.RuffleError)
        assert issubclass(ruffle.ResumeError, ruffle.RuffleError)
        assert issubclass(ruffle.MergeError, ruffle.RuffleError)

    def test_empty_merge_refuses(self) -> None:
        with pytest.raises(ruffle.MergeError, match="empty set of states"):
            RuffleState.merge([])

    def test_merge_policy_type_is_checked(self) -> None:
        with pytest.raises(TypeError, match="MergePolicy"):
            RuffleState.merge([], policy="strict")  # type: ignore[arg-type]


class TestFuse:
    def test_lower_is_better_orients_at_ingest(self) -> None:
        lexical = ChannelConfig(ChannelId("lex", "v1"), Direction.LOWER_IS_BETTER)
        fuser = Fuser([lexical])
        fused = fuser.fuse([ChannelInput.scored(lexical, [("worse", -1.0), ("best", -9.0)])])
        assert [item for item, _ in fused.ranking] == ["best", "worse"]

    def test_non_finite_scores_are_dropped(self) -> None:
        semantic = ChannelConfig(ChannelId("s", "v1"), Direction.HIGHER_IS_BETTER)
        fuser = Fuser([semantic])
        fused = fuser.fuse(
            [
                ChannelInput.scored(
                    semantic, [("ok", 0.5), ("nan", float("nan")), ("inf", float("inf"))]
                )
            ]
        )
        assert [item for item, _ in fused.ranking] == ["ok"]

    def test_unregistered_input_is_skipped(self) -> None:
        semantic = ChannelConfig(ChannelId("s", "v1"), Direction.HIGHER_IS_BETTER)
        rogue = ChannelConfig(ChannelId("rogue", "v1"), Direction.HIGHER_IS_BETTER)
        fuser = Fuser([semantic])
        fused = fuser.fuse(
            [
                ChannelInput.scored(semantic, [("a", 0.9)]),
                ChannelInput.scored(rogue, [("z", 99.0)]),
            ]
        )
        assert [item for item, _ in fused.ranking] == ["a"]
        assert "rogue" not in fused.weights

    def test_ranks_only_channel_is_flagged(self) -> None:
        recency = ChannelConfig(ChannelId("r", "v1"), Direction.HIGHER_IS_BETTER)
        fuser = Fuser([recency])
        fused = fuser.fuse([ChannelInput.ranked(recency, ["a", "b"])])
        assert fused.flags["r"] is ChannelFlag.RANKS_ONLY_DEFAULT_WEIGHTED
        assert fused.weights["r"] == 1.0

    def test_result_mappings_are_read_only(self) -> None:
        semantic = ChannelConfig(ChannelId("s", "v1"), Direction.HIGHER_IS_BETTER)
        fuser = Fuser([semantic])
        fused = fuser.fuse([ChannelInput.scored(semantic, spiked_pool())])
        with pytest.raises(TypeError):
            fused.weights["s"] = 2.0  # type: ignore[index]
        with pytest.raises(dataclasses.FrozenInstanceError):
            fused.confidence = 1.0  # type: ignore[misc]

    def test_stateless_with_empty_prior_is_unweighted(self) -> None:
        a = ChannelConfig(ChannelId("a", "v1"), Direction.HIGHER_IS_BETTER)
        b = ChannelConfig(ChannelId("b", "v1"), Direction.HIGHER_IS_BETTER)
        prior = Fuser([a, b]).state
        fused = Fuser.fuse_stateless(
            [
                ChannelInput.scored(a, spiked_pool()),
                ChannelInput.scored(b, spiked_pool()),
            ],
            [a, b],
            prior,
        )
        assert all(w == 1.0 for w in fused.weights.values())


class TestState:
    def make_state(self) -> RuffleState:
        semantic = ChannelConfig(ChannelId("s", "v1"), Direction.HIGHER_IS_BETTER)
        fuser = Fuser([semantic])
        fuser.fuse([ChannelInput.scored(semantic, spiked_pool())])
        return fuser.state

    def test_from_json_canonicalizes_formatting(self) -> None:
        state = self.make_state()
        pretty = state.to_json().replace(",", ", ")
        assert RuffleState.from_json(pretty).to_json() == state.to_json()

    def test_from_json_rejects_garbage(self) -> None:
        with pytest.raises(ValueError, match="invalid ruffle state"):
            RuffleState.from_json("{not json")

    def test_no_public_constructor(self) -> None:
        with pytest.raises(TypeError, match="no public constructor"):
            RuffleState()

    def test_state_snapshots_are_independent(self) -> None:
        semantic = ChannelConfig(ChannelId("s", "v1"), Direction.HIGHER_IS_BETTER)
        fuser = Fuser([semantic])
        before = fuser.state
        fuser.fuse([ChannelInput.scored(semantic, spiked_pool())])
        assert before != fuser.state
        assert before.channels == {}

    def test_views_expose_summaries(self) -> None:
        state = self.make_state()
        summary = state.channels["s"]
        assert summary.tag == "v1"
        assert summary.separation.count == 1.0
        assert summary.separation.variance >= 0.0
        fp = state.fingerprint
        assert fp.stat_version == ruffle.STAT_VERSION
        assert fp.directions["s"] is Direction.HIGHER_IS_BETTER
        assert state.format_version == ruffle.FORMAT_VERSION

    def test_decay_halves_counts_and_preserves_means(self) -> None:
        state = self.make_state()
        before = state.channels["s"].separation
        state.decay(0.5)
        after = state.channels["s"].separation
        assert after.count == pytest.approx(before.count * 0.5)
        assert after.mean == before.mean

    def test_rekey_moves_summaries(self) -> None:
        state = self.make_state()
        state.rekey("s", "dense")
        assert "s" not in state.channels
        assert state.channels["dense"].tag == "v1"
        assert state.fingerprint.directions["dense"] is Direction.HIGHER_IS_BETTER

    def test_merge_pools_counts(self) -> None:
        a, b = self.make_state(), self.make_state()
        merged, divergence = RuffleState.merge([a, b], MergePolicy.STRICT)
        assert merged.channels["s"].separation.count == 2.0
        assert divergence.max == 0.0

    def test_states_are_unhashable(self) -> None:
        with pytest.raises(TypeError):
            hash(self.make_state())


class TestResume:
    def test_round_trip(self) -> None:
        semantic = ChannelConfig(ChannelId("s", "v1"), Direction.HIGHER_IS_BETTER)
        fuser = Fuser([semantic])
        fuser.fuse([ChannelInput.scored(semantic, spiked_pool())])
        resumed = Fuser.resume([semantic], fuser.state)
        resumed.fuse([ChannelInput.scored(semantic, spiked_pool())])
        assert resumed.state.channels["s"].separation.count == 2.0

    def test_bumped_tag_refuses(self) -> None:
        v1 = ChannelConfig(ChannelId("s", "model-v1"), Direction.HIGHER_IS_BETTER)
        fuser = Fuser([v1])
        fuser.fuse([ChannelInput.scored(v1, spiked_pool())])
        v2 = ChannelConfig(ChannelId("s", "model-v2"), Direction.HIGHER_IS_BETTER)
        with pytest.raises(ruffle.ResumeError, match="model-v1 vs model-v2"):
            Fuser.resume([v2], fuser.state)


class TestAnchor:
    def test_build_calls_score_for_every_pair(self) -> None:
        a = ChannelConfig(ChannelId("a", "v1"), Direction.HIGHER_IS_BETTER)
        b = ChannelConfig(ChannelId("b", "v1"), Direction.LOWER_IS_BETTER)
        calls: dict[str, int] = {"n": 0}

        def score(candidate: str, key: str) -> float:
            calls["n"] += 1
            return float(len(candidate))

        candidates = [f"c{i}" for i in range(5)]
        anchor = Anchor.build(candidates, [a, b], score)
        assert calls["n"] == len(candidates) * 2
        assert "candidates=5" in repr(anchor)

    def test_refresh_coupling_accumulates_pairs(self) -> None:
        a = ChannelConfig(ChannelId("a", "v1"), Direction.HIGHER_IS_BETTER)
        b = ChannelConfig(ChannelId("b", "v1"), Direction.HIGHER_IS_BETTER)
        fuser = Fuser([a, b])
        candidates = [f"c{i}" for i in range(40)]
        anchor = Anchor.build(candidates, [a, b], lambda candidate, key: float(candidate[1:]))
        fuser.refresh_coupling(anchor)
        pair = fuser.state.pairs[("a", "b")]
        assert pair.refreshes == 1.0
        assert pair.redundancy.count == 40.0
        assert pair.redundancy.mean == pytest.approx(1.0)


class TestReprs:
    def test_reprs_are_informative(self) -> None:
        semantic = ChannelConfig(ChannelId("s", "v1"), Direction.HIGHER_IS_BETTER)
        fuser = Fuser([semantic])
        assert repr(fuser) == "Fuser(channels=['s'])"
        assert str(ChannelId("s", "v1")) == "s@v1"
        inp = ChannelInput.scored(semantic, [("a", 1.0)])
        assert "scored" in repr(inp) and "1 items" in repr(inp)
        assert "RuffleState" in repr(fuser.state)
