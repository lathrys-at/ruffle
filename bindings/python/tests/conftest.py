"""Shared helpers: fixture loading and construction from the parity schema."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from ruffle import (
    Anchor,
    BaselineMode,
    ChannelConfig,
    ChannelId,
    ChannelInput,
    CouplingConfig,
    DecayConfig,
    Direction,
    DiscriminationConfig,
    FuseConfig,
    Fused,
    GoodScore,
    RrfConfig,
)

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "parity"


def load_fixture(name: str) -> dict[str, Any]:
    with open(FIXTURE_DIR / name, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def all_fixture_names() -> list[str]:
    return sorted(p.name for p in FIXTURE_DIR.glob("*.json"))


def make_channel(d: dict[str, Any]) -> ChannelConfig:
    gs = d["good_score"]
    return ChannelConfig(
        ChannelId(d["key"], d["tag"]),
        Direction(d["direction"]),
        None
        if gs is None
        else GoodScore(typical=gs["typical"], good=gs["good"], weight=gs["weight"]),
    )


def make_channels(fixture: dict[str, Any]) -> dict[str, ChannelConfig]:
    """Every channel the fixture mentions, registered or not, keyed by join handle."""
    out = {}
    for d in fixture.get("channels", []):
        out[d["key"]] = make_channel(d)
    for d in fixture.get("unregistered_channels", []):
        out[d["key"]] = make_channel(d)
    return out


def make_config(d: dict[str, Any]) -> FuseConfig:
    return FuseConfig(
        discrimination=DiscriminationConfig(**d["discrimination"]),
        coupling=CouplingConfig(**d["coupling"]),
        fusion=RrfConfig(**d["fusion"]),
        decay=DecayConfig(**d["decay"]),
        baseline_mode=BaselineMode(d["baseline_mode"]),
    )


def make_input(d: dict[str, Any], channels: Mapping[str, ChannelConfig]) -> ChannelInput:
    cfg = channels[d["key"]]
    if "scored" in d:
        return ChannelInput.scored(cfg, [(item, score) for item, score in d["scored"]])
    return ChannelInput.ranked(cfg, d["ranked"])


def make_anchor(d: dict[str, Any], channels: Mapping[str, ChannelConfig]) -> Anchor:
    """Rebuilds the fixture's anchor through the public callback path, so the replay
    exercises Anchor.build rather than injecting the matrix directly."""
    candidates: list[str] = d["candidates"]
    keys: list[str] = d["channels"]
    rows: list[list[Any]] = d["scores"]
    index = {c: i for i, c in enumerate(candidates)}

    def score(candidate: str, key: str) -> Any:
        return rows[keys.index(key)][index[candidate]]

    return Anchor.build(candidates, [channels[k] for k in keys], score)


def fused_to_dict(f: Fused) -> dict[str, Any]:
    """A Fused result in the fixture's expected-output schema, for exact comparison."""
    return {
        "ranking": [[item, score] for item, score in f.ranking],
        "weights": dict(f.weights),
        "flags": {k: v.value for k, v in f.flags.items()},
        "discrimination": {
            k: {
                "g": d.g,
                "raw_separation": d.raw_separation,
                "top_m_average": d.top_m_average,
                "degenerate_separation": d.degenerate_separation,
                "reference_cold": d.reference_cold,
            }
            for k, d in f.discrimination.items()
        },
        "confidence": f.confidence,
        "conflict": f.conflict,
    }


@pytest.fixture
def quickstart_channels() -> dict[str, ChannelConfig]:
    return {
        "semantic": ChannelConfig(
            ChannelId("semantic", "text-embedding-v1"), Direction.HIGHER_IS_BETTER
        ),
        "lexical": ChannelConfig(
            ChannelId("lexical", "sqlite-fts5-trigram-bm25"),
            Direction.LOWER_IS_BETTER,
            GoodScore(typical=-4.0, good=-12.0, weight=8.0),
        ),
        "recency": ChannelConfig(
            ChannelId("recency", "recency-v1"), Direction.HIGHER_IS_BETTER
        ),
    }
