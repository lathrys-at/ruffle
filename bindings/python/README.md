# ruffle

Weighted, adaptive, calibration-free Reciprocal Rank Fusion (RRF) for Python.

Ruffle fuses the output of several retrieval channels into one ranking. It does this
without per-channel score calibration and without comparing one channel's raw scores
against another's. It requires no relevance labels, no representative query set, and
natively handles channels whose scores live on different scales. For each query, and
still without labels, it estimates how well each channel is discriminating and how
redundant the channels are with each other, and weights a rank-based RRF accordingly.
With the default configuration Ruffle stays close to plain RRF and tilts weights only
when the channels' own outputs support it.

This package binds the Rust crate [`ruffle`](https://crates.io/crates/ruffle). The
compiled engine does all the statistics; the Python layer is a typed, documented
surface over it. Behaviour and persisted state are identical across the two, down to
the serialized bytes: a state written here loads, merges, and resumes under the Rust
crate and its CLI, and vice versa.

## Install

```
pip install ruffle
```

Wheels ship for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows (x64,
arm64).
The wheels use the stable ABI (abi3), so one build covers CPython 3.10 and every
later version; each release is tested against CPython 3.10, 3.11, 3.12, 3.13, and
3.14. Support for the lowest version ends when it stops receiving security updates.
The sdist builds from source with a Rust toolchain.

## Quick start

The following fuses three channels for one query: semantic and lexical channels
scored on their own native scales, and a rank-only recency channel.

```python
from ruffle import (
    ChannelConfig, ChannelId, ChannelInput, Direction, Fuser, GoodScore,
)

# Channels represent different retrieval methods for the same query. Each channel's
# id is a stable key plus a model-version tag.
semantic = ChannelConfig(
    ChannelId("semantic", "text-embedding-v1"),
    Direction.HIGHER_IS_BETTER,  # higher cosine similarity is better
)

# SQLite FTS5 bm25() is negated BM25, so lower (more negative) is better. The
# GoodScore declares what typical and genuinely good scores look like in native
# units; without it, Ruffle learns the reference from traffic.
lexical = ChannelConfig(
    ChannelId("lexical", "sqlite-fts5-trigram-bm25"),
    Direction.LOWER_IS_BETTER,
    GoodScore(typical=-4.0, good=-12.0, weight=8.0),
)

# Channels can be rank-only (without score magnitudes), like a recency metric.
recency = ChannelConfig(
    ChannelId("recency", "recency-v1"),
    Direction.HIGHER_IS_BETTER,
)

fuser = Fuser([semantic, lexical, recency])

fused = fuser.fuse([
    ChannelInput.scored(semantic, [
        ("kelp-forest", 0.55), ("whale-sketch", 0.91), ("tide-chart", 0.42),
    ]),
    ChannelInput.scored(lexical, [
        ("whale-sketch", -3.7), ("field-notes", -1.4), ("kelp-forest", -6.4),
    ]),
    ChannelInput.ranked(recency, ["field-notes", "whale-sketch", "kelp-forest"]),
])

for item_id, score in fused.ranking:
    print(f"{item_id}: {score:.4f}")
```

`fused` also carries the weights used, per-channel flags explaining any non-standard
weighting, the discrimination reads behind the weights, and two agreement
diagnostics.

## State

Everything Ruffle persists is one mergeable summary, exposed as `RuffleState`. A
single merge operation serves as streaming update, operator prior, and
cross-deployment reconciliation, and it is gated on a required per-channel
model-version tag: a model swapped in under a kept channel name is refused rather
than silently blended.

```python
state_json = fuser.state.to_json()      # persist
# ... restart ...
from ruffle import RuffleState
fuser = Fuser.resume([semantic, lexical, recency], RuffleState.from_json(state_json))
```

## Documentation

The [tuning guide](https://github.com/lathrys-at/ruffle/blob/main/docs/tuning.md)
describes what to log, how to read the persisted state, and, for each configuration
default, when and why to change it. The
[design document](https://github.com/lathrys-at/ruffle/blob/main/docs/derivation.md)
contains the full derivation. API documentation lives on the docstrings; the package
is fully typed (`py.typed`).

## License

MIT OR Apache-2.0, at your option.
