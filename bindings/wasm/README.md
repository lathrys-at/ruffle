# @lathrys-at/ruffle

Weighted, adaptive, calibration-free Reciprocal Rank Fusion (RRF) for TypeScript and
JavaScript.

Ruffle fuses the output of several retrieval channels into one ranking. It does this
without per-channel score calibration and without comparing one channel's raw scores
against another's. It requires no relevance labels, no representative query set, and
natively handles channels whose scores live on different scales. For each query, and
still without labels, it estimates how well each channel is discriminating and how
redundant the channels are with each other, and weights a rank-based RRF accordingly.
With the default configuration Ruffle stays close to plain RRF and tilts weights only
when the channels' own outputs support it.

This package is the Rust crate [`ruffle`](https://crates.io/crates/ruffle) compiled
to WebAssembly, under a hand-written, fully typed TypeScript surface. The engine does
all the statistics, and WebAssembly's exact IEEE-754 semantics make its behaviour and
persisted state identical to the native crate down to the serialized bytes: a state
written in a browser or edge runtime loads, merges, and resumes under the Rust crate
and its CLI, and vice versa. One artifact runs in Node 20+, browsers, and edge
runtimes.

## Install

```
npm install @lathrys-at/ruffle
```

The module instantiates its wasm at import (top-level await), so everything is ready
to use synchronously after `import`.

## Quick start

The following fuses three channels for one query: semantic and lexical channels
scored on their own native scales, and a rank-only recency channel.

```ts
import { Direction, Fuser } from "@lathrys-at/ruffle";

// Channels represent different retrieval methods for the same query. Each channel's
// id is a stable key plus a model-version tag.
const semantic = {
  id: { key: "semantic", tag: "text-embedding-v1" },
  direction: Direction.HigherIsBetter, // higher cosine similarity is better
};

// SQLite FTS5 bm25() is negated BM25, so lower (more negative) is better. The
// goodScore declares what typical and genuinely good scores look like in native
// units; without it, Ruffle learns the reference from traffic.
const lexical = {
  id: { key: "lexical", tag: "sqlite-fts5-trigram-bm25" },
  direction: Direction.LowerIsBetter,
  goodScore: { typical: -4.0, good: -12.0, weight: 8 },
};

// Channels can be rank-only (without score magnitudes), like a recency metric.
const recency = {
  id: { key: "recency", tag: "recency-v1" },
  direction: Direction.HigherIsBetter,
};

const fuser = Fuser.create([semantic, lexical, recency]);

const fused = fuser.fuse([
  { key: "semantic", scored: [["kelp-forest", 0.55], ["whale-sketch", 0.91], ["tide-chart", 0.42]] },
  { key: "lexical", scored: [["whale-sketch", -3.7], ["field-notes", -1.4], ["kelp-forest", -6.4]] },
  { key: "recency", ranked: ["field-notes", "whale-sketch", "kelp-forest"] },
]);

for (const [id, score] of fused.ranking) {
  console.log(`${id}: ${score.toFixed(4)}`);
}
```

`fused` also carries the weights used (`Map`), per-channel flags explaining any
non-standard weighting, the discrimination reads behind the weights, and two
agreement diagnostics.

A `Fuser` owns a wasm-side allocation; `free()` releases it deterministically, and
the class supports `using` (`Symbol.dispose`) for scope-bound lifetimes. Everything
else crosses the boundary by value.

## State

Everything Ruffle persists is one mergeable summary, exposed as `RuffleState`. A
single merge operation serves as streaming update, operator prior, and
cross-deployment reconciliation, and it is gated on a required per-channel
model-version tag: a model swapped in under a kept channel name is refused rather
than silently blended.

```ts
import { Fuser, RuffleState } from "@lathrys-at/ruffle";

const json = fuser.state.toJson(); // persist
// ... restart ...
const resumed = Fuser.resume(
  [semantic, lexical, recency],
  RuffleState.fromJson(json),
);
```

## Documentation

The [tuning guide](https://github.com/lathrys-at/ruffle/blob/main/docs/tuning.md)
describes what to log, how to read the persisted state, and, for each configuration
default, when and why to change it. The
[design document](https://github.com/lathrys-at/ruffle/blob/main/docs/derivation.md)
contains the full derivation. API documentation lives on the declarations; the
package ships curated `.d.ts` types with documentation on every public item.

## License

MIT OR Apache-2.0, at your option.
