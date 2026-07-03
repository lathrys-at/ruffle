# Proposal: in-tree TypeScript bindings (WebAssembly)

Status: implemented (`bindings/wasm`), with three notes against the draft below. The
package name is `@lathrys-at/ruffle`, the recommended candidate; it is a one-line
change in `package.json` if reconsidered before the first npm publish. The single-ESM
question resolved to one artifact with top-level await: Node reads the wasm from
disk, browsers and bundlers fetch it from `import.meta.url`, and the module is ready
synchronously after import. The browser smoke test is not yet in CI; the vitest
suite covers Node, and a headless-browser lane remains open work. The shared
invariants (one engine, state interoperability, determinism, version lockstep,
parity suite) are stated in full in [`python-bindings.md`](python-bindings.md) and
apply here unchanged; the `libm` prerequisite is done in the core crate.

Departures from the draft after review: `RuffleState` is an immutable value
(`rekey` and `decay` return new states), matching the Python binding; result
objects are classes whose `toJSON` converts their ES `Map`s to plain records, so
`JSON.stringify` serializes them whole; a `StateError` covers a state document that
does not parse; and unknown configuration keys are refused with `TypeError` in the
TypeScript layer, because serde-wasm-bindgen reads known fields off a JS object and
ignores the rest, so `deny_unknown_fields` cannot catch a typo'd knob at the
boundary.

## Why wasm, and why it is a good home for ruffle

A TypeScript binding means compiling the crate to WebAssembly. The alternative for
Node-only reach would be a native addon (napi-rs), which is faster per call but
excludes browsers, Cloudflare Workers, Deno, and every other edge runtime; wasm runs
in all of them from a single artifact. Fusion is a microsecond-scale, allocation-light,
pure-`f64` computation with no I/O and no threads, which is close to the ideal wasm
workload: the boundary crossing dominates, and the boundary is a few thousand
id/score pairs per query.

Determinism survives the compilation target. WebAssembly specifies IEEE-754 `f64`
arithmetic exactly (no fast-math, no extended precision), and on the wasm target Rust
already links its own pure-Rust libm rather than a platform one. With the core crate
moved to explicit `libm` (the shared prerequisite), native and wasm builds produce
bit-identical rankings and state bytes, and the parity suite asserts it.

## Package name

`ruffle` on npm belongs to the Flash-emulator project, so a name is needed. Three
candidates:

1. **`@lathrys-at/ruffle`** (recommended). A scoped package under the org. Scopes are the
   npm-native answer to name collisions: provenance is explicit, the unscoped
   squatting problem disappears, and the import reads naturally
   (`import { Fuser } from "@lathrys-at/ruffle"`). Scoped packages are also the only
   names npm grants Trusted Publishing to without a claims process.
2. `ruffle-ts`. Available, but the `-ts` suffix conventionally signals "TypeScript
   port" or "type definitions," both of which misdescribe a wasm binding, and it
   invites confusion with `@types/*` conventions.
3. `ruffle-wasm`. Accurate but leads with the implementation detail rather than the
   library.

The draft assumes `@lathrys-at/ruffle`; nothing else in this proposal depends on the
choice.

## Architecture

wasm-bindgen, built with wasm-pack, as a workspace member:

```
bindings/wasm/
  Cargo.toml             # crate ruffle-wasm, cdylib, publish = false (crates.io)
  src/lib.rs             # the wasm-bindgen layer
  ts/                    # hand-written TypeScript wrapper + type definitions
  tests/                 # vitest suite + parity replay (Node), browser smoke test
```

The published npm package wraps the raw wasm-bindgen output in a thin hand-written
TypeScript layer. That layer owns the ergonomics: plain-object configs, `Map`-typed
results, and exceptions, so the generated glue stays an implementation detail and the
public `.d.ts` is curated rather than emitted.

Two boundary formats, chosen by role:

- **Per-query inputs and results** cross as structured JS values via
  `serde-wasm-bindgen` (no JSON stringification on the hot path). Ids are JS strings,
  scores are numbers; the same `Id = String` decision as Python, for the same reason.
- **State** crosses as a JSON string, produced by the same `serde_json` with
  `float_roundtrip` that the Rust crate uses. This is what makes a state file written
  by a browser deployment byte-identical to one written by a Rust service, and
  mergeable by either.

Expected artifact size is a few hundred kilobytes of wasm before gzip (the crate has
no heavy dependencies; `serde_json` is the largest contributor), reduced with
`wasm-opt -Oz`. Worth measuring early; if `serde_json` dominates, the state path can
feed the existing bytes through without re-parsing on the JS side.

## API mapping

The TypeScript surface follows the Rust tier-1 surface with the same three departures
as Python (string ids, plain number scores, config-objects-over-defaults), plus one
TS-specific choice: configuration is a nested `Partial<FuseConfig>` merged over the
crate defaults inside the binding, because TypeScript has no default-then-mutate
idiom and partial object literals are the native way to say "defaults except these."

```ts
import { Fuser, Direction, RuffleState } from "@lathrys-at/ruffle";

const dense = {
  id: { key: "dense", tag: "clip-v1" },
  direction: Direction.HigherIsBetter,
};
const lexical = {
  id: { key: "lexical", tag: "bm25-v1" },
  direction: Direction.HigherIsBetter,
  goodScore: { typical: 12.0, good: 24.0, weight: 8 },
};

const fuser = Fuser.create([dense, lexical], { coupling: { enabled: false } }); // throws on invalid config

const fused = fuser.fuse([
  { key: "dense", scored: [["doc-1", 0.91], ["doc-2", 0.55]] },
  { key: "lexical", scored: [["doc-2", 7.3], ["doc-1", 4.1]] },
]);
fused.ranking;        // ReadonlyArray<readonly [string, number]>
fused.weights;        // ReadonlyMap<string, number>
fused.discrimination; // ReadonlyMap<string, ChannelDiscrimination>

const json = fuser.state.toJson();       // persist
const resumed = Fuser.resume([dense, lexical], RuffleState.fromJson(json)); // throws ResumeError on gate failure
```

Errors map to a small exception hierarchy (`RuffleError` base; `ConfigError`,
`ResumeError`, `MergeError`, `StateError`) carrying the same variant information
the Rust enums do. The `components` tier is out of scope for the first release, as
in Python.

One wasm-specific lifecycle note: wasm-bindgen objects hold linear-memory allocations
that JS garbage collection does not free deterministically. The wrapper keeps the
long-lived `Fuser` as the only handle the caller manages (with a `free()` method and
`Symbol.dispose` support for `using`), and everything else crosses the boundary by
value, so there is nothing else to leak.

## Testing

The parity suite replayed under Node (vitest) is the core. Beyond it: a browser smoke
test (headless, via wasm-pack's harness) proving the artifact loads and fuses in a
real browser context; a type-level test compiling the public examples under
`tsc --strict`; and the cross-language state round-trip against the Rust CLI, as in
Python.

## Release integration

A `npm` job on `v*` tags: build with wasm-pack, run wasm-opt, assemble the package
(ESM entry, Node and bundler compatibility, curated `.d.ts`), assert the version
matches the tag, publish with npm Trusted Publishing (OIDC provenance). Same one-time
bootstrap caveat for the first publish of the name.

## Open questions

1. The package name (above). This is the one decision that blocks nothing else but
   should be settled before any announcement mentions it.
2. Single-artifact ESM versus per-target builds (`bundler`/`node`/`web`): wasm-pack
   can emit each; modern bundlers and Node ≥ 20 make a single ESM package with
   embedded init workable. Draft says single ESM, revisit if a consumer platform
   cannot load it.
3. A `napi-rs` native addon for Node-heavy deployments later, behind the same
   TypeScript API? Nothing in this design precludes it; the wrapper layer was kept
   hand-written partly so a second backend could slot under it.
