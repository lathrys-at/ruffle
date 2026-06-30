# Proposal: in-tree Go bindings

Status: draft. The shared invariants (one engine, state interoperability,
determinism, version lockstep, parity suite) are stated in full in
[`python-bindings.md`](python-bindings.md) and apply here unchanged, including the
prerequisite `libm` switch in the core crate.

## Naming: the problem dissolves

Go has no central package registry with claimable names; a module's name is its
import path. The squatted `ruffle` packages elsewhere on pkg.go.dev are other people's
paths and do not collide with ours. The in-tree module is

```
module github.com/lathrys-at/ruffle/bindings/go
```

with package name `ruffle`, imported as

```go
import "github.com/lathrys-at/ruffle/bindings/go"    // package ruffle
```

A caller who dislikes the path tail aliases it at the import site, which is ordinary
Go. The one mechanical consequence of a subdirectory module is the tag scheme: Go
tooling resolves versions of a nested module from tags prefixed with its path, so
releases carry a second tag, `bindings/go/vX.Y.Z`, alongside `vX.Y.Z`. The release
workflow can push both from the same commit; lockstep is preserved by making the Go
tag's version assert equality with the crate version, as the other bindings do.

## Architecture: wasm under wazero, not cgo

Two viable routes exist for Rust-backed Go packages, and they trade in opposite
directions:

- **cgo against a static library.** Maximum per-call performance, but it taxes every
  consumer: cgo disables trivial cross-compilation (`GOOS`/`GOARCH` builds now need a
  C toolchain and a prebuilt `.a` per target), slows Go builds, and either the
  repository vendors per-platform binaries or every `go get` user needs a Rust
  toolchain.
- **WebAssembly under [wazero](https://wazero.io).** The crate compiles once to a
  `wasm32-wasip1` artifact, the `.wasm` bytes are embedded in the Go package with
  `go:embed`, and wazero (a pure-Go, dependency-free wasm runtime with an ahead-of-time
  compiler on amd64/arm64) runs it. No cgo anywhere: `go get` works, cross-compilation
  works, static binaries work.

The draft recommends wazero. The deciding argument is who pays: cgo's costs land on
every downstream consumer forever, wasm's costs land on us once (an FFI layer) plus a
per-call overhead that the workload can afford. Native fusion at four channels and a
thousand candidates per channel costs ~0.4 ms; under wazero, with the boundary copy
and wasm execution, a conservative estimate is 1–3 ms per fuse. Retrieval fan-outs
this sits behind cost tens of milliseconds. If a profiled deployment ever needs the
native path, a cgo backend can be added behind the same Go API later; nothing in this
design precludes it.

Determinism is unimpaired: wasm's `f64` semantics are exact IEEE-754, and with the
core on explicit `libm`, the Go binding produces the same rankings and state bytes as
every other binding, enforced by the parity suite.

## The FFI layer

wasm-bindgen is a JavaScript-oriented tool; for a Go host the binding needs a plain
wasm ABI instead. A small new crate provides it:

```
bindings/ffi-wasm/
  Cargo.toml             # crate ruffle-ffi-wasm, cdylib for wasm32-wasip1, publish = false
  src/lib.rs             # exported functions over (ptr, len) byte buffers
bindings/go/
  go.mod
  ruffle.go              # the public Go API
  runtime.go             # wazero setup, instance pool, memory management
  ruffle.wasm            # embedded artifact, rebuilt in CI, checksummed
  ruffle_test.go         # Go tests + parity replay
```

The exported ABI is deliberately small and boring: a handle table inside the wasm
instance maps integer handles to live `Fuser`s, and every call passes and returns
JSON in linear memory (`alloc`, `free`, `fuser_new`, `fuser_resume`, `fuser_fuse`,
`fuser_state`, `fuser_refresh_coupling`, `state_merge`, `state_divergence`,
`last_error`). JSON is the right first choice because the state format is already
canonical JSON and the crate already depends on `serde_json` with `float_roundtrip`;
one format, no impedance. If profiling shows the per-query input serialization
matters, the input path (only) can move to a compact binary encoding later without
touching the state format.

Fallible calls return an error discriminant plus a JSON error payload that the Go
layer maps onto typed errors (`ruffle.ConfigError`, `ruffle.MismatchError` wrapping
the same variants as the Rust enums), all conforming to `error` with `errors.As`
support.

Concurrency: a wazero module instance is single-threaded, and a `Fuser` is stateful
anyway. The Go API mirrors that honestly: a `*ruffle.Fuser` is not safe for
concurrent use, same as any stateful Go object; the documentation says so and the
race detector enforces it in tests. For read-only fan-out (`FuseStateless`), the
binding maintains a `sync.Pool` of module instances so parallel callers do not
serialize on one interpreter.

## API mapping

The same three departures as the other bindings (string ids, plain float64 scores,
config structs with zero-value-means-default semantics), rendered in Go idiom:

```go
dense := ruffle.ChannelConfig{
    ID:        ruffle.ChannelID{Key: "dense", Tag: "clip-v1"},
    Direction: ruffle.HigherIsBetter,
}
lexical := ruffle.ChannelConfig{
    ID:        ruffle.ChannelID{Key: "lexical", Tag: "bm25-v1"},
    Direction: ruffle.HigherIsBetter,
    GoodScore: &ruffle.GoodScore{Typical: 12.0, Good: 24.0, Weight: 8},
}

fuser, err := ruffle.NewFuser([]ruffle.ChannelConfig{dense, lexical}, ruffle.DefaultConfig())
if err != nil { ... }                     // ConfigError: invalid knob, duplicate key, bad GoodScore

fused, err := fuser.Fuse([]ruffle.ChannelInput{
    {Key: "dense", Scored: []ruffle.ScoredItem{{ID: "doc-1", Score: 0.91}}},
    {Key: "lexical", Scored: []ruffle.ScoredItem{{ID: "doc-2", Score: 7.3}}},
})
fused.Ranking        // []ruffle.RankedItem{ID string; Score float64}
fused.Weights        // map[string]float64
fused.Discrimination // map[string]ruffle.ChannelDiscrimination

stateJSON, _ := fuser.StateJSON()          // persist
fuser2, err := ruffle.Resume(configs, stateJSON, cfg) // MismatchError on gate failure
```

One Go-specific decision: configuration uses explicit struct types whose zero values
are NOT the defaults (a zero `FuseConfig` would mean `top_eps = 0`, which is
invalid). `ruffle.DefaultConfig()` returns the crate defaults for mutation, and the
constructor validates exactly as Rust does, so the zero-value trap is caught loudly
at construction rather than silently misweighting. The `components` tier is out of
scope for the first release, as in the other bindings.

## Testing

The parity suite replayed under `go test` is the core, plus: the cross-language state
round-trip against the Rust CLI; a race-detector run over the documented concurrency
patterns; and a CI check that the embedded `ruffle.wasm` was built from the current
tree (rebuild and compare checksums), so the artifact can never silently lag the
sources it sits next to.

## Release integration

A `go` job on `v*` tags: build `ruffle-ffi-wasm` for `wasm32-wasip1`, verify the
embedded artifact checksum matches, run the Go tests, then push the
`bindings/go/vX.Y.Z` tag. There is no registry publish step; the Go proxy picks the
module up from the tag. The embedded-artifact-in-git question is the one aesthetic
wart (a ~few-hundred-KB binary committed per release); the alternative, fetching at
build time, breaks `go get`'s no-network-hooks model, so the draft accepts the
committed artifact.

## Open questions

1. Committed wasm artifact versus a generated-on-release approach that keeps the
   binary out of day-to-day diffs (for example, committing it only on release
   commits). Draft: commit it, refreshed by CI check, because a Go module must be
   complete at its tag.
2. Is 1–3 ms per fuse acceptable for the target Go audience, or does the cgo backend
   need to ship in the first release? Draft: wazero only; measure with the parity
   fixtures on real hardware before deciding anything else.
3. Should `FuseStateless` be exposed as a package-level function backed by the
   instance pool (ergonomic for read-only fan-out) or kept as a method to mirror the
   Rust surface? Draft: package-level function.
