# Proposal: in-tree Python bindings

Status: implemented (`bindings/python`), with three deltas from the draft below. The
floor is CPython 3.10 (`abi3-py310`) rather than 3.9, which reached end of life in
October 2025. The public surface is pure Python over a private compiled module
(`ruffle._core`), fully typed and documented, rather than "shims if any"; the FFI
boundary uses typed structures, not JSON, except for states, whose canonical JSON
string is the persistence format itself. The libm prerequisite in invariant 3 is done
in the core crate.

## Invariants shared by every binding

1. **One engine.** A binding links the Rust crate; it never reimplements a statistic.
   Behavioral parity is by construction, not by porting effort.
2. **State interoperability.** A `RuffleState` written by any binding loads, merges,
   and resumes in any other, byte-for-byte where the format is canonical JSON. The
   format/statistic version gates and the tag gate are enforced identically everywhere,
   because they are the same code.
3. **Determinism.** The same inputs, state, and configuration produce the same ranking
   in every binding. One caveat needs closing first, at the core-crate level: the
   engine calls `f64::sin` (copula map) and `f64::exp` (logistic squash) from `std`,
   which resolve to the platform libm on native targets. Those functions can differ in
   the last ulp between platform libms, which means the coupling correlation that
   enters persistent state can differ in the last ulp between, say, a glibc Linux
   deployment and a macOS one. Everything else in the state is arithmetic and `sqrt`,
   which are IEEE-exact. **Prerequisite work:** switch the core crate's `sin`/`exp`
   calls to the [`libm`](https://crates.io/crates/libm) crate, which is pure Rust and
   bit-identical on every target including wasm. Cost is a small constant factor on
   two functions that are not hot; the return is that "identical state bytes" and
   "identical rankings" become flat guarantees with no platform asterisk.
4. **Version lockstep.** A binding's version equals the crate version it wraps, and
   both release from the same tag. A binding never wraps an engine version other than
   the one it declares.
5. **A shared parity suite.** The Rust test suite gains a generator that emits golden
   fixtures as JSON: query inputs, configuration, prior state, expected ranking
   (ids and scores), expected posterior state bytes, and expected merge/refusal
   outcomes. Every binding's CI replays the fixtures and asserts equality. This is the
   enforcement mechanism for invariants 2 and 3; a binding without a green parity run
   does not release.

## Why Python first

Python is where retrieval pipelines are assembled: the embedding calls, the vector
store clients, and the evaluation harnesses live there. It is also where ruffle's
label-free story matters most, because the people prototyping RAG systems rarely have
judgment sets. The `ruffle` name is free on PyPI.

## Architecture

PyO3 with maturin, as a workspace member.

```
ruffle/                  # existing crate, unchanged, publishes to crates.io
bindings/python/
  Cargo.toml             # crate ruffle-py, cdylib, publish = false
  pyproject.toml         # PyPI package "ruffle", built by maturin
  src/lib.rs             # the PyO3 layer
  python/ruffle/         # pure-Python shims if any, py.typed, .pyi stubs
  tests/                 # pytest suite + parity replay
```

The repository root gains a `[workspace]` section with the root crate as a member;
`bindings/*` crates are `publish = false` on crates.io (they publish to their own
registries instead). Wheels are abi3 (`abi3-py39`), so there is one wheel per platform
and architecture rather than one per Python minor version: manylinux x86_64 and
aarch64, macOS x86_64 and arm64, Windows x64, plus an sdist that builds from source
with a Rust toolchain.

## API mapping

The Python surface mirrors the Rust tier-1 surface, with three deliberate departures
where Rust idioms do not translate.

**Ids are strings.** The core is generic over `Id: Hash + Eq + Clone`. Crossing the
FFI boundary with arbitrary Python objects as ids would mean thousands of
`__hash__`/`__eq__` calls back into the interpreter per query. Retrieval systems key
documents by string ids essentially always, so the binding fixes `Id = String` and
documents the restriction. Hosts with integer or composite ids map them at the edge.

**Scores are floats.** The Rust `Score` trait exists to force a compile-time
declaration of what a number means; Python has no equivalent discipline to enforce, so
the binding accepts `list[tuple[str, float]]` directly. The declaration burden the
trait carries in Rust is carried in Python by `ChannelConfig` alone (direction, tag,
good-score reference), which is where the semantics actually live.

**Configs are keyword arguments over defaults.** The Rust configs are
`#[non_exhaustive]` with default-then-mutate construction. The Pythonic equivalent is
constructors with keyword arguments defaulting to the crate defaults:
`FuseConfig(coupling=CouplingConfig(enabled=True))`. Validation runs at `Fuser`
construction exactly as in Rust and raises the mapped exceptions.

Everything else maps one to one:

| Rust | Python |
|---|---|
| `ChannelId::new(key, tag)` | `ChannelId(key, tag)` |
| `GoodScore::new(typical, good, weight)` | `GoodScore(typical, good, weight)` |
| `ChannelConfig::new(id, direction, good_score)` | `ChannelConfig(id, direction, good_score=None)` |
| `Direction::{HigherIsBetter, LowerIsBetter}` | `Direction.HIGHER_IS_BETTER`, `.LOWER_IS_BETTER` |
| `ChannelInput::scored / ::ranked` | `ChannelInput.scored(cfg, items)`, `.ranked(cfg, ids)` |
| `Fuser::new / resume / fuse / fuse_stateless / refresh_coupling / state` | same names, snake_case |
| `Fused { ranking, weights, flags, discrimination, confidence, conflict }` | frozen attributes on a result object |
| `RuffleState::{merge, divergence, rekey, decay}` + serde JSON | same, plus `to_json()` / `from_json()` |
| `Anchor::build(candidates, channels, score_fn)` | same, `score_fn: Callable[[str, str], float | None]` |
| `ConfigError` / `ResumeError` / `Mismatch` | exception classes under a common `RuffleError` base |

The `components` tier (the pure per-stage estimators) is deliberately out of scope for
the first release. It exists in Rust for composition and auditing; a Python caller who
needs it is better served by the Rust crate. Revisit on demand.

The GIL is released for the duration of `fuse` and `merge` once inputs are converted,
so a multi-threaded host can fuse on worker threads. The callback-driven
`Anchor.build` necessarily holds the GIL per call; anchor construction is offline, so
this does not matter.

## Testing

Three layers: the parity suite (invariant 5) replayed with pytest; a small idiomatic
test suite for the Python-only surface (exceptions, keyword construction, stub
accuracy under `mypy --strict`); and a round-trip test proving a state written by
Python merges under the Rust CLI and vice versa. Coverage of the Rust core is the core
suite's job, not the binding's.

## Release integration

The release workflow gains a `wheels` job on `v*` tags: maturin-action builds the
wheel matrix, the job publishes to PyPI via PyPI's Trusted Publishing (OIDC, no stored
token, same trust model as the crates.io publish). The version in `pyproject.toml` is
asserted equal to the crate version in the tag-check step. First publish of the PyPI
name requires the same one-time hand bootstrap as crates.io did.

## Open questions

1. Accept `int` ids alongside `str` (an enum id type internally)? Cheap to add, but it
   doubles the type surface of every signature; the draft says strings only.
2. Should `Fused.ranking` be a list of tuples or a NumPy-friendly pair of arrays?
   Tuples first; add a zero-copy accessor later if profiling in a real pipeline
   demands it.
3. Ship the CLI (`ruffle reconcile`/`rekey`) as a console script in the wheel? It is
   nearly free via a thin entry point, but it duplicates the cargo-installed binary
   and drags `clap`-equivalent argument parsing into Python. The draft says no.
