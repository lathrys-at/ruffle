/**
 * The parity replay itself, shared by the Node and browser suites: one function
 * per fixture kind, each asserting exact equality against the fixture's expected
 * outputs, including the serialized state bytes. Runtime-agnostic; fixture
 * loading is the caller's concern.
 */

import { expect } from "vitest";

import {
  ConfigError,
  Fuser,
  MergeError,
  MergePolicy,
  ResumeError,
  RuffleState,
} from "../ts/index.js";
import {
  fusedToFixture,
  makeAnchor,
  makeChannel,
  makeChannels,
  makeConfig,
  makeInput,
  type Fixture,
} from "./helpers.js";

export function replaySession(fixture: Fixture): void {
  expect(fixture.kind).toBe("session");
  const channels = makeChannels(fixture);
  const registered = (fixture.channels ?? []).map(makeChannel);
  const fuser = Fuser.create(registered, makeConfig(fixture.config!));
  try {
    for (const step of fixture.steps ?? []) {
      if (step.op === "fuse") {
        const fused = fuser.fuse(step.inputs.map(makeInput));
        expect(fusedToFixture(fused)).toEqual(step.expected);
      } else {
        fuser.refreshCoupling(makeAnchor(step.anchor, channels));
      }
    }
    expect(fuser.state.toJson()).toBe(fixture.expected_state);
  } finally {
    fuser.free();
  }
}

export function replayStateless(fixture: Fixture): void {
  expect(fixture.kind).toBe("stateless");
  const registered = (fixture.channels ?? []).map(makeChannel);
  const prior = RuffleState.fromJson(fixture.prior_state!);

  const fused = Fuser.fuseStateless(
    (fixture.inputs ?? []).map(makeInput),
    registered,
    prior,
    makeConfig(fixture.config!),
  );

  expect(fusedToFixture(fused)).toEqual(fixture.expected);
  // The prior is read, never written.
  expect(prior.toJson()).toBe(fixture.prior_state);
}

export function replayMerge(fixture: Fixture): void {
  expect(fixture.kind).toBe("merge");
  const parts = (fixture.parts ?? []).map((p) => RuffleState.fromJson(p));

  const [merged, divergence] = RuffleState.merge(parts, MergePolicy.Strict);
  expect(merged.toJson()).toBe(fixture.expected_state);
  expect({
    per_channel: Object.fromEntries(divergence.perChannel),
    max: divergence.max,
  }).toEqual(fixture.expected_divergence);

  // The merge is commutative: the reversed order produces identical bytes.
  const [remerged] = RuffleState.merge([...parts].reverse());
  expect(remerged.toJson()).toBe(fixture.expected_state);
}

export function replayStateOps(fixture: Fixture): void {
  expect(fixture.kind).toBe("state_ops");
  let state = RuffleState.fromJson(fixture.start_state!);

  for (const op of fixture.ops ?? []) {
    if (op.op === "rekey") {
      state = state.rekey(op.from, op.to);
    } else {
      state = state.decay(op.factor);
    }
  }

  expect(state.toJson()).toBe(fixture.expected_state);
}

// Each refusal kind maps to the exception a binding must raise and a stable
// fragment of the engine's message.
const MISMATCH_FRAGMENTS: Record<string, string> = {
  format_version: "format version mismatch",
  fingerprint: "statistic fingerprint mismatch",
  direction_conflict: "direction conflict",
  tag: "semantic tag mismatch",
  empty: "empty set of states",
};
const CONFIG_FRAGMENTS: Record<string, string> = {
  invalid_fuse_config: "invalid fuse configuration",
  invalid_good_score: "good score is unusable",
  duplicate_channel_key: "duplicate channel key",
};

export function replayRefusalCase(c: NonNullable<Fixture["cases"]>[number]): void {
  if (c.kind === "resume") {
    const channels = (c.channels ?? []).map(makeChannel);
    const config = makeConfig(c.config!);
    const state = RuffleState.fromJson(c.state!);
    const fragment = MISMATCH_FRAGMENTS[c.error]!;
    expect(() => Fuser.resume(channels, state, config)).toThrowError(ResumeError);
    expect(() => Fuser.resume(channels, state, config)).toThrowError(fragment);
    // fuseStateless runs the same gate against the prior.
    expect(() => Fuser.fuseStateless([], channels, state, config)).toThrowError(
      fragment,
    );
  } else if (c.kind === "merge") {
    const parts = (c.parts ?? []).map((p) => RuffleState.fromJson(p));
    const fragment = MISMATCH_FRAGMENTS[c.error]!;
    expect(() => RuffleState.merge(parts)).toThrowError(MergeError);
    expect(() => RuffleState.merge(parts)).toThrowError(fragment);
  } else {
    const channels = (c.channels ?? []).map(makeChannel);
    const config = makeConfig(c.config!);
    const fragment = CONFIG_FRAGMENTS[c.error]!;
    expect(() => Fuser.create(channels, config)).toThrowError(ConfigError);
    expect(() => Fuser.create(channels, config)).toThrowError(fragment);
  }
}

/** Replays any fixture by its kind, refusal cases included. */
export function replayFixture(fixture: Fixture): void {
  switch (fixture.kind) {
    case "session":
      replaySession(fixture);
      break;
    case "stateless":
      replayStateless(fixture);
      break;
    case "merge":
      replayMerge(fixture);
      break;
    case "state_ops":
      replayStateOps(fixture);
      break;
    case "refusals":
      for (const c of fixture.cases ?? []) {
        replayRefusalCase(c);
      }
      break;
    default:
      throw new Error(`unknown fixture kind ${fixture.kind}`);
  }
}
