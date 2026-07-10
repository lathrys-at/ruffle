/**
 * Idiomatic tests for the TypeScript-only surface: construction, immutability,
 * exceptions, defaults, lifecycle, and the read-only state views.
 */

import { describe, expect, test } from "vitest";

import {
  Anchor,
  ChannelFlag,
  ConfigError,
  Direction,
  FORMAT_VERSION,
  Fuser,
  MergeError,
  MergePolicy,
  ResumeError,
  RuffleError,
  RuffleState,
  STAT_VERSION,
  StateError,
  defaultConfig,
  version,
  type ChannelConfig,
  type ChannelInput,
  type FuseConfigInit,
} from "../ts/index.js";

function channel(
  key: string,
  tag = "v1",
  direction: Direction = Direction.HigherIsBetter,
): ChannelConfig {
  return { id: { key, tag }, direction };
}

function spikedPool(n = 30): Array<[string, number]> {
  const pool: Array<[string, number]> = [];
  for (let i = 0; i < n; i++) {
    pool.push([`doc${String(i).padStart(3, "0")}`, 0.01 * i]);
  }
  pool.push(["hit0", 10.0], ["hit1", 10.5]);
  return pool;
}

function makeState(): RuffleState {
  const semantic = channel("s");
  const fuser = Fuser.create([semantic]);
  try {
    fuser.fuse([{ key: "s", scored: spikedPool() }]);
    return fuser.state;
  } finally {
    fuser.free();
  }
}

describe("versioning", () => {
  test("versions are exposed and consistent", () => {
    expect(version).toMatch(/^\d+\.\d+\.\d+/);
    expect(FORMAT_VERSION).toBeGreaterThanOrEqual(2);
    expect(STAT_VERSION).toBeGreaterThanOrEqual(2);
  });
});

describe("configuration", () => {
  test("defaults come from the engine and partials lay over them", () => {
    const defaults = defaultConfig();
    expect(defaults.coupling.enabled).toBe(false);
    expect(defaults.fusion.rrfEta).toBe(20);
    expect(defaults.fusion.minGDispersion).toBe(0.45);

    const fuser = Fuser.create([channel("s")], { coupling: { enabled: true } });
    try {
      expect(fuser.config.coupling.enabled).toBe(true);
      expect(fuser.config.coupling.discountCap).toBe(
        defaults.coupling.discountCap,
      );
    } finally {
      fuser.free();
    }
  });

  test("resolved configs are frozen", () => {
    const fuser = Fuser.create([channel("s")]);
    try {
      expect(Object.isFrozen(fuser.config)).toBe(true);
      expect(Object.isFrozen(fuser.config.coupling)).toBe(true);
    } finally {
      fuser.free();
    }
  });

  test("an out-of-range knob names the field", () => {
    expect(() =>
      Fuser.create([channel("s")], {
        discrimination: { gFloor: 5.0, gUpperBound: 4.0 },
      }),
    ).toThrowError(/g_upper_bound/);
  });

  test("a base weight tilt renormalizes over present channels", () => {
    const a: ChannelConfig = { ...channel("a"), baseWeight: 3.0 };
    const b = channel("b");
    const holder = Fuser.create([a, b]);
    const prior = holder.state;
    holder.free();
    const fused = Fuser.fuseStateless(
      [
        { key: "a", scored: spikedPool() },
        { key: "b", scored: spikedPool() },
      ],
      [a, b],
      prior,
    );
    // Cold adaptive weights are neutral, so the fused weights are the declared
    // 3:1 tilt renormalized to sum to the channel count.
    expect(fused.weights.get("a")).toBeCloseTo(1.5, 9);
    expect(fused.weights.get("b")).toBeCloseTo(0.5, 9);
  });

  test("an invalid base weight is refused at construction", () => {
    const bad: ChannelConfig = { ...channel("s"), baseWeight: -1.0 };
    expect(() => Fuser.create([bad])).toThrowError(ConfigError);
    expect(() => Fuser.create([bad])).toThrowError(/base weight/);
  });

  test("an unusable good score is refused at construction", () => {
    const bad: ChannelConfig = {
      ...channel("s"),
      goodScore: { typical: 0.5, good: 0.3, weight: 4.0 },
    };
    expect(() => Fuser.create([bad])).toThrowError(ConfigError);
    expect(() => Fuser.create([bad])).toThrowError(/good score is unusable/);
  });
});

describe("exceptions", () => {
  test("hierarchy", () => {
    expect(new ConfigError("x")).toBeInstanceOf(RuffleError);
    expect(new ResumeError("x")).toBeInstanceOf(RuffleError);
    expect(new MergeError("x")).toBeInstanceOf(RuffleError);
    expect(new ConfigError("x").name).toBe("ConfigError");
  });

  test("an empty merge refuses", () => {
    expect(() => RuffleState.merge([])).toThrowError(MergeError);
    expect(() => RuffleState.merge([])).toThrowError(/empty set of states/);
  });

  test("an unknown merge policy is a TypeError", () => {
    expect(() =>
      RuffleState.merge([], "loose" as unknown as MergePolicy),
    ).toThrowError(TypeError);
  });
});

describe("fuse", () => {
  test("lower-is-better orients at ingest", () => {
    const lexical = channel("lex", "v1", Direction.LowerIsBetter);
    const fuser = Fuser.create([lexical]);
    try {
      const fused = fuser.fuse([
        { key: "lex", scored: [["worse", -1.0], ["best", -9.0]] },
      ]);
      expect(fused.ranking.map(([id]) => id)).toEqual(["best", "worse"]);
    } finally {
      fuser.free();
    }
  });

  test("non-finite scores are dropped", () => {
    const fuser = Fuser.create([channel("s")]);
    try {
      const fused = fuser.fuse([
        {
          key: "s",
          scored: [["ok", 0.5], ["nan", Number.NaN], ["inf", Number.POSITIVE_INFINITY]],
        },
      ]);
      expect(fused.ranking.map(([id]) => id)).toEqual(["ok"]);
    } finally {
      fuser.free();
    }
  });

  test("an unregistered input is skipped", () => {
    const fuser = Fuser.create([channel("s")]);
    try {
      const fused = fuser.fuse([
        { key: "s", scored: [["a", 0.9]] },
        { key: "rogue", scored: [["z", 99.0]] },
      ]);
      expect(fused.ranking.map(([id]) => id)).toEqual(["a"]);
      expect(fused.weights.has("rogue")).toBe(false);
    } finally {
      fuser.free();
    }
  });

  test("a rank-only channel is flagged at the neutral weight", () => {
    const fuser = Fuser.create([channel("r")]);
    try {
      const fused = fuser.fuse([{ key: "r", ranked: ["a", "b"] }]);
      expect(fused.flags.get("r")).toBe(ChannelFlag.RanksOnlyDefaultWeighted);
      expect(fused.weights.get("r")).toBe(1.0);
    } finally {
      fuser.free();
    }
  });

  test("stateless with an empty prior is unweighted", () => {
    const channels = [channel("a"), channel("b")];
    const empty = Fuser.create(channels);
    try {
      const fused = Fuser.fuseStateless(
        [
          { key: "a", scored: spikedPool() },
          { key: "b", scored: spikedPool() },
        ],
        channels,
        empty.state,
      );
      for (const w of fused.weights.values()) {
        expect(w).toBe(1.0);
      }
    } finally {
      empty.free();
    }
  });
});

describe("state", () => {
  test("fromJson canonicalizes formatting and rejects garbage", () => {
    const state = makeState();
    const pretty = state.toJson().replaceAll(",", ", ");
    expect(RuffleState.fromJson(pretty).toJson()).toBe(state.toJson());
    expect(() => RuffleState.fromJson("{not json")).toThrowError(
      /invalid ruffle state/,
    );
  });

  test("snapshots are independent", () => {
    const fuser = Fuser.create([channel("s")]);
    try {
      const before = fuser.state;
      fuser.fuse([{ key: "s", scored: spikedPool() }]);
      expect(before.equals(fuser.state)).toBe(false);
      expect(before.channels.size).toBe(0);
    } finally {
      fuser.free();
    }
  });

  test("views expose the summaries", () => {
    const state = makeState();
    const summary = state.channels.get("s")!;
    expect(summary.tag).toBe("v1");
    expect(summary.separation.count).toBe(1.0);
    expect(summary.separation.variance).toBeGreaterThanOrEqual(0);
    expect(state.fingerprint.statVersion).toBe(STAT_VERSION);
    expect(state.fingerprint.directions.get("s")).toBe(
      Direction.HigherIsBetter,
    );
    expect(state.formatVersion).toBe(FORMAT_VERSION);
  });

  test("decay returns a new state and preserves means", () => {
    const state = makeState();
    const before = state.channels.get("s")!.separation;
    const decayed = state.decay(0.5);
    const after = decayed.channels.get("s")!.separation;
    expect(after.count).toBeCloseTo(before.count * 0.5, 12);
    expect(after.mean).toBe(before.mean);
    expect(state.channels.get("s")!.separation.count).toBe(before.count);
  });

  test("rekey returns a new state with the summaries moved", () => {
    const state = makeState();
    const rekeyed = state.rekey("s", "dense");
    expect(rekeyed.channels.has("s")).toBe(false);
    expect(rekeyed.channels.get("dense")!.tag).toBe("v1");
    expect(rekeyed.fingerprint.directions.get("dense")).toBe(
      Direction.HigherIsBetter,
    );
    expect(state.channels.has("s")).toBe(true);
  });

  test("merge pools counts", () => {
    const [merged, divergence] = RuffleState.merge(
      [makeState(), makeState()],
      MergePolicy.Strict,
    );
    expect(merged.channels.get("s")!.separation.count).toBe(2.0);
    expect(divergence.max).toBe(0.0);
  });

  test("fromJson garbage raises StateError inside the hierarchy", () => {
    expect(() => RuffleState.fromJson("{not json")).toThrowError(StateError);
    expect(() => RuffleState.fromJson('{"format_version": 1}')).toThrowError(
      RuffleError,
    );
  });

  test("a state embedded in JSON.stringify carries the document", () => {
    const state = makeState();
    const embedded = JSON.parse(JSON.stringify({ state })) as {
      state: { format_version: number };
    };
    expect(embedded.state.format_version).toBe(FORMAT_VERSION);
    const reloaded = RuffleState.fromJson(JSON.stringify(state));
    expect(reloaded.equals(state)).toBe(true);
  });

  test("a single-part merge reproduces the state", () => {
    const state = makeState();
    const [merged] = RuffleState.merge([state]);
    expect(merged.equals(state)).toBe(true);
  });
});

describe("resume", () => {
  test("round trip continues accumulating", () => {
    const semantic = channel("s");
    const state = makeState();
    const resumed = Fuser.resume([semantic], state);
    try {
      resumed.fuse([{ key: "s", scored: spikedPool() }]);
      expect(resumed.state.channels.get("s")!.separation.count).toBe(2.0);
    } finally {
      resumed.free();
    }
  });

  test("a bumped tag refuses", () => {
    const state = makeState();
    const v2 = channel("s", "v2");
    expect(() => Fuser.resume([v2], state)).toThrowError(ResumeError);
    expect(() => Fuser.resume([v2], state)).toThrowError(/v1 vs v2/);
  });
});

describe("anchor", () => {
  test("build calls score for every pair and refresh accumulates", () => {
    const a = channel("a");
    const b = channel("b");
    let calls = 0;
    const candidates = Array.from({ length: 40 }, (_, i) => `c${i}`);
    const anchor = Anchor.build(candidates, [a, b], (candidate) => {
      calls += 1;
      return Number(candidate.slice(1));
    });
    expect(calls).toBe(candidates.length * 2);

    const fuser = Fuser.create([a, b]);
    try {
      fuser.refreshCoupling(anchor);
      const pairs = fuser.state.pairs;
      expect(pairs).toHaveLength(1);
      expect(pairs[0]!.channels).toEqual(["a", "b"]);
      expect(pairs[0]!.summary.refreshes).toBe(1.0);
      expect(pairs[0]!.summary.redundancy.count).toBe(40.0);
      expect(pairs[0]!.summary.redundancy.mean).toBeCloseTo(1.0, 9);
    } finally {
      fuser.free();
    }
  });
});

describe("results", () => {
  test("JSON.stringify serializes the whole result through toJSON", () => {
    const fuser = Fuser.create([channel("s")]);
    try {
      const fused = fuser.fuse([{ key: "s", scored: spikedPool() }]);
      const parsed = JSON.parse(JSON.stringify(fused)) as {
        ranking: Array<[string, number]>;
        weights: Record<string, number>;
        discrimination: Record<string, { g: number }>;
        confidence: number;
      };
      expect(parsed.ranking).toEqual(fused.ranking.map((entry) => [...entry]));
      expect(parsed.weights["s"]).toBe(fused.weights.get("s"));
      expect(parsed.discrimination["s"]!.g).toBe(
        fused.discrimination.get("s")!.g,
      );
      expect(parsed.confidence).toBe(fused.confidence);
    } finally {
      fuser.free();
    }
  });

  test("divergence stringifies through toJSON", () => {
    const a = makeState();
    const b = makeState();
    const parsed = JSON.parse(JSON.stringify(a.divergence(b))) as {
      perChannel: Record<string, number>;
      max: number;
    };
    expect(parsed.perChannel).toHaveProperty("s");
    expect(parsed.max).toBe(0);
  });
});

describe("error causes", () => {
  test("boundary errors carry the raw value as cause", () => {
    let caught: unknown;
    try {
      Fuser.create([channel("s")], {
        discrimination: { gFloor: 5.0, gUpperBound: 4.0 },
      });
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(ConfigError);
    const cause = (caught as ConfigError).cause as { kind: string };
    expect(cause.kind).toBe("config");
  });
});

describe("input validation", () => {
  test("an input with both scored and ranked is refused", () => {
    const fuser = Fuser.create([channel("s")]);
    try {
      const both = {
        key: "s",
        scored: [["a", 1.0]],
        ranked: ["a"],
      } as unknown as ChannelInput;
      expect(() => fuser.fuse([both])).toThrowError(TypeError);
      expect(() => fuser.fuse([both])).toThrowError(/one or the other/);
    } finally {
      fuser.free();
    }
  });

  test("an unknown configuration key is refused, top-level and nested", () => {
    const typo = {
      discrimination: { topeps: 0.1 },
    } as unknown as FuseConfigInit;
    expect(() => Fuser.create([channel("s")], typo)).toThrowError(TypeError);
    expect(() => Fuser.create([channel("s")], typo)).toThrowError(/topeps/);
    const rogue = { copuling: {} } as unknown as FuseConfigInit;
    expect(() => Fuser.create([channel("s")], rogue)).toThrowError(/copuling/);
  });

  test("prototype members do not pass as configuration keys", () => {
    const top = { toString: 1 } as unknown as FuseConfigInit;
    expect(() => Fuser.create([channel("s")], top)).toThrowError(/toString/);
    const nested = {
      coupling: { hasOwnProperty: 1 },
    } as unknown as FuseConfigInit;
    expect(() => Fuser.create([channel("s")], nested)).toThrowError(
      /hasOwnProperty/,
    );
  });

  test("a non-object configuration section is refused, not silently defaulted", () => {
    const primitive = { discrimination: 42 } as unknown as FuseConfigInit;
    expect(() => Fuser.create([channel("s")], primitive)).toThrowError(
      /must be an object, not number/,
    );
    const nullish = { coupling: null } as unknown as FuseConfigInit;
    expect(() => Fuser.create([channel("s")], nullish)).toThrowError(
      /must be an object, not null/,
    );
  });

  test("an input with neither scored nor ranked is refused by name", () => {
    const fuser = Fuser.create([channel("s")]);
    try {
      const bare = { key: "s" } as unknown as ChannelInput;
      expect(() => fuser.fuse([bare])).toThrowError(/neither scored nor ranked/);
    } finally {
      fuser.free();
    }
  });

  test("a ranked input with an explicit undefined scored field is accepted", () => {
    const fuser = Fuser.create([channel("s")]);
    try {
      const input = {
        key: "s",
        scored: undefined,
        ranked: ["a", "b"],
      } as unknown as ChannelInput;
      const fused = fuser.fuse([input]);
      expect(fused.ranking.map(([id]) => id)).toEqual(["a", "b"]);
    } finally {
      fuser.free();
    }
  });
});

describe("scale", () => {
  test("a large scored payload fuses deterministically", () => {
    const n = 20_000;
    const pool: Array<[string, number]> = Array.from({ length: n }, (_, i) => [
      `doc-${i}`,
      Math.sin(i) * 100,
    ]);
    const run = (): readonly (readonly [string, number])[] => {
      const fuser = Fuser.create([channel("s")]);
      try {
        return fuser.fuse([{ key: "s", scored: pool }]).ranking;
      } finally {
        fuser.free();
      }
    };
    const first = run();
    const second = run();
    expect(first).toHaveLength(n);
    expect(second).toEqual(first);
  });
});

describe("lifecycle", () => {
  test("Symbol.dispose frees the handle", () => {
    let stateJson: string;
    {
      using fuser = Fuser.create([channel("s")]);
      fuser.fuse([{ key: "s", scored: spikedPool() }]);
      stateJson = fuser.state.toJson();
    }
    expect(stateJson).toContain('"s"');
  });

  test("every operation on a freed fuser throws RuffleError", () => {
    const fuser = Fuser.create([channel("s")]);
    const anchor = Anchor.build(["c0"], [channel("s")], () => 1.0);
    fuser.free();
    expect(() => fuser.fuse([{ key: "s", scored: [["a", 1.0]] }])).toThrowError(
      RuffleError,
    );
    expect(() => fuser.fuse([{ key: "s", scored: [["a", 1.0]] }])).toThrowError(
      /freed/,
    );
    expect(() => fuser.state).toThrowError(/freed/);
    expect(() => fuser.refreshCoupling(anchor)).toThrowError(/freed/);
  });

  test("free is idempotent, including through Symbol.dispose", () => {
    const fuser = Fuser.create([channel("s")]);
    fuser.free();
    expect(() => fuser.free()).not.toThrow();
    expect(() => fuser[Symbol.dispose]()).not.toThrow();
  });
});
