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
  defaultConfig,
  version,
  type ChannelConfig,
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
    expect(defaults.fusion.rrfEta).toBe(60);

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
    expect(() => RuffleState.fromJson("{not json")).toThrowError(TypeError);
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

  test("decay halves counts and preserves means", () => {
    const state = makeState();
    const before = state.channels.get("s")!.separation;
    state.decay(0.5);
    const after = state.channels.get("s")!.separation;
    expect(after.count).toBeCloseTo(before.count * 0.5, 12);
    expect(after.mean).toBe(before.mean);
  });

  test("rekey moves summaries", () => {
    const state = makeState();
    state.rekey("s", "dense");
    expect(state.channels.has("s")).toBe(false);
    expect(state.channels.get("dense")!.tag).toBe("v1");
    expect(state.fingerprint.directions.get("dense")).toBe(
      Direction.HigherIsBetter,
    );
  });

  test("merge pools counts", () => {
    const [merged, divergence] = RuffleState.merge(
      [makeState(), makeState()],
      MergePolicy.Strict,
    );
    expect(merged.channels.get("s")!.separation.count).toBe(2.0);
    expect(divergence.max).toBe(0.0);
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

  test("a freed fuser throws rather than corrupting", () => {
    const fuser = Fuser.create([channel("s")]);
    fuser.free();
    expect(() => fuser.fuse([{ key: "s", scored: [["a", 1.0]] }])).toThrow();
  });
});
