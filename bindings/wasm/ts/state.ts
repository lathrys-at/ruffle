/**
 * Persistent state: the single mergeable object, plus its read-only views.
 *
 * Everything Ruffle persists is a confidence-weighted summary plus the identifiers
 * needed to merge it safely. The canonical representation is the engine's JSON
 * serialization: maps are stored ordered, so two states with identical contents
 * serialize byte-for-byte identically, making a serialized state
 * content-addressable and its diffs clean. A `RuffleState` holds those canonical
 * bytes and delegates every operation to the engine.
 */

import { core } from "./boundary.js";
import { rethrow } from "./errors.js";
import {
  MergePolicy,
  Direction,
  BaselineMode,
  type Divergence,
} from "./types.js";

/**
 * Confidence-weighted streaming mean and variance, as persisted by the engine.
 *
 * `count` is the effective observation count, fractional to support pseudo-counts
 * and decay; `mean` is the running mean; `m2` is the sum of squared deviations from
 * the mean, so the population variance is `m2 / count`.
 */
export class MeanVar {
  constructor(
    readonly count: number,
    readonly mean: number,
    readonly m2: number,
  ) {}

  /**
   * The population variance `m2 / count`, zero for an empty summary and clamped so
   * rounding never yields a negative value.
   */
  get variance(): number {
    if (this.count <= 0) {
      return 0;
    }
    return Math.max(this.m2 / this.count, 0);
  }

  /** The population standard deviation. */
  get std(): number {
    return Math.sqrt(this.variance);
  }
}

/**
 * The persistent statistics for one channel: the separation baseline, the
 * good-score reference, and the model-version tag that gates merging.
 */
export interface ChannelSummary {
  readonly separation: MeanVar;
  readonly reference: MeanVar;
  readonly tag: string;
}

/**
 * The persistent statistics for one pair of channels: the accumulated redundancy
 * correlation plus how many anchor refreshes back it.
 */
export interface PairSummary {
  readonly redundancy: MeanVar;
  readonly refreshes: number;
}

/** One pair's entry in a state: the canonical (sorted) channel pair and its summary. */
export interface PairEntry {
  readonly channels: readonly [string, string];
  readonly summary: PairSummary;
}

/**
 * A fingerprint answering whether two states were measuring the same thing the same
 * way: the statistic-definition version, the baseline mode, and the per-channel
 * orientation in force when the state was built.
 */
export interface StatFingerprint {
  readonly statVersion: number;
  readonly baselineMode: BaselineMode;
  readonly directions: ReadonlyMap<string, Direction>;
}

// The engine's canonical serialization schema (snake_case, as written by serde).
interface PersistedMeanVar {
  readonly count: number;
  readonly mean: number;
  readonly m2: number;
}
interface PersistedChannel {
  readonly separation: PersistedMeanVar;
  readonly reference: PersistedMeanVar;
  readonly tag: string;
}
interface PersistedPair {
  readonly redundancy: PersistedMeanVar;
  readonly refreshes: number;
}
interface PersistedState {
  readonly format_version: number;
  readonly fingerprint: {
    readonly stat_version: number;
    readonly baseline_mode: "ZScore";
    readonly directions: Readonly<
      Record<string, "HigherIsBetter" | "LowerIsBetter">
    >;
  };
  readonly channels: Readonly<Record<string, PersistedChannel>>;
  readonly pairs: ReadonlyArray<readonly [readonly [string, string], PersistedPair]>;
}

const PERSISTED_DIRECTIONS = {
  HigherIsBetter: Direction.HigherIsBetter,
  LowerIsBetter: Direction.LowerIsBetter,
} as const;

const PERSISTED_BASELINE_MODES = { ZScore: BaselineMode.ZScore } as const;

function meanVar(raw: PersistedMeanVar): MeanVar {
  return new MeanVar(raw.count, raw.mean, raw.m2);
}

/**
 * The persistent statistics Ruffle accumulates: a confidence-weighted summary per
 * channel and per channel pair, plus the versioning needed to merge two of them
 * safely.
 *
 * A state comes from `Fuser.state`, from `RuffleState.fromJson`, or from
 * `RuffleState.merge`; there is no public constructor, since an empty state is
 * created by building a `Fuser`. Its single merge operation serves three roles:
 * streaming update as new queries arrive, operator prior seeded before any traffic,
 * and cross-deployment reconciliation of states accumulated on separate machines.
 * Every merge is gated on a required per-channel model-version tag, so a model
 * swapped in under a kept channel name is refused rather than silently blended.
 *
 * A state written by this binding loads, merges, and resumes byte-for-byte under
 * the Rust crate and its command-line tool, and vice versa; the serialization is
 * the same canonical JSON everywhere, and wasm's exact IEEE-754 semantics make the
 * bytes identical across platforms and runtimes.
 */
export class RuffleState {
  #json: string;

  private constructor(canonical: string) {
    this.#json = canonical;
  }

  /**
   * Loads a state from its JSON serialization, validating it in the process.
   *
   * The input is re-serialized canonically, so `toJson` on the result yields the
   * engine's canonical bytes even when the input was formatted differently. Throws
   * `TypeError` when the input is not a well-formed serialized state.
   */
  static fromJson(data: string): RuffleState {
    try {
      return new RuffleState(core.stateCanonicalize(data));
    } catch (e) {
      rethrow(e);
    }
  }

  /** @internal */
  static _fromCanonical(canonical: string): RuffleState {
    return new RuffleState(canonical);
  }

  /**
   * The canonical JSON serialization. Byte-identical for equal contents, so the
   * output is content-addressable and safe to compare, hash, and diff.
   */
  toJson(): string {
    return this.#json;
  }

  /**
   * Combines several states into one, returning the merged state and an advisory
   * divergence between the inputs.
   *
   * The merge is associative and commutative and, with decay off, exact up to f64
   * rounding. Under `MergePolicy.Strict` it refuses with `MergeError` on the first
   * incompatibility: a foreign format or statistic version, a channel present in
   * more than one part with a conflicting orientation, or a channel present in more
   * than one part with a different model-version tag (the signature of a model
   * swap). An empty `parts` array also refuses.
   */
  static merge(
    parts: readonly RuffleState[],
    policy: MergePolicy = MergePolicy.Strict,
  ): [RuffleState, Divergence] {
    if (policy !== MergePolicy.Strict) {
      throw new TypeError(`unknown merge policy ${String(policy)}`);
    }
    try {
      const out = core.stateMerge(parts.map((p) => p.#json));
      return [
        new RuffleState(out.merged),
        { perChannel: out.divergence.perChannel, max: out.divergence.max },
      ];
    } catch (e) {
      rethrow(e);
    }
  }

  /**
   * The advisory divergence between this state and another, callable on its own
   * before any merge.
   *
   * For every channel present in both states it reports a standardized distance
   * over each of the channel's two baselines and keeps the larger. The good-score
   * reference lives in the channel's native units, so it is the baseline that jumps
   * under a silent model swap; the separation statistic is deliberately scale- and
   * shift-invariant, so a swap that rescales scores can leave it untouched.
   */
  divergence(other: RuffleState): Divergence {
    try {
      const out = core.stateDivergence(this.#json, other.#json);
      return { perChannel: out.perChannel, max: out.max };
    } catch (e) {
      rethrow(e);
    }
  }

  /**
   * Renames a channel's key, moving all of its statistics with it: the channel
   * summary, every pair summary that referenced the old key, and the channel's
   * orientation in the fingerprint.
   *
   * When the destination already exists, the moved data and the existing data are
   * merged, and the destination keeps its own model-version tag and orientation:
   * the caller is asserting that the old key's history belongs to the channel
   * already living under the new one. A no-op rename leaves the state unchanged.
   * Unlike `merge`, rekey runs no tag gate; it is a deliberate rename and cannot
   * fail.
   */
  rekey(fromKey: string, toKey: string): void {
    try {
      this.#json = core.stateRekey(this.#json, fromKey, toKey);
    } catch (e) {
      rethrow(e);
    }
  }

  /**
   * Scales the confidence of every persisted summary down by `factor`, shrinking
   * effective counts while leaving means and variances unchanged.
   *
   * `factor` is clamped to `[0, 1]`. Decay is the one operation that breaks the
   * exactness of `merge`: decaying then merging no longer gives the same result as
   * merging then decaying.
   */
  decay(factor: number): void {
    try {
      this.#json = core.stateDecay(this.#json, factor);
    } catch (e) {
      rethrow(e);
    }
  }

  /** The schema version this state was built or loaded at. */
  get formatVersion(): number {
    return this.#parsed().format_version;
  }

  /**
   * The statistic fingerprint: which statistic definitions, baseline mode, and
   * per-channel orientations the state was measured under.
   */
  get fingerprint(): StatFingerprint {
    const raw = this.#parsed().fingerprint;
    const directions = new Map<string, Direction>();
    for (const [key, dir] of Object.entries(raw.directions)) {
      directions.set(key, PERSISTED_DIRECTIONS[dir]);
    }
    return {
      statVersion: raw.stat_version,
      baselineMode: PERSISTED_BASELINE_MODES[raw.baseline_mode],
      directions,
    };
  }

  /**
   * The per-channel summaries, keyed by join handle. A snapshot: later mutations of
   * the state are not reflected in a previously returned map.
   */
  get channels(): ReadonlyMap<string, ChannelSummary> {
    const out = new Map<string, ChannelSummary>();
    for (const [key, raw] of Object.entries(this.#parsed().channels)) {
      out.set(key, {
        separation: meanVar(raw.separation),
        reference: meanVar(raw.reference),
        tag: raw.tag,
      });
    }
    return out;
  }

  /**
   * The per-pair coupling summaries, one entry per canonical (sorted) channel pair.
   * A snapshot, like `channels`.
   */
  get pairs(): readonly PairEntry[] {
    return this.#parsed().pairs.map(([pair, raw]) => ({
      channels: [pair[0], pair[1]],
      summary: { redundancy: meanVar(raw.redundancy), refreshes: raw.refreshes },
    }));
  }

  /** Whether this state serializes to the same canonical bytes as `other`. */
  equals(other: RuffleState): boolean {
    return this.#json === other.#json;
  }

  #parsed(): PersistedState {
    // The bytes come from the engine's canonical serializer (every constructor
    // funnels through it), so the shape assertion holds by construction.
    return JSON.parse(this.#json) as PersistedState;
  }
}
