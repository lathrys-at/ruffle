/** The public value types: channel identity, registration, inputs, and results. */

import type {
  DirectionValue,
  BaselineModeValue,
  FlagValue,
} from "./boundary.js";

/**
 * Whether a higher native score means a better match, or a lower one.
 *
 * Declared once per channel at configuration; Ruffle does not infer it from data.
 * Every score is oriented to higher-is-better at ingest. A channel registered with
 * the wrong direction ranks anti-relevantly and corrupts its own persistent
 * baseline.
 */
export const Direction = {
  /** A higher native score is a better match (already canonical). */
  HigherIsBetter: "higher_is_better",
  /** A lower native score is a better match (negated at ingest). */
  LowerIsBetter: "lower_is_better",
} as const;
export type Direction = DirectionValue;

/**
 * How a channel's scores are standardized within the channel before comparison.
 * Only z-score standardization ships today.
 */
export const BaselineMode = {
  /** Standardizes each score against the channel's running mean and variance. */
  ZScore: "z_score",
} as const;
export type BaselineMode = BaselineModeValue;

/** Why a channel was not weighted by its full discrimination score. */
export const ChannelFlag = {
  /**
   * The channel supplied ranks only, with no scores to compute a discrimination
   * statistic from, so it was carried at the neutral default weight.
   */
  RanksOnlyDefaultWeighted: "ranks_only_default_weighted",
  /**
   * The score pool's bulk had no usable scale to measure the top's elevation
   * against, so the separation read was floored rather than trusted.
   */
  DegenerateSeparation: "degenerate_separation",
  /**
   * The channel had no usable good-score reference yet, so its absolute-goodness
   * term could not be computed this query and it was weighted on separation alone.
   */
  NoReference: "no_reference",
} as const;
export type ChannelFlag = FlagValue;

/** How `RuffleState.merge` treats incompatible inputs. */
export const MergePolicy = {
  /** Refuses on any format, fingerprint, or tag mismatch. The only policy for now. */
  Strict: "strict",
} as const;
export type MergePolicy = (typeof MergePolicy)[keyof typeof MergePolicy];

/**
 * An operator-declared reference for how good a channel's scores are in absolute
 * terms, in the channel's native units (before orientation).
 *
 * The discrimination stage rewards a channel whose top results score well against
 * this reference, complementing the separation of top from bulk. `typical` is the
 * top score a typical, unremarkable query produces (the reference location); `good`
 * is the score a genuinely good match reaches (the gap from `typical` sets the
 * reference scale); `weight` is a pseudo-count for how firmly the declaration holds
 * before observed top scores refine it. Both anchors are oriented with the scores at
 * ingest, so for a `Direction.LowerIsBetter` channel a good match is a smaller
 * native value. After orientation `good` must exceed `typical`; a declaration that
 * cannot orient is refused at `Fuser` construction with `ConfigError`.
 */
export interface GoodScore {
  readonly typical: number;
  readonly good: number;
  readonly weight: number;
}

/**
 * A channel's identity: a stable join handle (`key`) plus a model-version `tag`.
 *
 * `key` is the stable join handle: every persistent map is keyed by it alone, and it
 * stays fixed across model versions. `tag` is the model version (for example
 * `"clip-vit-b32-rev1"`), changed whenever the model behind the channel changes;
 * Ruffle never interprets it, only checks it for equality on every merge, so a model
 * swapped in under a kept key is refused rather than silently blended. An
 * unnecessary tag change costs a cold start; a missed one corrupts the baseline.
 */
export interface ChannelId {
  readonly key: string;
  readonly tag: string;
}

/**
 * Per-channel registration.
 *
 * `id` and `direction` are declared once at channel configuration rather than per
 * query. `goodScore` is the optional declared reference for the absolute-goodness
 * statistic; when absent, the reference is learned from early traffic and the
 * absolute-goodness statistic cold-starts.
 */
export interface ChannelConfig {
  readonly id: ChannelId;
  readonly direction: Direction;
  readonly goodScore?: GoodScore | undefined;
}

/**
 * One channel's input for one query: the channel's key plus its surfaced items,
 * either scored or rank-only.
 *
 * Scored items are `[id, nativeScore]` pairs in the channel's native units; the
 * item order carries no meaning, and orientation plus non-finite filtering happen
 * inside the engine. A ranked input lists ids best first; a rank carries no
 * magnitude, and the channel is carried at the neutral default weight, flagged
 * `ChannelFlag.RanksOnlyDefaultWeighted` in the result. Each channel lists each item
 * at most once: a repeated id within one input is counted twice by the fusion.
 */
export type ChannelInput =
  | {
      readonly key: string;
      readonly scored: ReadonlyArray<readonly [string, number]>;
    }
  | { readonly key: string; readonly ranked: readonly string[] };

/**
 * One channel's discrimination read for one query: the combined weight and the raw
 * statistics behind it.
 *
 * `g` is the channel's combined discrimination weight, bounded, and near `1.0` when
 * the channel performs at its own norm. `rawSeparation` is the top-vs-bulk
 * separation statistic, `undefined` when the score pool is too degenerate to
 * measure it. `topMAverage` is the fixed-count top-m average exported for good-score
 * reference refinement, `undefined` when the pool is rank-only, empty, or shallower
 * than the fixed count. `degenerateSeparation` and `referenceCold` mirror the
 * conditions behind `ChannelFlag`.
 */
export interface ChannelDiscrimination {
  readonly g: number;
  readonly rawSeparation: number | undefined;
  readonly topMAverage: number | undefined;
  readonly degenerateSeparation: boolean;
  readonly referenceCold: boolean;
}

/**
 * The outcome of fusing one query: the merged ranking plus the weights, flags, and
 * diagnostics behind it.
 *
 * `ranking` is the fused order, best first, each id with its fused score. `weights`
 * holds the per-channel weights actually used. `flags` explains any non-standard
 * weighting; a channel absent from it was weighted on its full discrimination
 * score. `discrimination` holds the per-channel reads behind the weights, so the
 * reasoning is readable from the result alone. `confidence` is the top-set
 * agreement of the discriminating channels, in `[0, 1]`; `conflict` is its
 * complement, high when confident channels disagree on which items are relevant.
 */
export interface Fused {
  readonly ranking: ReadonlyArray<readonly [string, number]>;
  readonly weights: ReadonlyMap<string, number>;
  readonly flags: ReadonlyMap<string, ChannelFlag>;
  readonly discrimination: ReadonlyMap<string, ChannelDiscrimination>;
  readonly confidence: number;
  readonly conflict: number;
}

/**
 * An advisory standardized distance between two states' per-channel summaries.
 *
 * The number never gates a merge; gating is done by the model-version tag. It flags
 * a silent model swap, where two summaries have drifted far apart while their tags
 * still match. `max` is the largest per-channel distance, the single number a
 * caller can threshold on.
 */
export interface Divergence {
  readonly perChannel: ReadonlyMap<string, number>;
  readonly max: number;
}
