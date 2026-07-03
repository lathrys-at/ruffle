/**
 * The fusion configuration: full and partial shapes over the engine's defaults.
 *
 * Every default is read from the compiled engine at load, so the resolved values are
 * the crate's own defaults, never a copy that could drift. Every knob has a
 * conservative default, chosen so the shipped behaviour stays close to plain
 * reciprocal-rank fusion.
 */

import { core, type FuseConfigDto } from "./boundary.js";
import { BaselineMode } from "./types.js";

/**
 * Per-channel discrimination knobs: how each channel's separation and absolute
 * goodness are read and turned into a weight.
 *
 * - `topEps`: fraction of the result pool forming the "extreme top" whose mean is
 *   the numerator of the separation statistic (top versus bulk).
 * - `topM`: fixed number of top scores averaged for the absolute-goodness
 *   statistic. A fixed count is steadier than the single maximum.
 * - `minDistinctValues`: minimum number of distinct pool values required before the
 *   separation statistic is computed.
 * - `denomFloorFrac`: floors the separation statistic's denominator toward the
 *   inter-quartile gap by this fraction, so a near-tied bulk cannot inflate the
 *   ratio.
 * - `winsorZ`: a standardized separation read beyond this many standard deviations
 *   is winsorized before it touches the baseline.
 * - `minCountForZ`: minimum effective baseline count before a standardized
 *   separation read is trusted.
 * - `shrinkPoolSize`: pool size below which the channel's weight is shrunk toward
 *   its own running discrimination baseline.
 * - `gUpperBound`: upper bound on the discrimination weight `g`, so no single
 *   channel can dominate the fused order.
 * - `gFloor`: small positive floor on `g`, so an uncertain channel still
 *   contributes.
 * - `gSlope`: slope of the logistic squash mapping each standardized statistic to a
 *   `(0, 1)` factor in `g`.
 */
export interface DiscriminationConfig {
  readonly topEps: number;
  readonly topM: number;
  readonly minDistinctValues: number;
  readonly denomFloorFrac: number;
  readonly winsorZ: number;
  readonly minCountForZ: number;
  readonly shrinkPoolSize: number;
  readonly gUpperBound: number;
  readonly gFloor: number;
  readonly gSlope: number;
}

/**
 * Channel-coupling knobs: how the redundancy discount between channels is estimated
 * and applied. Independence is the only unconditionally recall-safe setting, so
 * coupling is off by default and every knob caps how far a discount can move
 * weight.
 *
 * - `enabled`: whether to apply any redundancy discount at all.
 * - `discountCap`: caps the discount well below the raw anchor point estimate.
 * - `shrinkToIdentity`: mandatory shrinkage intensity, in `[0, 1]`, of the
 *   redundancy correlation toward the identity.
 * - `minOverlap`: minimum number of anchor items scored by both channels before a
 *   pair correlation counts.
 * - `minReliability`: minimum accumulated overlap count before any discount
 *   applies.
 * - `minRefreshes`: minimum number of anchor refreshes backing a pair before any
 *   discount applies; stability across query strata is a between-refresh property.
 * - `stratumStabilityMaxVar`: maximum between-stratum variance of the anchor
 *   correlation that still allows a discount.
 */
export interface CouplingConfig {
  readonly enabled: boolean;
  readonly discountCap: number;
  readonly shrinkToIdentity: number;
  readonly minOverlap: number;
  readonly minReliability: number;
  readonly minRefreshes: number;
  readonly stratumStabilityMaxVar: number;
}

/**
 * Weighted reciprocal-rank fusion knobs. `rrfEta` is the RRF rank constant; larger
 * values flatten the rank contribution, and 60 is the common RRF default from
 * Cormack et al. (2009).
 */
export interface RrfConfig {
  readonly rrfEta: number;
}

/**
 * State-decay knobs: forgetting old observations to track corpus drift. Off by
 * default; decay ties a merge to an external clock, making the otherwise exact
 * merge identity approximate. The cadence is per observation rather than per
 * wall-clock interval, bounding each baseline's effective sample size at
 * `1 / (1 - factor)`. A caller who wants wall-clock decay instead can call
 * `RuffleState.decay` on its own schedule with this setting left off.
 */
export interface DecayConfig {
  readonly enabled: boolean;
  readonly factor: number;
}

/** The complete fusion configuration: the grouped sub-configs plus the baseline mode. */
export interface FuseConfig {
  readonly discrimination: DiscriminationConfig;
  readonly coupling: CouplingConfig;
  readonly fusion: RrfConfig;
  readonly decay: DecayConfig;
  readonly baselineMode: BaselineMode;
}

/**
 * A partial configuration merged over the engine's defaults, the native way to say
 * "defaults except these"::
 *
 *     Fuser.create(channels, { coupling: { enabled: true } })
 *
 * Validation runs when a `Fuser` is built, so an out-of-range knob fails at
 * construction with `ConfigError` rather than mid-query.
 */
export interface FuseConfigInit {
  readonly discrimination?: Partial<DiscriminationConfig>;
  readonly coupling?: Partial<CouplingConfig>;
  readonly fusion?: Partial<RrfConfig>;
  readonly decay?: Partial<DecayConfig>;
  readonly baselineMode?: BaselineMode;
}

/** The engine's default configuration. */
export function defaultConfig(): FuseConfig {
  return resolveConfig();
}

/**
 * The full configuration a partial one resolves to: each group is the engine's
 * defaults with the given fields laid over them.
 *
 * @internal
 */
export function resolveConfig(init?: FuseConfigInit): FuseConfig {
  const defaults: FuseConfigDto = core.defaultConfig();
  return Object.freeze({
    discrimination: Object.freeze({
      ...defaults.discrimination,
      ...init?.discrimination,
    }),
    coupling: Object.freeze({ ...defaults.coupling, ...init?.coupling }),
    fusion: Object.freeze({ ...defaults.fusion, ...init?.fusion }),
    decay: Object.freeze({ ...defaults.decay, ...init?.decay }),
    baselineMode: init?.baselineMode ?? BaselineMode.ZScore,
  });
}
