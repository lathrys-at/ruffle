/**
 * Weighted, adaptive, calibration-free reciprocal-rank fusion.
 *
 * Ruffle fuses the output of several retrieval channels into one ranking, without
 * per-channel score calibration and without comparing one channel's raw scores
 * against another's. It requires no relevance labels and natively handles channels
 * whose scores live on different scales. For each query it estimates two properties
 * from the channels' own outputs: per-channel discrimination (how far a channel's
 * top results stand above its bulk, and how good they are against a declared,
 * evidence-refined good-score reference) and pairwise redundancy (a correlation
 * measured on a shared full-scored anchor, away from the live pool's selection
 * bias). The estimates weight a rank-based RRF. Every estimate is conservative:
 * with the default configuration Ruffle stays close to plain RRF and tilts weights
 * only when the channels' own outputs support it.
 *
 * The package wraps the Rust crate `ruffle` compiled to WebAssembly; the engine
 * does all the statistics, so behaviour and persisted state are identical across
 * the two, down to the serialized bytes. The module awaits wasm instantiation at
 * import (top-level await), so everything is ready to use synchronously:
 *
 * ```ts
 * import { Fuser, Direction } from "@lathrys/ruffle";
 *
 * const semantic = {
 *   id: { key: "semantic", tag: "text-embedding-v1" },
 *   direction: Direction.HigherIsBetter,
 * };
 * const fuser = Fuser.create([semantic]);
 * const fused = fuser.fuse([{ key: "semantic", scored: [["doc-1", 0.91]] }]);
 * ```
 *
 * The design document and tuning guide live in the repository under `docs/`:
 * https://github.com/lathrys-at/ruffle
 */

import { initRuffle } from "./init.js";

await initRuffle();

export { Anchor } from "./anchor.js";
export {
  defaultConfig,
  type CouplingConfig,
  type DecayConfig,
  type DiscriminationConfig,
  type FuseConfig,
  type FuseConfigInit,
  type RrfConfig,
} from "./config.js";
export { ConfigError, MergeError, ResumeError, RuffleError } from "./errors.js";
export { Fuser } from "./fuser.js";
export {
  MeanVar,
  RuffleState,
  type ChannelSummary,
  type PairEntry,
  type PairSummary,
  type StatFingerprint,
} from "./state.js";
export {
  BaselineMode,
  ChannelFlag,
  Direction,
  MergePolicy,
  type ChannelConfig,
  type ChannelDiscrimination,
  type ChannelId,
  type ChannelInput,
  type Divergence,
  type Fused,
  type GoodScore,
} from "./types.js";

import { core } from "./boundary.js";

/** The engine version this artifact was built from, in lockstep with the crate. */
export const version: string = core.engineVersion();

/** The state schema version this build writes. */
export const FORMAT_VERSION: number = core.formatVersion();

/** The statistic-definition version this build writes. */
export const STAT_VERSION: number = core.statVersion();
