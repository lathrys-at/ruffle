/**
 * Typed surface of the compiled wasm module.
 *
 * The generated glue (`pkg/ruffle_wasm.js`) types every `JsValue` crossing as `any`;
 * this module is the one seam where those exports are asserted to their real shapes,
 * mirrored from the DTOs in `bindings/wasm/src/lib.rs`. Everything else in the
 * package imports the typed `core` from here, so no `any` reaches the public
 * surface.
 *
 * @internal
 */

import * as raw from "../pkg/ruffle_wasm.js";

export type DirectionValue = "higher_is_better" | "lower_is_better";
export type BaselineModeValue = "z_score";
export type FlagValue =
  | "ranks_only_default_weighted"
  | "degenerate_separation"
  | "no_reference";
export type ErrorKind =
  | "config"
  | "resume"
  | "merge"
  | "state"
  | "value"
  | "internal";

export interface GoodScoreDto {
  readonly typical: number;
  readonly good: number;
  readonly weight: number;
}

export interface ChannelDto {
  readonly key: string;
  readonly tag: string;
  readonly direction: DirectionValue;
  readonly goodScore: GoodScoreDto | undefined;
  readonly baseWeight: number;
}

export interface DiscriminationConfigDto {
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
  readonly gDeviationKeep: number;
}

export interface CouplingConfigDto {
  readonly enabled: boolean;
  readonly discountCap: number;
  readonly shrinkToIdentity: number;
  readonly minOverlap: number;
  readonly minReliability: number;
  readonly minRefreshes: number;
  readonly stratumStabilityMaxVar: number;
}

export interface RrfConfigDto {
  readonly rrfEta: number;
}

export interface DecayConfigDto {
  readonly enabled: boolean;
  readonly factor: number;
}

export interface FuseConfigDto {
  readonly discrimination: DiscriminationConfigDto;
  readonly coupling: CouplingConfigDto;
  readonly fusion: RrfConfigDto;
  readonly decay: DecayConfigDto;
  readonly baselineMode: BaselineModeValue;
}

export type InputDto =
  | {
      readonly key: string;
      readonly direction: DirectionValue;
      readonly scored: ReadonlyArray<readonly [string, number]>;
    }
  | { readonly key: string; readonly ranked: readonly string[] };

export interface AnchorDto {
  readonly channels: ReadonlyArray<readonly [string, DirectionValue]>;
  readonly rows: ReadonlyArray<ReadonlyArray<number | null>>;
}

export interface DiscriminationReadDto {
  readonly g: number;
  readonly rawSeparation: number | undefined;
  readonly topMAverage: number | undefined;
  readonly degenerateSeparation: boolean;
  readonly referenceCold: boolean;
}

export interface FusedDto {
  readonly ranking: ReadonlyArray<readonly [string, number]>;
  readonly weights: ReadonlyMap<string, number>;
  readonly flags: ReadonlyMap<string, FlagValue>;
  readonly discrimination: ReadonlyMap<string, DiscriminationReadDto>;
  readonly confidence: number;
  readonly conflict: number;
}

export interface DivergenceDto {
  readonly perChannel: ReadonlyMap<string, number>;
  readonly max: number;
}

export interface MergeDto {
  readonly merged: string;
  readonly divergence: DivergenceDto;
}

export interface BoundaryError {
  readonly kind: ErrorKind;
  readonly message: string;
}

export interface CoreFuser {
  free(): void;
  fuse(inputs: ReadonlyArray<InputDto>): FusedDto;
  refreshCoupling(anchor: AnchorDto): void;
  stateJson(): string;
}

interface CoreFuserConstructor {
  new (channels: ReadonlyArray<ChannelDto>, config: FuseConfigDto): CoreFuser;
  resume(
    channels: ReadonlyArray<ChannelDto>,
    config: FuseConfigDto,
    state: string,
  ): CoreFuser;
  fuseStateless(
    inputs: ReadonlyArray<InputDto>,
    channels: ReadonlyArray<ChannelDto>,
    config: FuseConfigDto,
    prior: string,
  ): FusedDto;
}

interface Core {
  readonly Fuser: CoreFuserConstructor;
  stateCanonicalize(state: string): string;
  stateMerge(parts: readonly string[]): MergeDto;
  stateDivergence(a: string, b: string): DivergenceDto;
  stateRekey(state: string, fromKey: string, toKey: string): string;
  stateDecay(state: string, factor: number): string;
  defaultConfig(): FuseConfigDto;
  engineVersion(): string;
  formatVersion(): number;
  statVersion(): number;
}

export const core: Core = raw as unknown as Core;
