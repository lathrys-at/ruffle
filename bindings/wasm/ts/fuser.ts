/** The entry point: fuses several retrieval channels' ranked outputs into one ranking. */

import {
  core,
  type ChannelDto,
  type CoreFuser,
  type FusedDto,
  type InputDto,
} from "./boundary.js";
import { resolveConfig, type FuseConfig, type FuseConfigInit } from "./config.js";
import { rethrow, RuffleError } from "./errors.js";
import { Anchor } from "./anchor.js";
import { RuffleState } from "./state.js";
import {
  Direction,
  Fused,
  type ChannelConfig,
  type ChannelInput,
} from "./types.js";

function channelDtos(channels: readonly ChannelConfig[]): ChannelDto[] {
  return channels.map((c) => ({
    key: c.id.key,
    tag: c.id.tag,
    direction: c.direction,
    goodScore: c.goodScore,
  }));
}

function directionMap(
  channels: readonly ChannelConfig[],
): ReadonlyMap<string, Direction> {
  return new Map(channels.map((c) => [c.id.key, c.direction]));
}

function inputDtos(
  inputs: readonly ChannelInput[],
  directions: ReadonlyMap<string, Direction>,
): InputDto[] {
  return inputs.map((input) => {
    // The union says one or the other. The checks compare values, not key
    // presence, so a ranked input carrying an explicit `scored: undefined` is
    // accepted; a plain-JS caller supplying both real fields would otherwise get
    // silent scored-wins, and one supplying neither would get an opaque boundary
    // error.
    const scored = "scored" in input ? input.scored : undefined;
    const ranked = "ranked" in input ? input.ranked : undefined;
    if (scored !== undefined && ranked !== undefined) {
      throw new TypeError(
        `channel input ${JSON.stringify(input.key)} carries both scored and ` +
          "ranked items; an input is one or the other",
      );
    }
    if (scored !== undefined) {
      // An unregistered key is skipped entirely by the engine, so its orientation
      // is never read; the fallback only keeps the boundary shape total.
      const direction = directions.get(input.key) ?? Direction.HigherIsBetter;
      return { key: input.key, direction, scored };
    }
    if (ranked === undefined) {
      throw new TypeError(
        `channel input ${JSON.stringify(input.key)} carries neither scored nor ` +
          "ranked items",
      );
    }
    return { key: input.key, ranked };
  });
}

function fusedFromDto(dto: FusedDto): Fused {
  return Fused._create({
    ranking: dto.ranking,
    weights: dto.weights,
    flags: dto.flags,
    discrimination: dto.discrimination,
    confidence: dto.confidence,
    conflict: dto.conflict,
  });
}

/**
 * The entry point: fuses several retrieval channels' ranked outputs into one
 * ranking.
 *
 * A fuser holds the channel registrations, the fusion configuration, and the
 * persistent baselines it accumulates across queries. Each `fuse` call weights the
 * channels by how well each is discriminating on this query and how redundant the
 * channels are with each other, then combines them by weighted reciprocal-rank
 * fusion. A fuser is built fresh with `Fuser.create` or resumed from saved state
 * with `Fuser.resume`; `state` exposes the baselines to persist.
 *
 * A fuser owns a wasm-side allocation. `free` releases it deterministically, and
 * the class implements `Symbol.dispose`, so `using fuser = Fuser.create(...)` frees
 * it at scope exit; everything else crosses the boundary by value, so there is
 * nothing else to manage.
 */
export class Fuser {
  #core: CoreFuser;
  #freed = false;
  readonly #channels: readonly ChannelConfig[];
  readonly #config: FuseConfig;
  readonly #directions: ReadonlyMap<string, Direction>;

  private constructor(
    inner: CoreFuser,
    channels: readonly ChannelConfig[],
    config: FuseConfig,
  ) {
    this.#core = inner;
    this.#channels = [...channels];
    this.#config = config;
    this.#directions = directionMap(channels);
  }

  /**
   * Builds a fresh fuser from channel registrations and a configuration, with empty
   * starting baselines.
   *
   * Throws `ConfigError` when the configuration holds an out-of-range knob, two
   * registrations share one join-handle key, or a declared `GoodScore` does not
   * orient to a usable reference.
   */
  static create(
    channels: readonly ChannelConfig[],
    config?: FuseConfigInit,
  ): Fuser {
    const cfg = resolveConfig(config);
    try {
      return new Fuser(new core.Fuser(channelDtos(channels), cfg), channels, cfg);
    } catch (e) {
      rethrow(e);
    }
  }

  /**
   * Builds a fuser from channel registrations, a previously persisted state, and a
   * configuration, continuing to accumulate from that state.
   *
   * Resume is the live boundary a real model change crosses (a swap happens across
   * a restart), so it runs the same compatibility gate a state merge does before
   * accepting the state. Without the gate, a model swapped in behind a bumped tag
   * would silently keep accumulating into the old model's baselines, which is the
   * corruption the tag exists to prevent. Throws `ConfigError` when the
   * registrations or configuration are invalid on their own, and `ResumeError` when
   * the state is incompatible with them: a foreign format or statistic version, a
   * flipped direction, or a changed model-version tag.
   */
  static resume(
    channels: readonly ChannelConfig[],
    state: RuffleState,
    config?: FuseConfigInit,
  ): Fuser {
    const cfg = resolveConfig(config);
    try {
      return new Fuser(
        core.Fuser.resume(channelDtos(channels), cfg, state.toJson()),
        channels,
        cfg,
      );
    } catch (e) {
      rethrow(e);
    }
  }

  /**
   * Fuses one query's per-channel results into a single ranking, and folds this
   * query's readings into the running baselines.
   *
   * An input whose key is not a registered channel is skipped entirely: without a
   * registration the engine has no direction, tag, or reference to interpret the
   * channel safely, so it is ignored rather than fused at a guessed weight. When
   * one channel key appears more than once, only the first input is fused; a later
   * duplicate would double-count the channel's vote under a single weight.
   */
  fuse(inputs: readonly ChannelInput[]): Fused {
    this.#alive();
    try {
      return fusedFromDto(this.#core.fuse(inputDtos(inputs, this.#directions)));
    } catch (e) {
      rethrow(e);
    }
  }

  /**
   * Fuses one query against the given registrations and a prior state, without
   * mutating any baseline.
   *
   * This runs the same weighting and fusion as `fuse` but updates nothing. With an
   * empty prior and no declared references, every weight lands at the neutral
   * `1.0` and the fusion reduces to standard, unweighted reciprocal-rank fusion.
   * Throws `ConfigError` when the registrations or configuration are invalid, and
   * `ResumeError` when the prior is incompatible with them, which would standardize
   * this query against baselines measured under a different model or orientation.
   */
  static fuseStateless(
    inputs: readonly ChannelInput[],
    channels: readonly ChannelConfig[],
    prior: RuffleState,
    config?: FuseConfigInit,
  ): Fused {
    const cfg = resolveConfig(config);
    try {
      return fusedFromDto(
        core.Fuser.fuseStateless(
          inputDtos(inputs, directionMap(channels)),
          channelDtos(channels),
          cfg,
          prior.toJson(),
        ),
      );
    } catch (e) {
      rethrow(e);
    }
  }

  /**
   * Folds a full-scored anchor's pairwise correlations into the persistent
   * redundancy baselines.
   *
   * Each pair's correlation is accumulated into its persistent summary, giving the
   * redundancy estimate its reliability (total both-scored overlap), its point
   * estimate (the overlap-weighted pooled correlation), and its stability signal
   * (the variability across refreshes and strata). Anchor construction is an
   * offline concern; the redundancy discount itself stays off unless
   * `coupling.enabled` is set.
   */
  refreshCoupling(anchor: Anchor): void {
    this.#alive();
    try {
      this.#core.refreshCoupling(anchor._toDto());
    } catch (e) {
      rethrow(e);
    }
  }

  /**
   * A snapshot of the persistent baseline state, for serialization and inspection.
   *
   * Each access crosses into the engine and returns an independent snapshot; later
   * fuses do not mutate a previously returned state. The snapshot is restored
   * through `Fuser.resume`.
   */
  get state(): RuffleState {
    this.#alive();
    try {
      return RuffleState._fromCanonical(this.#core.stateJson());
    } catch (e) {
      rethrow(e);
    }
  }

  /** The fusion configuration in force. */
  get config(): FuseConfig {
    return this.#config;
  }

  /** The channel registrations this fuser was built with. */
  get channels(): readonly ChannelConfig[] {
    return this.#channels;
  }

  /**
   * Releases the wasm-side allocation. The fuser is unusable afterwards (every
   * method throws `RuffleError`), and a second `free` is a no-op; JS garbage
   * collection does not free wasm linear memory deterministically, so a long-lived
   * host frees fusers it is done with (or holds them with `using`).
   */
  free(): void {
    if (this.#freed) {
      return;
    }
    this.#freed = true;
    this.#core.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }

  #alive(): void {
    if (this.#freed) {
      throw new RuffleError(
        "this fuser has been freed; its engine allocation is gone",
      );
    }
  }
}
