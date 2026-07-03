/** The full-scored anchor for coupling estimation. */

import type { AnchorDto, DirectionValue } from "./boundary.js";
import type { ChannelConfig } from "./types.js";

/**
 * A shared evaluation set in which every candidate is scored by every channel, used
 * to estimate how redundant the channels are with each other.
 *
 * Because every candidate is scored by every channel, a `null` entry unambiguously
 * means the channel's facet does not apply to that item, rather than that the item
 * was ranked below a cutoff and dropped. The candidate set must be an unselected
 * sample (a random or whole-corpus draw) rather than any channel's top-k results;
 * restricting the candidates to a top-k pool conditions on a selection effect
 * (Berkson's paradox) that pushes the channels spuriously anti-correlated and
 * destroys the redundancy estimate. Whether a candidate set is unselected cannot be
 * checked from the ids alone, so this contract rests with the caller.
 *
 * Instances come from `Anchor.build`; the anchor is fed to `Fuser.refreshCoupling`.
 */
export class Anchor {
  readonly #channels: Array<[string, DirectionValue]>;
  readonly #rows: Array<Array<number | null>>;

  private constructor(
    channels: Array<[string, DirectionValue]>,
    rows: Array<Array<number | null>>,
  ) {
    this.#channels = channels;
    this.#rows = rows;
  }

  /**
   * Builds an anchor by scoring every `(candidate, channel)` pair.
   *
   * `score(candidateId, channelKey)` is called once for each pair. A number return
   * is the channel's native score for that candidate, oriented to higher-is-better
   * by the channel's declared direction inside the engine; a non-finite value is
   * treated as absent. A `null` or `undefined` return means the channel's facet
   * does not apply to that candidate. Coverage is structural: because the callback
   * runs for every pair, the anchor is always full-scored, and an absent score is
   * never a hidden top-k cutoff.
   */
  static build(
    candidates: readonly string[],
    channels: readonly ChannelConfig[],
    score: (candidateId: string, channelKey: string) => number | null | undefined,
  ): Anchor {
    const rows: Array<Array<number | null>> = [];
    for (const config of channels) {
      const row: Array<number | null> = [];
      for (const candidate of candidates) {
        const value = score(candidate, config.id.key);
        row.push(value === null || value === undefined ? null : value);
      }
      rows.push(row);
    }
    return new Anchor(
      channels.map((c) => [c.id.key, c.direction]),
      rows,
    );
  }

  /** @internal */
  _toDto(): AnchorDto {
    return { channels: this.#channels, rows: this.#rows };
  }
}
