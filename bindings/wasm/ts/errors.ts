/** The exception hierarchy every error Ruffle raises belongs to. */

import type { BoundaryError } from "./boundary.js";

/** The base class for every error Ruffle raises. */
export class RuffleError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

/**
 * The channel registrations or the fusion configuration are invalid on their own:
 * an out-of-range knob, a duplicate channel key, or a declared good score that does
 * not orient to a usable reference.
 */
export class ConfigError extends RuffleError {}

/**
 * A persisted state is incompatible with the registrations or with this build: a
 * format or statistic-version mismatch, a flipped direction, or a changed
 * model-version tag (the signature of a model swap).
 */
export class ResumeError extends RuffleError {}

/**
 * Two states cannot be merged: they disagree on format, statistic definitions, a
 * channel's orientation, or a channel's model-version tag, or the merge received no
 * states at all.
 */
export class MergeError extends RuffleError {}

function isBoundaryError(e: unknown): e is BoundaryError {
  return (
    typeof e === "object" &&
    e !== null &&
    typeof (e as { kind?: unknown }).kind === "string" &&
    typeof (e as { message?: unknown }).message === "string"
  );
}

/**
 * Maps a value thrown across the wasm boundary to the typed hierarchy and rethrows.
 *
 * @internal
 */
export function rethrow(e: unknown): never {
  if (isBoundaryError(e)) {
    switch (e.kind) {
      case "config":
        throw new ConfigError(e.message);
      case "resume":
        throw new ResumeError(e.message);
      case "merge":
        throw new MergeError(e.message);
      case "value":
        throw new TypeError(e.message);
      case "internal":
        throw new RuffleError(e.message);
    }
  }
  throw e;
}
