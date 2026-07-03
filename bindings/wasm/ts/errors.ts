/** The exception hierarchy every error Ruffle raises belongs to. */

import type { BoundaryError } from "./boundary.js";

/** The base class for every error Ruffle raises. */
export class RuffleError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
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

/**
 * A serialized state document could not be parsed: it is not JSON, or not the state
 * schema.
 */
export class StateError extends RuffleError {}

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
        throw new ConfigError(e.message, { cause: e });
      case "resume":
        throw new ResumeError(e.message, { cause: e });
      case "merge":
        throw new MergeError(e.message, { cause: e });
      case "state":
        throw new StateError(e.message, { cause: e });
      case "value":
        throw new TypeError(e.message, { cause: e });
      case "internal":
        throw new RuffleError(e.message, { cause: e });
      default: {
        // Compile-time exhaustiveness over ErrorKind; at runtime a kind this
        // build does not know still lands in the hierarchy instead of escaping
        // as a raw object.
        const unhandled: never = e.kind;
        throw new RuffleError(
          `unknown error kind ${String(unhandled)}: ${e.message}`,
          { cause: e },
        );
      }
    }
  }
  throw e;
}
