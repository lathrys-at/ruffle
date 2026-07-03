/**
 * Cross-implementation interoperability: states written by this binding reconcile
 * under the Rust crate's command-line tool, and the CLI's output loads back here.
 *
 * Requires a Rust toolchain; skipped when cargo is unavailable.
 */

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "vitest";

import {
  Direction,
  Fuser,
  MergePolicy,
  RuffleState,
  type ChannelConfig,
} from "../ts/index.js";
import { REPO_ROOT } from "./node-fixtures.js";

function hasCargo(): boolean {
  try {
    execFileSync("cargo", ["--version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function makeState(offset: number): RuffleState {
  const semantic: ChannelConfig = {
    id: { key: "semantic", tag: "v1" },
    direction: Direction.HigherIsBetter,
  };
  const fuser = Fuser.create([semantic]);
  try {
    const pool: Array<[string, number]> = [];
    for (let i = 0; i < 30; i++) {
      pool.push([`doc${String(i).padStart(3, "0")}`, 0.01 * (i + offset)]);
    }
    pool.push(["hit0", 10.0], ["hit1", 10.5]);
    fuser.fuse([{ key: "semantic", scored: pool }]);
    return fuser.state;
  } finally {
    fuser.free();
  }
}

test.skipIf(!hasCargo())(
  "TypeScript-written states reconcile under the Rust CLI",
  () => {
    const a = makeState(0);
    const b = makeState(100);
    const dir = mkdtempSync(join(tmpdir(), "ruffle-interop-"));
    try {
      writeFileSync(join(dir, "a.json"), a.toJson(), "utf-8");
      writeFileSync(join(dir, "b.json"), b.toJson(), "utf-8");
      const out = join(dir, "merged.json");

      execFileSync(
        "cargo",
        [
          "run",
          "--quiet",
          "--features",
          "cli",
          "--bin",
          "ruffle",
          "--",
          "reconcile",
          join(dir, "a.json"),
          join(dir, "b.json"),
          "-o",
          out,
        ],
        { cwd: REPO_ROOT, stdio: ["ignore", "pipe", "pipe"] },
      );

      const cliMerged = RuffleState.fromJson(readFileSync(out, "utf-8"));
      const [tsMerged] = RuffleState.merge([a, b], MergePolicy.Strict);
      // Byte-for-byte: both implementations run the same merge on the same code.
      expect(cliMerged.toJson()).toBe(tsMerged.toJson());
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  },
  120_000,
);
