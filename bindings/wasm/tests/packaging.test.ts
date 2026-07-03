/**
 * Packaging assertions over the exact tarball `npm publish` would ship: the license
 * texts both licenses require in redistributions, the compiled surface, the wasm
 * artifact and its size, and the TypeScript sources the shipped source maps
 * reference.
 */

import { execFileSync } from "node:child_process";
import { statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { expect, test } from "vitest";

const PACKAGE_ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

// A dependency bump that balloons the artifact should fail loudly, not ship
// silently; the release build runs wasm-opt -Oz and currently lands well under
// this ceiling.
const WASM_SIZE_CEILING = 600_000;

interface PackEntry {
  readonly path: string;
}
interface PackReport {
  readonly files: readonly PackEntry[];
}

function packedPaths(): Set<string> {
  const out = execFileSync("npm", ["pack", "--dry-run", "--json"], {
    cwd: PACKAGE_ROOT,
    encoding: "utf8",
    // npm mixes progress output into stderr; stdout carries the JSON report.
    stdio: ["ignore", "pipe", "ignore"],
  });
  const report = JSON.parse(out) as PackReport[];
  return new Set(report[0]!.files.map((f) => f.path));
}

test("the published tarball carries the license texts and the full surface", () => {
  const paths = packedPaths();
  for (const required of [
    "LICENSE-MIT",
    "LICENSE-APACHE",
    "README.md",
    "package.json",
    "dist/index.js",
    "dist/index.d.ts",
    "ts/index.ts",
    "pkg/ruffle_wasm.js",
    "pkg/ruffle_wasm_bg.wasm",
  ]) {
    expect(paths, `missing ${required}`).toContain(required);
  }
});

test("the wasm artifact stays under the size ceiling", () => {
  const size = statSync(join(PACKAGE_ROOT, "pkg", "ruffle_wasm_bg.wasm")).size;
  expect(size).toBeGreaterThan(0);
  expect(size).toBeLessThan(WASM_SIZE_CEILING);
});
