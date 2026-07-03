/**
 * The typing gate: the package and its tests pass `tsc --strict` (the project's
 * tsconfig, which sets strict and noUncheckedIndexedAccess).
 *
 * The package ships curated declarations; this test keeps that promise enforced
 * from the test suite itself, so a hole in the annotations fails CI the same way a
 * behavioural regression does.
 */

import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { expect, test } from "vitest";

const PROJECT_DIR = fileURLToPath(new URL("..", import.meta.url));

test("tsc --noEmit passes on the package and tests", () => {
  const run = (): string =>
    execFileSync("npx", ["tsc", "--noEmit"], {
      cwd: PROJECT_DIR,
      encoding: "utf-8",
    });
  expect(run).not.toThrow();
});
