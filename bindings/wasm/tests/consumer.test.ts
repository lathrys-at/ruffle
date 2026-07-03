/**
 * The consumer's view of the published package: `npm pack` builds the exact
 * tarball a release would ship, it installs into a scratch project, a strict
 * consumer snippet type-checks against the packaged declarations through the
 * `exports`/`types` map, and the same package then loads and fuses under Node.
 * This is the lane that catches a broken types path, an exports-map regression,
 * or a tarball missing a file the runtime needs — failures the in-repo suites
 * cannot see because they import the sources directly.
 */

import { execFileSync } from "node:child_process";
import { mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { afterAll, beforeAll, expect, test } from "vitest";

const PACKAGE_ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

let scratch: string;

// A consumer-shaped tsconfig: strict, NodeNext resolution (the mode that honors
// the exports map), skipLibCheck on as in a typical consumer project — the
// declarations' own soundness is the repo typecheck's job; this checks that they
// resolve and type the consumer's code.
const CONSUMER_TSCONFIG = {
  compilerOptions: {
    target: "ES2022",
    lib: ["ES2023"],
    module: "NodeNext",
    moduleResolution: "NodeNext",
    strict: true,
    noEmit: true,
    skipLibCheck: true,
  },
  include: ["consumer.ts"],
};

const CONSUMER_TS = `
import {
  Direction,
  Fuser,
  RuffleState,
  StateError,
  type ChannelConfig,
  type Fused,
  type FusedJson,
} from "@lathrys-at/ruffle";

const semantic: ChannelConfig = {
  id: { key: "semantic", tag: "v1" },
  direction: Direction.HigherIsBetter,
};

const fuser = Fuser.create([semantic], { coupling: { enabled: false } });
const fused: Fused = fuser.fuse([
  { key: "semantic", scored: [["doc-1", 0.9], ["doc-2", 0.4]] },
]);

const ranking: ReadonlyArray<readonly [string, number]> = fused.ranking;
const weight: number | undefined = fused.weights.get("semantic");
const asJson: FusedJson = fused.toJSON();

// @ts-expect-error the result maps are read-only
fused.weights.set("semantic", 2);

const state: RuffleState = fuser.state;
const reloaded = RuffleState.fromJson(state.toJson());
const renamed: RuffleState = reloaded.rekey("semantic", "dense");

if (!(StateError.prototype instanceof Error)) {
  throw new TypeError("StateError must extend Error");
}

fuser.free();
export { ranking, weight, asJson, renamed };
`;

const CONSUMER_RUN = `
import { Direction, Fuser, RuffleState, version } from "@lathrys-at/ruffle";

const semantic = { id: { key: "semantic", tag: "v1" }, direction: Direction.HigherIsBetter };
const fuser = Fuser.create([semantic]);
const fused = fuser.fuse([{ key: "semantic", scored: [["doc-1", 0.9], ["doc-2", 0.4]] }]);
const state = RuffleState.fromJson(fuser.state.toJson());
fuser.free();

if (fused.ranking[0][0] !== "doc-1") throw new Error("wrong ranking");
if (!state.channels.has("semantic")) throw new Error("state missing channel");
console.log(JSON.stringify({ version, top: fused.ranking[0][0] }));
`;

beforeAll(() => {
  scratch = mkdtempSync(join(tmpdir(), "ruffle-consumer-"));

  execFileSync("npm", ["pack", "--pack-destination", scratch], {
    cwd: PACKAGE_ROOT,
    stdio: ["ignore", "pipe", "ignore"],
  });
  const tarball = readdirSync(scratch).find((f) => f.endsWith(".tgz"));
  expect(tarball).toBeDefined();

  writeFileSync(
    join(scratch, "package.json"),
    JSON.stringify({ name: "consumer", private: true, type: "module" }),
  );
  writeFileSync(
    join(scratch, "tsconfig.json"),
    JSON.stringify(CONSUMER_TSCONFIG),
  );
  writeFileSync(join(scratch, "consumer.ts"), CONSUMER_TS);
  writeFileSync(join(scratch, "consumer-run.mjs"), CONSUMER_RUN);

  // The package has no runtime dependencies, so installing the local tarball
  // needs no registry access.
  execFileSync(
    "npm",
    ["install", "--no-audit", "--no-fund", "--ignore-scripts", tarball!],
    { cwd: scratch, stdio: ["ignore", "pipe", "pipe"] },
  );
}, 120_000);

afterAll(() => {
  rmSync(scratch, { recursive: true, force: true });
});

test("a strict consumer type-checks against the packaged declarations", () => {
  const tsc = join(PACKAGE_ROOT, "node_modules", ".bin", "tsc");
  const run = (): string =>
    execFileSync(tsc, ["-p", scratch], { cwd: scratch, encoding: "utf-8" });
  expect(run).not.toThrow();
}, 120_000);

test("the installed package loads and fuses under Node", () => {
  const out = execFileSync(process.execPath, [join(scratch, "consumer-run.mjs")], {
    cwd: scratch,
    encoding: "utf-8",
  });
  const result = JSON.parse(out) as { version: string; top: string };
  expect(result.top).toBe("doc-1");
  expect(result.version).toMatch(/^\d+\.\d+\.\d+/);
}, 120_000);
