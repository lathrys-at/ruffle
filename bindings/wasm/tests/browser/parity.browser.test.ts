/**
 * The parity replay in a real browser: the same golden fixtures, the same exact
 * assertions as the Node suite, run in headless Chromium via vitest browser mode.
 *
 * This exercises the load path Node never touches: the module's top-level await
 * resolves the wasm artifact from `import.meta.url` and fetches it over HTTP, the
 * branch every browser, bundler, and edge deployment takes. The fixtures arrive
 * through Vite's raw imports, since a browser has no filesystem.
 */

import { describe, expect, test } from "vitest";

import { version } from "../../ts/index.js";
import type { Fixture } from "../helpers.js";
import { replayFixture } from "../replay.js";

const RAW_FIXTURES = import.meta.glob<string>(
  "../../../../tests/fixtures/parity/*.json",
  { query: "?raw", import: "default", eager: true },
);

const FIXTURES = Object.entries(RAW_FIXTURES)
  .map(([path, raw]) => {
    const name = path.split("/").at(-1)!;
    return [name, JSON.parse(raw) as Fixture] as const;
  })
  .sort(([a], [b]) => a.localeCompare(b));

// The project's lib deliberately excludes DOM, so the browser globals are read
// through a typed view of globalThis rather than by widening every module's
// ambient environment.
const browserGlobals = globalThis as unknown as {
  window?: unknown;
  location?: { href: string };
};

test("the module loaded over HTTP in a real browser", () => {
  expect(typeof browserGlobals.window).toBe("object");
  expect(new URL(browserGlobals.location!.href).protocol).toMatch(/^https?:$/);
  expect(version).toMatch(/^\d+\.\d+\.\d+/);
});

describe("parity fixtures replay bit-exact in the browser", () => {
  expect(FIXTURES.length).toBeGreaterThanOrEqual(8);
  test.each(FIXTURES)("%s", (_name, fixture) => {
    replayFixture(fixture);
  });
});
