/**
 * Node-only fixture loading. The pure fixture schema and builders live in
 * `helpers.ts`, which the browser suite also imports; everything touching the
 * filesystem sits here.
 */

import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { join } from "node:path";

import type { Fixture } from "./helpers.js";

export const REPO_ROOT = fileURLToPath(new URL("../../..", import.meta.url));
const FIXTURE_DIR = join(REPO_ROOT, "tests", "fixtures", "parity");

export function loadFixture(name: string): Fixture {
  return JSON.parse(readFileSync(join(FIXTURE_DIR, name), "utf-8")) as Fixture;
}

export function allFixtureNames(): string[] {
  return readdirSync(FIXTURE_DIR)
    .filter((n) => n.endsWith(".json"))
    .sort();
}
