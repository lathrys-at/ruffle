import { fileURLToPath } from "node:url";

import { defineConfig } from "vitest/config";

// The browser suite: the parity replay in headless Chromium, exercising the
// fetch-from-import.meta.url wasm load path that Node never takes. The dev
// server's filesystem allowlist reaches up to the repository root so the shared
// parity fixtures can be imported.
export default defineConfig({
  server: {
    fs: {
      allow: [fileURLToPath(new URL("../..", import.meta.url))],
    },
  },
  test: {
    include: ["tests/browser/**/*.test.ts"],
    browser: {
      enabled: true,
      headless: true,
      provider: "playwright",
      screenshotFailures: false,
      instances: [{ browser: "chromium" }],
    },
  },
});
