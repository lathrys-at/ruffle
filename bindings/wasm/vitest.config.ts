import { configDefaults, defineConfig } from "vitest/config";

// The default (Node) suite. The browser suite lives under tests/browser/ and runs
// through vitest.browser.config.ts, so it is excluded here rather than run twice.
export default defineConfig({
  test: {
    exclude: [...configDefaults.exclude, "tests/browser/**"],
  },
});
