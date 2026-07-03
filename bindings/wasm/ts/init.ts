/**
 * Loads and instantiates the wasm artifact.
 *
 * The package entry awaits this at module scope (top-level await), so the module is
 * ready to use once imported, in Node 20+ and in browsers and bundlers alike. Under
 * Node the artifact is read from disk; everywhere else it is fetched from the URL
 * the bundler or browser resolves for it.
 *
 * @internal
 */

import initWasm from "../pkg/ruffle_wasm.js";

let initialized = false;

export async function initRuffle(): Promise<void> {
  if (initialized) {
    return;
  }
  const wasmUrl = new URL("../pkg/ruffle_wasm_bg.wasm", import.meta.url);
  if (wasmUrl.protocol === "file:") {
    const { readFile } = await import("node:fs/promises");
    const { fileURLToPath } = await import("node:url");
    const bytes = await readFile(fileURLToPath(wasmUrl));
    await initWasm({ module_or_path: bytes });
  } else {
    await initWasm({ module_or_path: wasmUrl });
  }
  initialized = true;
}
