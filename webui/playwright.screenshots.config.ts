/* Playwright -- the README screenshots, on demand only (issue #20).
 *
 * Not the viewport smoke (playwright.config.ts, ./tests): this config is run by
 * scripts/readme_screenshots.py against a live IRIS API on a fresh demo home, and
 * captures what that real run shows. It starts no server of its own; the script does,
 * and passes the URL, a pairing code, the welcome turn's trace ID and the output
 * directory in IRIS_SHOTS_* variables. */
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./screenshots",
  workers: 1,
  retries: 0,
  reporter: [["list"]],
  timeout: 120_000,
  expect: { timeout: 20_000 },
  use: {
    ...devices["Desktop Chrome"],
    baseURL: process.env.IRIS_SHOTS_BASE_URL,
    viewport: { width: 1280, height: 800 },
    deviceScaleFactor: 1,
  },
});
