/* Playwright — the viewport smoke only (Track 2 PR 1, decision 22).
 *
 * WebKit at iPhone 13's 390x844, because the target is the owner's iPhone and
 * Safari is the engine that will actually render this. One project, no component
 * suite, no visual snapshots.
 *
 * The app under test is the built `dist` served by `vite preview`, not the dev
 * server: preview is what the server image ships (iris_api serves the same
 * bundle), and it needs no backend because tests/fixtures.ts answers every data
 * call. */
import { defineConfig, devices } from "@playwright/test";

const PORT = Number(process.env.IRIS_SMOKE_PORT ?? 4180);

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0, // a flaky layout test is a bug in the test; let it show
  reporter: process.env.CI ? [["list"], ["github"]] : [["list"]],
  timeout: 30_000,
  expect: { timeout: 5_000 },

  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "iphone-webkit",
      use: { ...devices["iPhone 13"] }, // 390x844, WebKit, touch, mobile UA
      testIgnore: /serviceworker\.spec\.ts/,
    },
    {
      // Service workers only. Playwright runs them in Chromium alone — its
      // WebKit exposes the API, reports a secure context, and then never
      // resolves `ready`, so the same assertion fails there on a correct
      // worker. One test, one engine, for the one thing WebKit cannot do.
      name: "chromium-serviceworker",
      use: { ...devices["Pixel 7"] },
      testMatch: /serviceworker\.spec\.ts/,
    },
  ],

  webServer: {
    // --host 127.0.0.1 is load-bearing: `vite preview` otherwise binds "localhost",
    // which resolves to ::1 first on macOS, and the readiness poll below never lands.
    command: `npm run build && npm run preview -- --port ${PORT} --strictPort --host 127.0.0.1`,
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
