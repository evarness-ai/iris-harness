/* The service worker registers and claims the page (Track 2b PR 8).
 *
 * CHROMIUM, not WebKit, and not by preference. Playwright executes service
 * workers in Chromium only: its WebKit exposes `navigator.serviceWorker` and
 * reports a secure context, then never resolves `ready`. Asserting
 * registration there fails on a perfectly good worker, which is worse than
 * not asserting it.
 *
 * So this proves the worker is registerable and root-scoped, on the one engine
 * that can run it. Whether *iOS* delivers push to it is not something any
 * local runner can answer — that is PR 9's on-device proof, and the plan says
 * PR 10 does not start until the owner has seen a banner on the real phone.
 *
 * Registration matters now because on iOS push is delivered to a service
 * worker or not at all. */
import { test, expect, devices } from "@playwright/test";
import { fixtureFor } from "./fixtures";

// Same viewport as the WebKit project, so a layout-dependent failure here
// would still be recognisable — but Chromium's engine.
test.use({ ...devices["Pixel 7"] });

test("the worker registers at the origin root and controls the page", async ({ page }) => {
  await page.route("**/*", async (route) => {
    const type = route.request().resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(fixtureFor(new URL(route.request().url()).pathname)),
    });
  });
  await page.goto("/chat", { waitUntil: "load" });

  const scope = await page.evaluate(async () => (await navigator.serviceWorker.ready).scope);
  // A worker only controls paths at or below where it was served. Anything
  // narrower than the root could not handle a navigation to /chat.
  expect(scope.endsWith("/"), `scope must be the origin root, got ${scope}`).toBe(true);

  // `controller` set proves it claimed this page, not merely that the file
  // parsed — `clients.claim()` in the activate handler is what does that.
  await page.waitForFunction(() => navigator.serviceWorker.controller !== null);
});
