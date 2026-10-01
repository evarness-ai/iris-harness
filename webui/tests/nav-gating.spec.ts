/* Plugin-gated navigation (OSS plan R17), in the release-1 shape.
 *
 * The API is stubbed with the nav a core + email install returns
 * (nav.core-email.json, written by the Python suite): Inbox is there because the
 * email plugin is mounted; Portfolio, Documents and the reminder sheet are not,
 * because their plugins are installed but not in the profile. The console must draw
 * the first and never the others -- and a direct link to one of the others must say
 * which plugin is missing instead of drawing a screen with no API behind it. */
import { test, expect, type Page } from "@playwright/test";
import { fixtureFor } from "./fixtures";
import { navFixture, navRoutes } from "./routes";

const CORE_EMAIL = navFixture("core-email");

async function serveCoreEmail(page: Page): Promise<string[]> {
  const called: string[] = [];
  await page.route("**/*", async (route) => {
    const type = route.request().resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const pathname = new URL(route.request().url()).pathname;
    called.push(pathname);
    const body = pathname === "/api/v1/webui/nav" ? CORE_EMAIL : fixtureFor(pathname);
    await route.fulfill({ json: body });
  });
  return called;
}

test("the sidebar is the core + email nav: Inbox in, the private domains out", async ({
  browser,
}) => {
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  await serveCoreEmail(page);
  await page.goto("/chat", { waitUntil: "networkidle" });

  const sidebar = page.getByRole("navigation", { name: "Main" });
  const hrefs = await sidebar
    .locator("a[href^='/']")
    .evaluateAll((els) => els.map((e) => (e as HTMLAnchorElement).getAttribute("href")!));
  expect(new Set(hrefs)).toEqual(new Set(navRoutes("core-email").map((r) => r.path)));
  expect(hrefs).toContain("/inbox");
  for (const gone of ["/portfolio", "/documents", "/reminders"]) expect(hrefs).not.toContain(gone);
  // No plugin screen left in it, so the Apps group is not drawn at all.
  await expect(sidebar.getByRole("button", { name: "Apps" })).toHaveCount(0);
  await context.close();
});

test("More on the phone lists the same core + email nav", async ({ page }) => {
  await serveCoreEmail(page);
  await page.goto("/more", { waitUntil: "networkidle" });
  const hrefs = await page
    .locator("main a[href^='/']")
    .evaluateAll((els) => els.map((e) => (e as HTMLAnchorElement).getAttribute("href")!));
  expect(hrefs).toContain("/inbox");
  expect(hrefs).not.toContain("/portfolio");
  expect(hrefs).not.toContain("/documents");
});

// `api`: the call the screen itself makes on mount. (The shell's attention bell reads
// reminders and dues on every page, tolerating their absence; that is not a screen.)
for (const { route, plugin, api } of [
  { route: "/portfolio", plugin: "finance_workflows", api: "/portfolio" },
  { route: "/documents", plugin: "file_organizer", api: "/rag/documents" },
  { route: "/reminders/r-1", plugin: "calendar", api: "/api/v1/reminders/r-1" },
]) {
  test(`a direct link to ${route} says ${plugin} is not installed`, async ({ page }) => {
    const called = await serveCoreEmail(page);
    await page.goto(route, { waitUntil: "networkidle" });
    const notice = page.getByRole("status").filter({ hasText: "isn't available" });
    await expect(notice).toBeVisible();
    await expect(notice).toContainText(plugin);
    await expect(notice).toContainText("not in the profile");
    // The screen itself never mounted, so it asked its missing API for nothing.
    expect(called).not.toContain(api);
  });

  // The control for the assertion above: with its plugin mounted, the same link
  // draws the screen, and the screen does make that call.
  test(`with ${plugin} mounted, ${route} draws its screen`, async ({ page }) => {
    const called: string[] = [];
    await page.route("**/*", async (r) => {
      const type = r.request().resourceType();
      if (type !== "fetch" && type !== "xhr") return r.continue();
      const pathname = new URL(r.request().url()).pathname;
      called.push(pathname);
      await r.fulfill({ json: fixtureFor(pathname) });
    });
    await page.goto(route, { waitUntil: "networkidle" });
    await expect(page.getByRole("status").filter({ hasText: "isn't available" })).toHaveCount(0);
    expect(called).toContain(api);
  });
}

test("a route no installed plugin declares shows the generic notice", async ({ page }) => {
  await serveCoreEmail(page);
  // /portfolio with an install that does not even have the finance plugin.
  await page.route("**/api/v1/webui/nav", (route) =>
    route.fulfill({ json: { ...CORE_EMAIL, unavailable: [] } }),
  );
  await page.goto("/portfolio", { waitUntil: "networkidle" });
  await expect(
    page.getByRole("status").filter({ hasText: "isn't part of this install" }),
  ).toBeVisible();
});

test("a mounted plugin's screen renders normally", async ({ page }) => {
  await serveCoreEmail(page);
  await page.goto("/inbox", { waitUntil: "networkidle" });
  await expect(page.getByRole("status").filter({ hasText: "isn't available" })).toHaveCount(0);
  await expect(page.getByRole("heading", { level: 1 })).toHaveText("Inbox");
});
