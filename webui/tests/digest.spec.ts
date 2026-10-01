/* The stored digest view (loop-proof plan PR 2, graph §7).
 *
 * Where the digest push lands. Pinned: the full body renders, https links stay
 * links, the `iris:not-useful/<sender>` link becomes a 👎 button that POSTs the
 * decoded sender to /api/digest/not-useful (stream D's endpoint), a partial
 * digest says so, and a push link to a digest that is gone says that instead of
 * breaking. Runs on the iPhone WebKit project at 390px. */
import { test, expect, type Page } from "@playwright/test";
import { fixtureFor } from "./fixtures";

interface Fake {
  notUseful: unknown[];
}

async function fakeServer(page: Page): Promise<Fake> {
  const fake: Fake = { notUseful: [] };
  await page.route("**/*", async (route) => {
    const req = route.request();
    const type = req.resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const { pathname } = new URL(req.url());
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

    if (pathname === "/api/digest/not-useful" && req.method() === "POST") {
      fake.notUseful.push(JSON.parse(req.postData() ?? "{}"));
      return json({ ok: true });
    }
    if (pathname === "/api/digest/0123456789abcdef0123456789abcdef") {
      return json(fixtureFor("/api/digest/latest"));
    }
    if (pathname.startsWith("/api/digest/") && pathname !== "/api/digest/latest") {
      return json({ detail: "no digest" }, 404);
    }
    return json(fixtureFor(pathname));
  });
  return fake;
}

test("the latest digest renders in full with its links", async ({ page }) => {
  await fakeServer(page);
  await page.goto("/digest");
  const body = page.getByTestId("digest-body");
  await expect(body.getByText("Morning digest", { exact: true })).toBeVisible();
  await expect(body.getByText("Bills due")).toBeVisible();
  await expect(body.getByText("partial · 2 sections failed")).toBeVisible();
  await expect(body.getByText(/couldn't build: Portfolio/)).toBeVisible();
  await expect(body.getByText(/couldn't build: AI news/)).toBeVisible();
  const news = body.getByRole("link", { name: "cnbc.com" }).first();
  await expect(news).toHaveAttribute("href", /^https:\/\/www\.cnbc\.com\//);
  const repo = body.getByRole("link", { name: /anthropics\// }).first();
  await expect(repo).toHaveAttribute("href", /^https:\/\/github\.com\/anthropics\//);
  // No iris: link survives as a link.
  await expect(body.locator('a[href^="iris:"]')).toHaveCount(0);
  // The page never scrolls sideways on a phone.
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
});

test("the grouped digest shows its groups, sections, quiet lines and footer", async ({ page }) => {
  await fakeServer(page);
  await page.goto("/digest");
  const body = page.getByTestId("digest-body");
  // Four groups, in order, as small uppercase accent labels.
  const groups = body.getByTestId("digest-group");
  await expect(groups).toHaveText(["☀️ Today", "💳 Money", "📬 Inbox", "📰 News"]);
  await expect(groups.first()).toHaveCSS("text-transform", "uppercase");
  await expect(body.getByTestId("digest-section")).toHaveCount(5);
  // Empty sections fold into one muted line per group.
  const quiet = body.getByTestId("digest-quiet");
  await expect(quiet).toHaveText(["No reminders due · Nothing due today.", "Spending looks normal."]);
  const [quietColour, textColour] = await Promise.all([
    quiet.first().evaluate((el) => getComputedStyle(el).color),
    body.getByText("16:30–17:15 Parent-teacher meeting").evaluate((el) => getComputedStyle(el).color),
  ]);
  expect(quietColour).not.toBe(textColour);
  // A failure is named once, under its own group.
  const failed = body.getByTestId("digest-failed");
  await expect(failed).toHaveCount(2);
  await expect(failed.first()).toContainText("Portfolio");
  // The footer is last, under the rule.
  await expect(body.locator("hr + p")).toHaveText("learned yesterday: nothing");
});

test("👎 posts the decoded sender and confirms", async ({ page }) => {
  const fake = await fakeServer(page);
  await page.goto("/digest/0123456789abcdef0123456789abcdef");
  const buttons = page.getByTestId("not-useful");
  await expect(buttons).toHaveCount(5);
  // A 44px tap target that does not make its row 44px tall.
  const [tap, row] = await buttons.first().evaluate((el) => [
    el.getBoundingClientRect().height,
    (el.closest("li") ?? el).getBoundingClientRect().height,
  ]);
  expect(tap).toBeGreaterThanOrEqual(44);
  expect(row).toBeLessThan(40);
  await buttons.nth(1).click();
  await expect(page.getByTestId("not-useful-done")).toHaveCount(1);
  await expect(buttons).toHaveCount(4);
  expect(fake.notUseful).toEqual([{ sender: "offers1@fabrikam.test" }]);
});

test("a push link to a digest that is gone says so", async ({ page }) => {
  await fakeServer(page);
  await page.goto("/digest/ffffffffffffffffffffffffffffffff");
  await expect(page.getByText("That digest is no longer stored.")).toBeVisible();
  await page.getByRole("link", { name: "Show the latest" }).click();
  await expect(page.getByTestId("digest-body")).toBeVisible();
});
