/* Real data only: after an install the console shows what the API answers, nothing else.
 *
 * The console once fell back to canned traces and sessions when the API was down (and,
 * for sessions, when it answered an empty list), so a fresh install or a stopped API
 * showed someone else's conversations under a small "mock data" badge. Now:
 *
 * - API up, nothing recorded yet: every screen renders its empty state, and the
 *   conversation screens say to start a chat;
 * - API unreachable: the "API unavailable" notice that names `iris serve`.
 *
 * Neither path may show any of the strings the old canned traces carried. */
import { test, expect, type Page } from "@playwright/test";
import { fixtureFor } from "./fixtures";
import { WELCOME, freshFor } from "./fresh-install";
import { navRoutes } from "./routes";

/** Distinctive strings from the canned traces the console used to ship. */
const CANNED = [
  "mock data",
  "what time is it today?",
  "refresh AI news and give me an updated pdf",
  "delete all my emails",
  "079d3fd77f7d",
  "c41a9b2e7f30",
  "a7e0c1d94b22",
];

const SCREENS = [...navRoutes("personal-assistant").map((r) => r.path), "/knowledge"];

const UNAVAILABLE = "API unavailable";

function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  return errors;
}

async function serveFresh(page: Page): Promise<void> {
  await page.route("**/*", async (route) => {
    const type = route.request().resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const body = freshFor(new URL(route.request().url()).pathname);
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
}

/** The API is down: every data call fails at the network, as with no `iris serve`.
 * The nav and this device are answered so the shell renders the screen under test; a
 * console with no nav at all is the shell's concern, not a screen's. */
async function serveDown(page: Page): Promise<void> {
  await page.route("**/*", async (route) => {
    const type = route.request().resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const pathname = new URL(route.request().url()).pathname;
    if (pathname === "/api/v1/webui/nav" || pathname === "/api/v1/devices/me") {
      const body = fixtureFor(pathname);
      return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
    }
    await route.abort("connectionrefused");
  });
}

async function expectNoCanned(page: Page): Promise<void> {
  const text = await page.locator("body").innerText();
  for (const s of CANNED) expect(text, `canned "${s}"`).not.toContain(s);
}

for (const path of SCREENS) {
  test(`fresh install: ${path} shows no canned data and no outage`, async ({ page }) => {
    const errors = collectErrors(page);
    await serveFresh(page);
    await page.goto(path, { waitUntil: "networkidle" });
    await expect(page.locator("main")).toBeVisible();
    await expect(page.getByText("Unexpected Application Error")).toHaveCount(0);
    await expect(page.getByText(UNAVAILABLE)).toHaveCount(0);
    await expectNoCanned(page);
    expect(errors).toEqual([]);
  });

  test(`API down: ${path} says so and shows no canned data`, async ({ page }) => {
    const errors = collectErrors(page);
    await serveDown(page);
    await page.goto(path, { waitUntil: "networkidle" });
    await expect(page.locator("main")).toBeVisible();
    await expect(page.getByText("Unexpected Application Error")).toHaveCount(0);
    // Chat's outage notice is in its history rail, which a phone does not show; the
    // desktop case is its own test below.
    if (path !== "/chat") {
      const notice = page.locator("main").getByText(UNAVAILABLE).first();
      await expect(notice).toBeVisible();
      await expect(notice).toContainText("iris serve");
    }
    await expectNoCanned(page);
    expect(errors).toEqual([]);
  });
}

test("fresh install: Sessions says to start a chat", async ({ page }) => {
  await serveFresh(page);
  await page.goto("/sessions", { waitUntil: "networkidle" });
  const empty = page.getByTestId("sessions-empty");
  await expect(empty).toContainText("No conversations yet");
  const open = empty.getByRole("link", { name: "Open Chat" });
  expect(await open.getAttribute("href")).toBe("/chat");
  const box = await open.boundingBox();
  expect(box?.height ?? 0, "Open Chat is a phone tap target").toBeGreaterThanOrEqual(44);
});

/** A fresh install that records what Chat's opening does: POST /chat/welcome runs the
 * welcome turn (ADR-0127) once, and from then on the API lists that one session, as
 * the real harness does. Before it, the lists are empty. */
async function serveFreshThenWelcomed(page: Page): Promise<{ welcomeCalls: number }> {
  const state = { welcomeCalls: 0 };
  const session = {
    session_id: WELCOME.session_id,
    title: "First-chat welcome",
    turn_count: 1,
    started_at: WELCOME.at,
    last_at: WELCOME.at,
    total_tokens: 0,
    total_duration_ms: 3.2,
    turns: [
      {
        session_id: WELCOME.session_id,
        trace_id: WELCOME.trace_id,
        request: "First-chat welcome",
        started_at: WELCOME.at,
        total_duration_ms: 3.2,
        total_tokens: 0,
      },
    ],
  };
  await page.route("**/*", async (route) => {
    const type = route.request().resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const pathname = new URL(route.request().url()).pathname;
    let body: unknown = freshFor(pathname);
    if (pathname === "/chat/welcome") {
      state.welcomeCalls += 1;
      body = { ...WELCOME, created: state.welcomeCalls === 1 };
    } else if (state.welcomeCalls > 0 && pathname === "/api/sessions") {
      body = [session];
    } else if (state.welcomeCalls > 0 && pathname === `/api/sessions/${WELCOME.session_id}/messages`) {
      body = [{ role: "assistant", text: WELCOME.response, ts: WELCOME.at, trace_id: WELCOME.trace_id }];
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  return state;
}

test("fresh install: Chat opens on IRIS's welcome, not canned data", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 }); // the history rail is md+
  const state = await serveFreshThenWelcomed(page);
  await page.goto("/chat", { waitUntil: "networkidle" });

  await expect(page.locator("main").getByText("Here is what I can do:")).toBeVisible();
  await expect(page.locator("main").getByText("Open Call trace to see")).toBeVisible();
  await expect(page).toHaveURL(new RegExp(`/chat/${WELCOME.session_id}$`));
  // The history lists the welcome's session: the conversation exists now.
  await expect(page.getByText("No conversations yet.")).toHaveCount(0);
  await expect(page.getByText("First-chat welcome").first()).toBeVisible();
  expect(state.welcomeCalls).toBe(1);
  await expectNoCanned(page);
});

test("a welcome that ran before is not shown again, and Chat asks once per page", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  let calls = 0;
  await page.route("**/*", async (route) => {
    const type = route.request().resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const pathname = new URL(route.request().url()).pathname;
    let body: unknown = freshFor(pathname);
    if (pathname === "/chat/welcome") {
      calls += 1;
      body = { ...WELCOME, created: false }; // it ran on another surface already
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  await page.goto("/chat", { waitUntil: "networkidle" });
  await expect(page.getByText("Here is what I can do:")).toHaveCount(0);
  await expect(page).toHaveURL(/\/chat$/);
  // Leave Chat and come back inside the app: Chat mounts again, the request does not.
  await page.getByRole("link", { name: "Sessions" }).first().click();
  await expect(page).toHaveURL(/\/sessions$/);
  await page.getByRole("link", { name: "Chat" }).first().click();
  await expect(page).toHaveURL(/\/chat/);
  await page.waitForLoadState("networkidle");
  expect(calls).toBe(1);
});

for (const path of ["/sessions", "/calltrace"]) {
  test(`API down: ${path} says the API is unavailable and names iris serve`, async ({ page }) => {
    await serveDown(page);
    await page.goto(path, { waitUntil: "networkidle" });
    const notice = page.getByText(UNAVAILABLE);
    await expect(notice).toBeVisible();
    await expect(notice).toContainText("iris serve");
    await expect(page).toHaveURL(new RegExp(`${path}$`));
  });
}

test("API down: Chat's history says the API is unavailable", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await serveDown(page);
  await page.goto("/chat", { waitUntil: "networkidle" });
  await expect(page.getByText(UNAVAILABLE).first()).toContainText("iris serve");
  await expectNoCanned(page);
});

test("a trace id the API does not know is said so, not loaded forever", async ({ page }) => {
  await page.route("**/*", async (route) => {
    const type = route.request().resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const pathname = new URL(route.request().url()).pathname;
    if (pathname.startsWith("/api/traces/")) {
      return route.fulfill({ status: 404, contentType: "application/json", body: '{"detail":"trace not found"}' });
    }
    const body =
      pathname === "/api/traces"
        ? [{ session_id: "s1", trace_id: "s1~0", request: "hello", started_at: "2026-10-01T09:00:00Z", total_duration_ms: 900, total_tokens: 120 }]
        : freshFor(pathname);
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  await page.goto("/calltrace/gone~3", { waitUntil: "networkidle" });
  await expect(page.getByText("No such trace")).toBeVisible();
  await expect(page.getByText("Loading trace")).toHaveCount(0);
});
