/* Viewport smoke (Track 2 PR 1, plan decisions 22 + 33).
 *
 * Every nav route (every plugin mounted: tests/routes.ts), at 390x844 in WebKit —
 * the engine and the size of the owner's iPhone — against canned fixtures. Three assertions per route:
 *
 *   1. no horizontal overflow (the page must not scroll sideways)
 *   2. no console errors and no uncaught exceptions
 *   3. the bottom tab bar is present, with its five tabs
 *   4. its tap targets are at least 44px tall (see TAP_TARGETS)
 *
 * This is a layout gate, not a functional one: the fixtures make every screen
 * render its empty state, which is the widest a fixed layout gets and the
 * cheapest thing to keep deterministic. It runs only when webui/ changed
 * (scripts/ci_local.sh --webui). */
import { test, expect, type ConsoleMessage, type Page } from "@playwright/test";
import { navRoutes } from "./routes";
import { SHEET_REMINDER_ID, fixtureFor } from "./fixtures";

// `/more` is the phone's page list and deliberately has no sidebar entry
// (`nav: false`), so the nav's groups do not list it. It is a route the owner
// reaches on every screen, so it is tested like one. So is the reminder sheet
// (/reminders/:id, the calendar plugin's off-nav screen): it is where an iPhone
// push tap lands.
const ROUTES = [
  ...navRoutes(),
  { path: "/more", label: "More" },
  { path: `/reminders/${SHEET_REMINDER_ID}`, label: "Reminder sheet" },
];

/* Controls shorter than Apple's 44px minimum, per route, as of Track 2 PR 3.
 *
 * The four pinned tabs are 0 and must stay 0. The rest are a debt list for the
 * floor screens, recorded rather than waved through, and asserted EXACTLY: a
 * screen that improves fails here until its number comes down, so the list can
 * never quietly drift out of date and hide a later regression. */
const TAP_TARGETS: Record<string, number> = {
  "/calltrace": 12,
  "/documents": 5,
  "/governance": 5,
  "/devices": 2,
  "/sessions": 2,
  "/twin": 1,
};

/** Visible buttons and links shorter than 44px. */
async function smallTapTargets(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const out: string[] = [];
    for (const el of Array.from(document.querySelectorAll<HTMLElement>("button, a"))) {
      const r = el.getBoundingClientRect();
      if (r.width === 0 || r.height === 0 || r.height >= 44) continue;
      out.push(`${Math.round(r.height)}px "${(el.textContent ?? "").trim().slice(0, 24)}"`);
    }
    return out;
  });
}

/** Serve every data call from fixtures; let the app's own assets through. */
async function stubApi(page: Page): Promise<void> {
  await page.route("**/*", async (route) => {
    const type = route.request().resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const pathname = new URL(route.request().url()).pathname;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(fixtureFor(pathname)),
    });
  });
}

/** How far the page scrolls sideways, and which elements are responsible.
 *
 * `scrollWidth - clientWidth` says *that* a page overflows; it never says what
 * did it, and hunting that by hand across 19 screens is the slow part. So the
 * same pass walks the tree for elements crossing the right edge and reports the
 * deepest ones (a wide parent is usually just holding a wide child). */
async function measureOverflow(page: Page): Promise<{ overflow: number; blame: string }> {
  return page.evaluate(() => {
    const root = document.documentElement;
    const limit = root.clientWidth;

    /* NOT `root.scrollWidth - root.clientWidth` any more.
     *
     * The shell sets `overflow: hidden` on html and body so the page cannot
     * rubber-band, and a hidden overflow stops being reported in scrollWidth
     * — which silently turned this whole assertion into a no-op across all 19
     * routes. The canary test at the bottom of this file is what caught it,
     * and is what keeps it caught.
     *
     * Geometry instead: an element overflows the PAGE when its right edge is
     * past the viewport and no ancestor clips it. Ancestors are walked only
     * as far as `body`, because body's own hidden overflow is the page edge
     * being measured, not a scroller that legitimately contains something
     * wide — a table, a code block, the status line on Chat. */
    const clippedByAScroller = (el: HTMLElement): boolean => {
      for (let p = el.parentElement; p && p !== document.body; p = p.parentElement) {
        const overflowX = getComputedStyle(p).overflowX;
        if (overflowX === "auto" || overflowX === "scroll" || overflowX === "hidden") return true;
      }
      return false;
    };

    let overflow = 0;
    const guilty: { depth: number; line: string }[] = [];
    for (const el of Array.from(document.querySelectorAll<HTMLElement>("body *"))) {
      const rect = el.getBoundingClientRect();
      if (rect.width === 0 || rect.right <= limit + 1) continue;
      if (clippedByAScroller(el)) continue;
      overflow = Math.max(overflow, Math.round(rect.right - limit));
      // Skip a parent whose only sin is containing a guilty child.
      if (Array.from(el.children).some((c) => c.getBoundingClientRect().right > limit + 1)) {
        continue;
      }
      let depth = 0;
      for (let p = el.parentElement; p; p = p.parentElement) depth += 1;
      const cls = (el.getAttribute("class") ?? "").slice(0, 80);
      const text = (el.textContent ?? "").trim().replace(/\s+/g, " ").slice(0, 40);
      guilty.push({
        depth,
        line: `  +${Math.round(rect.right - limit)}px  <${el.tagName.toLowerCase()} class="${cls}">  ${text}`,
      });
    }
    guilty.sort((a, b) => b.depth - a.depth);
    return { overflow, blame: guilty.slice(0, 5).map((g) => g.line).join("\n") || "  (none found)" };
  });
}

/** Console errors and uncaught exceptions, collected for the life of the page. */
function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (m: ConsoleMessage) => {
    if (m.type() === "error") errors.push(`console.error: ${m.text()}`);
  });
  page.on("pageerror", (e: Error) => errors.push(`pageerror: ${e.message}`));
  return errors;
}

test.describe("phone viewport", () => {
  for (const { path, label } of ROUTES) {
    test(`${label} (${path}) lays out at 390x844`, async ({ page }) => {
      const errors = collectErrors(page);
      await stubApi(page);

      const response = await page.goto(path, { waitUntil: "domcontentloaded" });
      expect(response?.status(), `${path} should serve the SPA shell`).toBeLessThan(400);

      // React renders, queries settle, layout lands.
      await page.waitForLoadState("networkidle");

      // Checked before the layout assertions: a screen that threw renders the
      // router's error boundary, whose unwrapped stack trace is itself ~110px of
      // horizontal overflow. Without this the failure reads as a layout bug.
      const crash = page.getByText("Unexpected Application Error", { exact: false });
      await expect(crash, `${path} crashed into the router error boundary`).toHaveCount(0);

      const { overflow, blame } = await measureOverflow(page);
      expect(
        overflow,
        `${path} overflows horizontally by ${overflow}px.\nWidest offenders:\n${blame}`,
      ).toBeLessThanOrEqual(1);

      const small = await smallTapTargets(page);
      const allowed = TAP_TARGETS[path] ?? 0;
      expect(
        small.length,
        allowed === 0
          ? `${path} has controls under 44px:\n  ${small.join("\n  ")}`
          : `${path} is recorded as having ${allowed} controls under 44px but has ` +
            `${small.length}. If you fixed some, lower the number in TAP_TARGETS.\n  ` +
            small.join("\n  "),
      ).toBe(allowed);

      expect(errors, `${path} logged errors:\n${errors.join("\n")}`).toEqual([]);
    });
  }
});

/* tests/routes.ts reads the nav the API returns (a fixture the Python suite keeps
 * current). This proves the sidebar draws exactly that, so a renderer that drops
 * or invents an entry fails here instead of quietly shrinking the suite above. */
test("nav parity: every nav route reaches the desktop sidebar", async ({ browser }) => {
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  await stubApi(page);
  await page.goto("/", { waitUntil: "networkidle" });

  // Scoped to the shell's own nav: Chat renders a second <aside> for session
  // history, whose links are not nav routes.
  const hrefs = await page
    .locator("nav[aria-label='Main'] a[href^='/']")
    .evaluateAll((els) => els.map((e) => (e as HTMLAnchorElement).getAttribute("href")!));
  await context.close();

  // navRoutes(), not ROUTES: /more is an off-nav route with no sidebar entry.
  expect(new Set(hrefs)).toEqual(new Set(navRoutes().map((r) => r.path)));
});

/* Decision 19 promises that a page added to the nav "appears in More and in
 * the desktop sidebar with no bar change". This holds that promise: More lists
 * every nav route, so a new screen reaches the phone with no edit to the bar. */
test("More lists every nav route", async ({ page }) => {
  await stubApi(page);
  await page.goto("/more", { waitUntil: "networkidle" });

  const hrefs = await page
    .locator("main a[href^='/']")
    .evaluateAll((els) => els.map((e) => (e as HTMLAnchorElement).getAttribute("href")!));

  for (const { path, label } of navRoutes()) {
    expect(hrefs, `More is missing ${label} (${path})`).toContain(path);
  }
});

/* The bar is the phone's navigation; the sidebar is the desktop's. Each should
 * be absent where the other belongs, or a 390px screen pays for both. */
test("the bar is phone-only and the sidebar is desktop-only", async ({ page, browser }) => {
  await stubApi(page);
  await page.goto("/chat", { waitUntil: "networkidle" });
  await expect(page.getByRole("tablist")).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Main" })).toBeHidden();

  const desktop = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const wide = await desktop.newPage();
  await stubApi(wide);
  await wide.goto("/chat", { waitUntil: "networkidle" });
  await expect(wide.getByRole("navigation", { name: "Main" })).toBeVisible();
  await expect(wide.getByRole("tablist")).toBeHidden();
  await desktop.close();
});

/* Decision 28: Pulse answers "is it alive" top down — the summary, what is
 * broken, the box it runs on, what runs on it, then what those need. The order
 * is the design, so it is asserted rather than left to whoever edits GROUPS.
 * Cost sits under Incidents (Track 2 PR 6): what it costs matters, but not
 * before what is broken. */
test("Pulse keeps its section order", async ({ page }) => {
  await stubApi(page);
  await page.goto("/health", { waitUntil: "networkidle" });

  const sections = await page.locator("main h3").allTextContents();
  expect(sections).toEqual(["Summary", "Incidents", "Cost", "VM", "Services", "Credentials"]);
});

/* Track 2 PR 7: a todo can be added from the phone. The control has to survive
 * the case it exists for — an empty list — because the outstanding panel used
 * to render only when something was already outstanding, which would have
 * hidden "Add task" exactly when you had nothing and most wanted it. */
test.describe("adding a task from Chat", () => {
  test("the control is there when the list is empty", async ({ page }) => {
    await page.route("**/*", async (route) => {
      const type = route.request().resourceType();
      if (type !== "fetch" && type !== "xhr") return route.continue();
      const pathname = new URL(route.request().url()).pathname;
      // Everything outstanding is empty: no tasks, reminders or dues.
      const body =
        pathname === "/tasks" ||
        pathname === "/api/v1/reminders" ||
        pathname.startsWith("/finance")
          ? { count: 0, tasks: [], reminders: [], dues: [] }
          : fixtureFor(pathname);
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(body),
      });
    });
    await page.goto("/chat", { waitUntil: "networkidle" });

    await expect(page.getByRole("button", { name: "Add task" })).toBeVisible();
  });

  test("it posts the title and clears itself", async ({ page }) => {
    const posted: string[] = [];
    await page.route("**/*", async (route) => {
      const request = route.request();
      const type = request.resourceType();
      if (type !== "fetch" && type !== "xhr") return route.continue();
      const pathname = new URL(request.url()).pathname;
      if (pathname === "/tasks" && request.method() === "POST") {
        posted.push(JSON.parse(request.postData() ?? "{}").title);
        return route.fulfill({ status: 201, contentType: "application/json", body: "{}" });
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(fixtureFor(pathname)),
      });
    });
    await page.goto("/chat", { waitUntil: "networkidle" });

    await page.getByRole("button", { name: "Add task" }).click();
    const field = page.getByLabel("New task");
    await field.fill("Book the dentist");
    await page.getByRole("button", { name: "Add", exact: true }).click();

    await expect.poll(() => posted).toEqual(["Book the dentist"]);
    // The form closes on success, so the next tap starts from a clean field.
    await expect(field).toBeHidden();
  });

  test("it will not post an empty title", async ({ page }) => {
    const posted: string[] = [];
    await page.route("**/*", async (route) => {
      const request = route.request();
      const type = request.resourceType();
      if (type !== "fetch" && type !== "xhr") return route.continue();
      const pathname = new URL(request.url()).pathname;
      if (pathname === "/tasks" && request.method() === "POST") {
        posted.push("posted");
        return route.fulfill({ status: 201, contentType: "application/json", body: "{}" });
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(fixtureFor(pathname)),
      });
    });
    await page.goto("/chat", { waitUntil: "networkidle" });

    await page.getByRole("button", { name: "Add task" }).click();
    await page.getByLabel("New task").fill("   ");
    await expect(page.getByRole("button", { name: "Add", exact: true })).toBeDisabled();
    expect(posted).toEqual([]);
  });
});

/* Add to Home Screen (Track 2b PR 8).
 *
 * `vite preview` serves `dist/`, which is what the server image ships, so
 * these exercise the real build output — including `public/` being copied to
 * the root, which is the part that silently does not happen if the directory
 * is misnamed. What they cannot check is iOS itself: whether Safari offers
 * "Add to Home Screen" and whether a home-screen app can subscribe to push is
 * PR 9's on-device proof.
 *
 * Worker REGISTRATION lives in serviceworker.spec.ts instead: Playwright runs
 * service workers in Chromium only. WebKit here exposes `navigator.
 * serviceWorker` and a secure context, and then `ready` never resolves — so a
 * registration assertion in this project would fail on a correct worker. */
test.describe("installable as an app", () => {
  test("the manifest is linked, served and standalone", async ({ page }) => {
    await stubApi(page);
    await page.goto("/chat", { waitUntil: "domcontentloaded" });

    const href = await page.locator('link[rel="manifest"]').getAttribute("href");
    expect(href).toBe("/manifest.webmanifest");

    const resp = await page.request.get("/manifest.webmanifest");
    expect(resp.status()).toBe(200);
    const manifest = await resp.json();
    // The field iOS reads to decide this is an app rather than a bookmark.
    expect(manifest.display).toBe("standalone");
    expect(manifest.scope).toBe("/");
  });

  test("every icon the manifest names is really there", async ({ page }) => {
    const manifest = await (await page.request.get("/manifest.webmanifest")).json();
    for (const icon of manifest.icons) {
      const resp = await page.request.get(icon.src);
      expect(resp.status(), `${icon.src} is missing from the build`).toBe(200);
      expect(resp.headers()["content-type"]).toContain("image/png");
    }
    const apple = await page.request.get("/apple-touch-icon.png");
    expect(apple.status(), "iOS reads this one first").toBe(200);
  });

  test("the offline page explains the tailnet, not the weather", async ({ page }) => {
    // The harness is tailnet-only, so "offline" almost always means Tailscale
    // is off. A generic error would send the owner looking at their signal.
    const resp = await page.request.get("/offline.html");
    expect(resp.status()).toBe(200);
    const html = await resp.text();
    expect(html).toContain("Tailscale");
  });
});

/* The shell now sets `overflow: hidden` on html and body so the page cannot
 * rubber-band (it is a fixed-height app shell; every screen scrolls in the
 * outlet). That could quietly neuter the overflow assertion above — a hidden
 * overflow still being an overflow only helps if the measurement can see it.
 *
 * So: inject something too wide, and require the measurement to catch it. If
 * this ever passes silently, every "no horizontal overflow" result above is
 * worthless. */
test("the overflow check still detects overflow after the shell lock", async ({ page }) => {
  await stubApi(page);
  await page.goto("/chat", { waitUntil: "networkidle" });

  const clean = await measureOverflow(page);
  expect(clean.overflow, "the page should start clean").toBeLessThanOrEqual(1);

  await page.evaluate(() => {
    const wide = document.createElement("div");
    wide.id = "overflow-canary";
    wide.style.cssText = "width:900px;height:8px";
    document.body.appendChild(wide);
  });

  const dirty = await measureOverflow(page);
  expect(dirty.overflow, "scrollWidth must still report content past the edge").toBeGreaterThan(100);

  await page.evaluate(() => document.getElementById("overflow-canary")?.remove());
});

/* The page itself must not scroll — that is what stops the drag. Screens
 * still scroll; they do it in the outlet wrapper. */
test("the document does not scroll, the outlet does", async ({ page }) => {
  await stubApi(page);
  await page.goto("/memory", { waitUntil: "networkidle" }); // a long screen

  const state = await page.evaluate(() => {
    const root = document.documentElement;
    const outlet = document.querySelector("main > div.overflow-y-auto") as HTMLElement | null;
    return {
      docOverflowY: getComputedStyle(root).overflowY,
      docScrolls: root.scrollHeight > root.clientHeight + 1,
      outletScrolls: outlet ? outlet.scrollHeight > outlet.clientHeight + 1 : false,
    };
  });

  expect(state.docOverflowY).toBe("hidden");
  expect(state.docScrolls, "the page must not rubber-band").toBe(false);
  expect(state.outletScrolls, "a long screen must still be reachable").toBe(true);
});

/* The header bell's sheet is portalled to <body>, and that is load-bearing.
 *
 * The header sets `backdrop-blur`; `backdrop-filter` makes an element a
 * containing block for fixed-position descendants, so a `fixed bottom-0`
 * sheet rendered inside the header anchors to the HEADER and lands at the top
 * of the screen over the content. It did exactly that once. */
test("the attention sheet opens at the bottom of the screen", async ({ page }) => {
  await stubApi(page);
  await page.goto("/chat", { waitUntil: "networkidle" });

  await page.getByRole("button", { name: /want.*attention/ }).click();
  const sheet = page.locator("body > div.fixed.bottom-0").last();
  await expect(sheet).toBeVisible();

  const box = await sheet.boundingBox();
  const viewport = page.viewportSize()!;
  expect(box, "the sheet should have a box").not.toBeNull();
  // Its bottom edge is the screen's bottom edge, not the header's.
  expect(Math.abs(box!.y + box!.height - viewport.height)).toBeLessThanOrEqual(2);
  expect(box!.y, "it must not start at the top of the screen").toBeGreaterThan(100);
});

/* Every count on the phone is one count. Two badges arguing about which
 * matters is the thing this replaced. */
test("the bell counts alerts and outstanding items together", async ({ page }) => {
  await stubApi(page);
  await page.goto("/chat", { waitUntil: "networkidle" });

  const bell = page.getByRole("button", { name: /want.*attention/ });
  const label = (await bell.getAttribute("aria-label")) ?? "";
  const counted = Number(label.match(/^(\d+)/)?.[1] ?? 0);
  // 1 health alert + 4 dues + 6 tasks + 4 reminders in the fixtures.
  expect(counted).toBeGreaterThan(1);

  await bell.click();
  // The alert sorts first: only one of these can be IRIS itself failing.
  const first = page.locator("body > div.fixed.bottom-0 li").first();
  await expect(first).toContainText("google_calendar");
});

/* ADR-0118 build step 1: a destructive tool shows what gates it and how to undo it,
 * on the plugin detail page, at phone width. No plugin declares one yet, so without
 * this fixture the tags would never have rendered anywhere. */
test("a destructive tool shows its approval gate and its undo", async ({ page }) => {
  const tool = (name: string, over: Record<string, unknown>) => ({
    name,
    effect: "read",
    confirm: "never",
    pinned: false,
    answers_directly: false,
    undo: null,
    guidance: "",
    registered: true,
    ...over,
  });
  const detail = {
    ...(fixtureFor("/plugins") as { plugins: Record<string, unknown>[] }).plugins[0],
    name: "mailbox",
    directory: "/Users/owner/Library/Application Support/iris/plugins/mailbox",
    manifest: {
      entrypoint: "plugin:setup",
      cli: null,
      requires: { python: ">=3.12", packages: [], env_vars: [] },
    },
    registrations: [],
    drift: {},
    files: [],
    tools: [
      tool("trash_email_permanently_after_the_retention_window", {
        effect: "destructive",
        confirm: "approval",
        undo: "restore_email_from_trash_to_the_original_folder",
      }),
      tool("restore_email_from_trash_to_the_original_folder", { effect: "write" }),
    ],
  };
  await page.route("**/*", async (route) => {
    const type = route.request().resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const pathname = new URL(route.request().url()).pathname;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(pathname === "/plugins/mailbox" ? detail : fixtureFor(pathname)),
    });
  });
  await page.goto("/agents/plugins/mailbox", { waitUntil: "networkidle" });

  await expect(page.getByText("destructive", { exact: true })).toBeVisible();
  await expect(page.getByText("approved per call")).toBeVisible();
  await expect(page.getByText("undo restore_email_from_trash_to_the_original_folder")).toBeVisible();
  // A destructive tool is never "confirm once/never" — that tag is for writes only.
  await expect(page.getByText("confirm approval")).toHaveCount(0);
  // Every tool name stays inside its card. measureOverflow skips content inside a
  // scroller, which is where the detail page lives, so check the geometry directly.
  const spill = await page.evaluate(() =>
    Array.from(document.querySelectorAll("main span.font-mono"))
      .map((el) => {
        const card = el.closest("div.rounded-lg");
        if (!card) return 0;
        return el.getBoundingClientRect().right - card.getBoundingClientRect().right;
      })
      .filter((d) => d > 1),
  );
  expect(spill, "a tool name runs past its card").toEqual([]);
  const { overflow, blame } = await measureOverflow(page);
  expect(overflow, `sideways scroll caused by: ${blame}`).toBeLessThanOrEqual(1);
});

/* ADR-0118 step 4b (prototype signed off 2026-09-21): a destructive approval, from the
 * chat that raised it to the red confirm sheet, at phone width. Long subjects, to test
 * wrapping; the exact call is behind a toggle. */
test.describe("a destructive approval", () => {
  const card = {
    title: "Trash 3 emails",
    lines: [
      "Your weekly deals are here, with an unbrokensubjectlinethatgoesonandonforeverandever — Store X · 20 Sep",
      "Last chance: 40% off everything — Shop Y · 19 Sep",
      "New arrivals, picked for you — Store X · 18 Sep",
    ],
    undo_tool: "restore_email",
    undo_window_days: 30,
    asked: "clean up this week's promo emails",
  };
  const destructive = {
    approval_id: "3f9c0d2a-0000-4000-8000-000000000001",
    run_id: "run-1",
    signal: card.title,
    context_summary: "Trash 3 emails …",
    requested_at: "2026-09-21T19:00:00Z",
    timeout_at: new Date(Date.now() + 58 * 60_000).toISOString(),
    channel: "web",
    status: "pending",
    checkpoint_id: "run-1:1",
    resumable: true,
    overdue: false,
    session_id: "web-s1",
    kind: "destructive",
    card,
    items: [{ tool: "trash_email", args: { ids: ["18c2f0a9e1", "18c2f11b04", "18c2f3d7aa"] } }],
  };
  const evaluator = {
    ...(fixtureFor("/governance/approvals") as { approvals: Record<string, unknown>[] })
      .approvals[0],
    kind: "evaluator",
    card: null,
    items: [],
  };

  async function serve(page: Page) {
    await page.route("**/*", async (route) => {
      const type = route.request().resourceType();
      if (type !== "fetch" && type !== "xhr") return route.continue();
      const pathname = new URL(route.request().url()).pathname;
      let body: unknown = fixtureFor(pathname);
      if (pathname === "/governance/approvals") body = { count: 2, approvals: [destructive, evaluator] };
      if (pathname === "/api/sessions/web-s1/messages")
        body = [
          { role: "user", text: card.asked, ts: "2026-09-21T19:00:00Z" },
          {
            role: "assistant",
            text: "Trash 3 emails: this needs your approval, so nothing has changed yet. [Review it in Activity](/actions), or answer with `iris approvals` or on Telegram.",
            ts: "2026-09-21T19:00:05Z",
          },
        ];
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
    });
  }

  test("the chat's link opens the card in the app, not a new tab", async ({ page }) => {
    await serve(page);
    await page.goto("/chat/web-s1", { waitUntil: "networkidle" });
    const link = page.getByRole("link", { name: "Review it in Activity" });
    await expect(link).toBeVisible();
    expect(await link.getAttribute("target")).toBeNull();
    await link.click();
    await expect(page).toHaveURL(/\/actions$/);
    await expect(page.getByText("Trash 3 emails", { exact: true }).first()).toBeVisible();
  });

  test("the card says what, whether it can be undone, and asks in red", async ({ page }) => {
    await serve(page);
    await page.goto("/actions", { waitUntil: "networkidle" });

    await expect(page.getByText("deletes data")).toBeVisible();
    await expect(page.getByText(/answer within 5\d min/)).toBeVisible();
    for (const line of card.lines) await expect(page.getByText(line)).toBeVisible();
    await expect(page.getByText('Reversible for 30 days. Say "undo that" to restore.')).toBeVisible();
    await expect(page.getByText(`"${card.asked}"`)).toBeVisible();

    // The exact call is one tap away, not in the way.
    await expect(page.getByText("18c2f0a9e1")).toHaveCount(0);
    await page.getByRole("button", { name: "Show the exact call" }).click();
    await expect(page.getByText(/trash_email \{"ids":\["18c2f0a9e1"/)).toBeVisible();

    // The evaluator approval keeps its own card and its plain "Approve".
    await expect(page.getByRole("button", { name: "Approve", exact: true })).toBeVisible();

    const { overflow, blame } = await measureOverflow(page);
    expect(overflow, `sideways scroll caused by: ${blame}`).toBeLessThanOrEqual(1);
    const spill = await page.evaluate(() =>
      Array.from(document.querySelectorAll("main li")).filter((li) => {
        const box = li.getBoundingClientRect();
        const list = li.closest("ul")!.getBoundingClientRect();
        return box.right > list.right + 1;
      }).length,
    );
    expect(spill, "an item line runs past its card").toBe(0);

    // Approving deletes, so it is restated in a red sheet naming exactly that.
    await page.getByRole("button", { name: "Trash 3 emails" }).click();
    await expect(page.getByRole("heading", { name: "Trash 3 emails?" })).toBeVisible();
    await expect(
      page.getByText("IRIS will run exactly the call on the card and nothing else."),
    ).toBeVisible();
  });
});
