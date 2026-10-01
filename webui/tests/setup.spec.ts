/* Email setup on the web (OSS plan R4 + R17).
 *
 * The API is a small stateful fake of /api/v1/email/onboarding, so each test asserts
 * what the page SENT as well as what it showed: Continue posts `until_waiting` and no
 * default-taking flag, step 6 posts to approve-writes only on the owner's click (and
 * never on load), a restart needs the in-page confirmation, and a slow step can be
 * submitted once. The connect command is copied, or selected when the clipboard refuses.
 * All accounts are made up. */
import { test, expect, type Page, type Route } from "@playwright/test";
import { fixtureFor } from "./fixtures";
import { navFixture, type NavFixture } from "./routes";
import {
  PROPOSALS,
  labelPreview,
  onboardingPath,
  overview,
  setupAt,
  waitingAtApproval,
  type SetupFixture,
} from "./setup-fixtures";

type Json = Record<string, unknown>;

const ACCOUNT = "gmail:owner@example.com";
const BASE = "/api/v1/email/onboarding";

interface Posted {
  path: string;
  body: Json;
}

function fakeSetupApi(initial: SetupFixture[], opts: { writes?: boolean; accounts?: string[] } = {}) {
  let setups = [...initial];
  const posted: Posted[] = [];
  let gets = 0;
  let gate: Promise<void> | null = null;
  let fail: { status: number; detail: string } | null = null;
  /** What the next advance answers with (the server's next state). */
  let next: ((s: SetupFixture | undefined, body: Json) => SetupFixture) | null = null;

  const find = (id: string) => setups.find((s) => s.account_id === id);
  const put = (state: SetupFixture) => {
    setups = setups.some((s) => s.account_id === state.account_id)
      ? setups.map((s) => (s.account_id === state.account_id ? state : s))
      : [...setups, state];
  };

  const handle = async (url: URL, method: string, body: Json, route: Route): Promise<boolean> => {
    if (url.pathname === "/capabilities") {
      await route.fulfill({ json: { writes_enabled: opts.writes ?? true, features: {} } });
      return true;
    }
    if (!url.pathname.startsWith(BASE)) return false;
    if (method === "GET" && url.pathname === BASE) {
      gets += 1;
      await route.fulfill({ json: overview(setups, opts.accounts ?? []) });
      return true;
    }
    const m = url.pathname.slice(BASE.length).match(/^\/([^/]+)\/([a-z-]+)$/);
    if (!m) return false;
    const id = decodeURIComponent(m[1]);
    if (method === "GET" && m[2] === "label-preview") {
      await route.fulfill({ json: labelPreview(id, find(id)?.approval_id ?? "appr-1") });
      return true;
    }
    if (method !== "POST") return false;
    posted.push({ path: `${m[2]}:${id}`, body });
    if (gate) await gate;
    if (fail) {
      await route.fulfill({ status: fail.status, json: { detail: fail.detail } });
      return true;
    }
    if (m[2] === "restart") {
      const had = Boolean(find(id));
      setups = setups.filter((s) => s.account_id !== id);
      await route.fulfill({ json: { restarted: had } });
      return true;
    }
    if (m[2] === "approve-writes") {
      const state = setupAt(id, "first_digest");
      put(state);
      await route.fulfill({ json: state });
      return true;
    }
    if (m[2] === "advance") {
      const state = next ? next(find(id), body) : setupAt(id, "complete");
      put(state);
      await route.fulfill({ json: state });
      return true;
    }
    return false;
  };

  return {
    posted,
    get gets() {
      return gets;
    },
    handle,
    answerAdvanceWith(fn: (s: SetupFixture | undefined, body: Json) => SetupFixture) {
      next = fn;
    },
    failWith(status: number, detail: string) {
      fail = { status, detail };
    },
    hold() {
      let release!: () => void;
      gate = new Promise<void>((r) => (release = r));
      return () => {
        gate = null;
        release();
      };
    },
  };
}

async function serve(page: Page, api: ReturnType<typeof fakeSetupApi>, nav?: NavFixture) {
  await page.route("**/*", async (route) => {
    const request = route.request();
    const type = request.resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const url = new URL(request.url());
    if (nav && url.pathname === "/api/v1/webui/nav") return route.fulfill({ json: nav });
    const body = (request.postData() ? JSON.parse(request.postData()!) : {}) as Json;
    if (await api.handle(url, request.method(), body, route)) return;
    await route.fulfill({ json: fixtureFor(url.pathname) });
  });
}

// -- the nav: Setup follows the email plugin ------------------------------------------

test("Setup is in the nav when email is mounted, and gone when it is not", async ({ page }) => {
  const coreEmail = navFixture("core-email");
  const api = fakeSetupApi([]);
  await serve(page, api, coreEmail);
  await page.goto("/more", { waitUntil: "networkidle" });
  await expect(page.locator("main a[href='/setup']")).toHaveCount(1);

  // The same install without the email plugin: its screens leave the nav, and a
  // direct link says which plugin is missing without calling the setup API.
  const withoutEmail: NavFixture = {
    ...coreEmail,
    groups: coreEmail.groups.map((g) => ({
      ...g,
      items: g.items.filter((i) => (i as unknown as Json).plugin !== "email_workflows"),
    })),
    unavailable: [
      ...coreEmail.unavailable,
      { route: "/setup", label: "Setup", plugin: "email_workflows", reason: "not in the profile" },
    ],
  };
  const bare = await page.context().newPage();
  const called: string[] = [];
  bare.on("request", (r) => called.push(new URL(r.url()).pathname));
  await serve(bare, fakeSetupApi([]), withoutEmail);
  await bare.goto("/more", { waitUntil: "networkidle" });
  await expect(bare.locator("main a[href='/setup']")).toHaveCount(0);
  await bare.goto("/setup", { waitUntil: "networkidle" });
  const notice = bare.getByRole("status").filter({ hasText: "isn't available" });
  await expect(notice).toContainText("email_workflows");
  expect(called).not.toContain(BASE);
});

// -- rendering each state ---------------------------------------------------------------

test("a blocked connect shows the login command, and Copy puts it on the clipboard", async ({
  page,
}) => {
  await page.addInitScript(() => {
    const w = window as unknown as { __copied: string[] };
    w.__copied = [];
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: (t: string) => (w.__copied.push(t), Promise.resolve()) },
    });
  });
  const blocked = setupAt(ACCOUNT, "connect", {
    status: "waiting",
    waiting_kind: "blocked",
    waiting_for: `${ACCOUNT} is not connected yet: run \`iris auth gmail login --user owner@example.com\`, then run setup again`,
    connect_command: "iris auth gmail login --user owner@example.com",
  });
  const api = fakeSetupApi([blocked]);
  await serve(page, api);
  await page.goto("/setup", { waitUntil: "networkidle" });

  const current = page.getByTestId("current-step");
  await expect(current).toHaveAttribute("data-waiting", "blocked");
  await expect(current.getByTestId("connect-command")).toHaveText(
    "iris auth gmail login --user owner@example.com",
  );
  // No password or login form: the provider's own CLI login owns that.
  await expect(page.locator("input[type='password']")).toHaveCount(0);
  await current.getByRole("button", { name: "Copy" }).click();
  await expect(current.getByRole("button", { name: "Copied" })).toBeVisible();
  const copied = await page.evaluate(() => (window as unknown as { __copied: string[] }).__copied);
  expect(copied).toEqual(["iris auth gmail login --user owner@example.com"]);
  expect(api.posted).toEqual([]);
});

test("when the clipboard refuses, the command is selected instead", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: () => Promise.reject(new Error("denied")) },
    });
  });
  const blocked = setupAt(ACCOUNT, "connect", {
    status: "waiting",
    waiting_kind: "blocked",
    waiting_for: "not connected yet",
    connect_command: "iris auth gmail login --user owner@example.com",
  });
  await serve(page, fakeSetupApi([blocked]));
  await page.goto("/setup", { waitUntil: "networkidle" });
  await page.getByTestId("current-step").getByRole("button", { name: "Copy" }).click();
  await expect(page.getByText("Selected: copy it with your keyboard or menu.")).toBeVisible();
  expect(await page.evaluate(() => window.getSelection()?.toString())).toBe(
    "iris auth gmail login --user owner@example.com",
  );
});

test("with no mailbox yet, each provider's login is offered", async ({ page }) => {
  await serve(page, fakeSetupApi([]));
  await page.goto("/setup", { waitUntil: "networkidle" });
  const commands = page.getByTestId("start-panel").getByTestId("connect-command");
  await expect(commands).toHaveText([
    "iris auth gmail login --user you@example.com",
    "iris auth imap login --user you@example.com",
  ]);
});

test("the steps show done / now / ahead, each finished one in the server's words", async ({
  page,
}) => {
  await serve(page, fakeSetupApi([setupAt(ACCOUNT, "classify")]));
  await page.goto("/setup", { waitUntil: "networkidle" });
  await expect(page.getByTestId("step-fetch")).toHaveAttribute("data-state", "done");
  await expect(page.getByTestId("step-fetch")).toContainText("fetched 412 email(s)");
  await expect(page.getByTestId("step-classify")).toHaveAttribute("data-state", "current");
  await expect(page.getByTestId("step-summary")).toHaveAttribute("data-state", "ahead");
  await expect(page.getByTestId("sweep")).toContainText("waiting for setup");
  await expect(page.getByTestId("current-step")).toContainText("Next: Classify");
});

test("a finished setup leads with what IRIS just did", async ({ page }) => {
  await serve(page, fakeSetupApi([setupAt(ACCOUNT, "complete")]));
  await page.goto("/setup", { waitUntil: "networkidle" });
  const summary = page.getByTestId("setup-summary");
  await expect(summary).toContainText("What IRIS just did");
  await expect(summary).toContainText("Mailbox writes approved: 180 label(s) written.");
  await expect(page.getByTestId("sweep")).toContainText("Scheduled sweep: on");
  await expect(page.getByTestId("current-step")).toHaveCount(0);
});

// -- advancing ----------------------------------------------------------------------------

test("Continue posts until_waiting and never takes a default for the owner", async ({ page }) => {
  const api = fakeSetupApi([setupAt(ACCOUNT, "fetch")]);
  api.answerAdvanceWith((s) =>
    setupAt(s!.account_id, "review_categories", {
      status: "waiting",
      waiting_kind: "decision",
      waiting_for: "review the 2 proposed categories: accept all, some or none",
    }),
  );
  await serve(page, api);
  await page.goto("/setup", { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Continue" }).click();

  await expect(page.getByTestId("category-review")).toBeVisible();
  expect(api.posted).toEqual([
    { path: `advance:${ACCOUNT}`, body: { until_waiting: true, actor: "web" } },
  ]);
});

test("the category review sends exactly the ticked categories", async ({ page }) => {
  const api = fakeSetupApi([
    setupAt(ACCOUNT, "review_categories", {
      status: "waiting",
      waiting_kind: "decision",
      waiting_for: "review the 2 proposed categories: accept all, some or none",
    }),
  ]);
  api.answerAdvanceWith((s) => setupAt(s!.account_id, "label_approval"));
  await serve(page, api);
  await page.goto("/setup", { waitUntil: "networkidle" });
  const review = page.getByTestId("category-review");
  // Every acceptable one starts ticked; the one the server refused cannot be.
  await expect(review.getByRole("checkbox", { name: "work/newsletters" })).toBeChecked();
  await expect(review.getByRole("checkbox", { name: "Category 2" })).toBeDisabled();
  await expect(review).toContainText(PROPOSALS[2].why_not);
  await review.getByRole("checkbox", { name: "work/newsletters" }).uncheck();
  await review.getByRole("button", { name: "Accept 1 selected" }).click();
  await expect(page.getByTestId("category-review")).toHaveCount(0);
  expect(api.posted).toEqual([
    {
      path: `advance:${ACCOUNT}`,
      body: { accept_categories: [1], until_waiting: true, actor: "web" },
    },
  ]);
});

// -- step 6: the owner's explicit answer ---------------------------------------------------

test("step 6 shows the preview and posts nothing until a button is clicked", async ({ page }) => {
  const api = fakeSetupApi([waitingAtApproval(ACCOUNT, "appr-42")]);
  await serve(page, api);
  await page.goto("/setup", { waitUntil: "networkidle" });
  const approval = page.getByTestId("label-approval");
  await expect(approval.getByTestId("preview-total")).toHaveText("168 label(s) would be written");
  await expect(approval.getByTestId("preview-group")).toHaveCount(3);
  await expect(approval).toContainText("Your statement is ready");
  await expect(approval).toContainText("IRIS labels removed from 2 email(s) marked promo.");
  await expect(approval).toContainText(`iris email writes revoke --account ${ACCOUNT}`);
  await expect(approval.getByTestId("action-center-note")).toContainText("appr-42");
  await expect(approval.getByRole("link", { name: "Open the Action Center" })).toHaveAttribute(
    "href",
    "/actions",
  );
  // Loading, re-rendering and refetching the step never answers it.
  await page.reload({ waitUntil: "networkidle" });
  await expect(page.getByTestId("label-approval")).toBeVisible();
  expect(api.posted).toEqual([]);
});

for (const [name, approve] of [
  ["Approve: let IRIS label mail", true],
  ["Decline: keep it read-only", false],
] as const) {
  test(`step 6: "${name}" posts approve=${approve} to approve-writes`, async ({ page }) => {
    const api = fakeSetupApi([waitingAtApproval(ACCOUNT)]);
    await serve(page, api);
    await page.goto("/setup", { waitUntil: "networkidle" });
    await page.getByRole("button", { name }).click();
    await expect(page.getByTestId("label-approval")).toHaveCount(0);
    expect(api.posted).toEqual([
      { path: `approve-writes:${ACCOUNT}`, body: { approve, actor: "web" } },
    ]);
  });
}

test("a read-only console shows step 6 but offers no answer to it", async ({ page }) => {
  const api = fakeSetupApi([waitingAtApproval(ACCOUNT)], { writes: false });
  await serve(page, api);
  await page.goto("/setup", { waitUntil: "networkidle" });
  await expect(page.getByTestId("label-approval")).toBeVisible();
  await expect(page.getByRole("button", { name: /^Approve/ })).toHaveCount(0);
  await expect(page.getByRole("button", { name: /^Decline/ })).toHaveCount(0);
  await expect(page.getByRole("button", { name: /Start over/ })).toHaveCount(0);
  await expect(page.getByTestId("setup-readonly")).toContainText("IRIS_WEBUI_ALLOW_WRITES");
  expect(api.posted).toEqual([]);
});

// -- restart ------------------------------------------------------------------------------

test("Start over asks in the page first; Cancel sends nothing, confirming restarts", async ({
  page,
}) => {
  const api = fakeSetupApi([setupAt(ACCOUNT, "classify")]);
  await serve(page, api);
  await page.goto("/setup", { waitUntil: "networkidle" });

  await page.getByRole("button", { name: "Start over…" }).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toContainText("Fetched mail, judgments");
  await dialog.getByRole("button", { name: "Cancel" }).click();
  await expect(dialog).toHaveCount(0);
  expect(api.posted).toEqual([]);

  await page.getByRole("button", { name: "Start over…" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Start over" }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  expect(api.posted).toEqual([{ path: `restart:${ACCOUNT}`, body: {} }]);
  await expect(page.getByRole("region", { name: `Setup for ${ACCOUNT}` })).toHaveCount(0);
});

// -- slow steps ------------------------------------------------------------------------------

test("a slow step is submitted once, shows it is working, and stays locked across a visit away", async ({
  page,
}) => {
  const api = fakeSetupApi([setupAt(ACCOUNT, "fetch")]);
  await serve(page, api);
  await page.goto("/setup", { waitUntil: "networkidle" });
  const release = api.hold();

  const cont = page.getByRole("button", { name: "Continue" });
  await cont.dblclick();
  await expect(page.getByTestId("setup-busy")).toContainText("Working on Fetch recent mail");
  await expect(cont).toBeDisabled();
  await expect(page.getByRole("button", { name: "Start over…" })).toBeDisabled();

  // Away and back while the request runs: the new screen knows it is still in flight.
  await page.locator("a[href='/actions']:visible").first().click();
  await expect(page).toHaveURL(/\/actions$/);
  await page.goBack();
  await expect(page.getByTestId("setup-busy")).toBeVisible();
  await expect(page.getByRole("button", { name: "Continue" })).toBeDisabled();

  release();
  await expect(page.getByTestId("setup-summary")).toBeVisible();
  await expect(page.getByTestId("setup-busy")).toHaveCount(0);
  expect(api.posted).toHaveLength(1);
});

test("a failed request says so and re-reads the state the server kept", async ({ page }) => {
  const api = fakeSetupApi([setupAt(ACCOUNT, "fetch")]);
  await serve(page, api);
  await page.goto("/setup", { waitUntil: "networkidle" });
  const before = api.gets;
  api.failWith(502, "Bad Gateway");
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(page.getByRole("alert")).toContainText("Bad Gateway");
  await expect(page.getByRole("alert")).toContainText("IRIS may have kept going");
  await expect.poll(() => api.gets).toBeGreaterThan(before);
  await expect(page.getByRole("button", { name: "Continue" })).toBeEnabled();
});

// -- starting --------------------------------------------------------------------------------

test("a connected account with no setup can be started", async ({ page }) => {
  const api = fakeSetupApi([], { accounts: ["imap:new@example.com"] });
  api.answerAdvanceWith((_s, _body) => setupAt("imap:new@example.com", "fetch"));
  await serve(page, api);
  await page.goto("/setup", { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Start setup" }).click();
  await expect(page.getByRole("region", { name: "Setup for imap:new@example.com" })).toBeVisible();
  expect(api.posted).toEqual([
    { path: "advance:imap:new@example.com", body: { until_waiting: true, actor: "web" } },
  ]);
  expect(onboardingPath("imap:new@example.com")).toContain("imap%3Anew%40example.com");
});

// -- dark mode (public issue #19) ----------------------------------------------------------
// Dark is the console's default (lib/theme.ts); it is set here explicitly so the test does
// not lean on that default. The light theme is not held to this yet: there the "done" and
// "now" tags read at 2.6:1 and 3.6:1, under AA -- a shared Tag/token issue, not Setup's.

/** WCAG contrast of each matched element's text against what is painted behind it,
 * compositing translucent backgrounds (`bg-primary/5`) over their ancestors. */
async function lowContrast(page: Page, selector: string): Promise<string[]> {
  return page.evaluate((sel) => {
    const parse = (c: string): number[] => {
      const m = c.match(/[\d.]+/g);
      if (!m) return [0, 0, 0, 0];
      const [r, g, b, a] = m.map(Number);
      return [r, g, b, a === undefined ? 1 : a];
    };
    const behind = (el: Element): number[] => {
      const layers: number[][] = [];
      for (let e: Element | null = el; e; e = e.parentElement) {
        const bg = parse(getComputedStyle(e).backgroundColor);
        if (bg[3] > 0) layers.push(bg);
        if (bg[3] >= 1) break;
      }
      let out = [255, 255, 255];
      for (const [r, g, b, a] of layers.reverse()) {
        out = [r * a + out[0] * (1 - a), g * a + out[1] * (1 - a), b * a + out[2] * (1 - a)];
      }
      return out;
    };
    const lum = ([r, g, b]: number[]): number => {
      const ch = (v: number) => {
        const s = v / 255;
        return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
      };
      return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b);
    };
    const bad: string[] = [];
    for (const el of Array.from(document.querySelectorAll(sel))) {
      const fg = lum(parse(getComputedStyle(el).color));
      const bg = lum(behind(el));
      const ratio = (Math.max(fg, bg) + 0.05) / (Math.min(fg, bg) + 0.05);
      if (ratio < 4.5) bad.push(`${ratio.toFixed(2)} "${(el.textContent ?? "").trim().slice(0, 30)}"`);
    }
    return bad;
  }, selector);
}

test("Setup reads at phone width in dark mode", async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem("iris-theme", "dark"));
  await serve(page, fakeSetupApi([setupAt(ACCOUNT, "classify")]));
  await page.goto("/setup", { waitUntil: "networkidle" });

  expect(await page.evaluate(() => document.documentElement.classList.contains("dark"))).toBe(true);
  await expect(page.getByTestId("current-step")).toContainText("Next: Classify");
  // The step titles and tags, their results and the current step's prompt: AA (4.5:1).
  const selector =
    "ol[aria-label='Setup steps'] li span, ol[aria-label='Setup steps'] li pre, [data-testid='current-step'] p";
  expect(await page.locator(selector).count()).toBeGreaterThan(5);
  expect(await lowContrast(page, selector)).toEqual([]);
  const sideways = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(sideways).toBeLessThanOrEqual(1);
});
