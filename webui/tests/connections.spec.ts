/* Settings > Connections and the Health Reconnect button (Reconnect Google
 * prototype, signed off 2026-09-26).
 *
 * The server is faked against the contract iris_api serves: credential rows in
 * GET /health carry a `reconnect` descriptor; GET <setup_route> says whether the
 * Web client is there; POST <route> answers { auth_url }; PUT <setup_route> stores
 * the client JSON. Google itself is a page on a fake host, so "Continue" is
 * proved by where the browser goes. Runs on the iPhone WebKit project at 390px. */
import { test, expect, type Page } from "@playwright/test";
import { fixtureFor } from "./fixtures";

const START = "/api/v1/connections/google/start";
const SETUP = "/api/v1/connections/google/client";
const GOOGLE = "https://accounts.google.test/o/oauth2/v2/auth";
const YOU = "you@example.com";
const WORK = "work.account.with.a.long.address@example-company.co.uk";

function row(target: string, provider: string, account: string | null, state: string) {
  return {
    kind: "credential",
    target,
    state,
    detail: account ? `${account}: ${state === "red" ? "token revoked — re-authenticate" : "connected"}` : "not connected",
    endpoint: null,
    action: state === "red" ? `iris auth ${provider} login --user ${account}` : null,
    subject: account,
    fix_url: state === "red" ? "/settings#connections" : null,
    reconnect: {
      route: START,
      setup_route: SETUP,
      group: "google",
      group_label: "Google",
      provider,
      label: target,
      account,
    },
  };
}

const CHECKS = [
  { kind: "service", target: "iris_api", state: "green", detail: "200", endpoint: null, action: null },
  row("Gmail", "gmail", YOU, "green"),
  row("Gmail", "gmail", WORK, "green"),
  row("Calendar", "gcalendar", YOU, "red"),
  row("Calendar", "gcalendar", WORK, "green"),
  row("Drive", "gdrive", null, "grey"),
  // A credential no plugin made reconnectable: no button, ever.
  { kind: "credential", target: "Anthropic", state: "green", detail: "API key present", endpoint: null, action: null },
];

interface Fake {
  configured: boolean;
  writes: boolean;
  starts: unknown[];
  uploads: unknown[];
  listed?: boolean;
}

async function fakeServer(page: Page, opts: Partial<Fake> = {}): Promise<Fake> {
  const fake: Fake = { configured: true, writes: true, starts: [], uploads: [], ...opts };
  const setup = () => ({
    configured: fake.configured,
    public_url_set: true,
    redirect_uri: "https://iris-vm.example.ts.net/api/v1/connections/google/callback",
    group: "google",
    group_label: "Google",
    providers: [
      { provider: "gmail", label: "Gmail" },
      { provider: "gcalendar", label: "Calendar" },
      { provider: "gdrive", label: "Drive" },
    ],
    add_provider: "gmail",
  });
  await page.route("https://accounts.google.test/**", (route) =>
    route.fulfill({ status: 200, contentType: "text/html", body: "<h1>Choose an account</h1>" }),
  );
  await page.route("**/*", async (route) => {
    const req = route.request();
    const type = req.resourceType();
    // fallback, not continue: the fake Google page above must still answer its navigation.
    if (type !== "fetch" && type !== "xhr") return route.fallback();
    const { pathname } = new URL(req.url());
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    if (pathname === "/capabilities") {
      return json({ ...(fixtureFor(pathname) as object), writes_enabled: fake.writes });
    }
    if (pathname === "/health") {
      const alerts = CHECKS.filter((c) => c.state === "red");
      return json({ state: "red", summary: "1 red", sampled_at: "2026-09-26T12:00:00Z", checks: CHECKS, alerts });
    }
    if (pathname === SETUP) {
      if (req.method() === "PUT") {
        fake.uploads.push(JSON.parse(req.postData() ?? "{}"));
        fake.configured = true;
        return json({ ...setup(), redirect_uri_listed: fake.listed ?? true });
      }
      return json(setup());
    }
    if (pathname === START && req.method() === "POST") {
      const body = JSON.parse(req.postData() ?? "{}") as { provider: string };
      fake.starts.push(body);
      return json({ auth_url: `${GOOGLE}?state=s&provider=${body.provider}`, expires_in: 600 });
    }
    return json(fixtureFor(pathname));
  });
  return fake;
}

test("one card per account, a state per service, and the right button on each", async ({ page }) => {
  await fakeServer(page);
  await page.goto("/settings#connections");

  const cards = page.getByTestId("connection-account");
  await expect(cards).toHaveCount(2);
  const you = cards.filter({ hasText: YOU });
  await expect(you.getByTestId("connection-row")).toHaveCount(3);
  const calendar = you.getByTestId("connection-row").filter({ hasText: "Calendar" });
  await expect(calendar).toContainText("access revoked — reconnect");
  await expect(calendar.getByRole("button", { name: "Reconnect" })).toBeEnabled();
  const drive = you.getByTestId("connection-row").filter({ hasText: "Drive" });
  await expect(drive).toContainText("not connected");
  await expect(drive.getByRole("button", { name: "Connect" })).toBeVisible();
  const gmail = you.getByTestId("connection-row").filter({ hasText: "Gmail" });
  await expect(gmail).toContainText("connected");
  await expect(gmail.getByRole("button")).toHaveCount(0);

  await expect(page.getByRole("button", { name: "Add a Google account" })).toBeEnabled();
  await expect(
    page.getByText(
      "Only a paired control device (or the owner's secret) can reconnect. Tokens stay on the server; this phone never sees them.",
    ),
  ).toBeVisible();
  await expect(page.getByTestId("connections-setup-banner")).toHaveCount(0);
});

test("Reconnect confirms in three steps, then leaves for Google with the right request", async ({ page }) => {
  const fake = await fakeServer(page);
  await page.goto("/settings#connections");

  await page
    .getByTestId("connection-account")
    .filter({ hasText: YOU })
    .getByRole("button", { name: "Reconnect" })
    .click();
  const sheet = page.getByTestId("reconnect-sheet");
  await expect(sheet.getByRole("heading", { name: "Reconnect Calendar" })).toBeVisible();
  await expect(sheet).toContainText(YOU);
  await expect(sheet).toContainText("You sign in on Google's own page and approve access.");
  await expect(sheet).toContainText("Google sends you straight back here.");
  await expect(sheet).toContainText("The server stores the new token and checks it live. It never reaches this phone.");

  await sheet.getByRole("button", { name: "Cancel" }).click();
  await expect(sheet).toHaveCount(0);
  expect(fake.starts).toEqual([]);

  await page.getByTestId("connection-account").filter({ hasText: YOU }).getByRole("button", { name: "Reconnect" }).click();
  await page.getByRole("button", { name: "Continue to Google" }).click();
  await page.waitForURL(/accounts\.google\.test/);
  expect(fake.starts).toEqual([{ provider: "gcalendar", account: YOU }]);
});

test("Add a Google account starts a connect for any account", async ({ page }) => {
  const fake = await fakeServer(page);
  await page.goto("/settings#connections");
  await page.getByRole("button", { name: "Add a Google account" }).click();
  await expect(page.getByRole("heading", { name: "Connect Gmail" })).toBeVisible();
  await expect(page.getByTestId("reconnect-sheet")).toContainText("a new account");
  await page.getByRole("button", { name: "Continue to Google" }).click();
  await page.waitForURL(/accounts\.google\.test/);
  expect(fake.starts).toEqual([{ provider: "gmail", account: null }]);
});

test("without the Web client the banner shows, the buttons are off, and the upload turns them on", async ({ page }) => {
  const fake = await fakeServer(page, { configured: false });
  await page.goto("/settings#connections");

  await expect(page.getByTestId("connections-setup-banner")).toContainText(
    "One-time setup needed: this server has no Google “Web application” client yet, so Reconnect is off.",
  );
  await expect(page.getByRole("button", { name: "Reconnect" })).toBeDisabled();
  await expect(page.getByRole("button", { name: "Add a Google account" })).toBeDisabled();
  await expect(page.getByTestId("connections-redirect-uri")).toHaveText(
    "https://iris-vm.example.ts.net/api/v1/connections/google/callback",
  );

  const client = JSON.stringify({ web: { client_id: "id", client_secret: "shh" } });
  await page.getByTestId("connections-client-file").setInputFiles({
    name: "client_secret.json",
    mimeType: "application/json",
    buffer: Buffer.from(client),
  });
  await expect(page.getByTestId("connections-setup-banner")).toHaveCount(0);
  expect(fake.uploads).toEqual([{ client_json: client }]);
  await expect(page.getByRole("button", { name: "Reconnect" })).toBeEnabled();
  // The outcome stays on the panel after the toast fades and the form folds away.
  await expect(page.getByTestId("connections-client-status")).toHaveText(
    "Google Web client saved on the server just now. Reconnect is ready.",
  );
});

test("a client that does not list the redirect URI is saved with a lasting warning", async ({ page }) => {
  await fakeServer(page, { configured: false, listed: false });
  await page.goto("/settings#connections");
  await page.getByTestId("connections-client-file").setInputFiles({
    name: "client_secret.json",
    mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify({ web: { client_id: "id", client_secret: "shh" } })),
  });
  await expect(page.getByTestId("connections-client-status")).toContainText(
    "Google Web client saved, but it does not list https://iris-vm.example.ts.net/api/v1/connections/google/callback as a redirect URI.",
  );
});

test("an already configured client says so on a fresh load", async ({ page }) => {
  await fakeServer(page);
  await page.goto("/settings#connections");
  await expect(page.getByTestId("connections-client-status")).toHaveText(
    "Google Web client saved on the server. Reconnect is ready.",
  );
});

test("a read-only phone sees the states but no buttons", async ({ page }) => {
  await fakeServer(page, { writes: false });
  await page.goto("/settings#connections");
  await expect(page.getByTestId("connection-account")).toHaveCount(2);
  await expect(page.getByTestId("connections").getByRole("button", { name: /Reconnect|Connect|Add a Google/ })).toHaveCount(0);
});

test("the server's redirect lands on the reconnected screen, once", async ({ page }) => {
  await fakeServer(page);
  await page.goto(`/settings?connect=connected&provider=gcalendar&account=${encodeURIComponent(YOU)}#connections`);

  const result = page.getByTestId("connect-result");
  await expect(result.getByRole("heading", { name: "Calendar reconnected" })).toBeVisible();
  await expect(result).toContainText(YOU);
  await expect(result).toContainText("Live check with Google: connected. The health alert clears on the next check.");
  await expect(page).toHaveURL(/\/settings#connections$/); // the result is read once
  await result.getByRole("button", { name: "Back to connections" }).click();
  const calendar = page
    .getByTestId("connection-account")
    .filter({ hasText: YOU })
    .getByTestId("connection-row")
    .filter({ hasText: "Calendar" });
  await expect(calendar).toContainText("connected — health updates on the next check");
});

const FAILURES: [string, string, string][] = [
  ["connect=cancelled&provider=gcalendar&account=" + encodeURIComponent(YOU), "Nothing changed", "You cancelled on Google’s page. The connection still needs attention."],
  [
    `connect=wrong_account&provider=gcalendar&account=${encodeURIComponent(YOU)}&approved=${encodeURIComponent(WORK)}`,
    "Different account",
    `You approved as ${WORK}, but this reconnect is for ${YOU}. The server saved nothing.`,
  ],
  ["connect=expired", "Link expired", "Each reconnect link works once, for 10 minutes. Start again."],
];

for (const [query, title, body] of FAILURES) {
  test(`the ${title} screen says what happened`, async ({ page }) => {
    await fakeServer(page);
    await page.goto(`/settings?${query}#connections`);
    const result = page.getByTestId("connect-result");
    await expect(result.getByRole("heading", { name: title })).toBeVisible();
    await expect(result).toContainText(body);
    await expect(result.getByRole("button", { name: "Back to connections" })).toBeVisible();
    // Try again needs to know what to retry; an expired link does not say.
    await expect(result.getByRole("button", { name: "Try again" })).toHaveCount(query.startsWith("connect=expired") ? 0 : 1);
  });
}

test("Try again after a wrong account starts the same reconnect", async ({ page }) => {
  const fake = await fakeServer(page);
  await page.goto(`/settings?${FAILURES[1][0]}#connections`);
  await page.getByRole("button", { name: "Try again" }).click();
  await page.waitForURL(/accounts\.google\.test/);
  expect(fake.starts).toEqual([{ provider: "gcalendar", account: YOU }]);
});

test("Health: a revoked, reconnectable row has Reconnect; Manage connections opens the tab", async ({ page }) => {
  const fake = await fakeServer(page);
  await page.goto("/health");

  const rows = page.getByTestId("health-check");
  const calendar = rows.filter({ hasText: "Calendar" }).filter({ hasText: "revoked" });
  const button = calendar.getByRole("button", { name: "Reconnect" });
  await expect(button).toBeVisible();
  expect((await button.boundingBox())!.height).toBeGreaterThanOrEqual(44);
  // Green rows and a row no plugin made reconnectable carry no button.
  await expect(rows.getByRole("button", { name: "Reconnect" })).toHaveCount(1);

  await button.click();
  await expect(page.getByRole("heading", { name: "Reconnect Calendar" })).toBeVisible();
  await page.getByRole("button", { name: "Continue to Google" }).click();
  await page.waitForURL(/accounts\.google\.test/);
  expect(fake.starts).toEqual([{ provider: "gcalendar", account: YOU }]);

  await page.goto("/health");
  await page.getByRole("link", { name: "Manage connections" }).click();
  await expect(page).toHaveURL(/\/settings#connections$/);
  await expect(page.getByTestId("connection-account")).toHaveCount(2);
});

test("the connections tab and the sheet fit the phone", async ({ page }) => {
  await fakeServer(page);
  await page.goto("/settings#connections");
  await expect(page.getByTestId("connection-account")).toHaveCount(2);
  const overflow = () => page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(await overflow()).toBeLessThanOrEqual(0);
  const small = await page.evaluate(() =>
    Array.from(document.querySelectorAll<HTMLElement>("[data-testid=connections] button"))
      .map((el) => el.getBoundingClientRect())
      .filter((r) => r.width > 0 && r.height > 0 && r.height < 44).length,
  );
  expect(small).toBe(0);
  await page.getByRole("button", { name: "Reconnect" }).click();
  await expect(page.getByTestId("reconnect-sheet")).toBeVisible();
  expect(await overflow()).toBeLessThanOrEqual(0);
});
