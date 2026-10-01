/* Done and Snooze on the web surfaces (loop-proof PR 3b, D14).
 *
 * Three surfaces, one API (POST /api/v1/reminders/<id>/done | snooze | undo):
 *
 *   - the reminder sheet (/reminders/:id), where an iPhone push tap lands;
 *   - the Chat panel's Active reminders (desktop), Done + a Snooze menu per row;
 *   - the service worker's notification buttons (Chrome/Android/desktop).
 *
 * The API is stubbed per test with a small stateful fake, so each test asserts what
 * the page SENT, not just what it rendered. The worker is exercised in Node against
 * a fake `self`: Playwright cannot click a system notification, and the logic worth
 * pinning (which request a button makes, what a tap opens, that a failure still
 * shows something) is plain code. */
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import { test, expect, type Page, type Route } from "@playwright/test";
import { SHEET_REMINDER, SHEET_REMINDER_ID, fixtureFor } from "./fixtures";

type Json = Record<string, unknown>;

interface Posted {
  path: string;
  body: Json;
}

/** Serve the shell from fixtures and the reminder API from `reminder` handlers. */
async function serve(
  page: Page,
  handle: (pathname: string, method: string, body: Json, route: Route) => Promise<boolean>,
): Promise<void> {
  await page.route("**/*", async (route) => {
    const request = route.request();
    const type = request.resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const pathname = new URL(request.url()).pathname;
    const body = (request.postData() ? JSON.parse(request.postData()!) : {}) as Json;
    if (await handle(pathname, request.method(), body, route)) return;
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(fixtureFor(pathname)),
    });
  });
}

const json = (route: Route, body: unknown, status = 200) =>
  route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

/** A fake of stream D's reminder API for one reminder, recording every POST. */
function fakeReminderApi(start: Json = SHEET_REMINDER) {
  const base = `/api/v1/reminders/${start.id}`;
  let view: Json = { ...start };
  const posted: Posted[] = [];
  const handle = async (pathname: string, method: string, body: Json, route: Route) => {
    if (!pathname.startsWith(base)) return false;
    if (method === "GET" && pathname === base) {
      await json(route, view);
      return true;
    }
    posted.push({ path: pathname.slice(base.length), body });
    if (pathname === `${base}/done`) {
      view = { ...view, status: "done", actions: [] };
      await json(route, { reminder: view, undo: { kind: "done" } });
    } else if (pathname === `${base}/snooze`) {
      const until: Record<string, string> = {
        "10m": "Mon Sep 28, 8:13 AM",
        "1h": "Mon Sep 28, 9:03 AM",
        tomorrow_9am: "Tue Sep 29, 9:00 AM",
      };
      if (!until[String(body.for)]) {
        await json(route, { detail: "I can snooze 10m, 1h or tomorrow 9am." }, 422);
        return true;
      }
      view = { ...view, status: "pending", remind_at_local: until[String(body.for)] };
      await json(route, {
        reminder: view,
        undo: { kind: "snooze", until_before: "2026-09-28T13:00:00+00:00" },
      });
    } else if (pathname === `${base}/undo`) {
      view = { ...start };
      // The contract's undo answer is the row view; `actions` may be absent.
      const { actions: _omit, ...rest } = view;
      void _omit;
      await json(route, { reminder: rest });
    } else {
      return false;
    }
    return true;
  };
  return { handle, posted };
}

const SHEET = `/reminders/${SHEET_REMINDER_ID}`;

test.describe("the reminder sheet", () => {
  test("shows the reminder and its four choices, big enough to tap", async ({ page }) => {
    const api = fakeReminderApi();
    await serve(page, api.handle);
    await page.goto(SHEET, { waitUntil: "networkidle" });

    await expect(page.getByRole("heading", { name: `⏰ ${SHEET_REMINDER.text}` })).toBeVisible();
    await expect(page.getByText("Due Mon Sep 28, 8:00 AM · repeats every Monday")).toBeVisible();
    for (const name of ["✅ Done", "Snooze 10 min", "Snooze 1 hour", "Tomorrow 9:00 AM"]) {
      const button = page.getByRole("button", { name, exact: true });
      await expect(button).toBeVisible();
      expect((await button.boundingBox())!.height).toBeGreaterThanOrEqual(44);
    }
    await page.screenshot({ path: test.info().outputPath("reminder-sheet.png"), fullPage: true });
  });

  test("Done posts from the sheet, says so, and Undo puts it back", async ({ page }) => {
    const api = fakeReminderApi();
    await serve(page, api.handle);
    await page.goto(SHEET, { waitUntil: "networkidle" });

    await page.getByRole("button", { name: "✅ Done" }).click();
    await expect(page.getByTestId("reminder-status")).toHaveText(/^✓ Done at \d{1,2}:\d{2}/);
    await expect(page.getByRole("button", { name: "Snooze 1 hour" })).toHaveCount(0);
    expect(api.posted).toEqual([{ path: "/done", body: { source: "sheet" } }]);
    await page.screenshot({ path: test.info().outputPath("reminder-sheet-done.png"), fullPage: true });

    await page.getByRole("button", { name: "Undo" }).click();
    await expect(page.getByRole("button", { name: "✅ Done" })).toBeVisible();
    expect(api.posted[1]).toEqual({ path: "/undo", body: { kind: "done" } });
  });

  test("Snooze 1 hour says until when, and Undo sends the snooze's token", async ({ page }) => {
    const api = fakeReminderApi();
    await serve(page, api.handle);
    await page.goto(SHEET, { waitUntil: "networkidle" });

    await page.getByRole("button", { name: "Snooze 1 hour" }).click();
    await expect(page.getByTestId("reminder-status")).toHaveText(
      "⏰ Snoozed until Mon Sep 28, 9:03 AM",
    );
    expect(api.posted).toEqual([{ path: "/snooze", body: { for: "1h", source: "sheet" } }]);

    await page.getByRole("button", { name: "Undo" }).click();
    await expect(page.getByRole("button", { name: "Snooze 1 hour" })).toBeVisible();
    expect(api.posted[1]).toEqual({
      path: "/undo",
      body: { kind: "snooze", until_before: "2026-09-28T13:00:00+00:00" },
    });
  });

  test("a closed reminder shows its state and no buttons", async ({ page }) => {
    const api = fakeReminderApi({ ...SHEET_REMINDER, status: "done", actions: [] });
    await serve(page, api.handle);
    await page.goto(SHEET, { waitUntil: "networkidle" });

    await expect(page.getByTestId("reminder-status")).toHaveText("✓ Done");
    await expect(page.locator("main article button")).toHaveCount(0);
  });

  test("an expired reminder says it expired", async ({ page }) => {
    const api = fakeReminderApi({ ...SHEET_REMINDER, status: "expired", actions: [] });
    await serve(page, api.handle);
    await page.goto(SHEET, { waitUntil: "networkidle" });
    await expect(page.getByTestId("reminder-status")).toContainText("Expired");
    await expect(page.locator("main article button")).toHaveCount(0);
  });

  test("a failed one says it could not be delivered and can still be answered", async ({
    page,
  }) => {
    const api = fakeReminderApi({ ...SHEET_REMINDER, status: "failed" });
    await serve(page, api.handle);
    await page.goto(SHEET, { waitUntil: "networkidle" });
    await expect(page.getByText("Couldn't be delivered")).toBeVisible();
    await expect(page.getByRole("button", { name: "✅ Done" })).toBeVisible();
  });

  test("a reminder that no longer exists is gone, not an error", async ({ page }) => {
    await serve(page, async (pathname, _m, _b, route) => {
      if (!pathname.startsWith("/api/v1/reminders/")) return false;
      await json(route, { detail: "not found" }, 404);
      return true;
    });
    await page.goto("/reminders/does-not-exist", { waitUntil: "networkidle" });
    await expect(page.getByRole("heading", { name: "This reminder is gone" })).toBeVisible();
  });

  test("a refused snooze shows the server's reason and keeps the buttons", async ({ page }) => {
    await serve(page, async (pathname, method, _b, route) => {
      if (pathname === `/api/v1/reminders/${SHEET_REMINDER_ID}` && method === "GET") {
        await json(route, SHEET_REMINDER);
        return true;
      }
      if (pathname.endsWith("/snooze")) {
        await json(route, { detail: "That reminder already fired twice today." }, 422);
        return true;
      }
      return false;
    });
    await page.goto(SHEET, { waitUntil: "networkidle" });
    await page.getByRole("button", { name: "Snooze 10 min" }).click();
    await expect(page.locator("main").getByText("That reminder already fired twice today.")).toBeVisible();
    await expect(page.getByRole("button", { name: "Snooze 10 min" })).toBeVisible();
  });
});

/* ── Chat → Active reminders (desktop; the phone reads the list in the bell) ── */

const LIST = [
  { ...SHEET_REMINDER, status: "pending", kind: "reminder" },
  {
    ...SHEET_REMINDER,
    id: "task-failed",
    text: "Renew the car insurance",
    kind: "task",
    status: "failed",
    recurrence: null,
    recurrence_label: "",
  },
  {
    ...SHEET_REMINDER,
    id: "lead-pending",
    text: "Standup starts in 15 min",
    kind: "event_lead",
    status: "pending",
  },
  {
    ...SHEET_REMINDER,
    id: "bill-failed",
    text: "Electricity bill due",
    kind: "bill",
    status: "failed",
  },
];

async function openPanel(page: Page, posted: Posted[], listQueries: string[]) {
  await page.setViewportSize({ width: 1280, height: 900 });
  await serve(page, async (pathname, method, body, route) => {
    if (pathname === "/api/v1/reminders" && method === "GET") {
      listQueries.push(new URL(route.request().url()).search);
      await json(route, { count: LIST.length, reminders: LIST });
      return true;
    }
    if (pathname.startsWith("/api/v1/reminders/") && method === "POST") {
      posted.push({ path: pathname, body });
      const view = { ...LIST[0], status: "pending", remind_at_local: "Mon Sep 28, 9:03 AM" };
      await json(route, { reminder: view, undo: { kind: "snooze", until_before: "x" } });
      return true;
    }
    return false;
  });
  await page.goto("/chat", { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Show details" }).click();
}

test.describe("Chat's Active reminders", () => {
  test("every row gets Done and Snooze; failed ones of any kind stay red", async ({ page }) => {
    const posted: Posted[] = [];
    const queries: string[] = [];
    await openPanel(page, posted, queries);

    // No kind filter on the request: a failed task reminder must be seen.
    expect(queries.some((q) => q.includes("kind="))).toBe(false);

    const rows = page.getByTestId("active-reminder");
    // The pending event lead-time notice is left out (generated noise); the failed
    // bill reminder is kept, because nothing else says it never arrived.
    await expect(rows).toHaveCount(3);
    await expect(rows.filter({ hasText: "Standup starts in 15 min" })).toHaveCount(0);
    const failed = rows.filter({ hasText: "Renew the car insurance" });
    await expect(failed.getByText("Couldn't be delivered")).toBeVisible();
    await expect(failed.getByRole("button", { name: "Done" })).toBeVisible();
    await expect(rows.filter({ hasText: "Electricity bill due" })).toHaveCount(1);

    await failed.getByRole("button", { name: "Done" }).click();
    await expect.poll(() => posted).toEqual([
      { path: "/api/v1/reminders/task-failed/done", body: { source: "chat_panel" } },
    ]);
  });

  test("the Snooze menu offers 10 min, 1 hour and tomorrow 9am", async ({ page }) => {
    const posted: Posted[] = [];
    await openPanel(page, posted, []);

    const row = page.getByTestId("active-reminder").first();
    await expect(row.getByRole("button", { name: "1 hour" })).toHaveCount(0);
    await row.getByRole("button", { name: "Snooze ▾" }).click();
    for (const name of ["10 min", "1 hour", "Tomorrow 9am"]) {
      await expect(row.getByRole("button", { name })).toBeVisible();
    }
    await row.getByRole("button", { name: "1 hour" }).click();
    await expect.poll(() => posted).toEqual([
      {
        path: `/api/v1/reminders/${SHEET_REMINDER_ID}/snooze`,
        body: { for: "1h", source: "chat_panel" },
      },
    ]);
    await expect(page.getByText("Snoozed until Mon Sep 28, 9:03 AM")).toBeVisible();
  });

  test("a reminder's Action Center card is not listed again as a task", async ({ page }) => {
    // The failed reminder raises an Action Center card, which is a task with
    // source_kind "reminder"; Outstanding lists the reminder once, with its own buttons.
    await page.setViewportSize({ width: 1280, height: 900 });
    await serve(page, async (pathname, method, _body, route) => {
      if (pathname === "/tasks" && method === "GET") {
        await json(route, {
          tasks: [
            { id: "card-1", title: "⏰ Renew the car insurance", status: "open", source_kind: "reminder" },
            { id: "todo-1", title: "Book the plumber", status: "open", source_kind: "manual" },
          ],
        });
        return true;
      }
      if (pathname === "/api/v1/reminders" && method === "GET") {
        await json(route, { count: LIST.length, reminders: LIST });
        return true;
      }
      return false;
    });
    await page.goto("/chat", { waitUntil: "networkidle" });
    await page.getByRole("button", { name: "Show details" }).click();

    await expect(page.getByText("Book the plumber")).toBeVisible();
    await expect(page.getByText("⏰ Renew the car insurance")).toHaveCount(0);
    await expect(page.getByTestId("active-reminder").filter({ hasText: "Renew the car insurance" })).toHaveCount(1);
  });

  test("the reminder's text opens its sheet", async ({ page }) => {
    await openPanel(page, [], []);
    await page.getByRole("link", { name: SHEET_REMINDER.text }).click();
    await expect(page).toHaveURL(new RegExp(`/reminders/${SHEET_REMINDER_ID}$`));
  });
});

/* ── The service worker's push + notificationclick, in Node ── */

const SW_SOURCE = readFileSync(
  path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "public", "sw.js"),
  "utf8",
);

interface Shown {
  title: string;
  options: Json;
}

/** sw.js evaluated against a fake worker global. `maxActions: 0` is Safari. */
function loadWorker(opts: {
  maxActions?: number;
  fetchStatus?: number;
  fetchThrows?: boolean;
  answer?: Json;
}) {
  const handlers: Record<string, (e: unknown) => void> = {};
  const shown: Shown[] = [];
  const fetched: { url: string; init: Json }[] = [];
  const opened: string[] = [];
  const ctx: Json = {
    URL,
    console,
    Response: class {},
    Request: class {},
    caches: { open: async () => ({}), keys: async () => [] },
    Notification: opts.maxActions === undefined ? undefined : { maxActions: opts.maxActions },
    location: { origin: "https://iris.example.ts.net" },
    registration: {
      showNotification: async (title: string, options: Json) => {
        shown.push({ title, options });
      },
    },
    clients: {
      matchAll: async () => [],
      openWindow: async (url: string) => {
        opened.push(url);
      },
      claim: async () => undefined,
    },
    fetch: async (url: string, init: Json) => {
      fetched.push({ url, init });
      if (opts.fetchThrows) throw new Error("offline");
      return {
        ok: (opts.fetchStatus ?? 200) < 400,
        status: opts.fetchStatus ?? 200,
        json: async () =>
          opts.answer ?? {
            reminder: { text: "Take out the recycling", remind_at_local: "Mon Sep 28, 9:03 AM" },
            undo: { kind: "snooze" },
          },
      };
    },
    addEventListener: (name: string, fn: (e: unknown) => void) => {
      handlers[name] = fn;
    },
  };
  ctx.self = ctx;
  vm.runInNewContext(SW_SOURCE, ctx);

  async function dispatch(name: string, event: Json): Promise<void> {
    const pending: Promise<unknown>[] = [];
    handlers[name]({ ...event, waitUntil: (p: Promise<unknown>) => pending.push(p) });
    await Promise.all(pending);
  }
  return { dispatch, shown, fetched, opened };
}

const REMINDER_PUSH = {
  title: "⏰ Take out the recycling",
  body: "8:00 AM · repeats every Monday",
  url: "/reminders/r1",
  data: { url: "/reminders/r1", reminder_id: "r1" },
  tag: "reminder:r1",
  renotify: true,
  actions: [
    { action: "done", title: "Done" },
    { action: "snooze_1h", title: "Snooze 1h" },
  ],
};

const reminderNotification = { data: { url: "/reminders/r1", reminder_id: "r1" }, close() {} };

test.describe("the service worker", () => {
  test("a reminder push shows Done and Snooze 1h where buttons exist", async () => {
    const w = loadWorker({ maxActions: 2 });
    await w.dispatch("push", { data: { json: () => REMINDER_PUSH } });
    expect(w.shown).toHaveLength(1);
    const { title, options } = w.shown[0];
    expect(title).toBe("⏰ Take out the recycling");
    expect(options.actions).toEqual(REMINDER_PUSH.actions);
    expect(options.data).toEqual({ url: "/reminders/r1", reminder_id: "r1" });
    expect(options.tag).toBe("reminder:r1");
    expect(options.renotify).toBe(true);
  });

  test("on Safari (no maxActions) the same push shows, without buttons", async () => {
    const w = loadWorker({});
    await w.dispatch("push", { data: { json: () => REMINDER_PUSH } });
    expect(w.shown).toHaveLength(1);
    expect(w.shown[0].options.actions).toBeUndefined();
  });

  test("an old payload with only a url still opens where it points", async () => {
    const w = loadWorker({ maxActions: 2 });
    await w.dispatch("push", { data: { json: () => ({ title: "IRIS", url: "/health" }) } });
    expect(w.shown[0].options.data).toEqual({ url: "/health", reminder_id: null });
  });

  test("Done answers the reminder without opening the app", async () => {
    const w = loadWorker({ maxActions: 2 });
    await w.dispatch("notificationclick", { action: "done", notification: reminderNotification });
    expect(w.fetched).toHaveLength(1);
    expect(w.fetched[0].url).toBe("/api/v1/reminders/r1/done");
    expect(w.fetched[0].init.method).toBe("POST");
    expect(w.fetched[0].init.credentials).toBe("include");
    expect(JSON.parse(String(w.fetched[0].init.body))).toEqual({ source: "push" });
    expect(w.shown.map((s) => s.title)).toEqual(["Done ✓"]);
    expect(w.shown[0].options.tag).toBe("reminder:r1");
    expect(w.opened).toEqual([]);
  });

  test("Snooze 1h posts the snooze and says until when", async () => {
    const w = loadWorker({ maxActions: 2 });
    await w.dispatch("notificationclick", {
      action: "snooze_1h",
      notification: reminderNotification,
    });
    expect(w.fetched[0].url).toBe("/api/v1/reminders/r1/snooze");
    expect(JSON.parse(String(w.fetched[0].init.body))).toEqual({ for: "1h", source: "push" });
    expect(w.shown.map((s) => s.title)).toEqual(["Snoozed until 9:03 AM"]);
  });

  test("a button that could not reach IRIS says so, and its tap opens the sheet", async () => {
    for (const w of [loadWorker({ maxActions: 2, fetchThrows: true }), loadWorker({ maxActions: 2, fetchStatus: 401 })]) {
      await w.dispatch("notificationclick", { action: "done", notification: reminderNotification });
      expect(w.shown.map((s) => s.title)).toEqual(["Couldn't update the reminder"]);
      expect(w.shown[0].options.data).toEqual({ url: "/reminders/r1", reminder_id: "r1" });
    }
  });

  test("a tap with no button (iOS) opens the reminder's sheet", async () => {
    const w = loadWorker({});
    await w.dispatch("notificationclick", { action: "", notification: reminderNotification });
    expect(w.fetched).toEqual([]);
    expect(w.opened).toEqual(["https://iris.example.ts.net/reminders/r1"]);
  });
});

/* ── A bill's reminder (loop-proof PR 4, prototype "Push + sheet") ── */

const BILL_ID = "0f8fad5b-d9cb-469f-a165-70867728950e";
const BILL_OPEN: Json = {
  id: BILL_ID,
  text: "❓ Did you pay Example Card $35.00?",
  kind: "bill",
  remind_at: "2026-10-13T14:00:00+00:00",
  remind_at_local: "Tue Oct 13, 9:00 AM",
  status: "sent",
  recurrence: null,
  recurrence_label: "",
  closed_reason: null,
  actions: ["paid", "not_yet", "1h", "tomorrow_9am"],
  bill: {
    step: "ask1",
    entity: "Example Card",
    amount: "$35.00",
    statement: "$1,284.50",
    due: "2026-10-12",
    due_local: "Mon Oct 12",
    headline: "❓ Did you pay Example Card $35.00?",
    line: "It was due Mon Oct 12 · no payment email seen",
    paid_by: null,
    not_yet: false,
    reopen_id: null,
  },
};

/** The bill's reminder API: Paid (done), Not yet (snooze not_yet), undo. */
function fakeBillApi(start: Json) {
  const base = `/api/v1/reminders/${BILL_ID}`;
  let view: Json = { ...start };
  const posted: Posted[] = [];
  const bill = () => view.bill as Json;
  const handle = async (pathname: string, method: string, body: Json, route: Route) => {
    if (!pathname.startsWith(base)) return false;
    if (method === "GET" && pathname === base) {
      await json(route, view);
      return true;
    }
    posted.push({ path: pathname.slice(base.length), body });
    if (pathname === `${base}/done`) {
      view = {
        ...view,
        status: "done",
        actions: [],
        bill: { ...bill(), paid_by: "the reminder sheet", reopen_id: BILL_ID },
      };
      await json(route, { reminder: view, undo: { kind: "done" }, next: null });
    } else if (pathname === `${base}/snooze` && body.for === "not_yet") {
      view = { ...view, status: "expired", actions: [], bill: { ...bill(), not_yet: true } };
      await json(route, {
        reminder: view,
        undo: { kind: "not_yet" },
        next: { remind_at: "2026-10-14T14:00:00+00:00", remind_at_local: "Wed Oct 14, 9:00 AM" },
      });
    } else if (pathname === `${base}/undo`) {
      view = { ...BILL_OPEN };
      await json(route, { reminder: view });
    } else {
      return false;
    }
    return true;
  };
  return { handle, posted };
}

const BILL_SHEET = `/reminders/${BILL_ID}`;

test.describe("a bill's reminder sheet", () => {
  test("shows the bill, ✅ Paid, Not yet and the two snoozes", async ({ page }) => {
    const api = fakeBillApi(BILL_OPEN);
    await serve(page, api.handle);
    await page.goto(BILL_SHEET, { waitUntil: "networkidle" });

    await expect(page.getByRole("heading", { name: "💳 Example Card" })).toBeVisible();
    await expect(page.getByText("$35.00 min due Mon Oct 12 · statement $1,284.50")).toBeVisible();
    for (const name of ["✅ Paid", "Not yet", "Remind me in 1 hour", "Tomorrow 9:00 AM"]) {
      const button = page.getByRole("button", { name, exact: true });
      await expect(button).toBeVisible();
      expect((await button.boundingBox())!.height).toBeGreaterThanOrEqual(44);
    }
    await expect(page.getByRole("button", { name: "✅ Done" })).toHaveCount(0);
    await page.screenshot({ path: test.info().outputPath("bill-sheet.png"), fullPage: true });
  });

  test("Paid is the reminder's Done and says who paid it", async ({ page }) => {
    const api = fakeBillApi(BILL_OPEN);
    await serve(page, api.handle);
    await page.goto(BILL_SHEET, { waitUntil: "networkidle" });

    await page.getByRole("button", { name: "✅ Paid" }).click();
    await expect(page.getByTestId("reminder-status")).toHaveText("✓ Paid — the reminder sheet");
    expect(api.posted).toEqual([{ path: "/done", body: { source: "sheet" } }]);
    await page.screenshot({ path: test.info().outputPath("bill-sheet-paid.png"), fullPage: true });
  });

  test("Not yet acknowledges and says when it asks again", async ({ page }) => {
    const api = fakeBillApi(BILL_OPEN);
    await serve(page, api.handle);
    await page.goto(BILL_SHEET, { waitUntil: "networkidle" });

    await page.getByRole("button", { name: "Not yet" }).click();
    await expect(page.getByTestId("reminder-status")).toHaveText(
      "Noted — not paid yet. I'll ask again Wed Oct 14, 9:00 AM.",
    );
    expect(api.posted).toEqual([{ path: "/snooze", body: { for: "not_yet", source: "sheet" } }]);
    await page.getByRole("button", { name: "Undo" }).click();
    await expect(page.getByRole("button", { name: "Not yet" })).toBeVisible();
    expect(api.posted[1]).toEqual({ path: "/undo", body: { kind: "not_yet" } });
  });

  test("a bill a payment email closed says so, and Not paid reopens it", async ({ page }) => {
    const closed: Json = {
      ...BILL_OPEN,
      status: "expired",
      closed_reason: "closed: paid (payment_email)",
      actions: [],
      bill: {
        ...(BILL_OPEN.bill as Json),
        step: "dayof",
        paid_by: "a payment email",
        reopen_id: BILL_ID,
      },
    };
    const api = fakeBillApi(closed);
    await serve(page, api.handle);
    await page.goto(BILL_SHEET, { waitUntil: "networkidle" });

    await expect(page.getByTestId("reminder-status")).toHaveText("✓ Paid — a payment email");
    await page.screenshot({ path: test.info().outputPath("bill-sheet-closed.png"), fullPage: true });
    await page.getByRole("button", { name: "Not paid — reopen" }).click();
    await expect(page.getByRole("button", { name: "✅ Paid" })).toBeVisible();
    expect(api.posted).toEqual([{ path: "/undo", body: { kind: "done", source: "sheet" } }]);
  });
});

test.describe("the service worker, for a bill", () => {
  const billNotification = {
    data: { url: `/reminders/${BILL_ID}`, reminder_id: BILL_ID },
    close() {},
  };

  test("Paid posts the Done and confirms Paid ✓", async () => {
    const w = loadWorker({
      maxActions: 2,
      answer: { reminder: { text: "x", bill: { entity: "Example Card" } }, undo: { kind: "done" } },
    });
    await w.dispatch("notificationclick", { action: "paid", notification: billNotification });
    expect(w.fetched[0].url).toBe(`/api/v1/reminders/${BILL_ID}/done`);
    expect(JSON.parse(String(w.fetched[0].init.body))).toEqual({ source: "push" });
    expect(w.shown.map((s) => [s.title, s.options.body])).toEqual([
      ["Paid ✓", "Example Card: no more reminders for this bill."],
    ]);
  });

  test("Not yet posts the acknowledge and says when it asks again", async () => {
    const w = loadWorker({
      maxActions: 2,
      answer: {
        reminder: { text: "x" },
        undo: { kind: "not_yet" },
        next: { remind_at_local: "Wed Oct 14, 9:00 AM" },
      },
    });
    await w.dispatch("notificationclick", { action: "not_yet", notification: billNotification });
    expect(w.fetched[0].url).toBe(`/api/v1/reminders/${BILL_ID}/snooze`);
    expect(JSON.parse(String(w.fetched[0].init.body))).toEqual({
      for: "not_yet",
      source: "push",
    });
    expect(w.shown.map((s) => [s.title, s.options.body])).toEqual([
      ["Noted — not paid yet", "I'll ask again Wed Oct 14, 9:00 AM."],
    ]);
  });

  test("Tomorrow snoozes to 9:00", async () => {
    const w = loadWorker({ maxActions: 2 });
    await w.dispatch("notificationclick", {
      action: "snooze_tomorrow",
      notification: billNotification,
    });
    expect(JSON.parse(String(w.fetched[0].init.body))).toEqual({
      for: "tomorrow_9am",
      source: "push",
    });
  });
});
