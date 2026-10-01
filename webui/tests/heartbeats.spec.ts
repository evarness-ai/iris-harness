/* Which job ran and which did not, on the Heartbeats screen (loop-proof D13).
 *
 * The owner's surface for "did the 12:15 sweep run?" is the Heartbeats list: each
 * heartbeat carries its most recent slot's verdict from the runs the server keeps in
 * heartbeat_runs.db. The server words it ("Missed 12:15 — last success 06:15"); the
 * page shows it with a tag, on an iPhone, without overflowing. The API is stubbed. */
import { test, expect, type Page } from "@playwright/test";
import { fixtureFor } from "./fixtures";

const beat = (name: string, extra: Record<string, unknown>) => ({
  name,
  schedule: "15 6,12,18 * * *",
  schedule_text: "daily at 06:15, 12:15 and 18:15",
  enabled: true,
  description: "",
  default_schedule: "15 6,12,18 * * *",
  default_enabled: true,
  overridden: false,
  runnable: true,
  unavailable_reason: null,
  platforms: [],
  next_run_at: "2026-09-26T23:15:00Z",
  ...extra,
});

const HEARTBEATS = {
  timezone: "America/Chicago",
  heartbeats: [
    beat("email_sweep", {
      watched: true,
      last_run: null,
      last_success_at: "2026-09-26T11:15:04Z",
      job: {
        state: "red",
        detail: "Missed 12:15 — last success 06:15",
        slot: "2026-09-26T17:15:00Z",
        missed: true,
        ran: false,
        last_error: "",
      },
    }),
    beat("email_judge", {
      schedule: "30 6,12,18 * * *",
      watched: true,
      last_success_at: "2026-09-26T17:30:02Z",
      last_run: {
        name: "email_judge",
        status: "success",
        started_at: "2026-09-26T17:30:02Z",
        finished_at: "2026-09-26T17:31:40Z",
        output: "judged 12 · waiting 0",
        error: "",
        result: { judged: 12, waiting: 0, unsure: 1, errors: 0, unreachable: false },
      },
      job: {
        state: "green",
        detail: "ran 12:30 ✓ (judged 12 · waiting 0)",
        slot: "2026-09-26T17:30:00Z",
        missed: false,
        ran: true,
        last_error: "",
      },
    }),
    beat("routine_tick", { schedule: "interval:60", schedule_text: "every minute", job: null }),
  ],
};

async function serve(page: Page): Promise<void> {
  await page.route("**/*", async (route) => {
    const request = route.request();
    const type = request.resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const pathname = new URL(request.url()).pathname;
    let body: unknown = fixtureFor(pathname);
    if (pathname.endsWith("/heartbeat")) body = HEARTBEATS;
    else if (pathname.endsWith("/heartbeat/runs")) {
      body = { count: 1, total: 1, source: "kept", runs: [HEARTBEATS.heartbeats[1].last_run] };
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });
}

test("a missed slot and the judge's run show on the Heartbeats list", async ({ page }) => {
  await serve(page);
  await page.goto("/heartbeats", { waitUntil: "networkidle" });

  const sweep = page.getByTestId("beat-job-email_sweep");
  await expect(sweep).toContainText("missed");
  await expect(sweep).toContainText("Missed 12:15 — last success 06:15");
  await expect(sweep).toContainText("alerts when missed");

  const judge = page.getByTestId("beat-job-email_judge");
  await expect(judge).toContainText("ran");
  await expect(judge).toContainText("judged 12 · waiting 0");

  // A heartbeat with no kept verdict (a fast tick) shows no job line.
  await expect(page.getByTestId("beat-job-routine_tick")).toHaveCount(0);

  // Recent runs are the kept ones, with the handler's one-line summary.
  await expect(page.getByText("Recent runs (1)")).toBeVisible();

  // Nothing overflows the phone's width.
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
