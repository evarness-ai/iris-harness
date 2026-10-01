/* Inbox → Judged (loop-proof PR 5): the web list of what the email judge sorted.
 *
 * The API is a small stateful fake of /api/v1/email/judgments, so each test asserts
 * what the page SENT as well as what it showed: "Change" moves a row at once
 * (optimistic), posts {"bucket"}, and the toast's Undo posts the previous bucket
 * back. All senders are made up. */
import { test, expect, type Page, type Route } from "@playwright/test";
import { JUDGE_BUCKETS, fixtureFor } from "./fixtures";

type Json = Record<string, unknown>;

interface Row {
  message_id: string;
  sender: string;
  subject: string;
  bucket: string;
  bucket_name: string;
  judge_bucket: string;
  owner_bucket: string | null;
  confidence: number;
  figures: Json;
  [key: string]: unknown;
}

const nameOf = (key: string) => JUDGE_BUCKETS.find((b) => b.key === key)?.name ?? key;

function judged(id: string, sender: string, subject: string, bucket: string, extra: Partial<Row> = {}): Row {
  return {
    message_id: id,
    account_id: "gmail:owner@example.com",
    sender,
    from_address: `hello@${id}.example`,
    subject,
    snippet: "",
    received_at: "2026-09-26T13:00:00Z",
    bucket,
    bucket_name: nameOf(bucket),
    judge_bucket: bucket,
    owner_bucket: null,
    owner_source: null,
    confidence: 0.81,
    figures: {},
    judged_at: "2026-09-26T13:05:00Z",
    corrected_at: null,
    ...extra,
  };
}

function fakeJudgeApi() {
  const rows: Row[] = [
    judged("m-case", "Harbor Bank", "About your recent inquiry", "fyi"),
    judged("m-bill", "Northwind Card", "Your statement is ready", "bill", {
      confidence: 0.98,
      figures: { min_due: "$35.00", due_date: "2026-10-12" },
    }),
    judged("m-plan", "Nimbus Utilities", "Important information about your account", "unsure", {
      confidence: 0.52,
    }),
  ];
  const posted: { id: string; body: Json }[] = [];
  const gets: string[] = [];
  let gate: Promise<void> | null = null;
  let refuse: string | null = null;

  const handle = async (url: URL, method: string, body: Json, route: Route): Promise<boolean> => {
    const base = "/api/v1/email/judgments";
    if (!url.pathname.startsWith(base)) return false;
    if (method === "GET" && url.pathname === base) {
      const bucket = url.searchParams.get("bucket");
      gets.push(bucket ?? "");
      const list = bucket ? rows.filter((r) => r.bucket === bucket) : rows;
      await route.fulfill({ json: { judgments: list, buckets: JUDGE_BUCKETS } });
      return true;
    }
    const m = url.pathname.match(/^\/api\/v1\/email\/judgments\/([^/]+)\/bucket$/);
    if (method === "POST" && m) {
      posted.push({ id: m[1], body });
      if (gate) await gate;
      if (refuse) {
        await route.fulfill({ status: 403, json: { detail: refuse } });
        return true;
      }
      const row = rows.find((r) => r.message_id === m[1])!;
      const previous = row.bucket;
      row.bucket = String(body.bucket);
      row.bucket_name = nameOf(row.bucket);
      row.owner_bucket = row.bucket;
      await route.fulfill({ json: { judgment: row, previous, changed: previous !== row.bucket } });
      return true;
    }
    return false;
  };
  return {
    rows,
    posted,
    gets,
    handle,
    hold() {
      let release!: () => void;
      gate = new Promise<void>((r) => (release = r));
      return () => {
        gate = null;
        release();
      };
    },
    refuseWith(detail: string) {
      refuse = detail;
    },
  };
}

async function serve(page: Page, api: ReturnType<typeof fakeJudgeApi>): Promise<void> {
  await page.route("**/*", async (route) => {
    const request = route.request();
    const type = request.resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const url = new URL(request.url());
    const body = (request.postData() ? JSON.parse(request.postData()!) : {}) as Json;
    if (await api.handle(url, request.method(), body, route)) return;
    await route.fulfill({ json: fixtureFor(url.pathname) });
  });
}

const rowOf = (page: Page, subject: string) =>
  page.getByTestId("judged-row").filter({ hasText: subject });

test("lists each judged email with its bucket, confidence and figures", async ({ page }) => {
  const api = fakeJudgeApi();
  await serve(page, api);
  await page.goto("/inbox", { waitUntil: "networkidle" });

  await expect(page.getByRole("tab", { name: "Judged" })).toBeVisible();
  await expect(page.getByTestId("judged-row")).toHaveCount(3);
  const bill = rowOf(page, "Your statement is ready");
  await expect(bill).toContainText("Northwind Card");
  await expect(bill.getByTestId("bucket")).toHaveText("Bill");
  await expect(bill).toContainText("confidence 0.98");
  await expect(bill).toContainText("min due $35.00 · due date 2026-10-12");
  await expect(rowOf(page, "Important information").getByTestId("bucket")).toHaveText("Unsure");

  // Filter chips: All + every bucket the API names, each a 44px target.
  for (const name of ["All", ...JUDGE_BUCKETS.map((b) => b.name)]) {
    const chip = page.getByRole("button", { name, exact: true });
    await expect(chip).toBeVisible();
    expect((await chip.boundingBox())!.height).toBeGreaterThanOrEqual(44);
  }
  for (const change of await page.getByRole("button", { name: "Change" }).all()) {
    expect((await change.boundingBox())!.height).toBeGreaterThanOrEqual(44);
  }
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
  await page.screenshot({ path: test.info().outputPath("inbox-judged.png"), fullPage: true });
});

test("Change moves the row at once, posts the bucket, and Undo posts it back", async ({ page }) => {
  const api = fakeJudgeApi();
  await serve(page, api);
  await page.goto("/inbox", { waitUntil: "networkidle" });

  const row = rowOf(page, "About your recent inquiry");
  await row.getByRole("button", { name: "Change" }).click();
  const menu = row.getByRole("menu");
  // The other buckets and Promo; not the one it already has.
  await expect(menu.getByRole("menuitem")).toHaveText(["Bill", "Event", "Needs reply", "Unsure", "Promo"]);
  for (const item of await menu.getByRole("menuitem").all()) {
    expect((await item.boundingBox())!.height).toBeGreaterThanOrEqual(44);
  }

  const release = api.hold();
  await menu.getByRole("menuitem", { name: "Needs reply" }).click();
  // Optimistic: the badge moves before the server answers.
  await expect(row.getByTestId("bucket")).toHaveText("Needs reply");
  expect(api.posted).toEqual([{ id: "m-case", body: { bucket: "needs_reply" } }]);
  release();

  await expect(page.getByText("Moved to Needs reply — its Gmail label follows")).toBeVisible();
  await page.screenshot({ path: test.info().outputPath("inbox-moved.png"), fullPage: true });
  await page.getByRole("button", { name: "Undo" }).click();
  await expect(row.getByTestId("bucket")).toHaveText("FYI");
  expect(api.posted[1]).toEqual({ id: "m-case", body: { bucket: "fyi" } });
  await expect(page.getByText("Moved back to FYI")).toBeVisible();
});

test("Promo is a choice too", async ({ page }) => {
  const api = fakeJudgeApi();
  await serve(page, api);
  await page.goto("/inbox", { waitUntil: "networkidle" });
  const row = rowOf(page, "Important information");
  await row.getByRole("button", { name: "Change" }).click();
  await row.getByRole("menuitem", { name: "Promo" }).click();
  await expect(row.getByTestId("bucket")).toHaveText("Promo");
  expect(api.posted).toEqual([{ id: "m-plan", body: { bucket: "promo" } }]);
});

test("a filter chip asks the API for that bucket only", async ({ page }) => {
  const api = fakeJudgeApi();
  await serve(page, api);
  await page.goto("/inbox", { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Bill", exact: true }).click();
  await expect(page.getByTestId("judged-row")).toHaveCount(1);
  await expect(page.getByRole("button", { name: "Bill", exact: true })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  expect(api.gets).toContain("bill");
  await page.getByRole("button", { name: "Event", exact: true }).click();
  await expect(page.getByTestId("judged-empty")).toHaveText("Nothing in Event.");
});

test("a refused change rolls the row back and says why", async ({ page }) => {
  const api = fakeJudgeApi();
  api.refuseWith("this device is paired read-only");
  await serve(page, api);
  await page.goto("/inbox", { waitUntil: "networkidle" });
  const row = rowOf(page, "Your statement is ready");
  await row.getByRole("button", { name: "Change" }).click();
  await row.getByRole("menuitem", { name: "FYI" }).click();
  await expect(page.getByText("this device is paired read-only")).toBeVisible();
  await expect(row.getByTestId("bucket")).toHaveText("Bill");
});
