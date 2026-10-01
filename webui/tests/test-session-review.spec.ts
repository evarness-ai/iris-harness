/* "Looks like a test session" — the one-time bulk review (ADR-0119, cleanup PR 5).
 *
 * The server is faked against the contract: GET /memory/review/test-sessions lists the
 * sessions whose id is not shaped like a real conversation's, and removal is the same
 * /memory/removed contract the Map uses. The fake keeps state, so a removed session
 * really leaves the list. Runs on the iPhone WebKit project at 390px. */
import { test, expect, type Page } from "@playwright/test";
import { fixtureFor } from "./fixtures";

interface Row {
  session_id: string;
  summary_goal: string;
  turns: number;
  last_activity: string;
  reasons: string[];
}

const ROWS: Row[] = [
  {
    session_id: "cal6-verify",
    summary_goal: "Goal: check the calendar sync",
    turns: 6,
    last_activity: "2026-06-24T02:33:14Z",
    reasons: ["its id is not shaped like a real conversation's"],
  },
  {
    session_id: "cascade",
    summary_goal: "",
    turns: 2,
    last_activity: "2026-06-25T14:41:17Z",
    reasons: ["its id is not shaped like a real conversation's", "only 2 turns"],
  },
  {
    session_id: "clr-probe",
    summary_goal: "",
    turns: 4,
    last_activity: "2026-07-01T10:00:00Z",
    reasons: ["its id is not shaped like a real conversation's", "only 4 turns"],
  },
];

interface Fake {
  removed: Set<string>;
  previewBodies: unknown[];
  removeBodies: unknown[];
}

async function fakeServer(
  page: Page,
  opts: { writes?: boolean; rows?: Row[] } = {},
): Promise<Fake> {
  const fake: Fake = { removed: new Set(), previewBodies: [], removeBodies: [] };
  const rows = opts.rows ?? ROWS;
  await page.route("**/*", async (route) => {
    const req = route.request();
    const type = req.resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const { pathname } = new URL(req.url());
    const json = (body: unknown) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
    const body = () => JSON.parse(req.postData() ?? "{}");

    if (pathname === "/capabilities") {
      return json({ ...(fixtureFor(pathname) as object), writes_enabled: opts.writes ?? true });
    }
    if (pathname === "/memory/review/test-sessions") {
      return json({
        sessions: rows.filter((r) => !fake.removed.has(r.session_id)),
        real_shapes: ["a web console chat", "a Telegram chat"],
      });
    }
    if (pathname === "/memory/removed/preview") {
      const b = body();
      fake.previewBodies.push(b);
      return json({
        effects: b.targets.map((t: { kind: string; id: string }) => ({
          target: t,
          label: t.id,
          lines: ["leaves the Map, recall and the chat list"],
        })),
      });
    }
    if (pathname === "/memory/removed" && req.method() === "POST") {
      const b = body();
      fake.removeBodies.push(b);
      const items = b.targets.map((t: { kind: string; id: string }) => {
        fake.removed.add(t.id);
        return {
          id: `rm-${t.id}`,
          kind: "session",
          label: t.id,
          removed_at: "2026-09-22T20:00:00Z",
          cascade: [],
          permanent: false,
        };
      });
      return json({ items });
    }
    if (pathname === "/memory/removed") return json({ items: [] });
    return json(fixtureFor(pathname));
  });
  return fake;
}

async function openReview(page: Page): Promise<void> {
  await page.goto("/memory");
  await page.getByRole("button", { name: /^Review/ }).click();
  await expect(page.getByTestId("test-session-review")).toBeVisible();
}

test("every flagged session is ticked; untick one and only the rest are removed", async ({
  page,
}) => {
  const fake = await fakeServer(page);
  await openReview(page);

  const section = page.getByTestId("test-session-review");
  await expect(section.getByText("cal6-verify")).toBeVisible();
  await expect(section.getByText("Goal: check the calendar sync")).toBeVisible();
  await expect(section.getByText(/only 2 turns/)).toBeVisible();
  await expect(section.getByRole("button", { name: "Remove ticked (3)" })).toBeVisible();

  await section.getByLabel("Keep ticked to remove clr-probe").uncheck();
  await section.getByRole("button", { name: "Remove ticked (2)" }).click();

  const dialog = page.getByRole("dialog");
  await expect(dialog.getByText("Remove 2 items?")).toBeVisible();
  await expect(dialog.getByText("leaves the Map, recall and the chat list").first()).toBeVisible();
  await dialog.getByRole("button", { name: "Remove", exact: true }).click();

  expect(fake.removeBodies).toEqual([
    {
      targets: [
        { kind: "session", id: "cal6-verify" },
        { kind: "session", id: "cascade" },
      ],
    },
  ]);
  await expect(section.getByText("cal6-verify")).toHaveCount(0);
  await expect(section.getByText("clr-probe")).toBeVisible();
  await expect(section.getByRole("button", { name: "Remove ticked (1)" })).toBeVisible();
});

test("untick all disables removal; tick all brings every one back", async ({ page }) => {
  await fakeServer(page);
  await openReview(page);
  const section = page.getByTestId("test-session-review");

  await section.getByRole("button", { name: "Untick all", exact: true }).click();
  await expect(section.getByRole("button", { name: "Remove ticked (0)" })).toBeDisabled();
  await section.getByRole("button", { name: "Tick all", exact: true }).click();
  await expect(section.getByRole("button", { name: "Remove ticked (3)" })).toBeEnabled();
});

test("nothing flagged means no section at all", async ({ page }) => {
  await fakeServer(page, { rows: [] });
  await page.goto("/memory");
  await page.getByRole("button", { name: /^Review/ }).click();
  await expect(page.getByText(/item\(s\) waiting/)).toBeVisible();
  await expect(page.getByTestId("test-session-review")).toHaveCount(0);
});

test("with writes off the list shows but nothing can be ticked or removed", async ({ page }) => {
  await fakeServer(page, { writes: false });
  await openReview(page);
  const section = page.getByTestId("test-session-review");
  await expect(section.getByText("cascade")).toBeVisible();
  await expect(section.getByRole("checkbox")).toHaveCount(0);
  await expect(section.getByRole("button", { name: /Remove ticked/ })).toHaveCount(0);
});

test("the section fits a 390px phone", async ({ page }) => {
  await fakeServer(page);
  await openReview(page);
  const box = await page.getByTestId("test-session-review").boundingBox();
  expect(box).not.toBeNull();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(page.viewportSize()!.width);
});
