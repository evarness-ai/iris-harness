/* Settings → Digest (loop-proof plan PR 2, prototype v4 signed off 2026-09-25).
 *
 * The server is faked against the contract iris_api serves: GET /digest/config returns
 * the fields in effect, the file's values and every section the digest knows; PATCH
 * takes any of the fields (the tab sends `sections` and `sections_off` together).
 * The fake keeps state, so a change shows the way it would against the real API.
 * Runs on the iPhone WebKit project at 390px. */
import { test, expect, type Page } from "@playwright/test";
import { fixtureFor } from "./fixtures";

const FILE = {
  enabled: true,
  time: "07:00",
  channel: "all",
  sections: ["bills_due", "todays_events", "focus", "news_ai", "news_local", "learned_yesterday"],
  sections_off: [] as string[],
  section_config: { news_local: { line_cap: 3 } } as Record<string, Record<string, unknown>>,
  news_topics: ["AI", "world"],
  news_groups: {
    news_ai: { title: "AI / Tech", topics: ["AI", "technology"] },
    news_local: { title: "Local — {news_local_area}", topics: ["{news_local_area}"] },
  } as Record<string, { title: string; topics: string[] }>,
  news_local_area: "St. Louis",
  news_sources: [] as string[],
  news_language: "en",
  focus_categories: ["email/personal", "email/finance"],
  focus_limit: 10,
  focus_per_account: 5,
};
type Fields = typeof FILE;

interface Fake {
  fields: Fields;
  patches: Partial<Fields>[];
}

async function fakeServer(page: Page): Promise<Fake> {
  const fake: Fake = { fields: structuredClone(FILE), patches: [] };
  const view = () => ({
    fields: fake.fields,
    file: FILE,
    changed: (Object.keys(FILE) as (keyof Fields)[]).filter(
      (k) => JSON.stringify(fake.fields[k]) !== JSON.stringify(FILE[k]),
    ),
    all_sections: [...fake.fields.sections, ...fake.fields.sections_off],
    locked_sections: ["learned_yesterday"],
    groups: [
      { id: "money", title: "Money", icon: "💳", sections: ["bills_due"] },
      { id: "news", title: "News", icon: "📰", sections: ["news_ai", "news_local"] },
    ],
    section_titles: {
      news_ai: "AI / Tech",
      news_local: `Local — ${fake.fields.news_local_area}`,
    },
    timezone: "America/Chicago",
    migration: { at: "2026-09-25T06:00:00+00:00", status: "migrated", dropped: ["bills_due.categories:images"] },
  });
  await page.route("**/*", async (route) => {
    const req = route.request();
    const type = req.resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const { pathname } = new URL(req.url());
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

    if (pathname === "/capabilities") {
      return json({ ...(fixtureFor(pathname) as object), writes_enabled: true });
    }
    if (pathname === "/digest/config") {
      if (req.method() === "PATCH") {
        const body = JSON.parse(req.postData() ?? "{}") as Partial<Fields>;
        fake.patches.push(body);
        // Like the server: a news group's change merges over the others.
        const groups = { ...fake.fields.news_groups, ...(body.news_groups ?? {}) };
        fake.fields = { ...fake.fields, ...body, news_groups: groups };
      } else if (req.method() === "DELETE") {
        fake.fields = structuredClone(FILE);
      }
      return json(view());
    }
    return json(fixtureFor(pathname));
  });
  return fake;
}

test("the digest tab shows when, where and what, and where the settings came from", async ({ page }) => {
  await fakeServer(page);
  await page.goto("/settings#digest");

  await expect(page.getByLabel("Delivery time")).toHaveValue("07:00");
  await expect(page.getByText("America/Chicago (IRIS_TZ)")).toBeVisible();
  await expect(page.getByLabel("Digest channel")).toHaveValue("all");
  await expect(page.getByRole("switch", { name: "Bills due" })).toHaveAttribute("aria-checked", "true");
  await expect(page.getByText("Preferred, never strict", { exact: false })).toBeVisible();
  await expect(page.getByText("dropped: bills_due.categories:images", { exact: false })).toBeVisible();
});

test("turning a section off and moving one up send the whole order", async ({ page }) => {
  const fake = await fakeServer(page);
  await page.goto("/settings#digest");

  await page.getByRole("switch", { name: "Todays events" }).click();
  await expect(page.getByRole("switch", { name: "Todays events" })).toHaveAttribute("aria-checked", "false");
  expect(fake.patches.at(-1)).toEqual({
    sections: ["bills_due", "focus", "news_ai", "news_local", "learned_yesterday"],
    sections_off: ["todays_events"],
  });

  await page.getByRole("button", { name: "Move News: AI / Tech up" }).click();
  await expect.poll(() => fake.patches.length).toBe(2);
  expect(fake.patches[1]).toEqual({
    sections: ["bills_due", "news_ai", "focus", "news_local", "learned_yesterday"],
    sections_off: ["todays_events"],
  });
});

test("the footer is locked on and a section's lines are capped", async ({ page }) => {
  const fake = await fakeServer(page);
  await page.goto("/settings#digest");

  const footer = page.getByRole("switch", { name: "Learned yesterday" });
  await expect(footer).toHaveAttribute("aria-checked", "true");
  await expect(footer).toBeDisabled();
  await expect(page.getByRole("button", { name: "Move Learned yesterday up" })).toHaveCount(0);

  await page.getByRole("button", { name: "Lines for News: AI / Tech: all lines" }).click();
  await page.getByLabel("Lines for News: AI / Tech", { exact: true }).fill("5");
  await page.getByLabel("Lines for News: AI / Tech", { exact: true }).press("Enter");
  await expect(page.getByRole("button", { name: "Lines for News: AI / Tech: ≤ 5 lines" })).toBeVisible();
  expect(fake.patches.at(-1)).toEqual({
    section_config: { news_local: { line_cap: 3 }, news_ai: { line_cap: 5 } },
  });
});

test("every line for a section the file caps sends an empty mapping", async ({ page }) => {
  const fake = await fakeServer(page);
  await page.goto("/settings#digest");

  const name = "News: Local — St. Louis";
  await page.getByRole("button", { name: `Lines for ${name}: ≤ 3 lines` }).click();
  await page.getByLabel(`Lines for ${name}`, { exact: true }).fill("");
  await page.getByLabel(`Lines for ${name}`, { exact: true }).press("Enter");
  await expect.poll(() => fake.patches.at(-1)).toEqual({ section_config: { news_local: {} } });
});

test("a preferred source and a topic are added and removed", async ({ page }) => {
  const fake = await fakeServer(page);
  await page.goto("/settings#digest");

  await page.getByLabel("Add to News sources").fill("bbc.com");
  await page.getByLabel("Add to News sources").press("Enter");
  await expect(page.getByText("★ bbc.com")).toBeVisible();
  expect(fake.patches.at(-1)).toEqual({ news_sources: ["bbc.com"] });

  await page.getByRole("button", { name: "Remove technology from AI / Tech topics" }).click();
  await expect.poll(() => fake.patches.at(-1)).toEqual({
    news_groups: { news_ai: { title: "AI / Tech", topics: ["AI"] } },
  });
});

test("the news groups show their titles and the local area can change", async ({ page }) => {
  const fake = await fakeServer(page);
  await page.goto("/settings#digest");

  await expect(page.getByText("💳 Money · 📰 News · More")).toBeVisible();
  await expect(page.getByRole("switch", { name: "News: Local — St. Louis" })).toBeVisible();
  await page.getByLabel("Add to Local — St. Louis topics").fill("Missouri");
  await page.getByLabel("Add to Local — St. Louis topics").press("Enter");
  await expect.poll(() => fake.patches.at(-1)).toEqual({
    news_groups: {
      news_local: {
        title: "Local — {news_local_area}",
        topics: ["{news_local_area}", "Missouri"],
      },
    },
  });

  const area = page.getByLabel("Local news area");
  await expect(area).toHaveValue("St. Louis");
  await area.fill("Kansas City");
  await area.locator("xpath=ancestor::form").getByRole("button", { name: "Save" }).click();
  await expect.poll(() => fake.patches.at(-1)).toEqual({ news_local_area: "Kansas City" });
  await expect(page.getByRole("switch", { name: "News: Local — Kansas City" })).toBeVisible();
});

test("the digest tab fits the phone", async ({ page }) => {
  await fakeServer(page);
  await page.goto("/settings#digest");
  await expect(page.getByLabel("Delivery time")).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  const small = await page.evaluate(() =>
    Array.from(document.querySelectorAll<HTMLElement>("section button"))
      .map((el) => el.getBoundingClientRect())
      .filter((r) => r.width > 0 && r.height > 0 && r.height < 44).length,
  );
  expect(small).toBe(0);
});

test("the news language is a code the owner can change", async ({ page }) => {
  const fake = await fakeServer(page);
  await page.goto("/settings#digest");

  const field = page.getByLabel("News language");
  await expect(field).toHaveValue("en");
  await expect(page.getByText("file: en")).toBeVisible();
  await field.fill("JA");
  await field.locator("xpath=ancestor::form").getByRole("button", { name: "Save" }).click();
  await expect.poll(() => fake.patches.at(-1)).toEqual({ news_language: "ja" });
  await expect(field).toHaveValue("ja");
});
