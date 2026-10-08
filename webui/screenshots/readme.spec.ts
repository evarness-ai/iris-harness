/* The README screenshots (issue #20): Governance, the welcome turn's Call trace, Inbox
 * and Setup, at 1280x800, light and dark. scripts/readme_screenshots.py runs this
 * against a live API on a fresh demo home; see that script for the whole flow. */
import { expect, test, type Page } from "@playwright/test";

const env = (name: string): string => {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is not set: run scripts/readme_screenshots.py`);
  return value;
};

const OUT = env("IRIS_SHOTS_OUT");
const TRACE_ID = env("IRIS_SHOTS_TRACE_ID");

interface Screen {
  name: string;
  path: string;
  height?: number;
  ready: (page: Page) => Promise<void>;
}

const SCREENS: Screen[] = [
  {
    name: "governance",
    path: "/governance",
    ready: (page) => expect(page.getByRole("heading", { name: /governance/i }).first()).toBeVisible(),
  },
  {
    name: "call-trace",
    path: `/calltrace/${encodeURIComponent(TRACE_ID)}`,
    // Taller than the rest, so the whole turn fits; the last node (the model-free
    // response guard) is opened in the detail panel.
    height: 1100,
    ready: async (page) => {
      const nodes = page.locator(".react-flow__node");
      await expect(nodes.first()).toBeVisible();
      await nodes.last().click();
    },
  },
  {
    name: "inbox",
    path: "/inbox",
    ready: (page) => expect(page.getByText(/Needs reply/).first()).toBeVisible(),
  },
  {
    name: "setup",
    path: "/setup",
    ready: (page) => expect(page.getByText("example.com").first()).toBeVisible(),
  },
];

test("README screenshots", async ({ page, context }) => {
  // Pair the browser as the Pair screen does: claim the one-time code. The server
  // answers with the HttpOnly device cookie, which this context keeps.
  const claim = await page.request.post("/api/v1/devices/pair/claim", {
    data: { code: env("IRIS_SHOTS_PAIR_CODE"), name: "readme-screenshots", kind: "browser" },
  });
  expect(claim.ok(), await claim.text()).toBeTruthy();

  for (const theme of ["light", "dark"] as const) {
    // Init scripts run in the order added, so the last theme added wins.
    await context.addInitScript((t) => localStorage.setItem("iris-theme", t), theme);
    await page.emulateMedia({ colorScheme: theme });
    for (const screen of SCREENS) {
      await page.setViewportSize({ width: 1280, height: screen.height ?? 800 });
      await page.goto(screen.path);
      await screen.ready(page);
      await expect(page).not.toHaveURL(/\/pair/);
      await page.waitForLoadState("networkidle");
      await page.waitForTimeout(800); // let the graph's fit-view and fades settle
      await page.screenshot({ path: `${OUT}/${screen.name}-${theme}.png` });
    }
  }
});
