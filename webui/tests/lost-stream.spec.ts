/* A lost stream is not a lost turn (2026-09-21).
 *
 * The owner answered "yes" to a trash request and switched apps; iOS suspended the
 * home-screen app, the stream died with "TypeError: Load failed", and the chat showed
 * that error. The turn now runs on the server regardless, so the chat waits for it
 * and shows the recorded answer. These fake the server: the stream fails at the
 * network level (the turn starts, as on the real server), /api/chat-status says a
 * turn is running until the test ends it, then the session's messages hold the
 * answer. The test, not a poll count, decides when the turn ends: the chat also
 * polls status in the background. */
import { test, expect, type Page } from "@playwright/test";
import { fixtureFor } from "./fixtures";

const ANSWER = "Trash 3 emails: this needs your approval, so nothing has changed yet.";

interface Fake {
  cancels: number;
  running: boolean;
  answered: boolean;
}

async function fakeServer(page: Page, opts: { running: boolean }): Promise<Fake> {
  const fake: Fake = { cancels: 0, running: opts.running, answered: false };
  await page.route("**/*", async (route) => {
    const req = route.request();
    const type = req.resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const { pathname } = new URL(req.url());
    const json = (body: unknown) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });

    if (pathname === "/chat/stream") {
      fake.running = true; // the server took the turn; the connection then dies
      return route.abort("failed");
    }
    if (pathname === "/chat/cancel") {
      fake.cancels += 1;
      return json({ cancelled: true });
    }
    if (pathname === "/api/chat-status") {
      return json({
        session_id: "s",
        iris_version: "test",
        provider: "local",
        provider_label: "Local",
        model: "m",
        window: null,
        turn_in_progress: fake.running,
      });
    }
    if (/^\/api\/sessions\/[^/]+\/messages$/.test(pathname)) {
      return json(
        fake.answered
          ? [
              { role: "user", text: "yes" },
              { role: "assistant", text: ANSWER },
            ]
          : [],
      );
    }
    return json(fixtureFor(pathname));
  });
  return fake;
}

function endTurn(fake: Fake): void {
  fake.running = false;
  fake.answered = true;
}

test("a stream that dies mid-turn waits for the answer instead of an error", async ({
  page,
}) => {
  const fake = await fakeServer(page, { running: false });
  await page.goto("/chat/web-lost");
  await page.getByRole("textbox").fill("yes");
  await page.getByRole("button", { name: "Send message" }).click();

  await expect(page.getByText("Still working on it.")).toBeVisible();
  endTurn(fake);
  await expect(page.getByText(ANSWER)).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText("Still working on it.")).toHaveCount(0);
  await expect(page.getByText(/Load failed|Error:/)).toHaveCount(0);
});

test("reopening a session whose turn is still running waits for it", async ({ page }) => {
  const fake = await fakeServer(page, { running: true });
  await page.goto("/chat/web-reopened");

  await expect(page.getByText("Still working on it.")).toBeVisible();
  endTurn(fake);
  await expect(page.getByText(ANSWER)).toBeVisible({ timeout: 20_000 });
});

test("Stop cancels the turn on the server", async ({ page }) => {
  const fake = await fakeServer(page, { running: true });
  await page.goto("/chat/web-stop");
  await expect(page.getByText("Still working on it.")).toBeVisible();

  await page.getByRole("button", { name: "Stop" }).click();

  await expect(page.getByText("Still working on it.")).toHaveCount(0);
  await expect.poll(() => fake.cancels).toBe(1);
});
