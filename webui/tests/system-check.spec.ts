/* System Check screen (issue #67's decision-support ask): the install preflight over
 * GET /health/doctor, and its one write -- pulling a missing starter model through
 * POST /health/doctor/pull-model, polled via the existing GET /activities feed. The
 * API is a small stateful fake so each test asserts what the page did, not just what
 * it showed. */
import { test, expect, type Page, type Route } from "@playwright/test";

type Json = Record<string, unknown>;

const REPORT = (overrides: Partial<Json> = {}): Json => ({
  verdict: "demo_only",
  exit_code: 1,
  checks: [
    { name: "Python", status: "pass", state: "green", detail: "3.12.4", fix: null, blocks: "none" },
    { name: "RAM", status: "pass", state: "green", detail: "32.0 GB", fix: null, blocks: "none" },
    {
      name: "Starter models",
      status: "fail",
      state: "red",
      detail: "qwen2.5:7b-instruct not pulled",
      fix: "iris doctor --fix",
      blocks: "use",
    },
  ],
  missing_models: [{ name: "qwen2.5:7b-instruct", size_gb: 4.7 }],
  key: { source: "not_read", detail: "resolved by the first governed call" },
  ollama_url: "http://127.0.0.1:11434",
  fixable: true,
  ...overrides,
});

function fakeApi(opts: { writes?: boolean } = {}) {
  let report = REPORT();
  const activities: Json[] = [];
  const posted: { path: string; body: Json }[] = [];
  let nextActivityStatus: "completed" | "failed" = "completed";

  const handle = async (url: URL, method: string, body: Json, route: Route): Promise<boolean> => {
    if (url.pathname === "/capabilities") {
      await route.fulfill({ json: { writes_enabled: opts.writes ?? true, features: {} } });
      return true;
    }
    if (url.pathname === "/health/doctor" && method === "GET") {
      await route.fulfill({ json: report });
      return true;
    }
    if (url.pathname === "/health/doctor/pull-model" && method === "POST") {
      posted.push({ path: "pull-model", body });
      const id = `act-${activities.length + 1}`;
      activities.push({
        id,
        kind: "doctor.pull_model",
        title: `Pull ${body.model}`,
        status: "running",
        progress: 0.1,
        progress_message: "pulling 10%",
        origin: "api",
        result_summary: "",
        error: "",
        undo_ref: null,
        metadata: {},
        started_at: "2026-10-01T00:00:00Z",
        finished_at: null,
        created_at: "2026-10-01T00:00:00Z",
        updated_at: "2026-10-01T00:00:00Z",
      });
      await route.fulfill({ json: { activity_id: id } });
      return true;
    }
    if (url.pathname === "/activities" && method === "GET") {
      await route.fulfill({
        json: {
          count: activities.length,
          running: activities.filter((a) => a.status === "running").length,
          activities,
        },
      });
      return true;
    }
    return false;
  };

  return {
    posted,
    finishTheRunningPull(status: "completed" | "failed" = "completed") {
      nextActivityStatus = status;
      const a = activities[activities.length - 1];
      if (!a) return;
      a.status = nextActivityStatus;
      a.progress = 1.0;
      a.progress_message = "success";
      a.finished_at = "2026-10-01T00:01:00Z";
      if (nextActivityStatus === "completed") a.result_summary = `pulled ${(a.title as string).slice(5)}`;
      else a.error = "pull failed: disk full";
    },
    setReport(next: Json) {
      report = next;
    },
    handle,
  };
}

async function serve(page: Page, api: ReturnType<typeof fakeApi>): Promise<void> {
  await page.route("**/*", async (route) => {
    const request = route.request();
    const type = request.resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const url = new URL(request.url());
    const body = (request.postData() ? JSON.parse(request.postData()!) : {}) as Json;
    if (await api.handle(url, request.method(), body, route)) return;
    await route.fulfill({ json: {} });
  });
}

test("shows the verdict, the checks, and the missing model", async ({ page }) => {
  const api = fakeApi();
  await serve(page, api);
  await page.goto("/system-check", { waitUntil: "networkidle" });

  await expect(page.getByText("Ready for the demo only")).toBeVisible();
  await expect(page.getByTestId("doctor-check")).toHaveCount(3);
  await expect(page.getByTestId("missing-model")).toContainText("qwen2.5:7b-instruct");
  await expect(page.getByTestId("missing-model")).toContainText("4.7 GB");
});

test("pulling a model asks first, then shows live progress to completion", async ({ page }) => {
  const api = fakeApi();
  await serve(page, api);
  await page.goto("/system-check", { waitUntil: "networkidle" });

  await page.getByRole("button", { name: "Pull" }).click();
  await expect(page.getByRole("dialog")).toContainText("Pull qwen2.5:7b-instruct?");
  await page.getByRole("dialog").getByRole("button", { name: "Pull" }).click();

  expect(api.posted).toEqual([{ path: "pull-model", body: { model: "qwen2.5:7b-instruct" } }]);
  await expect(page.getByTestId("missing-model")).toContainText("pulling 10%");

  api.finishTheRunningPull("completed");
  await expect(page.getByText("qwen2.5:7b-instruct pulled")).toBeVisible({ timeout: 10_000 });
});

test("a failed pull toasts the error instead of claiming success", async ({ page }) => {
  const api = fakeApi();
  await serve(page, api);
  await page.goto("/system-check", { waitUntil: "networkidle" });

  await page.getByRole("button", { name: "Pull" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Pull" }).click();
  api.finishTheRunningPull("failed");

  await expect(
    page.getByRole("region", { name: /Notifications/ }).getByText("disk full").first(),
  ).toBeVisible({ timeout: 10_000 });
});

test("a read-only console shows the missing model but no way to pull it", async ({ page }) => {
  const api = fakeApi({ writes: false });
  await serve(page, api);
  await page.goto("/system-check", { waitUntil: "networkidle" });

  await expect(page.getByTestId("missing-model")).toContainText("qwen2.5:7b-instruct");
  await expect(page.getByRole("button", { name: "Pull" })).toHaveCount(0);
  await expect(page.getByText("control-paired device")).toBeVisible();
});

test("a ready report shows no missing-model section", async ({ page }) => {
  const api = fakeApi();
  api.setReport(REPORT({ verdict: "ready", exit_code: 0, missing_models: [] }));
  await serve(page, api);
  await page.goto("/system-check", { waitUntil: "networkidle" });

  await expect(page.getByText("Nothing stands in the way.")).toBeVisible();
  await expect(page.getByTestId("missing-model")).toHaveCount(0);
});
