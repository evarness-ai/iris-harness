/* The builder showcase (OSS plan R17/R12): Governance and Call trace.
 *
 * What a builder evaluating IRIS should see on a phone: every governed call with who
 * made it (an MCP client is `mcp:<client>`), whether a deterministic handler answered,
 * local or cloud; the proof bundle's three invariants verified over the last week, with
 * the evidence and the CLI to reproduce it; the owner-PII shadow counts; and, on a call
 * trace, every hook decision of the turn in order. The API is stubbed; what the screens
 * must NEVER show is a payload field the server did not send or an unmasked address. */
import { test, expect, type Page } from "@playwright/test";
import { fixtureFor } from "./fixtures";

const ADDRESS = "jordan.owner@example.com";

const AUDIT_ALL = {
  count: 3,
  total: 3,
  audit_db: "/tmp/audit.db",
  callers: ["mcp:desktop", "plugin:email_workflows"],
  entries: [
    {
      id: 3,
      ts: "2026-09-30T10:00:03Z",
      run_id: "r3",
      agent_type: "chat",
      hook_point: "pre_response",
      plugin: "curator",
      decision: "allow",
      classification: "personal",
      tier: "tier_1",
      locality: "local",
      severity: "info",
      reason: "answered by a deterministic handler",
      deterministic: true,
      handler: "dues_intercept",
    },
    {
      id: 2,
      ts: "2026-09-30T10:00:02Z",
      run_id: "r2",
      agent_type: "chat",
      hook_point: "post_tool_use",
      plugin: "mcp_client_egress",
      decision: "deny",
      classification: "personal",
      tier: "tier_3",
      locality: "cloud",
      severity: "warn",
      reason: "the result holds the owner's personal data; withheld from mcp:desktop (jo***er@example.com)",
      caller: "mcp:desktop",
      tool_name: "email_search",
      digest_alg: "hmac-sha256/v1/k1",
    },
    {
      id: 1,
      ts: "2026-09-30T10:00:01Z",
      run_id: "r1",
      agent_type: "chat",
      hook_point: "pre_tool_use",
      plugin: "caller_policy",
      decision: "allow",
      classification: null,
      tier: null,
      locality: null,
      severity: "info",
      reason: "caller_policy: plugin:email_workflows may call 'email_search'",
      caller: "plugin:email_workflows",
      tool_name: "email_search",
    },
  ],
};

const PROOF = {
  since: "2026-09-23T10:00:00+00:00",
  until: null,
  ledger_rows: 412,
  ok: false,
  integrity: [],
  invariants: [
    {
      id: "no-private-to-cloud",
      statement: "No model call whose data was classified personal or secret was allowed to a cloud tier (tier_3).",
      ok: false,
      violation_count: 1,
      violations: ["ledger row 12: a personal model call was allowed to tier_3"],
      evidence: { model_call_rows: 80 },
    },
    {
      id: "mailbox-write-approved",
      statement: "No mailbox write without an approved approval row.",
      ok: true,
      violation_count: 0,
      violations: [],
      evidence: { approval_rows: 0, writes_observed: 0 },
    },
    {
      id: "every-call-and-answer-audited",
      statement: "Every model call and every answer has an audit row.",
      ok: true,
      violation_count: 0,
      violations: [],
      evidence: { model_calls_observed: 80, answers_observed: 31 },
    },
  ],
};

const PII = {
  mode: "shadow",
  since: "2026-09-23T10:00:00+00:00",
  rows: 40,
  checked: { owner_pii_egress: 40 },
  unobserved: {},
  cells: [
    {
      hook_point: "pre_llm_call",
      guard: "owner_pii_egress",
      kind: "email",
      action: "mask",
      first_name_alone: false,
      log_only_destination: false,
      occurrences: 9,
      calls: 4,
      distinct: 1,
    },
  ],
};

const TRACE_ID = "sess-1~0";
const TRACE = {
  session_id: "sess-1",
  trace_id: TRACE_ID,
  request: "any dues this week?",
  started_at: "2026-09-30T10:00:00Z",
  total_duration_ms: 900,
  total_tokens: 0,
  nodes: [],
  edges: [],
  steps: [],
  governance: [
    { id: 1, t_offset_ms: 2, hook_point: "pre_turn", plugin: "input_safety", decision: "allow", reason: "" },
    {
      id: 2,
      t_offset_ms: 40,
      step_id: 0,
      hook_point: "pre_llm_call",
      plugin: "egress_gate",
      decision: "allow",
      classification: "personal",
      tier: "tier_1",
      locality: "local",
      reason: "egress_gate: personal to tier_1 permitted",
    },
    {
      id: 3,
      t_offset_ms: 120,
      hook_point: "pre_tool_use",
      plugin: "caller_policy",
      decision: "allow",
      caller: "plugin:finance_workflows",
      tool_name: "bills_due",
      digest_alg: "hmac-sha256/v1/k1",
      reason: "caller_policy: plugin:finance_workflows may call 'bills_due'",
    },
    {
      id: 4,
      t_offset_ms: 180,
      hook_point: "post_tool_use",
      plugin: "output_classifier",
      decision: "allow",
      reason: "statement for jo***er@example.com",
    },
    {
      id: 5,
      t_offset_ms: 850,
      hook_point: "pre_response",
      plugin: "curator",
      decision: "allow",
      deterministic: true,
      handler: "dues_intercept",
      reason: "",
    },
  ],
};

/** Stub every data call; the Governance endpoints answer from the constants above. */
async function serve(page: Page, seen: string[] = []): Promise<void> {
  await page.route("**/*", async (route) => {
    const request = route.request();
    const type = request.resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const url = new URL(request.url());
    const pathname = url.pathname;
    seen.push(`${pathname}${url.search}`);
    let body: unknown = fixtureFor(pathname);
    if (pathname.endsWith("/governance/audit")) {
      const caller = url.searchParams.get("caller");
      const entries = caller
        ? AUDIT_ALL.entries.filter((e) =>
            caller.endsWith(":") ? e.caller?.startsWith(caller) : e.caller === caller,
          )
        : AUDIT_ALL.entries;
      body = { ...AUDIT_ALL, count: entries.length, entries };
    } else if (pathname.endsWith("/governance/proof-bundle/check")) body = PROOF;
    else if (pathname.endsWith("/governance/pii-shadow")) body = PII;
    else if (pathname.endsWith("/api/traces")) {
      body = [{ session_id: "sess-1", trace_id: TRACE_ID, request: TRACE.request, started_at: TRACE.started_at, total_duration_ms: 900, total_tokens: 0 }];
    } else if (pathname.endsWith(`/api/traces/${encodeURIComponent(TRACE_ID)}`)) body = TRACE;
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
}

async function assertNoLeak(page: Page): Promise<void> {
  const text = await page.locator("body").innerText();
  expect(text).not.toContain(ADDRESS);
  for (const raw of ["payload_json", "args_digest", "result_digest", "\"args\""]) {
    expect(text).not.toContain(raw);
  }
}

test("Governance shows who called, how it was answered and where it ran", async ({ page }) => {
  await serve(page);
  await page.goto("/governance", { waitUntil: "networkidle" });

  const entries = page.getByTestId("audit-entry");
  await expect(entries).toHaveCount(3);
  const mcp = entries.nth(1);
  await expect(mcp).toContainText("caller mcp:desktop");
  await expect(mcp).toContainText("cloud · tier_3");
  await expect(mcp).toContainText("email_search");
  await expect(mcp).toContainText("keyed digest");
  await expect(entries.nth(0)).toContainText("deterministic · dues_intercept");
  await expect(entries.nth(0)).toContainText("local · tier_1");
  await assertNoLeak(page);
});

test("the caller filter asks the server for MCP clients only", async ({ page }) => {
  const seen: string[] = [];
  await serve(page, seen);
  await page.goto("/governance", { waitUntil: "networkidle" });

  await page.getByLabel("Filter by caller").selectOption("mcp:");
  await expect(page.getByTestId("audit-entry")).toHaveCount(1);
  await expect(page.getByTestId("audit-caller")).toHaveText("caller mcp:desktop");
  expect(seen.some((u) => u.includes("/governance/audit?caller=mcp%3A"))).toBe(true);
});

test("the proof bundle shows each invariant, its evidence and the CLI", async ({ page }) => {
  await serve(page);
  await page.goto("/governance", { waitUntil: "networkidle" });

  const proof = page.getByTestId("proof-bundle");
  const invariants = proof.getByTestId("proof-invariant");
  await expect(invariants).toHaveCount(3);
  await expect(invariants.nth(0)).toContainText("violated");
  await expect(invariants.nth(0)).toContainText("ledger row 12: a personal model call was allowed to tier_3");
  await expect(invariants.nth(0)).toContainText("80 model-call rows checked");
  // An invariant over nothing observed is not presented as a plain pass.
  await expect(invariants.nth(1)).toContainText("holds (nothing observed)");
  await expect(invariants.nth(2)).toContainText("holds");
  await expect(invariants.nth(2)).toContainText("80 model calls observed");
  await expect(proof).toContainText("iris governance proof-bundle check --days 7");
  await expect(proof).toContainText("iris governance proof-bundle verify proof-bundle.json");
  await expect(page.getByText("violations", { exact: true })).toBeVisible();
});

test("the owner-PII shadow summary is a table of counts", async ({ page }) => {
  await serve(page);
  await page.goto("/governance", { waitUntil: "networkidle" });

  const pii = page.getByTestId("pii-shadow");
  await expect(pii).toContainText("calls read per guard: owner_pii_egress 40");
  const row = pii.locator("tbody tr");
  await expect(row).toHaveCount(1);
  await expect(row).toContainText("owner_pii_egress");
  await expect(row).toContainText("mask");
  await assertNoLeak(page);
});

test("a call trace lists every hook decision of the turn, in order", async ({ page }) => {
  await serve(page);
  await page.goto(`/calltrace/${encodeURIComponent(TRACE_ID)}`, { waitUntil: "networkidle" });

  const events = page.getByTestId("governance-event");
  await expect(events).toHaveCount(5);
  const hooks = await events.evaluateAll((els) =>
    els.map((el) => el.querySelector(".font-semibold")?.textContent ?? ""),
  );
  expect(hooks).toEqual(["PRE_TURN", "PRE_LLM_CALL", "PRE_TOOL_USE", "POST_TOOL_USE", "PRE_RESPONSE"]);
  await expect(events.nth(1)).toContainText("local · tier_1");
  await expect(events.nth(1)).toContainText("personal");
  await expect(events.nth(2)).toContainText("caller plugin:finance_workflows");
  await expect(events.nth(4)).toContainText("deterministic · dues_intercept");
  await assertNoLeak(page);

  // Lays out on the phone it is demoed on.
  const overflow = await page.evaluate(() => {
    const limit = document.documentElement.clientWidth;
    let worst = 0;
    for (const el of Array.from(document.querySelectorAll<HTMLElement>("[data-testid='governance-timeline'] *"))) {
      worst = Math.max(worst, Math.round(el.getBoundingClientRect().right - limit));
    }
    return worst;
  });
  expect(overflow).toBeLessThanOrEqual(1);
});

/* A fresh install (OSS plan R17, public issue #20): no turn has run, so the ledger, the
 * proof window and the trace list are empty. The bodies below are what the IRIS API
 * answers on an empty IRIS_HOME. Each screen must say what to do next, never sit on a
 * spinner, and never show the canned mock traces as if they were the owner's. */
const FRESH: Record<string, unknown> = {
  "/governance/state": {
    enabled: true,
    audit_db: "/home/owner/.iris/governance/audit.db",
    audit_count: 0,
    flags: [{ key: "IRIS_GOVERNANCE_ENABLED", label: "Kernel enabled", on: true }],
  },
  "/governance/audit": {
    count: 0,
    total: 0,
    audit_db: "/home/owner/.iris/governance/audit.db",
    callers: [],
    entries: [],
  },
  "/governance/pii-shadow": {
    mode: "off",
    since: "2026-09-24T10:00:00+00:00",
    rows: 0,
    checked: {},
    unobserved: {},
    cells: [],
  },
  "/governance/proof-bundle/check": {
    since: "2026-09-24T10:00:00+00:00",
    until: null,
    ledger_rows: 0,
    ok: true,
    integrity: [],
    invariants: [
      { id: "no-private-to-cloud", statement: "No private call to a cloud tier.", ok: true, violation_count: 0, violations: [], evidence: { model_call_rows: 0 } },
      { id: "mailbox-write-approved", statement: "No mailbox write without an approval.", ok: true, violation_count: 0, violations: [], evidence: { approval_rows: 0, writes_observed: 0 } },
      { id: "every-call-and-answer-audited", statement: "Every call and answer audited.", ok: true, violation_count: 0, violations: [], evidence: { model_calls_observed: 0, answers_observed: 0 } },
    ],
  },
  "/api/traces": [],
};

async function serveFresh(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/*", async (route) => {
    const type = route.request().resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const pathname = new URL(route.request().url()).pathname;
    const fresh = Object.keys(FRESH).find((p) => pathname.endsWith(p));
    const body = fresh ? FRESH[fresh] : fixtureFor(pathname);
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  return errors;
}

test("on a fresh install, Governance says what to do next", async ({ page }) => {
  const errors = await serveFresh(page);
  await page.goto("/governance", { waitUntil: "networkidle" });

  await expect(page.getByText("Unexpected Application Error")).toHaveCount(0);
  await expect(page.getByText("No governance decisions yet")).toContainText(
    "ask IRIS something in Chat",
  );
  // The three invariants hold over nothing, and the card says so instead of a bare pass.
  const invariants = page.getByTestId("proof-invariant");
  await expect(invariants).toHaveCount(3);
  for (let i = 0; i < 3; i++) await expect(invariants.nth(i)).toContainText("holds (nothing observed)");
  await expect(page.getByTestId("proof-bundle")).toContainText("Nothing recorded in this window yet");
  await expect(page.getByTestId("pii-shadow")).toContainText("IRIS_GOVERNANCE_OWNER_PII is off here");
  expect(errors).toEqual([]);
});

test("a decision filter that matches nothing is not mistaken for a fresh install", async ({ page }) => {
  await page.route("**/*", async (route) => {
    const type = route.request().resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const url = new URL(route.request().url());
    let body: unknown = fixtureFor(url.pathname);
    if (url.pathname.endsWith("/governance/audit")) {
      // Three rows in the ledger, none of them a transform.
      body = url.searchParams.get("decision") === "transform"
        ? { ...AUDIT_ALL, count: 0, entries: [] }
        : AUDIT_ALL;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  await page.goto("/governance", { waitUntil: "networkidle" });
  await expect(page.getByTestId("audit-entry")).toHaveCount(3);

  await page.getByRole("button", { name: "transform", exact: true }).click();
  await expect(page.getByText("No decisions match this filter.")).toBeVisible();
  await expect(page.getByText("No governance decisions yet")).toHaveCount(0);
});

test("on a fresh install, Call trace says what to do next, not mock traces", async ({ page }) => {
  const errors = await serveFresh(page);
  await page.goto("/calltrace", { waitUntil: "networkidle" });

  const empty = page.getByTestId("calltrace-empty");
  await expect(empty).toContainText("No turns traced yet");
  await expect(empty).toContainText("Ask IRIS something in Chat");
  await expect(page.getByText("mock data")).toHaveCount(0);
  await expect(page.getByText("Loading trace")).toHaveCount(0);
  await expect(page).toHaveURL(/\/calltrace$/);

  const open = empty.getByRole("link", { name: "Open Chat" });
  expect(await open.getAttribute("href")).toBe("/chat");
  const box = await open.boundingBox();
  expect(box?.height ?? 0, "Open Chat is a phone tap target").toBeGreaterThanOrEqual(44);
  expect(errors).toEqual([]);
});
