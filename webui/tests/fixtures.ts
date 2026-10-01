/* Canned API responses for the viewport smoke (Track 2 PR 1, decision 22).
 *
 * The smoke answers one question — does this screen lay out on a phone — so the
 * data only has to be shaped right, not real. Every non-asset request is served
 * from here, so a screen never sees a 404, never sees a 401 (which would bounce
 * it to /pair) and never depends on a backend being up.
 *
 * DEFAULT is the catch-all: an object carrying every collection name the lib
 * modules destructure. A screen reading `data.routines ?? []` gets an array; a
 * screen reading `data.count` gets 0. Endpoints whose screens need a particular
 * field to render at all get an entry in OVERRIDES.
 *
 * The collections are POPULATED, and deliberately hostile. An empty screen is
 * the one layout that cannot overflow, so empty fixtures prove nothing — that is
 * the #562 lesson, where a composer verified against 3 sessions broke on the
 * owner's 60. So the rows here carry what real data carries and CSS forgets:
 * unbroken 40-character ids, absolute paths, model names with no spaces, and
 * enough rows to push a list past one screen. */
import { navFixture } from "./routes";
import { labelPreview, onboardingPath, overview, setupAt, waitingAtApproval } from "./setup-fixtures";

/** An unbreakable token — no spaces, no hyphens a browser may wrap on. */
const LONG_ID = "a7f3c9e10b24d85f6e3a1c07b9d4f28e6c1a5b93";
/** A path long enough to blow a row open if nothing truncates it. */
const LONG_PATH = "/Users/owner/Library/Application Support/iris/data/memory/statements.db";

/** Email setup (R4): an account id no browser can wrap, waiting at step 6. */
const SETUP_ACCOUNT = `gmail:${LONG_ID}${LONG_ID}@example.com`;

/** `n` rows from `make`, so a list is tested at length, not at zero. */
function rows<T>(n: number, make: (i: number) => T): T[] {
  return Array.from({ length: n }, (_, i) => make(i));
}

/** The email judge's buckets as the API lists them (judge.yaml order, + promo). */
export const JUDGE_BUCKETS = [
  { key: "bill", name: "Bill" },
  { key: "event", name: "Event" },
  { key: "needs_reply", name: "Needs reply" },
  { key: "fyi", name: "FYI" },
  { key: "unsure", name: "Unsure" },
  { key: "promo", name: "Promo" },
];

/** Empty-but-valid for anything not named in OVERRIDES. */
export const DEFAULT: Record<string, unknown> = {
  available: true,
  enabled: true,
  count: 0,
  actions: rows(6, (i) => ({
    id: `${LONG_ID}-${i}`,
    origin: i % 2 ? "health" : "finance",
    title: `Rotate the credential for google_calendar (${LONG_ID})`,
    description: `Raised by the health watch after refresh failed twice. Source: ${LONG_PATH}`,
    source_kind: "credential",
    created_at: "2026-09-20T06:14:00Z",
    action: { kind: "copy_command", label: "Re-authenticate", safe: false, command: `iris auth google --config ${LONG_PATH}` },
    feedback_ref: `ref:${LONG_ID}`,
  })),
  agents: rows(5, (i) => ({
    name: `background_finance_analyst_${i}`,
    title: `Background finance analyst ${i}`,
    description: "Reconciles statements against known dues and raises anything unmatched.",
    source_kind: "plugin",
    plugin: "iris_personal.plugins.finance",
  })),
  approvals: rows(3, (i) => ({
    approval_id: `${LONG_ID}-ap${i}`,
    run_id: `${LONG_ID}-run${i}`,
    signal: "send_email",
    context_summary: `Reply to the landlord about the boiler service date, drafted from Thursday's thread and checked against ${LONG_PATH}.`,
    timeout_at: "2026-09-20T11:30:00Z",
    overdue: false,
    resumable: true,
  })),
  checks: [],
  devices: rows(4, (i) => ({
    id: `${LONG_ID}-d${i}`,
    name: i === 0 ? "iPhone (owner, control scope, paired over tailnet)" : `device-${i}`,
    kind: i === 0 ? "app" : "browser",
    scope: i === 0 ? "control" : "read",
    created_at: "2026-09-20T18:00:00Z",
    last_seen_at: "2026-09-20T20:41:00Z",
    revoked_at: null,
  })),
  documents: rows(5, (i) => ({
    file_id: `${LONG_ID}-doc${i}`,
    filename: `octopus-energy-statement-2026-09-${LONG_ID}.pdf`,
    kind: "pdf",
    classification: "private",
    byte_size: 284_512,
    location_tier: "local",
    storage_path: LONG_PATH,
    created_at: "2026-09-19T03:00:00Z",
    updated_at: "2026-09-19T03:00:00Z",
  })),
  dues: rows(4, (i) => ({
    id: `${LONG_ID}-due${i}`,
    label: "Electricity bill — Octopus Energy, account 123456789012",
    due_date: "2026-09-24",
    amount: 84.2,
    status: "open",
  })),
  edges: [],
  entities: rows(5, (i) => ({
    id: `${LONG_ID}-e${i}`,
    name: `entity_with_a_deliberately_long_name_${i}`,
    kind: "person",
    mentions: 3,
  })),
  events: [],
  experiments: [],
  facts: rows(5, (i) => ({
    key: `fact.${LONG_ID}.${i}`,
    value: "The owner prefers briefings at 07:30 on weekdays and nothing at weekends.",
    confidence: 0.8,
    confirmed: false,
  })),
  flags: {},
  heartbeats: rows(5, (i) => ({
    name: `health_tick_${i}`,
    enabled: true,
    schedule: "*/15 * * * *",
    last_run_at: "2026-09-20T08:33:00Z",
    last_status: "ok",
  })),
  incidents: [
    {
      id: 1,
      key: `credential:google_calendar:${LONG_ID}`,
      target: "google_calendar",
      subject: "robin@example.com",
      kind: "credential",
      state: "needs_user",
      detail: "token revoked — refresh returned invalid_grant and a retry did not help",
      action: "iris auth google",
      opened_at: "2026-09-20T06:12:00Z",
      updated_at: "2026-09-20T06:14:00Z",
      attempts: 2,
      repairs: [
        { tried: "refresh token", ok: false, detail: "invalid_grant" },
        { tried: "retry with backoff", ok: false, detail: "still revoked" },
      ],
      notified_at: "2026-09-20T06:14:00Z",
      notify_count: 2,
      resolved_at: null,
      resolution: null,
    },
    {
      id: 2,
      key: `disk:iris_api.log:${LONG_ID}`,
      target: "iris_api.log",
      subject: LONG_PATH,
      kind: "disk",
      state: "resolved",
      detail: "99 MB over the rotation budget",
      action: null,
      opened_at: "2026-09-20T03:00:00Z",
      updated_at: "2026-09-20T03:01:00Z",
      attempts: 1,
      repairs: [{ tried: "rotate + compress", ok: true, detail: null }],
      notified_at: null,
      notify_count: 0,
      resolved_at: "2026-09-20T03:01:00Z",
      resolution: "self_healed",
    },
  ],
  items: [],
  lessons: [],
  nodes: [],
  plugins: rows(5, (i) => ({
    name: `iris_personal.plugins.module_number_${i}`,
    version: "0.1.0",
    agents: 2,
    tools: 7,
  })),
  proposals: [],
  reminders: rows(4, (i) => ({
    id: `${LONG_ID}-r${i}`,
    text: "Book the dentist and confirm the referral letter arrived",
    kind: "reminder",
    remind_at: "2026-09-22T14:00:00+00:00",
    remind_at_local: "Tue Sep 22, 9:00 AM",
    status: i === 0 ? "failed" : "pending",
    recurrence: i === 1 ? "weekly" : null,
    recurrence_label: i === 1 ? "every Tuesday" : "",
  })),
  routines: rows(5, (i) => ({
    id: `${LONG_ID}-rt${i}`,
    name: `weekday_morning_briefing_variant_${i}`,
    approval_status: i === 0 ? "draft" : "scheduled",
    schedule: "30 7 * * 1-5",
    last_run_at: "2026-09-20T07:30:00Z",
  })),
  runs: rows(5, (i) => ({
    id: `${LONG_ID}-run${i}`,
    started_at: "2026-09-20T07:30:00Z",
    status: "ok",
    duration_ms: 1840,
  })),
  sessions: rows(12, (i) => ({
    session_id: `${LONG_ID}-s${i}`,
    title: `What did the overnight ticks do, and why did the calendar credential fail? (${i})`,
    turn_count: 14,
    started_at: "2026-09-20T06:00:00Z",
    last_at: "2026-09-20T08:40:00Z",
    total_tokens: 48210,
    total_duration_ms: 92140,
    turns: [],
  })),
  suites: [],
  tasks: rows(6, (i) => ({
    id: `${LONG_ID}-t${i}`,
    title: "Rotate the leaked Telegram bot token and update the VM env file",
    status: i % 2 ? "open" : "doing",
    source_kind: "agent",
    due_at: "2026-09-23T09:00:00Z",
  })),
  terms: [],
  tiers: [],
  traces: rows(6, (i) => ({
    session_id: `${LONG_ID}-s${i}`,
    trace_id: `${LONG_ID}-tr${i}`,
    request: "what did the overnight ticks do?",
    started_at: "2026-09-20T08:40:00Z",
    total_duration_ms: 4120,
    total_tokens: 3180,
  })),
  providers: {},
  stores: {},
};

/* Paths whose screen needs a specific field before it renders anything.
 *
 * Matched EXACTLY, never by prefix. Prefix matching looks tidier and is a trap:
 * it served the `/health` snapshot to `/health/incidents`, whose screen then read
 * `data.incidents.filter` on an object that had no `incidents`, crashed into the
 * router's error boundary, and reported itself as a 111px layout overflow — the
 * error boundary renders an unbreakable stack trace. Anything not named here
 * falls through to DEFAULT, which carries every collection name empty. */
/** The reminder the sheet (/reminders/:id) is smoke-tested with: long text, repeating,
 * all four actions open, and an unbroken id. */
export const SHEET_REMINDER_ID = `${LONG_ID}-r1`;
export const SHEET_REMINDER = {
  id: SHEET_REMINDER_ID,
  text: "Book the dentist and confirm the referral letter arrived before the appointment on Thursday",
  kind: "reminder",
  remind_at: "2026-09-28T13:00:00+00:00",
  remind_at_local: "Mon Sep 28, 8:00 AM",
  status: "sent",
  recurrence: "weekly",
  recurrence_label: "every Monday",
  actions: ["done", "10m", "1h", "tomorrow_9am"],
};

export const OVERRIDES: Record<string, unknown> = {
  // Every plugin mounted, so every screen is reachable (tests/routes.ts).
  "/api/v1/webui/nav": navFixture("personal-assistant"),
  [`/api/v1/reminders/${SHEET_REMINDER_ID}`]: SHEET_REMINDER,
  "/capabilities": {
    writes_enabled: true,
    deployment_label: "viewport-smoke",
    features: {},
  },
  "/api/v1/devices/me": {
    id: "smoke-device",
    name: "Viewport smoke",
    kind: "browser",
    scope: "control",
  },
  "/health": {
    state: "yellow",
    summary: "1 credential revoked, 1 service slow, everything else nominal.",
    // The red projection the server computes from `checks`. Absent here until
    // Track 2 PR 11, which meant the header bell's urgent path — and the
    // banner's, before it — was never exercised by the smoke.
    alerts: [
      {
        kind: "credential",
        target: "google_calendar",
        state: "red",
        detail: "token revoked — refresh returned invalid_grant",
        action: "iris auth google",
      },
    ],
    sampled_at: "2026-09-20T08:41:00Z",
    checks: [
      {
        kind: "service",
        target: "iris_api",
        state: "green",
        detail: "200 in 38 ms",
        endpoint: "http://127.0.0.1:8003/healthz?verbose=1&include=services,credentials",
      },
      {
        // Track 2 PR 5b: the failover proxy had no probe at all until now.
        kind: "service",
        target: "llm_proxy",
        state: "green",
        detail: "HTTP 200",
        endpoint: "http://127.0.0.1:4000/health",
      },
      {
        kind: "service",
        target: "ollama",
        state: "yellow",
        detail: "reachable over the tailnet but the first token took 11.4s",
        endpoint: "http://192.0.2.10:11434/api/tags",
      },
      {
        kind: "credential",
        target: "google_calendar",
        state: "red",
        detail: "token revoked — refresh returned invalid_grant",
        action: "iris auth google",
      },
      {
        kind: "hardware",
        target: "host",
        state: "green",
        detail: "CPU 23%, RAM 0.6/1.0 GB free, no thermal throttling reported",
      },
      {
        // Track 2 PR 5. The trial VM's likely failure, and nothing watched it
        // before: 29 GB shared by the image, the logs and every store.
        kind: "hardware",
        target: "disk",
        state: "yellow",
        detail: "24.1 GB/29.0 GB used (83%) on /app/data — filling up",
        action: "iris housekeeping run",
      },
      {
        kind: "hardware",
        target: "uptime",
        state: "green",
        detail: "up 4d 2h",
      },
    ],
  },
  "/cost": {
    recording: true,
    enforcing: true,
    daily_cap_usd: 1.0,
    user_id: "local",
    today_usd: 0.42,
    month_usd: 6.18,
    by_tier_usd: { tier_1: 0, tier_2: 5.03, tier_3: 1.15 },
    entries: 2381,
    ledger_db: LONG_PATH,
    enable_hint: null,
  },
  "/llm/mode": {
    mode: "active",
    auto_mode: "adaptive",
    pin: null,
    adaptive: true,
    snapshot: { cpu_percent: 23, ram_free_gb: 9.4, thermal_throttled: false },
  },
  "/settings": {
    tiers: [
      {
        name: "tier1",
        model: "granite4",
        provider: "ollama",
        temperature: 0.2,
        max_tokens: 2048,
        use_for: ["intent"],
      },
    ],
    intent_tier_map: {},
    providers: { ollama: true, azure: true },
    paths: {},
    host: { ram_total_gb: 16, ram_free_gb: 9.4, cpu_percent: 23, thermal_throttled: false },
    stores: {
      skill_count: 0,
      heartbeat_count: 0,
      database_sizes: {},
      filemanager_roots: 0,
      accounts: {},
    },
    flags: [],
  },
  "/runtime/inventory": {
    iris_version: "0.0.0-smoke",
    git_branch: "smoke",
    git_rev: "0000000",
    python_version: "3.12.0",
    packages: {},
    ollama_models: [],
  },
  "/governance/state": { enabled: true, audit_db: ":memory:", audit_count: 0, flags: [] },
  "/governance/audit": {
    count: 8,
    total: 8,
    audit_db: LONG_PATH,
    callers: [`mcp:${LONG_ID}`, "plugin:email_workflows", "model:react"],
    entries: rows(8, (i) => ({
      id: i,
      ts: "2026-09-20T08:41:00Z",
      run_id: `${LONG_ID}-run${i}`,
      agent_type: "background_finance_analyst",
      hook_point: "pre_tool_use",
      decision: i % 3 === 0 ? "require_approval" : "allow",
      classification: i % 2 ? "personal" : null,
      tier: i % 2 ? "tier_3" : "tier_1",
      locality: i % 2 ? "cloud" : "local",
      reason: `Matched the egress policy for an outbound email with an attachment ${LONG_ID}.`,
      caller: i % 2 ? `mcp:${LONG_ID}` : "plugin:email_workflows",
      tool_name: `email_search_${LONG_ID}`,
      digest_alg: `hmac-sha256/v1/${LONG_ID}`,
      deterministic: i === 4,
      handler: i === 4 ? `dues_intercept_${LONG_ID}` : undefined,
    })),
  },
  "/governance/proof-bundle/check": {
    since: "2026-09-23T08:41:00+00:00",
    until: null,
    ledger_rows: 4210,
    ok: false,
    integrity: [],
    invariants: [
      {
        id: "no-private-to-cloud",
        statement: "No model call whose data was classified personal or secret was allowed to a cloud tier (tier_3).",
        ok: false,
        violation_count: 23,
        violations: rows(20, (i) => `ledger row ${i}${LONG_ID}: a personal model call was allowed to tier_3`),
        evidence: { model_call_rows: 1180 },
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
        evidence: { model_calls_observed: 1180, answers_observed: 402 },
      },
    ],
  },
  "/governance/pii-shadow": {
    mode: "shadow",
    since: "2026-09-23T08:41:00+00:00",
    rows: 311,
    checked: { owner_pii_egress: 120, [`guard_${LONG_ID}`]: 3 },
    unobserved: { no_corpus: 2 },
    cells: rows(6, (i) => ({
      hook_point: "pre_llm_call",
      guard: `owner_pii_egress_${LONG_ID}`,
      kind: i % 2 ? "email" : "name",
      action: "mask",
      first_name_alone: i === 2,
      log_only_destination: i === 3,
      occurrences: 40 + i,
      calls: 12,
      distinct: i % 2 ? null : 2,
    })),
  },
  "/observability/llm-metrics": {
    backend: {
      kind: "none",
      enabled: false,
      healthy: false,
      phoenix_url: null,
      instrumented_targets: [],
    },
    summary: {
      sessions_scanned: 0,
      llm_call_count: 0,
      llm_error_count: 0,
      total_tokens: 0,
      total_duration_ms: 0,
      provider_counts: {},
      model_counts: {},
      learning: {},
    },
  },
  "/portfolio": {
    count: 6,
    priced_count: 6,
    as_of_dates: ["2026-09-19"],
    positions: rows(6, (i) => ({
      name: `A holding whose name does not wrap politely ${i}`,
      isin: `INE${LONG_ID.slice(0, 9).toUpperCase()}`,
      symbol: `VERY_LONG_INSTRUMENT_SYMBOL_${i}.NS`,
      asset_type: "equity",
      currency: "INR",
      quantity: "120",
      cost_basis: "221226.00",
      stored_value: "229224.00",
      as_of_date: "2026-09-19",
      live_price: 1910.2,
      day_change_pct: -0.42,
      market_value: "229224.00",
      pnl: "7998.00",
      pnl_pct: 3.62,
    })),
    totals: [
      {
        currency: "INR",
        cost_basis: "221226.00",
        market_value: "229224.00",
        pnl: "7998.00",
        pnl_pct: 3.62,
      },
    ],
  },
  "/plugins": {
    profile: null,
    count: 5,
    totals: { loaded: 4, degraded: 1, failed: 0, disabled: 0, unsupported: 0 },
    plugins: rows(5, (i) => ({
      name: `iris_personal.plugins.module_number_${i}`,
      status: i === 4 ? "degraded" : "loaded",
      source: LONG_PATH,
      version: "0.1.0",
      description: "Registers two agents and seven tools, plus a router mounted on the API.",
      trust: "first-party",
      flavor: null,
      provides: ["agents", "tools", "api_routes"],
      enabled: true,
      set_by: "config",
      registration_counts: { agents: 2, tools: 7, channels: 0 },
      agents: [`background_finance_analyst_${i}`],
      declared_tools: 7,
      failure_count: 0,
      last_error: null,
      load_error: null,
    })),
  },
  "/market/indexes": {
    indexes: rows(4, (i) => ({
      symbol: `NIFTY_50_INDEX_LONG_SYMBOL_${i}`,
      name: `Nifty 50 with a deliberately long index name ${i}`,
      price: 24_815.35,
      currency: "INR",
      change_pct: -0.42,
    })),
  },
  // The stored digest (loop-proof PR 2, grouped as digest v5): group headings,
  // section headings, a folded "nothing" line, a long body with an unbroken URL,
  // the 👎 action links, a failure line under each group that had one, and the
  // footer after a rule — what a real partial morning looks like.
  "/api/digest/latest": {
    id: "0123456789abcdef0123456789abcdef",
    created_at: "2026-09-25T12:00:00Z",
    heartbeat: "morning-digest",
    skill_id: "morning-brief",
    subject: "Morning digest",
    body: [
      "Good morning. Briefing for Friday, September 25, 2026.",
      "",
      "## ☀️ Today",
      "",
      "### Today's events",
      "- 16:30–17:15 Parent-teacher meeting",
      "",
      "*No reminders due · Nothing due today.*",
      "",
      "## 💳 Money",
      "",
      "### Bills due",
      ...rows(8, (i) => `- Amazon Pay Wingtip credit card ${i}: ₹3,150.40 credit — due Sep 30 (${LONG_ID})`),
      "",
      "*Spending looks normal.*",
      "",
      `⚠ couldn't build: Portfolio (research engine timeout after 20 s at ${LONG_PATH}) — everything else is current.`,
      "",
      "## 📬 Inbox",
      "",
      "### Focus — personal · family · finance, newest 5 per inbox",
      ...rows(5, (i) => `- Wexly ${i} · Card perks this week · finance [👎](iris:not-useful/offers${i}%40fabrikam.test)`),
      "",
      "## 📰 News",
      "",
      "### Trending repos",
      ...rows(4, (i) => `- [anthropics/${LONG_ID}-${i}](https://github.com/anthropics/${LONG_ID}) — ★1,240 today`),
      "",
      "### AI / Tech",
      ...rows(3, (i) => `- AI can only move at the speed of trust ${i} ([cnbc.com](https://www.cnbc.com/${LONG_ID}))`),
      "",
      "⚠ couldn't build: AI news (research engine timeout after 20 s) — everything else is current.",
      "",
      "---",
      "",
      "learned yesterday: nothing",
    ].join("\n"),
    failed_sections: [
      { name: "portfolio", title: "Portfolio", reason: "research engine timeout after 20 s" },
      { name: "ai_news", title: "AI news", reason: "research engine timeout after 20 s" },
    ],
  },
  // Inbox → Judged (loop-proof PR 5): made-up senders, a subject with no spaces to
  // break on, a long figure value, and enough rows to push the list past one screen.
  "/api/v1/email/judgments": {
    buckets: JUDGE_BUCKETS,
    judgments: rows(9, (i) => ({
      message_id: `${LONG_ID}-m${i}`,
      account_id: "gmail:owner@example.com",
      sender: i === 0 ? `Northwind Card Services International Billing Department ${i}` : `Sample Sender ${i}`,
      from_address: `statements-${LONG_ID}@northwind.example`,
      subject: i === 1 ? `Statement_${LONG_ID}_${LONG_ID}` : `Your statement is ready (${i})`,
      snippet: "New balance and minimum payment inside.",
      received_at: "2026-09-26T13:00:00Z",
      bucket: JUDGE_BUCKETS[i % JUDGE_BUCKETS.length].key,
      bucket_name: JUDGE_BUCKETS[i % JUDGE_BUCKETS.length].name,
      judge_bucket: "fyi",
      owner_bucket: i % 3 === 0 ? JUDGE_BUCKETS[i % JUDGE_BUCKETS.length].key : null,
      owner_source: i % 3 === 0 ? "gmail" : null,
      confidence: 0.52 + i / 100,
      figures: i % 2 ? { min_due: "$35.00", statement_balance: "$1,284.50", due_date: LONG_ID } : {},
      judged_at: "2026-09-26T13:05:00Z",
      corrected_at: null,
    })),
  },
  // Email setup (R4): the step-6 decision (the widest the screen gets: preview groups,
  // Approve / Decline, the Action Center link) beside a finished setup and a fresh account.
  "/api/v1/email/onboarding": overview(
    [waitingAtApproval(SETUP_ACCOUNT), setupAt("imap:done@example.com", "complete")],
    [SETUP_ACCOUNT, "imap:done@example.com", `imap:${LONG_ID}@example.com`],
  ),
  [`${onboardingPath(SETUP_ACCOUNT)}/label-preview`]: labelPreview(SETUP_ACCOUNT),
  "/api/digest": {
    digests: rows(6, (i) => ({
      id: `${LONG_ID.slice(0, 31)}${i}`,
      created_at: `2026-09-2${i}T12:00:00Z`,
      subject: "Morning digest",
      skill_id: "morning-brief",
      failed_sections: i % 3 === 0 ? 2 : 0,
    })),
  },
};

/** The fixture body for one request path. Exact match, else DEFAULT. */
export function fixtureFor(pathname: string): unknown {
  return pathname in OVERRIDES ? OVERRIDES[pathname] : DEFAULT;
}
