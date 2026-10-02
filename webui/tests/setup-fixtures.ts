/* Email setup's API bodies (OSS plan R4), shaped as onboarding_api.py returns them.
 *
 * Shared by the viewport smoke (fixtures.ts serves a hostile overview: an unbreakable
 * account id, long subjects, every step rendered) and setup.spec.ts (a stateful fake).
 * All addresses and subjects are made up. */

export const STEP_TITLES: [string, string][] = [
  ["connect", "Connect the mailbox"],
  ["fetch", "Fetch recent mail"],
  ["discover", "Discover categories"],
  ["review_categories", "Review the categories"],
  ["classify", "Classify (the judge, then kNN)"],
  ["label_approval", "Label preview and approval"],
  ["first_digest", "First digest"],
  ["review_queue", "Review queue"],
  ["enable_sweep", "Keep it current"],
  ["summary", "What IRIS just did"],
];

type Json = Record<string, unknown>;

export interface SetupFixture {
  account_id: string;
  step: string;
  status: "in_progress" | "waiting" | "done";
  waiting_kind: "" | "decision" | "blocked" | "activity";
  waiting_for: string;
  approval_id: string | null;
  activity_id: string | null;
  results: Record<string, Json>;
  rendered: Record<string, string>;
  connect_command: string;
  notice: string;
  sweep: { swept: boolean; state: string; reason: string; since: string | null };
  [key: string]: unknown;
}

const RENDERED: Record<string, string> = {
  connect: "gmail account owner@example.com (vault key: env)",
  fetch:
    "fetched 412 email(s); 180 wait for the judge, 232 released at once (promotions, social, your own sent mail)",
  discover: "3 categories proposed",
  review_categories: "accepted 2: work/newsletters, finance/statements",
  classify:
    "judged 180 (40 Bill, 12 Event, 30 Needs reply, 98 FYI); 0 still wait for the next judge run. kNN filed 120, 8 left for review. No label written.",
  label_approval: "mailbox writes approved (appr-1 [audit #12]); 180 label(s) written, 0 failed",
  first_digest: "Good morning. 3 bills are due this week and 2 emails need a reply.",
  review_queue: "4 email(s) the judge was unsure of wait in the Action Center",
  enable_sweep: "email_sweep: on, daily at 06:15\nemail_judge: on, daily at 06:30",
  summary:
    "- Fetched 412 email(s).\n- Proposed 3 categories; you accepted 2.\n- Judged 180; kNN filed 120; 4 wait for you as Unsure.\n- Mailbox writes approved: 180 label(s) written.\n- Next: the sweep runs daily at 06:15, the judge daily at 06:30.",
};

export const PROPOSALS = [
  {
    cluster_id: 0,
    size: 64,
    cohesion: 0.71,
    path: "work/newsletters",
    acceptable: true,
    why_not: "",
    top_domain: "news.example.com",
    samples: ["Weekly roundup", "What changed this week", "Your digest"],
  },
  {
    cluster_id: 1,
    size: 41,
    cohesion: 0.66,
    path: "finance/statements",
    acceptable: true,
    why_not: "",
    top_domain: "bank.example.com",
    samples: ["Your statement is ready", "Payment received"],
  },
  {
    cluster_id: 2,
    size: 12,
    cohesion: 0.31,
    path: null,
    acceptable: false,
    why_not: "cluster 2: too loose to name (cohesion 0.31)",
    top_domain: "misc.example.com",
    samples: ["Hello"],
  },
];

/** A setup at `step`; every step before it finished (results + rendered). */
export function setupAt(
  accountId: string,
  step: string,
  extra: Partial<SetupFixture> = {},
): SetupFixture {
  const index = step === "complete" ? STEP_TITLES.length : STEP_TITLES.findIndex(([s]) => s === step);
  const done = STEP_TITLES.slice(0, index).map(([s]) => s);
  const results: Record<string, Json> = {};
  const rendered: Record<string, string> = {};
  for (const s of done) {
    results[s] = s === "discover" ? { corpus: 412, proposals: PROPOSALS } : { ok: true };
    rendered[s] = RENDERED[s];
  }
  return {
    account_id: accountId,
    run_id: "onboard-0123456789ab",
    provider: accountId.split(":")[0],
    step,
    status: step === "complete" ? "done" : "in_progress",
    waiting_kind: "",
    waiting_for: "",
    approval_id: null,
    activity_id: null,
    steps: STEP_TITLES.map(([s, title]) => ({
      step: s,
      title,
      done: done.includes(s),
      current: s === step,
    })),
    results,
    rendered,
    started_at: "2026-09-30T08:00:00Z",
    updated_at: "2026-09-30T08:05:00Z",
    completed_at: step === "complete" ? "2026-09-30T08:09:00Z" : null,
    notice: "",
    sweep:
      step === "complete"
        ? { swept: true, state: "released", reason: "email setup turned the sweep on", since: null }
        : {
            swept: false,
            state: "held",
            reason: "email setup (onboard-0123456789ab) has not reached 'Keep it current'",
            since: null,
          },
    connect_command: `iris auth ${accountId.split(":")[0]} login --user ${accountId.split(":")[1] ?? ""}`,
    ...extra,
  };
}

export function waitingAtApproval(accountId: string, approvalId = "appr-7f3c9e10"): SetupFixture {
  return setupAt(accountId, "label_approval", {
    status: "waiting",
    waiting_kind: "decision",
    waiting_for: `approve the label preview (approval ${approvalId}) to let IRIS change ${accountId}, or decline to keep it read-only`,
    approval_id: approvalId,
  });
}

/** A long step (fetch, the judge) running in the background on the Activity spine
 * (issue #67): the screen polls this rather than waiting out one long request. */
export function waitingOnActivity(
  accountId: string,
  step: string,
  activityId = "act-fetch-1",
  progress = "fetching mail…",
): SetupFixture {
  return setupAt(accountId, step, {
    status: "waiting",
    waiting_kind: "activity",
    waiting_for: progress,
    activity_id: activityId,
  });
}

export function labelPreview(accountId: string, approvalId = "appr-7f3c9e10"): Json {
  const groups = [
    { bucket: "bill", label: "IRIS/Bill", count: 40, samples: ["Your statement is ready", "Invoice 2026-09"] },
    { bucket: "needs_reply", label: "IRIS/Needs reply", count: 30, samples: ["Can you confirm Tuesday?"] },
    { bucket: "fyi", label: "IRIS/FYI", count: 98, samples: [] },
  ];
  const notes = [
    "IRIS never archives, never marks read and never deletes on its own.",
    `Take it back any time: iris email writes revoke --account ${accountId}`,
  ];
  return {
    account_id: accountId,
    groups,
    removals: 2,
    total: 168,
    labelling: true,
    labels_enabled: true,
    approval_id: approvalId,
    already_approved: false,
    status: "",
    notes,
    lines: [...groups.map((g) => `${g.label}: ${g.count} email(s)`), ...notes],
  };
}

export const CONNECT_HINTS = [
  { provider: "gmail", command: "iris auth gmail login --user you@example.com" },
  { provider: "imap", command: "iris auth imap login --user you@example.com" },
];

export function overview(setups: SetupFixture[], accounts: string[] = [], demo: string | null = null): Json {
  return {
    steps: STEP_TITLES.map(([step, title]) => ({ step, title })),
    setups,
    accounts,
    connect_hints: CONNECT_HINTS,
    demo_account: demo,
  };
}

export const onboardingPath = (accountId: string) =>
  `/api/v1/email/onboarding/${encodeURIComponent(accountId)}`;
