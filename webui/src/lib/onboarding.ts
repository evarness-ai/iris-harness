/* Email setup's API (OSS plan R4): the same state machine `iris email setup` drives.
 *
 *   GET  /api/v1/email/onboarding                        overview: steps, setups, accounts
 *   GET  /api/v1/email/onboarding/<id>/label-preview     step 6's preview (read-only)
 *   POST /api/v1/email/onboarding/<id>/advance           {until_waiting, ...} -> state
 *   POST /api/v1/email/onboarding/<id>/approve-writes    {approve} -> state (step 6 only)
 *   POST /api/v1/email/onboarding/<id>/restart           -> {restarted}
 *
 * Every decision is the server's (onboarding.py); this module only carries it. The
 * POSTs are gated writes (R17): a control-paired device or IRIS_WEBUI_ALLOW_WRITES. The
 * long steps (fetch, discover, the judge) run inside the POST, so a call can take
 * minutes; each account has one write in flight at a time (see `useSetupWrite`). */
import { useIsMutating, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "./http";

export interface SetupStep {
  step: string;
  title: string;
  done: boolean;
  current: boolean;
}

export interface SweepStatus {
  swept: boolean;
  state: string;
  reason: string;
  since: string | null;
}

export interface SetupState {
  account_id: string;
  run_id: string;
  provider: string;
  /** The current step, `complete` at the end. */
  step: string;
  status: "in_progress" | "waiting" | "done";
  /** `decision`: the owner decides. `blocked`: something outside setup comes first. */
  waiting_kind: "" | "decision" | "blocked";
  waiting_for: string;
  approval_id: string | null;
  steps: SetupStep[];
  results: Record<string, Record<string, unknown>>;
  /** Each finished step in plain lines: the words the CLI prints. */
  rendered: Record<string, string>;
  started_at: string;
  updated_at: string;
  completed_at: string | null;
  /** Shown once, never stored (a new vault key's `export` line). */
  notice: string;
  sweep: SweepStatus;
  /** The provider plugin's login for this account; "" when it has none. */
  connect_command: string;
}

export interface ConnectHint {
  provider: string;
  command: string;
}

export interface SetupOverview {
  steps: { step: string; title: string }[];
  setups: SetupState[];
  /** Connected mailbox accounts, setup or not. */
  accounts: string[];
  connect_hints: ConnectHint[];
  /** The synthetic mailbox's account id inside a demo home, else null. */
  demo_account: string | null;
}

export interface LabelGroup {
  bucket: string;
  label: string;
  count: number;
  samples: string[];
}

export interface LabelPreview {
  account_id: string;
  groups: LabelGroup[];
  removals: number;
  total: number;
  labelling: boolean;
  labels_enabled: boolean;
  approval_id: string | null;
  already_approved: boolean;
  /** Why nothing would be labelled, or "" when labels are due. */
  status: string;
  /** What IRIS never does, and how to take the approval back. */
  notes: string[];
  /** The approval card's lines (status, one per group, removals, notes). */
  lines: string[];
}

/** One proposed category, as the discover step recorded it. */
export interface Proposal {
  cluster_id: number;
  size: number;
  cohesion: number;
  path: string | null;
  acceptable: boolean;
  why_not: string;
  top_domain: string;
  samples: string[];
}

export interface AdvanceBody {
  create_master_key?: boolean;
  accept_categories?: number[];
}

const BASE = "/api/v1/email/onboarding";
const KEY = ["email-onboarding"] as const;

const accountPath = (accountId: string) => `${BASE}/${encodeURIComponent(accountId)}`;

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await apiFetch(url, { headers: { accept: "application/json" }, ...init });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: unknown };
      if (typeof j?.detail === "string") msg = j.detail;
    } catch {
      /* non-JSON (a proxy's error page) */
    }
    throw new Error(msg);
  }
  return (await r.json()) as T;
}

function post<T>(url: string, body: unknown): Promise<T> {
  return call<T>(url, {
    method: "POST",
    headers: { accept: "application/json", "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getOverview(): Promise<SetupOverview> {
  return call<SetupOverview>(BASE);
}

export function getLabelPreview(accountId: string): Promise<LabelPreview> {
  return call<LabelPreview>(`${accountPath(accountId)}/label-preview`);
}

/** Run until setup waits or completes (`until_waiting`). Never `accept_defaults`: on
 * the web every decision is the owner's own click. */
export function advance(accountId: string, body: AdvanceBody = {}): Promise<SetupState> {
  return post<SetupState>(`${accountPath(accountId)}/advance`, {
    ...body,
    until_waiting: true,
    actor: "web",
  });
}

export function approveWrites(accountId: string, approve: boolean): Promise<SetupState> {
  return post<SetupState>(`${accountPath(accountId)}/approve-writes`, { approve, actor: "web" });
}

export function restart(accountId: string): Promise<{ restarted: boolean }> {
  return post<{ restarted: boolean }>(`${accountPath(accountId)}/restart`, {});
}

export function useSetupOverview() {
  return useQuery({ queryKey: KEY, queryFn: getOverview });
}

export function useLabelPreview(accountId: string, enabled: boolean) {
  return useQuery({
    queryKey: [...KEY, accountId, "label-preview"],
    queryFn: () => getLabelPreview(accountId),
    enabled,
  });
}

export type SetupWrite =
  | { kind: "advance"; body?: AdvanceBody }
  | { kind: "approve"; approve: boolean }
  | { kind: "restart" };

const writeKey = (accountId: string) => [...KEY, "write", accountId] as const;

/** Whether a write for `accountId` is in flight, from any mount of the screen: a
 * request that outlives a navigation still blocks a second one when the owner comes
 * back. */
export function useSetupBusy(accountId: string): boolean {
  return useIsMutating({ mutationKey: writeKey(accountId) }) > 0;
}

/** One account's writes. The answer replaces that account's setup in the overview;
 * a failure re-reads it, because the server keeps working on a request whose
 * connection dropped and the state it saved is the truth. */
export function useSetupWrite(accountId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationKey: writeKey(accountId),
    mutationFn: (w: SetupWrite): Promise<SetupState | { restarted: boolean }> => {
      if (w.kind === "approve") return approveWrites(accountId, w.approve);
      if (w.kind === "restart") return restart(accountId);
      return advance(accountId, w.body);
    },
    onSuccess: (data) => {
      if ("account_id" in data) {
        qc.setQueryData<SetupOverview>(KEY, (old) =>
          old
            ? {
                ...old,
                setups: old.setups.some((s) => s.account_id === data.account_id)
                  ? old.setups.map((s) => (s.account_id === data.account_id ? data : s))
                  : [...old.setups, data],
              }
            : old,
        );
      }
    },
    onSettled: () => qc.invalidateQueries({ queryKey: KEY }),
  });
}
