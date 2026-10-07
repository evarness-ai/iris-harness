/* Read-only control-surface client (Phase 3).
 *
 * Thin typed wrappers over the IRIS API's operator endpoints — LLM mode +
 * pressure, self-learning metrics, routines, heartbeats. All GET / read-only;
 * UI-driven writes (approve a routine, pin a mode, trigger a heartbeat) come in
 * a later phase behind the approval flow. Like the trace/session client these
 * have no fallback data: when the API is down the React Query hook surfaces an
 * error state and the screen shows an "API unavailable" notice. */
import { apiFetch } from "./http";

export interface Capabilities {
  writes_enabled: boolean;
  feedback_capture?: boolean;
  /* IRIS_DEPLOYMENT_LABEL on the server; "" when unset (or absent on an older API). */
  deployment_label?: string;
}

async function getJSON<T>(url: string): Promise<T> {
  const r = await apiFetch(url, { headers: { accept: "application/json" } });
  if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`.trim());
  return (await r.json()) as T;
}

// ---- LLM mode + pressure (governor) -------------------------------------

export interface GovernorSnapshot {
  ram_free_gb: number;
  cpu_percent: number;
  cpu_speed_limit: number | null;
  thermal_throttled: boolean;
  sampled_at: string;
}

export interface LlmMode {
  mode: string;
  auto_mode: string;
  pin: string | null;
  adaptive: boolean;
  snapshot: GovernorSnapshot | null;
}

export const getCapabilities = () => getJSON<Capabilities>("/capabilities");

export const getLlmMode = () => getJSON<LlmMode>("/llm/mode");

async function writeOk(url: string, init: RequestInit): Promise<void> {
  const r = await apiFetch(url, init);
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: string };
      if (j?.detail) msg = j.detail;
    } catch {
      /* non-JSON */
    }
    throw new Error(msg);
  }
}

/** Pin the LLM mode (active | idle). Reversible via unpinLlmMode. */
export function pinLlmMode(mode: string): Promise<void> {
  return writeOk("/llm/mode/pin", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ mode }),
  });
}

export function unpinLlmMode(): Promise<void> {
  return writeOk("/llm/mode/pin", { method: "DELETE" });
}

// ---- Self-learning / LLM metrics ----------------------------------------

export interface MetricsBackend {
  kind: string;
  enabled: boolean;
  healthy: boolean;
  endpoint: string | null;
  instrumented_targets: string[];
  error: string | null;
}

export interface LearningSummary {
  sessions_scanned: number;
  llm_call_count: number;
  llm_error_count: number;
  total_tokens: number;
  total_duration_ms: number;
  provider_counts: Record<string, number>;
  model_counts: Record<string, number>;
  learning?: Record<string, unknown>;
}

export interface LearningMetrics {
  backend: MetricsBackend;
  summary: LearningSummary;
}

/** 404 means the operator hasn't enabled metrics — a normal state, not an error. */
export async function getLearningMetrics(): Promise<LearningMetrics | { disabled: true }> {
  const r = await apiFetch("/observability/llm-metrics", { headers: { accept: "application/json" } });
  if (r.status === 404) return { disabled: true };
  if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`.trim());
  return (await r.json()) as LearningMetrics;
}

export function isMetricsDisabled(
  m: LearningMetrics | { disabled: true } | undefined,
): m is { disabled: true } {
  return Boolean(m && "disabled" in m);
}

// ---- Routines ------------------------------------------------------------

export type RoutineStatus =
  | "draft"
  | "clarify"
  | "template"
  | "approved"
  | "scheduled"
  | "paused"
  | "retired";

export interface Routine {
  id: string;
  title: string;
  goal: string;
  schedule: string;
  template: string;
  delivery_channel: string;
  approval_status: RoutineStatus;
  run_count: number;
  success_count: number;
  failure_count: number;
  last_run_at: string | null;
  promotion_candidate: boolean;
  source_preferences: string[];
  required_capabilities: string[];
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export async function getRoutines(): Promise<Routine[]> {
  const data = await getJSON<{ routines: Routine[] }>("/routines");
  return data.routines ?? [];
}

/** Change a routine's approval status (approve / pause / resume). */
export function updateRoutineStatus(id: string, approvalStatus: RoutineStatus): Promise<void> {
  return writeOk(`/routines/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ approval_status: approvalStatus }),
  });
}

// ---- Settings catalog (ADR-0120) ------------------------------------------

export interface CatalogSetting {
  name: string;
  /* "core", "plugin:<name>" or "proxy". */
  owner: string;
  kind: string;
  default: unknown;
  applies: "now" | "next_run" | "restart";
  label: string;
  description: string;
  tab: string;
  guarded: boolean;
  guard_reason: string;
  editable: boolean;
  not_editable_reason: string;
  is_set: boolean;
  /* null for secrets, paths and URLs: the API never sends those values. */
  value: string | null;
  /* Changed from the app (saved on the data volume), and the deploy's own value. */
  overridden?: boolean;
  deploy_value?: string | null;
}

export async function getSettingsCatalog(): Promise<CatalogSetting[]> {
  const data = await getJSON<{ settings: CatalogSetting[] }>("/settings/catalog");
  return data.settings ?? [];
}

export type SettingResult = CatalogSetting & { restart_required: boolean };

/** Change one setting. A guarded one needs ``confirm`` (the API refuses without it). */
export const patchSetting = (name: string, value: string | boolean | number, confirm = false) =>
  postJSON<SettingResult>(`/settings/${encodeURIComponent(name)}`, { value, confirm }, "PATCH");

/** Back to what the deploy's environment says. */
export const resetSetting = (name: string) =>
  postJSON<SettingResult>(`/settings/${encodeURIComponent(name)}`, undefined, "DELETE");

// ---- Models (llm_tiers.yaml edits, ADR-0120) -----------------------------

export interface TierFields {
  provider: string;
  model: string;
  max_tokens: number;
  temperature: number;
  timeout_seconds: number;
}

export interface ModelTier extends TierFields {
  name: string;
  title: string;
  use_for: string[];
  /* The file's own values; ``changed`` names the fields the owner edited. */
  file: TierFields;
  changed: (keyof TierFields)[];
}

export interface ModelsView {
  tiers: ModelTier[];
  intents: Record<string, string>;
  moved_intents: string[];
  providers: string[];
}

export const getModels = () => getJSON<ModelsView>("/models");

export const patchTier = (name: string, fields: Partial<TierFields>, confirm = false) =>
  postJSON<ModelsView>(
    `/models/tiers/${encodeURIComponent(name)}`,
    { ...fields, confirm },
    "PATCH",
  );

export const resetTier = (name: string) =>
  postJSON<ModelsView>(`/models/tiers/${encodeURIComponent(name)}/override`, undefined, "DELETE");

export const moveIntent = (intent: string, tier: string, confirm = false) =>
  postJSON<ModelsView>(`/models/intents/${encodeURIComponent(intent)}`, { tier, confirm }, "PATCH");

export const resetIntent = (intent: string) =>
  postJSON<ModelsView>(
    `/models/intents/${encodeURIComponent(intent)}/override`,
    undefined,
    "DELETE",
  );
// ---- Health watch (health_watch.yaml edits, ADR-0120) ---------------------

export interface WatchField {
  name: string;
  kind: "int" | "float" | "bool";
  min: number | null;
  max: number | null;
  value: number | boolean;
  file: number | boolean;
  changed: boolean;
}

export interface WatchConfigView {
  fields: WatchField[];
  enabled: boolean;
  running: boolean;
}

export const getWatchConfig = () => getJSON<WatchConfigView>("/health/watch/config");
export const patchWatchConfig = (changes: Record<string, number | boolean>) =>
  postJSON<WatchConfigView>("/health/watch/config", changes, "PATCH");
export const resetWatchConfig = () =>
  postJSON<WatchConfigView>("/health/watch/config", undefined, "DELETE");

// ---- Morning digest (config/digest.yaml edits, ADR-0120, loop-proof D4) ---------

export interface DigestFields {
  enabled: boolean;
  time: string;
  channel: string;
  sections: string[];
  sections_off: string[];
  section_config: Record<string, { line_cap?: number } & Record<string, unknown>>;
  /** Topics for a news fetch that names no group; the digest's news uses news_groups. */
  news_topics: string[];
  /** Each news section's heading and topics; either may say {news_local_area}. */
  news_groups: Record<string, { title: string; topics: string[] }>;
  /** The place the local news is about ("St. Louis"). */
  news_local_area: string;
  news_sources: string[];
  /** ISO 639-1 code the news is in ("en"), or "any". */
  news_language: string;
  focus_categories: string[];
  focus_limit: number;
  /** The newest this many per inbox, then newest first up to focus_limit. */
  focus_per_account: number;
}

export interface DigestConfigView {
  fields: DigestFields;
  file: DigestFields;
  changed: (keyof DigestFields)[];
  all_sections: string[];
  locked_sections: string[];
  /** digest.yaml's layout (read-only): the groups the sections render under. */
  groups: { id: string; title: string; icon: string; sections: string[] }[];
  /** A section's heading where it has its own (the news groups: "Local — St. Louis"). */
  section_titles: Record<string, string>;
  timezone: string;
  migration: { at: string; status: string; dropped: string[] } | null;
}

export const getDigestConfig = () => getJSON<DigestConfigView>("/digest/config");
export const patchDigestConfig = (changes: Partial<DigestFields>) =>
  postJSON<DigestConfigView>("/digest/config", changes, "PATCH");
export const resetDigestConfig = () =>
  postJSON<DigestConfigView>("/digest/config", undefined, "DELETE");

export interface RestartStatus {
  supervised: boolean;
  started_at: string;
  waiting_for_restart: string[];
}

export const getRestartStatus = () => getJSON<RestartStatus>("/system/restart");

/** Ask every server to restart (only where a supervisor brings them back). */
export const requestRestart = () =>
  postJSON<{ restarting: boolean; requested_at: string }>(
    "/system/restart",
    { confirm: true },
    "POST",
  );

export interface SettingChangeRow {
  id: number;
  at: string;
  section: string;
  key: string;
  action: "set" | "reset";
  old: unknown;
  new: unknown;
  actor: string;
  actor_name: string | null;
}

export const getSettingsHistory = (limit = 100) =>
  getJSON<{ count: number; changes: SettingChangeRow[] }>(`/settings/history?limit=${limit}`);

// ---- Heartbeats ----------------------------------------------------------

export interface Heartbeat {
  name: string;
  schedule: string;
  enabled: boolean;
  description: string;
  /* ADR-0120: the app may change schedule + enabled; these describe the change. */
  schedule_text?: string;
  default_schedule?: string;
  default_enabled?: boolean;
  overridden?: boolean;
  /* false when this harness has no handler for it (e.g. a Mac-only job on the VM). */
  runnable?: boolean;
  /* Why it cannot run here, e.g. "needs darwin; this harness runs on linux". */
  unavailable_reason?: string | null;
  platforms?: string[];
  next_run_at?: string | null;
  /* Loop-proof D13, from the runs kept in heartbeat_runs.db (they survive a restart).
   * Null where this harness keeps no runs. */
  watched?: boolean;
  last_run?: HeartbeatRun | null;
  last_success_at?: string | null;
  job?: HeartbeatJobStatus | null;
}

/** Did the heartbeat's most recent scheduled slot run? `detail` is the server's
 * sentence in the owner's zone: "Missed 12:15 — last success 06:15", "ran 06:15 ✓ (23 new)". */
export interface HeartbeatJobStatus {
  state: "green" | "yellow" | "red" | "grey";
  detail: string;
  slot: string | null;
  missed: boolean;
  ran: boolean;
  last_error: string;
}

export interface HeartbeatList {
  heartbeats: Heartbeat[];
  /* The zone the scheduler reads a daily/cron time in. */
  timezone: string | null;
}

export interface HeartbeatRun {
  name: string;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  output: string | null;
  error: string | null;
  /* Kept runs only: the slot it served, how it started, the handler's counts. */
  slot?: string | null;
  trigger?: string;
  result?: Record<string, unknown>;
}

export async function getHeartbeats(): Promise<HeartbeatList> {
  const data = await getJSON<Partial<HeartbeatList>>("/heartbeat");
  return { heartbeats: data.heartbeats ?? [], timezone: data.timezone ?? null };
}

export interface HeartbeatPatch {
  schedule?: string;
  enabled?: boolean;
}

/** Change a heartbeat's schedule and/or turn it on or off; applies at once (ADR-0120). */
export const patchHeartbeat = (name: string, patch: HeartbeatPatch) =>
  postJSON<Heartbeat>(`/heartbeat/${encodeURIComponent(name)}`, patch, "PATCH");

/** Back to the schedule the deploy ships. */
export const resetHeartbeat = (name: string) =>
  postJSON<Heartbeat>(`/heartbeat/${encodeURIComponent(name)}/override`, undefined, "DELETE");

export async function getHeartbeatRuns(limit = 50): Promise<HeartbeatRun[]> {
  const data = await getJSON<{ runs: HeartbeatRun[] }>(`/heartbeat/runs?limit=${limit}`);
  return data.runs ?? [];
}

export interface HeartbeatTriggerResult {
  name: string;
  status: string;
  output: string | null;
  error: string | null;
}

/** Run a heartbeat now (manual trigger). */
export async function triggerHeartbeat(name: string): Promise<HeartbeatTriggerResult> {
  const r = await apiFetch(`/heartbeat/trigger/${encodeURIComponent(name)}`, { method: "POST" });
  if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`.trim());
  return (await r.json()) as HeartbeatTriggerResult;
}

// ---- Activities (async background jobs, read-only) ------------------------

export interface Activity {
  id: string;
  kind: string;
  title: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  progress_message: string;
  origin: string;
  result_summary: string;
  error: string;
  undo_ref: string | null;
  metadata: Record<string, unknown>;
  started_at: string | null;
  finished_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface ActivitiesResponse {
  count: number;
  running: number;
  activities: Activity[];
}

/** Durable feed of background system jobs (FileManager categorize/cleanup/
 * organize). Pure read; the in-process ActivityRunner is the sole writer. */
export const getActivities = () => getJSON<ActivitiesResponse>("/activities");

// ---- Governance (read-only) ----------------------------------------------

export interface GovFlag {
  key: string;
  label: string;
  on: boolean;
}

export interface GovernanceState {
  enabled: boolean;
  audit_db: string;
  audit_count: number;
  write_health?: AuditWriteHealth;
  flags: GovFlag[];
}

/** How the ledger's own writes are doing (counts only; issue #134). */
export interface AuditWriteHealth {
  ok: boolean;
  spool_pending: number;
  spool_rejected: number;
  writes_spooled: number;
  writes_lost: number;
  last_error_class: string | null;
}

export interface AuditEntry {
  id: number;
  ts: string;
  run_id: string;
  step_id?: number | null;
  agent_type: string;
  hook_point: string;
  plugin: string;
  decision: string;
  classification: string | null;
  tier: string | null;
  /** Where the tier runs: tier_3 is cloud, any other tier is local. */
  locality?: "local" | "cloud" | null;
  severity: string;
  /** Email addresses masked by the server (display_mask); the ledger keeps the original. */
  reason: string;
  /* The documented payload fields only (foundation/observability/audit_view.py). */
  caller?: string;
  deterministic?: boolean;
  handler?: string;
  tool_name?: string;
  capability?: string;
  method?: string;
  capability_provider?: string;
  digest_alg?: string;
  session_id?: string;
  audience?: string;
}

export interface AuditResponse {
  count: number;
  total: number;
  audit_db: string;
  /** Every caller the ledger names, for the filter. */
  callers?: string[];
  entries: AuditEntry[];
}

export const getGovernanceState = () => getJSON<GovernanceState>("/governance/state");

export const getGovernanceAudit = (decision?: string, caller?: string) => {
  const q = new URLSearchParams();
  if (decision) q.set("decision", decision);
  if (caller) q.set("caller", caller);
  const qs = q.toString();
  return getJSON<AuditResponse>(`/governance/audit${qs ? `?${qs}` : ""}`);
};

/** Owner-PII shadow: what the guards would have done (counts only, never a literal). */
export interface PiiShadowCell {
  hook_point: string;
  guard: string;
  kind: string;
  action: string;
  first_name_alone: boolean;
  log_only_destination: boolean;
  occurrences: number;
  calls: number;
  distinct: number | null;
}

export interface PiiShadowSummary {
  mode: string;
  since: string;
  rows: number;
  checked: Record<string, number>;
  unobserved: Record<string, number>;
  cells: PiiShadowCell[];
}

export const getPiiShadow = (days = 7) =>
  getJSON<PiiShadowSummary>(`/governance/pii-shadow?days=${days}`);

/** The R14 proof bundle of a recent window, verified per invariant. */
export interface ProofInvariant {
  id: string;
  statement: string;
  ok: boolean;
  violation_count: number;
  violations: string[];
  evidence: Record<string, number>;
}

export interface ProofBundleCheck {
  since: string | null;
  until: string | null;
  ledger_rows: number;
  ok: boolean;
  integrity: string[];
  invariants: ProofInvariant[];
}

export const getProofBundleCheck = (days = 7) =>
  getJSON<ProofBundleCheck>(`/governance/proof-bundle/check?days=${days}`);

// ---- HITL approval queue -------------------------------------------------
//
// A `require_approval` verdict halts a run and waits for a human. The queue is
// channel-agnostic and so is the capability behind these two calls — the CLI
// (`iris approvals`) and the Telegram handler go through the same server-side
// function, so answering here does exactly what answering there does.

/** How a destructive-tool approval reads to the owner (ADR-0118 step 4): the plugin's
 * title and one line per item, the declared undo, and what they asked. Frozen when the
 * approval was created, so it is exactly what the owner is approving. */
export interface ApprovalCard {
  title: string;
  lines: string[];
  undo_tool: string | null;
  undo_window_days: number | null;
  asked: string | null;
  /** "write" for a write approved per call (sending an email); absent or
   * "destructive" for data loss (ADR-0118 amendment). */
  effect?: "destructive" | "write" | null;
}

export interface PendingApproval {
  approval_id: string;
  run_id: string;
  signal: string;
  context_summary: string;
  requested_at: string;
  timeout_at: string;
  channel: string;
  status: string;
  checkpoint_id: string | null;
  /** Whether approving can actually continue the run, so we don't promise a resume. */
  resumable: boolean;
  /** Past its deadline but not yet swept by approval_timeout_tick (runs every 60s). */
  overdue: boolean;
  session_id: string | null;
  /** "destructive": it deletes or overwrites data and pins exact calls; "evaluator": a paused run. */
  kind: "destructive" | "evaluator";
  card: ApprovalCard | null;
  /** The exact calls approving will run, for "Show the exact call". */
  items: { tool: string; args: Record<string, unknown> }[];
}

export interface ApprovalsResponse {
  count: number;
  approvals: PendingApproval[];
}

export interface ApprovalOutcome {
  approval_id: string;
  run_id: string;
  status: string;
  signal: string;
  responded_at: string | null;
  response_actor: string | null;
  checkpoint_id: string | null;
  /** True when the halted run was continued; `detail` is then its answer. */
  resumed: boolean;
  detail: string;
}

export const getApprovals = () => getJSON<ApprovalsResponse>("/governance/approvals");

/** Approve or reject. Approving continues the halted run, so this can take a while. */
export async function respondToApproval(
  approvalId: string,
  status: "approved" | "rejected",
): Promise<ApprovalOutcome> {
  const r = await apiFetch(`/governance/approvals/${encodeURIComponent(approvalId)}/respond`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ status, actor: "web" }),
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: string };
      if (j?.detail) msg = j.detail;
    } catch {
      /* non-JSON */
    }
    throw new Error(msg);
  }
  return (await r.json()) as ApprovalOutcome;
}

// ---- Settings (read-only) ------------------------------------------------

export interface TierInfo {
  name: string;
  provider: string;
  model: string;
  max_tokens: number;
  temperature: number;
  use_for: string[];
}

export interface Settings {
  tiers: TierInfo[];
  intent_tier_map: Record<string, string>;
  providers: Record<string, boolean>;
  paths: Record<string, string>;
  host: { ram_total_gb: number; ram_free_gb: number; cpu_percent: number; thermal_throttled: boolean };
  stores: {
    skill_count: number;
    heartbeat_count: number;
    database_sizes: Record<string, number>;
    filemanager_roots: number;
    accounts: Record<string, number>;
  };
  flags: GovFlag[];
}

export const getSettings = () => getJSON<Settings>("/settings");

// ---- Conversation memory (read-only) -------------------------------------

export interface MemoryTurn {
  role: string;
  content: string;
}

export interface MemoryResponse {
  session_id: string;
  turn_count: number;
  turns: MemoryTurn[];
}

export const getMemory = (sessionId: string) =>
  getJSON<MemoryResponse>(`/memory/${encodeURIComponent(sessionId)}`);

// ---- Fact-history retention review (human-reviewed; never auto-deletes) ----

export interface FactHistoryEntry {
  /** Statement id (memris PR 2c-ii); "<id>#forget" for a forget event. */
  id: string;
  key: string;
  old_value: string | null;
  new_value: string | null;
  source: string;
  reason: string;
  changed_at: string;
}

export interface FactHistoryRetention {
  older_than_days: number;
  total: number;
  count: number;
  entries: FactHistoryEntry[];
}

/** Review queue: history entries older than the window. Read-only (never deletes). */
export const getFactHistoryRetention = (olderThanDays = 180, limit = 200) =>
  getJSON<FactHistoryRetention>(
    `/memory/history/retention?older_than_days=${olderThanDays}&limit=${limit}`,
  );

/** Prune chosen history entries (terminal). Write-gated by IRIS_WEBUI_ALLOW_WRITES. */
export async function pruneFactHistory(entryIds: string[]): Promise<{ removed: number }> {
  const r = await apiFetch("/memory/history/prune", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ entry_ids: entryIds }),
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: string };
      if (j?.detail) msg = j.detail;
    } catch {
      /* non-JSON */
    }
    throw new Error(msg);
  }
  return (await r.json()) as { removed: number };
}

// ---- Fact contradictions review (detected same-key conflicts) -------------

export interface FactContradiction {
  /** The statement that replaced, or was refused (memris PR 2c-ii). */
  id: string;
  key: string;
  stored_value: string;
  stored_confidence: number | null;
  incoming_value: string;
  incoming_confidence: number | null;
  resolution: string; // superseded | blocked
  source: string;
  detected_at: string;
  seen_count: number;
}

export interface FactContradictions {
  count: number;
  contradictions: FactContradiction[];
}

/** Review queue: detected same-key value conflicts. Read-only. */
export const getContradictions = (includeAcknowledged = false) =>
  getJSON<FactContradictions>(
    `/memory/contradictions?include_acknowledged=${includeAcknowledged}`,
  );

/** Acknowledge reviewed conflicts. Write-gated by IRIS_WEBUI_ALLOW_WRITES. */
export async function ackContradictions(ids: string[]): Promise<{ acknowledged: number }> {
  const r = await apiFetch("/memory/contradictions/ack", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ ids }),
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: string };
      if (j?.detail) msg = j.detail;
    } catch {
      /* non-JSON */
    }
    throw new Error(msg);
  }
  return (await r.json()) as { acknowledged: number };
}

// ---- System Health (ADR-0069, read-only) ---------------------------------

export type HealthState = "green" | "yellow" | "red" | "grey";

export interface HealthCheck {
  target: string;
  kind: "service" | "credential" | "hardware" | "plugin" | "governance";
  state: HealthState;
  detail: string;
  endpoint: string | null;
  /** Exact remediation command for a red check (e.g. `iris auth gmail login`). */
  action: string | null;
  /** The one thing under `target` the row is about (a Gmail row's account). */
  subject?: string | null;
  /** The console page that fixes this row, e.g. `/settings#connections`. */
  fix_url?: string | null;
  /** How to reconnect the credential, when its plugin offers it (lib/connections.ts). */
  reconnect?: Reconnect | null;
}

/** A reconnectable credential row's descriptor; the server names every value. */
export interface Reconnect {
  route: string;
  setup_route: string;
  group: string;
  group_label: string;
  provider: string;
  label: string;
  account: string | null;
}

export interface HealthSnapshot {
  state: HealthState; // worst across all checks
  summary: string; // e.g. "11 green, 2 grey"
  sampled_at: string;
  checks: HealthCheck[];
  alerts: HealthCheck[]; // red projection — what the banner shows
}

export const getHealth = () => getJSON<HealthSnapshot>("/health");

// ---- Health watch incidents (ADR-0116) ------------------------------------

export interface HealthRepair {
  at: string;
  tried: string; // e.g. "token refresh", "re-run email_sweep", "restart governor"
  ok: boolean;
  detail: string;
  final: boolean; // retrying could not help (a revoked token)
}

export interface HealthIncident {
  id: number;
  key: string;
  target: string;
  subject: string | null;
  kind: string;
  state: "repairing" | "needs_user" | "resolved";
  detail: string;
  action: string | null;
  opened_at: string;
  updated_at: string;
  attempts: number;
  repairs: HealthRepair[];
  notified_at: string | null;
  notify_count: number;
  resolved_at: string | null;
  resolution: "self_healed" | "user_fixed" | null;
}

export interface HealthIncidents {
  enabled: boolean; // false when the watch is off or not installed
  count: number;
  incidents: HealthIncident[]; // newest first
}

/** LLM spend from the governance cost ledger (GET /cost, Track 2 PR 6).
 *
 * `recording` is not decoration: `CostLimiter` is opt-in and is the ledger's
 * only writer, so a 0.00 from a ledger nobody writes looks exactly like a
 * genuine 0.00. The card renders the two differently. */
export interface CostSummary {
  recording: boolean;
  /** Only meaningful while recording: does passing the cap refuse calls. */
  enforcing: boolean;
  /** The cap in force, or null when recording without enforcing. */
  daily_cap_usd: number | null;
  user_id: string;
  today_usd: number;
  month_usd: number;
  by_tier_usd: Record<string, number>;
  entries: number;
  ledger_db: string;
  enable_hint: string | null;
}

export const getCost = () => getJSON<CostSummary>("/cost");

export const getHealthIncidents = (limit = 25) =>
  getJSON<HealthIncidents>(`/health/incidents?limit=${limit}`);

export interface HealthWatchPass {
  state: HealthState;
  summary: string;
  events: string[];
  open: HealthIncident[];
}

/** Run one health-watch pass now: refresh, repair, notify (ADR-0116). Write-gated. */
export function runHealthWatch(): Promise<HealthWatchPass> {
  return postJSON<HealthWatchPass>("/health/watch");
}

// ---- System Check (iris doctor's install preflight, OSS plan R5) ---------

export interface DoctorCheck {
  name: string;
  status: "pass" | "warn" | "fail" | "info";
  state: string;
  detail: string;
  fix: string | null;
  /** What a failing check stops: real use, everything (incl. the demo), or nothing. */
  blocks: "all" | "use" | "none";
}

export interface StarterModel {
  name: string;
  size_gb: number;
}

export interface DoctorReport {
  verdict: "ready" | "demo_only" | "not_ready";
  exit_code: number;
  checks: DoctorCheck[];
  missing_models: StarterModel[];
  key: { source: string; detail: string };
  ollama_url: string;
  /** Something a fix (a model pull, this screen's one write) could do. */
  fixable: boolean;
}

/** The same preflight `iris doctor` prints: hardware, Ollama, the starter model,
 * the vault key. Pure read. */
export const getDoctorReport = () => getJSON<DoctorReport>("/health/doctor");

/** Pull a missing starter model in the background (write-gated): returns the
 * Activity id to poll through `useActivities()`. */
export function pullStarterModel(model: string): Promise<{ activity_id: string }> {
  return postJSON<{ activity_id: string }>("/health/doctor/pull-model", { model });
}

// ---- Setup (iris setup's own progress) -----------------------------------

export type SetupStepName = "preflight" | "home_secret" | "services" | "telegram" | "email";
export type SetupStepStatus = "done" | "skipped" | "failed";

export interface SetupStepRecord {
  status: SetupStepStatus;
  at: string;
  detail: string;
}

export interface SetupProgress {
  order: SetupStepName[];
  mandatory: SetupStepName[];
  mandatory_done: boolean;
  next_step: SetupStepName | null;
  steps: Partial<Record<SetupStepName, SetupStepRecord>>;
}

/** `iris setup`'s own progress (`$IRIS_HOME/setup.json`), read-only -- running or
 * resuming a step stays the CLI's. */
export const getSetupProgress = () => getJSON<SetupProgress>("/health/setup");

// ---- Context health (ADR-0081) ------------------------------------------

export interface ContextHealth {
  available: boolean;
  session_id?: string;
  window?: {
    budget_tokens: number;
    current_tokens: number;
    fill_pct: number;
    compaction_ratio: number;
    near_full: boolean;
    last_compaction: {
      trigger: string;
      archived_count: number;
      tokens_before: number;
      tokens_after: number;
      kept_turns: number;
    } | null;
  };
  budgets?: {
    transcript_budget: number;
    memory_budget: number;
    last_context_tokens: number | null;
    last_transcript_evicted: number | null;
  };
  suppression?: {
    total_feedback: number;
    active_suppressions: number;
    by_subsystem: Record<string, number>;
  };
}

export const getContextHealth = (sessionId = "default") =>
  getJSON<ContextHealth>(`/context-health?session_id=${encodeURIComponent(sessionId)}`);

/** Force-compact a session's conversation now (ADR-0084). Write-gated. */
export function compactContext(sessionId = "default"): Promise<void> {
  return writeOk(`/context-health/compact?session_id=${encodeURIComponent(sessionId)}`, {
    method: "POST",
  });
}

// ---- Learning experiment console (ADR-0083) -----------------------------

export interface LearningFlag {
  enabled: boolean; // live runtime state
  env_default: boolean; // what a restart reverts to
}

export interface LearningFlags {
  available: boolean;
  flags: Record<string, LearningFlag>;
}

export const getLearningFlags = () => getJSON<LearningFlags>("/learning/flags");

export interface ProposalQualityRow {
  subsystem: string;
  awaiting: number;
  accepted: number;
  rejected: number;
  reviewed: number;
  acceptance_rate: number;
}

export interface ProposalQuality {
  available: boolean;
  behaviors?: ProposalQualityRow;
  intentions?: ProposalQualityRow;
}

export const getProposalQuality = () => getJSON<ProposalQuality>("/learning/proposal-quality");

/** Hot-toggle a learning capability (no restart). Write-gated. */
export function setLearningFlag(name: string, enabled: boolean): Promise<void> {
  return writeOk(`/learning/flags/${encodeURIComponent(name)}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
}

/** Run a learning capability now instead of waiting for its heartbeat. Write-gated. */
export function runLearningNow(name: string): Promise<void> {
  return writeOk(`/learning/flags/${encodeURIComponent(name)}/run`, { method: "POST" });
}

// ---- Tasks + reminders (chat outstanding controls) -----------------------

export type TaskStatus = "open" | "doing" | "done" | "dropped" | "expired"; // expired: aged out (digest.yaml expiry)

export interface UserTask {
  id: string;
  title: string;
  status: TaskStatus;
  source_kind: string | null;
  due_at: string | null;
  updated_at: string;
}

export interface TasksResponse {
  count: number;
  tasks: UserTask[];
}

/** Add a todo (POST /tasks). Write-gated; the server forces source_kind=manual. */
export function createTask(input: { title: string; description?: string; due_at?: string }) {
  return writeOk("/tasks", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(input),
  });
}

export function getTasks(status?: TaskStatus) {
  const q = status ? `?status=${encodeURIComponent(status)}` : "";
  return getJSON<TasksResponse>(`/tasks${q}`);
}

export function updateTaskStatus(id: string, status: TaskStatus): Promise<void> {
  return writeOk(`/tasks/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ status }),
  });
}

/* The one reminder store (loop-proof D14): GET /api/v1/reminders, and from PR 3b one
 * reminder with Done / Snooze / Undo. `remind_at` is ISO UTC; `remind_at_local` is the
 * owner's zone (IRIS_TZ), ready to show. */
/** The backend's reminder lifecycle (notifications/models.py ReminderStatus). */
export type ReminderStatus =
  | "pending"
  | "sending"
  | "sent"
  | "failed"
  | "done"
  | "cancelled"
  | "expired";

export interface ReminderItem {
  id: string;
  text: string;
  kind: string; // reminder | event_lead | bill | task | goal
  remind_at: string;
  remind_at_local: string;
  status: ReminderStatus;
  recurrence: string | null;
  recurrence_label: string;
  /** Why it ended: "done: sheet", "closed: paid (payment_email)", "not yet: push". */
  closed_reason?: string | null;
  /** A bill's reminder (loop-proof PR 4): its own words and who closed it. */
  bill?: ReminderBill;
}

/** The step of a bill's reminder: 3 days before, the due day, the three "Did you
 * pay?" mornings after, an amount change, or the "marked paid" confirmation. */
export type BillStep = "t3d" | "dayof" | "ask1" | "ask2" | "ask3" | "changed" | "paid";

export interface ReminderBill {
  step: BillStep | string;
  entity: string;
  amount: string; // "$35.00", "" when unknown
  statement: string; // "$1,284.50", "" when none
  due: string; // ISO date
  due_local: string; // "Mon Oct 13"
  headline: string; // "💳 Discover — $35.00 min due Mon Oct 13"
  line: string; // "in 3 days · statement $1,284.50"
  /** Who closed it as paid ("a payment email", "Telegram"); null while open. */
  paid_by: string | null;
  /** The owner answered this question "Not yet". */
  not_yet: boolean;
  /** The reminder "Not paid — reopen" undoes (this one, or on the "marked paid"
   * confirmation the bill's reminder the close ended); null while the bill is open. */
  reopen_id?: string | null;
}

export interface RemindersResponse {
  count: number;
  timezone?: string;
  reminders: ReminderItem[];
}

/** The owner's reminders still wanting something: not yet sent (a snooze moves one back here) or undeliverable. */
export const ACTIVE_REMINDER_STATUSES = "pending,sending,failed";

/* Kinds the Active list shows while they are still on their way. `event_lead` (the
 * lead-time notice every calendar event gets) and `bill` (a due's reminder) are left
 * out: they are generated, one per event or due, and would bury the owner's own. A
 * FAILED one of any kind is shown, because nothing else will tell the owner it never
 * arrived. */
const ACTIVE_REMINDER_KINDS = new Set(["reminder", "task", "goal"]);

export function isActiveListed(r: ReminderItem): boolean {
  return r.status === "failed" || ACTIVE_REMINDER_KINDS.has(r.kind);
}

/** `kind` narrows server-side (one kind); omitted, every kind comes back. */
export function getReminders(status = ACTIVE_REMINDER_STATUSES, kind?: string) {
  const q = new URLSearchParams({ status });
  if (kind) q.set("kind", kind);
  return getJSON<RemindersResponse>(`/api/v1/reminders?${q.toString()}`);
}

/** What the owner can do to a reminder right now; empty once it is closed. A bill's
 * reminder offers `paid` (its Done) and, on a "Did you pay?" question, `not_yet`. */
export type ReminderAction = "done" | "paid" | "not_yet" | "10m" | "1h" | "tomorrow_9am";
/** What the snooze call takes: a time, or `not_yet` (a bill question's acknowledge —
 * not moved; the next question is its own reminder). */
export type SnoozeFor = "10m" | "1h" | "tomorrow_9am" | "not_yet";
/** Which surface acted: the server records it (snooze learning, the audit). */
export type ReminderSource = "sheet" | "push" | "chat_panel";

export interface ReminderDetail extends ReminderItem {
  actions: ReminderAction[];
}

/** Hand this back to `undoReminder` unchanged: the server knows what it means. */
export type ReminderUndo =
  | { kind: "done"; source?: ReminderSource }
  | { kind: "not_yet"; source?: ReminderSource }
  | { kind: "snooze"; until_before: string }
  | Record<string, unknown>;

export interface ReminderActionResult {
  reminder: ReminderDetail;
  undo: ReminderUndo;
  /** A Done on a repeating reminder: the next one. A bill's Not yet: when the next
   * question comes (`remind_at`, `remind_at_local` only), null after the last. */
  next?: Partial<ReminderDetail> | null;
  /** Paid on a bill already closed as paid (PR 4): nothing changed. */
  already_paid?: boolean;
}

/** The reminder is not in the store (deleted, or a link from another IRIS). */
export class ReminderGone extends Error {
  constructor() {
    super("reminder not found");
    this.name = "ReminderGone";
  }
}

async function reminderCall<T>(path: string, body?: unknown): Promise<T> {
  const init: RequestInit = body === undefined
    ? { headers: { accept: "application/json" } }
    : {
        method: "POST",
        headers: { accept: "application/json", "content-type": "application/json" },
        body: JSON.stringify(body),
      };
  const r = await apiFetch(`/api/v1/reminders/${path}`, init);
  if (r.status === 404) throw new ReminderGone();
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: unknown };
      if (typeof j?.detail === "string") msg = j.detail;
    } catch {
      /* non-JSON */
    }
    throw new Error(msg);
  }
  return (await r.json()) as T;
}

export function getReminder(id: string) {
  return reminderCall<ReminderDetail>(encodeURIComponent(id));
}

/** Done / Snooze are allowed from a read-only paired device (the server's rule, D14). */
export function markReminderDone(id: string, source: ReminderSource) {
  return reminderCall<ReminderActionResult>(`${encodeURIComponent(id)}/done`, { source });
}

export function snoozeReminder(id: string, until: SnoozeFor, source: ReminderSource) {
  return reminderCall<ReminderActionResult>(`${encodeURIComponent(id)}/snooze`, {
    for: until,
    source,
  });
}

export function undoReminder(id: string, undo: ReminderUndo) {
  return reminderCall<{ reminder: ReminderDetail }>(`${encodeURIComponent(id)}/undo`, undo);
}

export interface FinanceDueItem {
  id: string;
  label: string;
  amount: string | null;
  currency: string;
  due_date: string | null;
  overdue: boolean;
  source: "bill" | "statement" | "notification";
  category: string;
  status: "open" | "resolved" | "dismissed";
}

export interface FinanceDuesResponse {
  count: number;
  dues: FinanceDueItem[];
  filtered_to: string | null;
  named_absent: boolean;
  generated_at: string;
}

export const getFinanceDues = (includeResolved = false) =>
  getJSON<FinanceDuesResponse>(
    `/finance/dues?include_resolved=${includeResolved ? "true" : "false"}`,
  );

export type FinanceDueStatus = "open" | "resolved" | "dismissed";

export function updateFinanceDueStatus(
  id: string,
  status: FinanceDueStatus,
  note?: string,
): Promise<void> {
  return writeOk(`/finance/dues/${encodeURIComponent(id)}/status`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ status, note }),
  });
}

// ---- Action Center (ADR-0073) -------------------------------------------

export type ActionKind = "copy_command" | "re_extract" | "register" | "review";

/** One answer on a choice card (ADR-0121). `needs_option` answers carry the picked option. */
export interface ActionChoice {
  value: string;
  label: string;
  primary: boolean;
  needs_option: boolean;
}

export interface ActionOptions {
  name: string;
  label: string;
  values: { value: string; label: string }[];
  /** null: the owner must pick before a needs_option answer. */
  default: string | null;
}

export interface ActionCard {
  tag: string | null;
  facts: { label: string; value: string }[];
  evidence: { when: string; text: string }[];
  evidence_label: string;
  note: string;
}

export interface TaskAction {
  kind: ActionKind;
  label: string;
  command: string | null;
  target_id: string | null;
  safe: boolean;
  /** A choice card: answered with one of these instead of a single button (ADR-0121). */
  choices?: ActionChoice[];
  options?: ActionOptions | null;
  card?: ActionCard | null;
}

/** One actionable blocker — a persisted task ("task") or a live Health item
 * ("health"), unioned at the read layer (ADR-0073 §2b). */
export interface PendingAction {
  id: string;
  origin: "task" | "health" | "approval";
  source_kind: string;
  title: string;
  description: string;
  action: TaskAction;
  created_at: string | null;
  /** Self-describing surface-feedback token (issue 0028); null when the item has
   * no stable suppression key. Present → a "not useful" affordance can be shown. */
  feedback_ref: string | null;
}

export interface ActionsResponse {
  count: number;
  actions: PendingAction[];
}

export const getActions = () => getJSON<ActionsResponse>("/actions");

export interface InvokeResult {
  id: string;
  result: string;
}

export interface ActionAnswer {
  choice: string;
  option?: string | null;
}

/** Run a safe pending action. Write-gated; the server rejects display-only ones. A
 * choice card is answered with `answer` (ADR-0121). */
export async function invokeAction(id: string, answer?: ActionAnswer): Promise<InvokeResult> {
  const r = await apiFetch(`/actions/${encodeURIComponent(id)}/invoke`, {
    method: "POST",
    ...(answer
      ? { headers: { "content-type": "application/json" }, body: JSON.stringify(answer) }
      : {}),
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: string };
      if (j?.detail) msg = j.detail;
    } catch {
      /* non-JSON */
    }
    throw new Error(msg);
  }
  return (await r.json()) as InvokeResult;
}

/** Mark a surfaced item "not useful" (or useful) so IRIS suppresses similar ones
 * (issue 0028). Records a learning signal only — no external effect — so it is NOT
 * write-gated; a read-only user can still suppress noise. Pass the item's
 * `feedback_ref`. */
export async function submitSurfaceFeedback(
  ref: string,
  verdict: "not_useful" | "useful" = "not_useful",
): Promise<void> {
  const r = await apiFetch("/surface-feedback", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ ref, verdict }),
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: string };
      if (j?.detail) msg = j.detail;
    } catch {
      /* non-JSON */
    }
    throw new Error(msg);
  }
}

// ---- Per-agent console (ADR-0074) ---------------------------------------

export interface AgentInfo {
  name: string;
  title: string;
  description: string;
  source_kind: string | null;
  /** The plugin that registered this agent; null when the core registered it;
   * absent from an API that predates plugin reporting. */
  plugin?: string | null;
}

export const getAgents = () => getJSON<{ count: number; agents: AgentInfo[] }>("/agents");

// ---- Plugins (read-only inventory over the live registry) ----------------

export type PluginStatus = "loaded" | "degraded" | "failed" | "disabled" | "unsupported";

export interface ConfigFile {
  path: string;
  size: number;
  content: string | null;
  truncated: boolean;
}

export interface PluginSummary {
  name: string;
  status: PluginStatus;
  source: string;
  version: string | null;
  description: string;
  trust: string;
  /* Provenance (ADR-0136); null when the plugin has no manifest. */
  party: string | null;
  flavor: string | null;
  provides: string[];
  enabled: boolean | null;
  set_by: string | null;
  /* ADR-0120: why the app may not turn it off (e.g. the app runs through it). */
  locked?: string | null;
  registration_counts: Record<string, number>;
  agents: string[];
  declared_tools: number;
  failure_count: number;
  last_error: string | null;
  load_error: string | null;
  degraded_reason: string | null;
}

export interface PluginProfile {
  name: string;
  description: string;
  layers: string[];
  intercept_order: string[];
  available_profiles: string[];
  files: ConfigFile[];
}

export interface PluginInventory {
  profile: PluginProfile | null;
  count: number;
  totals: Record<PluginStatus, number>;
  plugins: PluginSummary[];
}

export interface PluginTool {
  name: string;
  effect: "read" | "write" | "destructive";
  /** "approval" for a destructive tool: approved per call, never confirmed (ADR-0118). */
  confirm: "once" | "never" | "approval";
  pinned: boolean;
  answers_directly: boolean;
  /** The tool that reverses a destructive one, when the service has a reversible form. */
  undo: string | null;
  /** How long that undo stays possible, in days. */
  undo_window_days: number | null;
  guidance: string;
  registered: boolean;
}

export interface PluginManifestDoc {
  entrypoint: string;
  cli: string | null;
  requires: { python: string; packages: string[]; env_vars: string[] };
}

export interface PluginDetail extends PluginSummary {
  directory: string | null;
  manifest: PluginManifestDoc | null;
  registrations: { kind: string; name: string; detail: string }[];
  tools: PluginTool[];
  drift: Record<string, string[]>;
  files: ConfigFile[];
}

export const getPlugins = () => getJSON<PluginInventory>("/plugins");

export const getPlugin = (name: string) =>
  getJSON<PluginDetail>(`/plugins/${encodeURIComponent(name)}`);

export interface AgentRun {
  name: string;
  status: string;
  finished_at: string | null;
  output: string | null;
  error: string | null;
}

export interface AgentStatementGroup {
  institution: string;
  doc_label: string;
  count: number;
  latest: string | null;
  extracted: number;
}

export interface AgentStores {
  accounts: number;
  statements: { total: number; unextracted: number; groups: AgentStatementGroup[] };
}

export interface AgentLlmSetting {
  intent: string;
  tier: string;
  model: string;
  provider: string;
}

export interface AgentToggle {
  key: string;
  label: string;
  enabled: boolean;
  default: boolean;
  applies: string; // "now" | "next run" | "restart"
  /* ADR-0120: needs the owner's confirm, which only Settings asks for. */
  guarded?: boolean;
}

export interface AgentHeartbeatSetting {
  name: string;
  schedule: string;
  enabled: boolean;
}

export interface AgentSettings {
  llm: AgentLlmSetting[];
  toggles: AgentToggle[];
  heartbeats: AgentHeartbeatSetting[];
}

export interface AgentDashboard {
  name: string;
  title: string;
  description: string;
  source_kind: string | null;
  /** The plugin that registered this agent; null = the core; absent = an older API. */
  plugin?: string | null;
  pending_actions: { count: number; actions: PendingAction[] };
  recent_runs: AgentRun[];
  stores: AgentStores | null;
  settings: AgentSettings | null;
}

export const getAgentDashboard = (name: string) =>
  getJSON<AgentDashboard>(`/agents/${encodeURIComponent(name)}`);

export interface AgentMetricsTier {
  tier: string;
  volume: number;
  success_rate: number;
}

export interface AgentMetrics {
  window_hours: number;
  sampled_at: string;
  volume: number;
  success_rate: number | null;
  correction_rate: number | null;
  reuse_count: number;
  avg_tokens: number | null;
  by_tier: AgentMetricsTier[];
}

export const getAgentMetrics = (name: string, windowDays = 7) =>
  getJSON<{ name: string; available: boolean; metrics: AgentMetrics | null }>(
    `/agents/${encodeURIComponent(name)}/metrics?window_days=${windowDays}`,
  );

export interface ToggleApplied {
  key: string;
  enabled: boolean;
  applies: string;
  restart_required: boolean;
}

/** Edit a curated agent toggle (write-gated; only toggles the agent owns). */
export async function setAgentToggle(
  name: string,
  key: string,
  enabled: boolean,
): Promise<{ applied: ToggleApplied[] }> {
  const r = await apiFetch(`/agents/${encodeURIComponent(name)}/settings`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ toggles: { [key]: enabled } }),
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: string };
      if (j?.detail) msg = j.detail;
    } catch {
      /* non-JSON */
    }
    throw new Error(msg);
  }
  return (await r.json()) as { applied: ToggleApplied[] };
}

// ---- Runtime inventory (fast-follow #5, read-only) -----------------------

export interface OllamaModel {
  name: string | null;
  size: number | null;
  parameter_size: string | null;
}

export interface RuntimeInventory {
  iris_version: string;
  git_branch: string | null;
  git_rev: string | null;
  python_version: string;
  packages: Record<string, string>;
  ollama_models: OllamaModel[];
}

export const getRuntimeInventory = () => getJSON<RuntimeInventory>("/runtime/inventory");

// ---- Self-learning experiments (fast-follow #4, read-only) ----------------

/** Shape of the learning-health block embedded in /observability/llm-metrics. */
export interface LearningHealth {
  signal_volume?: Record<string, number>;
  signals_recorded_total?: number;
  signals_dropped_total?: number;
  drop_rate?: number;
  experiments?: Record<string, number>;
}

export interface SandboxVerdict {
  verdict: "better" | "worse" | "inconclusive";
  metric: string;
  improvement_pct: number | null;
  note: string;
  workload_size?: number;
  evaluated_at?: string;
}

export interface Experiment {
  id: string;
  domain: string;
  hypothesis: string;
  variant_description: string;
  baseline_metric: number;
  current_metric: number | null;
  status: string;
  created_at: string | null;
  started_at: string | null;
  evaluated_at: string | null;
  evaluation_window_hours: number;
  // Sandbox pre-flight verdict (ADR-0070), when one has run.
  sandbox?: SandboxVerdict | null;
}

export interface LearningExperiments {
  experiments: Experiment[];
  status_counts: Record<string, number>;
}

export const getExperiments = () => getJSON<LearningExperiments>("/learning/experiments");

/* Measured learning intelligence (fast-follow #4, slice 1) — deterministic
 * aggregates over the existing signals. Pure measurement; an agentic analyst
 * interprets these numbers in a later slice. */
export interface SignalAccuracy {
  escalation_shadow_total: number;
  escalation_would_act: number;
  escalation_would_act_rate: number;
  escalation_precision: number | null;
  signals_recorded_total: number;
  signals_dropped_total: number;
  drop_rate: number;
}

export interface OutcomeCell {
  intent: string;
  tier: string;
  samples: number;
  completion_rate: number;
  correction_rate: number | null;
  correction_samples: number;
  reuse_count: number;
  avg_tokens: number | null;
}

export interface FeedbackIntent {
  intent: string;
  positive: number;
  negative: number;
  satisfaction: number;
}

export interface FeedbackSummary {
  total: number;
  positive: number;
  negative: number;
  satisfaction: number | null;
  by_intent: FeedbackIntent[];
}

export interface LearningIntelligence {
  available: boolean;
  sampled_at?: string;
  window_hours?: number;
  accuracy?: SignalAccuracy;
  matrix?: OutcomeCell[];
  experiments?: Record<string, number>;
  feedback?: FeedbackSummary;
}

export const getLearningIntelligence = () =>
  getJSON<LearningIntelligence>("/learning/intelligence");

/* Learning analyst recommendations (fast-follow #4, slice 2) — advisory, the
 * agentic interpretation of the measured report. Surfaced read-only; the user
 * (or, later, the experiment loop) acts on them — never auto-applied. */
export interface LearningRecommendation {
  title: string;
  finding: string;
  action: string;
  evidence: string[];
  confidence: "low" | "medium" | "high";
}

export interface LearningAnalysisResult {
  available: boolean;
  generated_at?: string;
  model?: string;
  summary?: string;
  recommendations?: LearningRecommendation[];
}

export const getLearningAnalysis = () =>
  getJSON<LearningAnalysisResult>("/learning/analysis");

export interface PromoteResult {
  experiment_id: string;
  hypothesis: string;
  baseline_metric: number;
  measurable: boolean;
  note: string;
}

/** Promote recommendation #index (1-based) to a tracked experiment (slice 3). Write-gated. */
export async function promoteRecommendation(index: number): Promise<PromoteResult> {
  const r = await apiFetch(`/learning/recommendations/${index}/promote`, { method: "POST" });
  if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`.trim());
  return (await r.json()) as PromoteResult;
}

/* Digital-twin review surfaces (HITL). Three propose-only layers IRIS builds about
 * the user over time: mined habits (behaviors), how the user steers (signals), and
 * rolled-up longitudinal goals (intentions). Reads are open; approve/reject/dismiss
 * are write-gated and close the loop into durable identity layers. */

export interface BehaviorProposal {
  pattern_id: string;
  text: string;
  confidence: string; // low | medium | high
  evidence: string[];
  status: string; // pending | approved | rejected
  created_at: string;
}

export interface BehaviorProposals {
  status: string;
  count: number;
  behaviors: BehaviorProposal[];
}

export const getBehaviors = (status = "pending") =>
  getJSON<BehaviorProposals>(`/learning/behaviors?status=${encodeURIComponent(status)}`);

/** Approve a mined habit → appends it to episodic memory. Write-gated. */
export const approveBehavior = (id: string) =>
  writeOk(`/learning/behaviors/${encodeURIComponent(id)}/approve`, { method: "POST" });

/** Reject a mined habit (won't be re-proposed). Write-gated. */
export const rejectBehavior = (id: string) =>
  writeOk(`/learning/behaviors/${encodeURIComponent(id)}/reject`, { method: "POST" });

export interface UserBehaviorSignal {
  id: number;
  kind: string;
  subject: string;
  detail: string;
  created_at: string;
}

export interface UserBehaviorSignals {
  count: number;
  signals: UserBehaviorSignal[];
  summary: Record<string, number>;
}

export const getSignals = (kind?: string, limit = 100) => {
  const q = new URLSearchParams({ limit: String(limit) });
  if (kind) q.set("kind", kind);
  return getJSON<UserBehaviorSignals>(`/learning/signals?${q.toString()}`);
};

export interface IntentionProposal {
  intention_id: string;
  title: string;
  summary: string;
  supporting: string[];
  status: string; // proposed | active | dismissed
  created_at: string;
}

export interface IntentionProposals {
  status: string;
  count: number;
  intentions: IntentionProposal[];
}

export const getIntentions = (status = "proposed") =>
  getJSON<IntentionProposals>(`/learning/intentions?status=${encodeURIComponent(status)}`);

/** Approve an intention → writes it to the ACTIVE identity layer. Write-gated. */
export const approveIntention = (id: string) =>
  writeOk(`/learning/intentions/${encodeURIComponent(id)}/approve`, { method: "POST" });

/** Dismiss a proposed intention (won't be re-proposed). Write-gated. */
export const dismissIntention = (id: string) =>
  writeOk(`/learning/intentions/${encodeURIComponent(id)}/dismiss`, { method: "POST" });

// ---- Portfolio (investments) --------------------------------------------

export type AssetType = "stock" | "etf" | "mutual_fund" | "other";

export interface PortfolioPosition {
  name: string;
  isin: string | null;
  symbol: string | null; // resolved live ticker (e.g. INFY.NS)
  asset_type: AssetType;
  currency: string;
  quantity: string | null;
  cost_basis: string | null;
  stored_value: string | null;
  as_of_date: string | null;
  live_price: number | null;
  day_change_pct: number | null;
  market_value: string | null;
  pnl: string | null;
  pnl_pct: number | null;
  priced: boolean;
}

export interface PortfolioTotal {
  currency: string;
  market_value: string | null;
  cost_basis: string | null;
  pnl: string | null;
  pnl_pct: number | null;
  position_count: number;
}

export interface PortfolioSnapshot {
  count: number;
  priced_count: number;
  as_of_dates: string[];
  positions: PortfolioPosition[];
  totals: PortfolioTotal[];
}

export const getPortfolio = () => getJSON<PortfolioSnapshot>("/portfolio");

export interface HoldingsImportResult {
  parsed: number;
  mapped: number;
  written: number;
  unmapped_symbols: string[];
}

/** Import a broker holdings CSV (server-side local path). Write-gated. */
export function importHoldings(filePath: string): Promise<void> {
  return writeOk("/portfolio/import-holdings", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ path: filePath }),
  });
}

// ---- Market indexes (context strip) -------------------------------------

export interface MarketIndex {
  symbol: string;
  name: string;
  price: number;
  currency: string;
  change_pct: number;
}

export const getMarketIndexes = () =>
  getJSON<{ indexes: MarketIndex[] }>("/market/indexes");

// ---- Playground (harness test bench) ------------------------------------

export interface PlaygroundScenarioMeta {
  name: string;
  message: string;
  tags: string[];
}

export interface PlaygroundSuiteMeta {
  name: string;
  description?: string;
  path: string;
  error?: string;
  scenarios?: PlaygroundScenarioMeta[];
}

export interface PlaygroundAssertion {
  field: string;
  expected: unknown;
  actual: unknown;
}

export interface PlaygroundScenarioResult {
  name: string;
  passed: boolean;
  intent: string | null;
  handler: string | null;
  sources: string[];
  duration_ms: number;
  error: string | null;
  response: string;
  failed_assertions: PlaygroundAssertion[];
}

export interface PlaygroundRunResult {
  suite: string;
  ok: boolean;
  passed: number;
  total: number;
  results: PlaygroundScenarioResult[];
}

export interface DriftSurface {
  surface: string;
  ok: boolean;
  declared_only: string[];
  registered_only: string[];
  in_sync: string[];
}

export interface DriftReport {
  ok: boolean;
  surfaces: DriftSurface[];
}

export const getPlaygroundSuites = () =>
  getJSON<{ suites: PlaygroundSuiteMeta[] }>("/playground/suites");

export const getPlaygroundDrift = () => getJSON<DriftReport>("/playground/drift");

export async function runPlaygroundSuite(suite: string): Promise<PlaygroundRunResult> {
  const r = await apiFetch("/playground/run", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ suite }),
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: string };
      if (j?.detail) msg = j.detail;
    } catch {
      /* non-JSON */
    }
    throw new Error(msg);
  }
  return (await r.json()) as PlaygroundRunResult;
}

/* ── Memory: profile, facts, review queue, sessions, context, housekeeping ──
 * Every screen action below has a CLI twin (`iris facts …`, `iris memory …`) —
 * the UI is a client of the same API, never the only way to do something. */

export interface CuratedProfile {
  profile: string;
  chars: number;
}

export interface StoredFact {
  /** The fact's own id — a key can hold several values (two cards). */
  id: string | null;
  key: string;
  value: string;
  confidence: number;
  source: string;
  confirmed: boolean;
  first_seen: string;
  last_confirmed: string;
  times_confirmed: number;
}

export interface FactProposalRow {
  /** The proposed statement's id (memris PR 2c): a string like "st_01j9…". */
  id: string;
  key: string;
  value: string;
  current_value: string | null;
  confidence: number;
  source: string;
  evidence: string;
  created_at: string;
  kind: "new" | "changed";
  /** Whom the fact is about when not the owner — someone one hop away (memris PR 3b). */
  subject?: string | null;
}

export interface LessonRow {
  id: number;
  trigger: string;
  lesson: string;
  evidence: string;
  source: string;
  created_at: string;
}

/** One queue: since memris PR 2c a fact never reviewed IS a proposal. */
export interface MemoryReview {
  pending_count: number;
  proposals: FactProposalRow[];
}

/** A pair of similar names memory keeps apart until someone decides (ADR-0115 d4). */
export interface EntityDecisionRow {
  id: string;
  decision: "candidate" | "same" | "distinct" | "undone";
  a: { id: string; label: string };
  b: { id: string; label: string };
  score: number | null;
  evidence_count: number;
  decided_by: string | null;
  decided_at: string;
  asked_at: string | null;
}

/** A word memory learned on its own (ADR-0115 decision 7): data, never YAML. */
export interface LearnedTermRow {
  name: string;
  kind: "attribute" | "relation";
  label: string;
  status: "candidate" | "alias" | "active" | "rejected";
  alias_of: string | null;
  observations: number;
  conversations: number;
  examples: string[];
  first_seen: string;
  last_seen: string;
  activated_at: string | null;
  decided_by: string | null;
}

export interface StoredSession {
  session_id: string;
  last_activity: string;
  turns: number;
  summary: string;
  state: "hot" | "cold-eligible";
  is_run: boolean;
}

export interface ContextBlock {
  block: string;
  tokens: number;
  present: boolean;
  preview: string;
}

export interface ContextInspection {
  session_id: string;
  /** The message the memory-graph block was computed for (its last user message). */
  message?: string;
  blocks: ContextBlock[];
  pointers: string[];
  total_tokens: number;
  compaction_in_flight: boolean;
}

export interface HousekeepingRun {
  started_at: string;
  dry_run: boolean;
  sessions_cooled: number;
  sessions_kept_unsummarized: number;
  turns_deleted: number;
  vectors_deleted: number;
  orphan_vectors_swept: number;
  logs_compressed: number;
  logs_deleted: number;
  log_bytes_reclaimed: number;
  vacuumed: boolean;
  errors: string[];
}

export interface ForgetPreviewRow {
  deleted: boolean;
  preview?: {
    needle: string;
    total: number;
    turns: { id: number; session_id: string; role: string; content: string }[];
    summaries: { session_id: string; summary: string }[];
    facts: { key: string; value: string }[];
  };
  turns?: number;
  vectors?: number;
  summaries?: number;
  facts?: number;
}

export const getCuratedProfile = () => getJSON<CuratedProfile>("/memory/profile");
export const getStoredFacts = (confirmedOnly = true) =>
  getJSON<{ count: number; facts: StoredFact[] }>(
    `/memory/facts?confirmed_only=${confirmedOnly}`,
  );
export const getMemoryReview = () => getJSON<MemoryReview>("/memory/review");
export const getLookalikes = () =>
  getJSON<{ count: number; decisions: EntityDecisionRow[] }>("/memory/entities");
export const getLearnedTerms = () =>
  getJSON<{ count: number; terms: LearnedTermRow[] }>("/memory/terms");
export const getLessons = () =>
  getJSON<{ count: number; lessons: LessonRow[] }>("/memory/lessons");
export const getStoredSessions = () =>
  getJSON<{ count: number; hot_days: number; sessions: StoredSession[] }>("/memory/sessions");
export const getContextInspection = (sessionId: string) =>
  getJSON<ContextInspection>(`/memory/context/${encodeURIComponent(sessionId)}`);
export const getHousekeepingRuns = () =>
  getJSON<{ runs: HousekeepingRun[] }>("/memory/housekeeping");

/** A sender IRIS saw but did not propose, or one the owner ignored (ADR-0121). */
export interface SkippedSender {
  domain: string;
  status: "not_proposed" | "ignored";
  verdict: string | null;
  reason: string | null;
  email_count: number;
  latest_at: string | null;
}

export interface FinanceSenders {
  open: number;
  waiting: number;
  not_proposed: SkippedSender[];
}

export const getFinanceSenders = () => getJSON<FinanceSenders>("/finance/senders");

/** "Add anyway" / undo Ignore: open the sender's card now. Write-gated. */
export const proposeFinanceSender = (domain: string) =>
  postJSON<{ domain: string; result: string }>(
    `/finance/senders/${encodeURIComponent(domain)}/propose`,
  );

async function postJSON<T>(path: string, body?: unknown, method = "POST"): Promise<T> {
  const r = await apiFetch(path, {
    method,
    headers: { "content-type": "application/json" },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: string };
      if (j?.detail) msg = j.detail;
    } catch {
      /* non-JSON */
    }
    throw new Error(msg);
  }
  return (await r.json()) as T;
}

export const writeCuratedProfile = (profile: string) =>
  postJSON<{ written: boolean }>("/memory/profile", { profile }, "PUT");
export const approveProposal = (id: string) =>
  postJSON<{ approved: boolean }>(`/memory/review/${encodeURIComponent(id)}/approve`);
export const rejectProposal = (id: string) =>
  postJSON<{ rejected: boolean }>(`/memory/review/${encodeURIComponent(id)}/reject`);
export const markEntitiesSame = (id: string) =>
  postJSON<EntityDecisionRow>(`/memory/entities/${encodeURIComponent(id)}/same`);
export const markEntitiesDistinct = (id: string) =>
  postJSON<EntityDecisionRow>(`/memory/entities/${encodeURIComponent(id)}/distinct`);
export const rejectTerm = (name: string) =>
  postJSON<LearnedTermRow>(`/memory/terms/${encodeURIComponent(name)}/reject`);
export const activateTerm = (name: string) =>
  postJSON<LearnedTermRow>(`/memory/terms/${encodeURIComponent(name)}/activate`);
export const confirmFact = (key: string) =>
  postJSON<{ confirmed: boolean }>(`/memory/facts/${encodeURIComponent(key)}/confirm`);
export const forgetFact = ({ key, id }: { key: string; id?: string | null }) =>
  postJSON<{ forgotten: boolean }>(
    `/memory/facts/${encodeURIComponent(key)}/forget${id ? `?id=${encodeURIComponent(id)}` : ""}`,
  );
export const correctFact = (key: string, value: string, id?: string | null) =>
  postJSON<{ updated: boolean }>(
    `/memory/facts/${encodeURIComponent(key)}`,
    id ? { value, id } : { value },
    "PATCH",
  );
export const approveLesson = (id: number) =>
  postJSON<{ approved: boolean; behavior: string }>(`/memory/lessons/${id}/approve`);
export const rejectLesson = (id: number) =>
  postJSON<{ rejected: boolean }>(`/memory/lessons/${id}/reject`);
export const forgetMatching = (needle: string, confirm = false) =>
  postJSON<ForgetPreviewRow>(
    `/memory/forget?needle=${encodeURIComponent(needle)}&confirm=${confirm}`,
  );
export const runHousekeeping = (dryRun = true) =>
  postJSON<HousekeepingRun>(`/memory/housekeeping/run?dry_run=${dryRun}`);
export const compactSession = (sessionId: string) =>
  postJSON<{ compacted: boolean }>(
    `/context-health/compact?session_id=${encodeURIComponent(sessionId)}`,
  );

export type MemoryGraphKind = "you" | "entity" | "fact" | "session" | "lesson" | "pattern" | "more";

export interface MemoryGraphNode {
  id: string;
  kind: MemoryGraphKind;
  label: string;
  confirmed: boolean;
  meta: Record<string, unknown>;
  /** What Remove acts on (ADR-0119). Absent for "you", lessons, patterns and "more":
   * those cannot be removed from the Map. */
  ref?: RemovalTarget | null;
  /** Suppressed once, then named again by a confirmed fact (plan decision 4). */
  previously_removed?: boolean;
}

/* Removal (ADR-0119, Memory Map cleanup plan decisions 2-13). Reversible: every removal
 * is listed, restorable, and only then deletable for good. */

/** A summary mention that is not a memris entity is removed as a suppressed `name`. */
export type RemovableKind = "entity" | "name" | "session" | "fact";

/** For a `name`, the id is the label. */
export interface RemovalTarget {
  kind: RemovableKind;
  id: string;
}

export interface RemovedItem {
  id: string;
  kind: RemovableKind;
  label: string;
  removed_at: string;
  /** The facts that went with it (an entity's live statements). */
  cascade: { statement_id: string; text: string }[];
  /** Deleted for good; a deleted entity's or name's suppression stays. */
  permanent: boolean;
}

export interface RemovalEffect {
  target: RemovalTarget;
  label: string;
  lines: string[];
}

/** A conversation whose id looks like a test run (retention.yaml `test_session_review`). */
export interface TestSessionRow {
  session_id: string;
  /** The summary's first line, or "" when the session was never summarized. */
  summary_goal: string;
  turns: number;
  last_activity: string;
  reasons: string[];
}

export const getTestSessions = () =>
  getJSON<{ sessions: TestSessionRow[]; real_shapes: string[] }>("/memory/review/test-sessions");
export const getRemoved = () => getJSON<{ items: RemovedItem[] }>("/memory/removed");
export const previewRemoval = (targets: RemovalTarget[]) =>
  postJSON<{ effects: RemovalEffect[] }>("/memory/removed/preview", { targets });
export const removeFromMemory = (targets: RemovalTarget[]) =>
  postJSON<{ items: RemovedItem[] }>("/memory/removed", { targets });
export const restoreRemoved = (id: string) =>
  postJSON<{ restored: RemovedItem }>(`/memory/removed/${encodeURIComponent(id)}/restore`);
export const deleteRemoved = (ids: string[]) =>
  postJSON<{ deleted: string[]; refused: { id: string; reason: string }[] }>(
    "/memory/removed/delete",
    { ids, confirm: "delete" },
  );

/** An edge's label is its property's label in the ontology (memris PR 5); a fact edge's
 * meta carries the statement's predicate, status and time bounds. */
export interface MemoryGraphEdge {
  id: string;
  source: string;
  target: string;
  label: string;
  meta?: Record<string, unknown>;
}
export interface MemoryGraph {
  nodes: MemoryGraphNode[];
  edges: MemoryGraphEdge[];
  stats: { total_nodes: number; shown: number; by_kind: Record<string, number> };
  focus: string;
  /** The moment the graph was drawn at (ISO-8601), or null for now. */
  as_of?: string | null;
}

/** The memory graph — computed per request from the stores, never persisted. */
export const getMemoryGraph = (opts: {
  focus?: string | null;
  kinds?: string[];
  confirmedOnly?: boolean;
  /** Draw what memory held true at this moment (ISO-8601 with a timezone). */
  asOf?: string | null;
}) => {
  const params = new URLSearchParams();
  if (opts.focus) params.set("focus", opts.focus);
  if (opts.kinds?.length) params.set("kinds", opts.kinds.join(","));
  if (opts.confirmedOnly) params.set("confirmed_only", "true");
  if (opts.asOf) params.set("as_of", opts.asOf);
  return getJSON<MemoryGraph>(`/memory/graph?${params.toString()}`);
};
