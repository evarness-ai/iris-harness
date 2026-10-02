/* TanStack Query hooks over the API client — real loading/error/caching.
 * Every read is the live API's answer: when the API is down the hook is in its
 * error state and the screen says so; there is no canned fallback. */
import { useMutation, useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";
import {
  getTrace,
  getChatStatus,
  listSessionMessages,
  listSessions,
  listSlashCommands,
  listTraces,
} from "./client";
import {
  ackContradictions,
  approveBehavior,
  approveIntention,
  approveLesson,
  approveProposal,
  compactContext,
  compactSession,
  confirmFact,
  correctFact,
  dismissIntention,
  forgetFact,
  forgetMatching,
  getActions,
  getActivities,
  getAgentDashboard,
  getAgentMetrics,
  getAgents,
  getApprovals,
  getBehaviors,
  getPlugin,
  getPlugins,
  getCapabilities,
  getContextHealth,
  getContextInspection,
  getContradictions,
  getCuratedProfile,
  getExperiments,
  getFactHistoryRetention,
  getFinanceDues,
  getGovernanceAudit,
  getGovernanceState,
  getPiiShadow,
  getProofBundleCheck,
  getHealth,
  createTask,
  getCost,
  getHealthIncidents,
  runHealthWatch,
  getDoctorReport,
  pullStarterModel,
  getHeartbeatRuns,
  getHeartbeats,
  getSettingsCatalog,
  getRestartStatus,
  getModels,
  moveIntent,
  patchTier,
  resetIntent,
  resetTier,
  getWatchConfig,
  patchWatchConfig,
  resetWatchConfig,
  getDigestConfig,
  patchDigestConfig,
  resetDigestConfig,
  type DigestFields,
  getSettingsHistory,
  patchSetting,
  requestRestart,
  resetSetting,
  getHousekeepingRuns,
  getIntentions,
  getLearningAnalysis,
  getLearningFlags,
  getLearningIntelligence,
  getLearningMetrics,
  getLessons,
  getLlmMode,
  getMarketIndexes,
  getMemory,
  getMemoryGraph,
  getRemoved,
  getTestSessions,
  previewRemoval,
  removeFromMemory,
  restoreRemoved,
  deleteRemoved,
  getMemoryReview,
  getLookalikes,
  getLearnedTerms,
  rejectTerm,
  activateTerm,
  markEntitiesDistinct,
  markEntitiesSame,
  getPlaygroundDrift,
  getPlaygroundSuites,
  getPortfolio,
  getProposalQuality,
  getReminder,
  getReminders,
  isActiveListed,
  markReminderDone,
  ReminderGone,
  snoozeReminder,
  undoReminder,
  type ReminderActionResult,
  type ReminderDetail,
  type ReminderSource,
  type ReminderUndo,
  type SnoozeFor,
  getRoutines,
  getRuntimeInventory,
  getSettings,
  getSignals,
  getStoredFacts,
  getStoredSessions,
  getTasks,
  invokeAction,
  getFinanceSenders,
  proposeFinanceSender,
  type ActionAnswer,
  pinLlmMode,
  promoteRecommendation,
  pruneFactHistory,
  rejectBehavior,
  rejectLesson,
  rejectProposal,
  respondToApproval,
  runHousekeeping,
  runLearningNow,
  runPlaygroundSuite,
  setAgentToggle,
  setLearningFlag,
  submitSurfaceFeedback,
  triggerHeartbeat,
  patchHeartbeat,
  resetHeartbeat,
  type HeartbeatPatch,
  type TierFields,
  type RoutineStatus,
  unpinLlmMode,
  updateFinanceDueStatus,
  updateRoutineStatus,
  updateTaskStatus,
  writeCuratedProfile,
} from "./control";
import { deleteRagDocument, getRagDocuments, uploadRagDocument } from "./rag";
import { getKnowledgeGraph } from "./knowledge";
import {
  type DeviceScope,
  getPrincipal,
  listDevices,
  revokeDevice,
  startPairing,
} from "./devices";

export function useSessions() {
  return useQuery({ queryKey: ["sessions"], queryFn: listSessions });
}

export function useSessionMessages(sessionId: string, enabled = true) {
  return useQuery({
    queryKey: ["session-messages", sessionId],
    queryFn: () => listSessionMessages(sessionId),
    enabled: enabled && Boolean(sessionId),
  });
}

export function useTraces() {
  return useQuery({ queryKey: ["traces"], queryFn: listTraces });
}

export function useSlashCommands() {
  return useQuery({ queryKey: ["slash-commands"], queryFn: listSlashCommands, staleTime: 300_000 });
}

export function useTrace(traceId: string) {
  return useQuery({
    queryKey: ["trace", traceId],
    queryFn: () => getTrace(traceId),
    enabled: Boolean(traceId),
  });
}

export function useChatStatus(sessionId: string) {
  return useQuery({
    queryKey: ["chat-status", sessionId],
    queryFn: () => getChatStatus(sessionId),
    enabled: Boolean(sessionId),
    refetchInterval: 15_000,
  });
}

/* Capabilities — what the web UI is allowed to do (gated by IRIS_WEBUI_ALLOW_WRITES).
 * Defaults to read-only (false) until confirmed, so write controls stay hidden
 * unless the server explicitly allows them. */
export function useCapabilities() {
  return useQuery({ queryKey: ["capabilities"], queryFn: getCapabilities, staleTime: 60_000 });
}

export function useWritesEnabled(): boolean {
  return useCapabilities().data?.writes_enabled ?? false;
}

/* Whether the non-blocking answer-feedback affordance is on (IRIS_FEEDBACK_CAPTURE). */
export function useFeedbackEnabled(): boolean {
  return useCapabilities().data?.feedback_capture ?? false;
}

/* Control surfaces (Phase 3) — read-only. Light polling on the live-ish ones so
 * the dashboard reflects the running runtime without a manual refresh. */

export function useLlmMode() {
  return useQuery({ queryKey: ["llm-mode"], queryFn: getLlmMode, refetchInterval: 15_000 });
}

export function useLearningMetrics() {
  return useQuery({ queryKey: ["learning-metrics"], queryFn: getLearningMetrics });
}

export function useRoutines() {
  return useQuery({ queryKey: ["routines"], queryFn: getRoutines });
}

export function usePortfolio() {
  // Live prices move; refresh on an interval like the other read-only screens.
  return useQuery({ queryKey: ["portfolio"], queryFn: getPortfolio, refetchInterval: 60_000 });
}

export function useMarketIndexes() {
  return useQuery({
    queryKey: ["market-indexes"],
    queryFn: getMarketIndexes,
    refetchInterval: 60_000,
  });
}

export function useHeartbeats() {
  return useQuery({ queryKey: ["heartbeats"], queryFn: getHeartbeats });
}

/* Async Activity feed (P1): background system jobs (FileManager categorize/
 * cleanup/organize). Polls so a running job's progress updates live and a
 * just-completed one shows up without a manual refresh. */
export function useActivities() {
  return useQuery({
    queryKey: ["activities"],
    queryFn: getActivities,
    // Poll fast and keep polling even when the tab is blurred — users alt-tab to
    // Finder to watch a folder while a categorize/cleanup job runs, and a paused
    // poll is why "no progress updates" was reported.
    refetchInterval: 4_000,
    refetchIntervalInBackground: true,
  });
}

/** Badge count: running jobs + unseen completed ones (mirrors useActionCount). */
export function useActivityBadge(): number {
  const { data } = useActivities();
  return data?.running ?? 0;
}

export function useHeartbeatRuns(limit = 50) {
  return useQuery({
    queryKey: ["heartbeat-runs", limit],
    queryFn: () => getHeartbeatRuns(limit),
    refetchInterval: 20_000,
  });
}

/* RAG documents (Phase 4) — catalog listing + an upload mutation that refreshes
 * the list on success. */

export function useRagDocuments() {
  return useQuery({ queryKey: ["rag-documents"], queryFn: getRagDocuments });
}

export function useUploadDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: uploadRagDocument,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rag-documents"] }),
  });
}

/* Unified knowledge graph / context map (Phase 5). */
export function useKnowledgeGraph() {
  return useQuery({ queryKey: ["knowledge-graph"], queryFn: getKnowledgeGraph });
}

/* Fact-history retention review — list entries past the window + prune chosen ids.
 * The prune mutation refreshes the review queue on success. */
export function useFactHistoryRetention(olderThanDays = 180) {
  return useQuery({
    queryKey: ["fact-history-retention", olderThanDays],
    queryFn: () => getFactHistoryRetention(olderThanDays),
  });
}

export function usePruneFactHistory() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (entryIds: string[]) => pruneFactHistory(entryIds),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["fact-history-retention"] }),
  });
}

/* Fact contradictions review — list detected same-key conflicts + acknowledge them.
 * The ack mutation refreshes the queue on success. */
export function useContradictions() {
  return useQuery({
    queryKey: ["fact-contradictions"],
    queryFn: () => getContradictions(),
  });
}

export function useAckContradictions() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (ids: string[]) => ackContradictions(ids),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["fact-contradictions"] }),
  });
}

/* Governance + settings (Phase 6) — read-only. */
export function useGovernanceState() {
  return useQuery({ queryKey: ["governance-state"], queryFn: getGovernanceState });
}

export function useGovernanceAudit(decision?: string, caller?: string) {
  return useQuery({
    queryKey: ["governance-audit", decision ?? "all", caller ?? "all"],
    queryFn: () => getGovernanceAudit(decision, caller),
  });
}

export function usePiiShadow(days = 7) {
  return useQuery({ queryKey: ["governance-pii-shadow", days], queryFn: () => getPiiShadow(days) });
}

export function useProofBundleCheck(days = 7) {
  return useQuery({
    queryKey: ["governance-proof-bundle", days],
    queryFn: () => getProofBundleCheck(days),
  });
}

export function useSettingsCatalog() {
  return useQuery({ queryKey: ["settings-catalog"], queryFn: getSettingsCatalog });
}

export function useRestartStatus() {
  return useQuery({
    queryKey: ["restart-status"],
    queryFn: getRestartStatus,
    // While changes wait for a restart, keep asking: after a restart the server's
    // start time moves on and the list empties, and the page must notice by itself.
    refetchInterval: (q) => ((q.state.data?.waiting_for_restart?.length ?? 0) > 0 ? 4_000 : false),
  });
}

/** Everything a settings change or a restart can make stale. */
export function useInvalidateSettings() {
  const qc = useQueryClient();
  return () => invalidateSettings(qc);
}

export function useSettingsHistory() {
  return useQuery({ queryKey: ["settings-history"], queryFn: () => getSettingsHistory(100) });
}

/* A setting change touches the catalog, the restart banner, the history and plugins. */
function invalidateSettings(qc: ReturnType<typeof useQueryClient>) {
  for (const key of ["settings-catalog", "restart-status", "settings-history", "plugins"]) {
    void qc.invalidateQueries({ queryKey: [key] });
  }
}

export function useModels() {
  return useQuery({ queryKey: ["models"], queryFn: getModels });
}

/* A model edit changes the tiers, the restart banner and the history. */
function useModelMutation<A>(fn: (a: A) => Promise<unknown>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => {
      for (const key of ["models", "restart-status", "settings-history"]) {
        void qc.invalidateQueries({ queryKey: [key] });
      }
    },
  });
}

export const usePatchTier = () =>
  useModelMutation((a: { name: string; fields: Partial<TierFields>; confirm?: boolean }) =>
    patchTier(a.name, a.fields, a.confirm ?? false),
  );
export const useResetTier = () => useModelMutation((name: string) => resetTier(name));
export const useMoveIntent = () =>
  useModelMutation((a: { intent: string; tier: string; confirm?: boolean }) =>
    moveIntent(a.intent, a.tier, a.confirm ?? false),
  );
export const useResetIntent = () => useModelMutation((intent: string) => resetIntent(intent));
export function useWatchConfig() {
  return useQuery({ queryKey: ["watch-config"], queryFn: getWatchConfig });
}

function useWatchMutation<A>(fn: (a: A) => Promise<unknown>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => {
      for (const key of ["watch-config", "settings-history"]) {
        void qc.invalidateQueries({ queryKey: [key] });
      }
    },
  });
}

export const usePatchWatchConfig = () =>
  useWatchMutation((changes: Record<string, number | boolean>) => patchWatchConfig(changes));
export const useResetWatchConfig = () => useWatchMutation(() => resetWatchConfig());

export function useDigestConfig() {
  return useQuery({ queryKey: ["digest-config"], queryFn: getDigestConfig });
}

function useDigestMutation<A>(fn: (a: A) => Promise<unknown>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => {
      for (const key of ["digest-config", "settings-history"]) {
        void qc.invalidateQueries({ queryKey: [key] });
      }
    },
  });
}

export const usePatchDigestConfig = () =>
  useDigestMutation((changes: Partial<DigestFields>) => patchDigestConfig(changes));
export const useResetDigestConfig = () => useDigestMutation(() => resetDigestConfig());

export function usePatchSetting() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (v: { name: string; value: string | boolean | number; confirm?: boolean }) =>
      patchSetting(v.name, v.value, v.confirm ?? false),
    onSuccess: () => invalidateSettings(qc),
  });
}

export function useResetSetting() {
  const qc = useQueryClient();
  return useMutation({ mutationFn: resetSetting, onSuccess: () => invalidateSettings(qc) });
}

export function useRequestRestart() {
  return useMutation({ mutationFn: requestRestart });
}

export function useSettings() {
  return useQuery({ queryKey: ["settings"], queryFn: getSettings });
}

/* System Health (ADR-0069) — read-only. Polled so the banner + screen reflect
 * the background health_tick without a manual refresh. */
export function useHealth() {
  return useQuery({ queryKey: ["health"], queryFn: getHealth, refetchInterval: 30_000 });
}

/* LLM spend (Track 2 PR 6). Polled with the rest of Pulse; the ledger is a
 * local SQLite read, so this costs nothing to keep fresh. */
export function useCost() {
  return useQuery({ queryKey: ["cost"], queryFn: getCost, refetchInterval: 60_000 });
}

/* Health-watch incidents (ADR-0116): what broke, what IRIS tried, whether it asked
 * the owner. Polled with the snapshot so a new incident shows without a reload. */
export function useHealthIncidents() {
  return useQuery({
    queryKey: ["health-incidents"],
    queryFn: () => getHealthIncidents(),
    refetchInterval: 30_000,
  });
}

/* One watch pass now; refetches the snapshot and the incidents it may have changed. */
export function useRunHealthWatch() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: runHealthWatch,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["health"] });
      qc.invalidateQueries({ queryKey: ["health-incidents"] });
    },
  });
}

/* System Check (iris doctor's install preflight, OSS plan R5). Polled gently --
 * this doesn't change on its own between a model pull completing and the owner
 * fixing something at a terminal. */
export function useDoctorReport() {
  return useQuery({
    queryKey: ["doctor"],
    queryFn: getDoctorReport,
    refetchInterval: 30_000,
  });
}

/* Pull a missing starter model: invalidate the report (and the Activity feed
 * already polls on its own) once the job is submitted, so "missing" clears as
 * soon as the background pull finishes. */
export function usePullStarterModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: pullStarterModel,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["activities"] });
    },
  });
}

/* Context health (ADR-0081) — how the harness manages its context budget
 * (window fill, in-loop budgets, suppression). Polled so it tracks live turns. */
export function useContextHealth(sessionId = "default") {
  return useQuery({
    queryKey: ["context-health", sessionId],
    queryFn: () => getContextHealth(sessionId),
    refetchInterval: 20_000,
  });
}

/* Compact a session's conversation on demand (ADR-0084) — the action half of the
 * context-health control loop. Refetches the snapshot so the window fill drops. */
export function useCompactContext() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) => compactContext(sessionId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["context-health"] }),
  });
}

/* Learning experiment console (ADR-0083) — hot-toggle the miners + run-now,
 * then watch the measurement. Flags poll so a heartbeat-driven change shows. */
export function useLearningFlags() {
  return useQuery({
    queryKey: ["learning-flags"],
    queryFn: getLearningFlags,
    refetchInterval: 15_000,
  });
}

export function useSetLearningFlag() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, enabled }: { name: string; enabled: boolean }) =>
      setLearningFlag(name, enabled),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["learning-flags"] }),
  });
}

export function useRunLearningNow() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => runLearningNow(name),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["learning-proposal-quality"] });
      qc.invalidateQueries({ queryKey: ["learning-metrics"] });
    },
  });
}

export function useProposalQuality() {
  return useQuery({
    queryKey: ["learning-proposal-quality"],
    queryFn: getProposalQuality,
    refetchInterval: 20_000,
  });
}

/* Action Center (ADR-0073) — the unified pending-actions inbox (persisted
 * agent actions + live Health items). Polled so it reflects background ticks. */
export function useActions() {
  return useQuery({ queryKey: ["actions"], queryFn: getActions, refetchInterval: 30_000 });
}

export function useOutstandingItems(includeResolvedDues = false) {
  return useQuery({
    queryKey: ["chat-outstanding-items", includeResolvedDues],
    refetchInterval: 30_000,
    queryFn: async () => {
      const [tasksResp, remindersResp, duesResp] = await Promise.allSettled([
        getTasks(),
        getReminders(),
        getFinanceDues(includeResolvedDues),
      ]);
      const tasks =
        tasksResp.status === "fulfilled"
          ? (tasksResp.value.tasks ?? []).filter(
              (t) =>
                (t.status === "open" || t.status === "doing") &&
                // A reminder's Action Center card is a task too; the reminder itself is
                // already listed below with its own Done / Snooze, so don't list it twice.
                t.source_kind !== "reminder",
            )
          : [];
      const reminders =
        remindersResp.status === "fulfilled"
          ? (remindersResp.value.reminders ?? []).filter(isActiveListed)
          : [];
      const dues = duesResp.status === "fulfilled" ? duesResp.value.dues ?? [] : [];
      return {
        tasks,
        reminders,
        dues,
        count: tasks.length + reminders.length + dues.length,
      };
    },
  });
}

/* After Done / Snooze / Undo: the answer's view goes straight into the sheet when it
 * says what can be done next (`actions`); either way the sheet refetches, and the
 * Chat panel and the bell drop or regain the row. */
function freshReminder(qc: QueryClient, id: string, view: ReminderDetail | undefined) {
  if (view && Array.isArray(view.actions)) qc.setQueryData(["reminder", id], view);
  qc.invalidateQueries({ queryKey: ["reminder", id] });
  qc.invalidateQueries({ queryKey: ["chat-outstanding-items"] });
}

/* One reminder, for the sheet a push tap opens (/reminders/:id). A 404 is final: the
 * sheet says the reminder is gone instead of retrying. */
export function useReminder(id: string | undefined) {
  return useQuery({
    queryKey: ["reminder", id],
    queryFn: () => getReminder(id!),
    enabled: Boolean(id),
    retry: (count, err) => !(err instanceof ReminderGone) && count < 2,
  });
}

export interface ReminderActInput {
  id: string;
  action: "done" | SnoozeFor;
  source: ReminderSource;
}

/* Done or Snooze, from the sheet or the Chat panel. The answer carries the updated
 * reminder (written into the sheet's cache at once) and the undo token. */
export function useReminderAct() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, action, source }: ReminderActInput): Promise<ReminderActionResult> =>
      action === "done" ? markReminderDone(id, source) : snoozeReminder(id, action, source),
    onSuccess: (res, { id }) => freshReminder(qc, id, res.reminder),
  });
}

export function useReminderUndo() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, undo }: { id: string; undo: ReminderUndo }) => undoReminder(id, undo),
    onSuccess: (res, { id }) => freshReminder(qc, id, res.reminder),
  });
}

/* Open-action count for the sidebar badge (0 while loading / on error). */
export function useActionCount(): number {
  return useActions().data?.count ?? 0;
}

/* Pending approvals — a halted run waiting on a human. Polled on the same cadence as
 * the actions inbox it sits above, so a halt raised by another surface shows up here. */
export function usePendingApprovals() {
  return useQuery({
    queryKey: ["governance-approvals"],
    queryFn: getApprovals,
    refetchInterval: 15_000,
  });
}

export function useRespondToApproval() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, status }: { id: string; status: "approved" | "rejected" }) =>
      respondToApproval(id, status),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["governance-approvals"] });
      // The approval also appears in the Action Center, and approving may have
      // produced a chat turn in the session the run belongs to.
      qc.invalidateQueries({ queryKey: ["actions"] });
      qc.invalidateQueries({ queryKey: ["governance-audit"] });
    },
  });
}

export function useInvokeAction() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (arg: string | { id: string; answer: ActionAnswer }) =>
      typeof arg === "string" ? invokeAction(arg) : invokeAction(arg.id, arg.answer),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["actions"] });
      qc.invalidateQueries({ queryKey: ["finance-senders"] });
    },
  });
}

/** Senders IRIS skipped or the owner ignored (ADR-0121). Absent when finance isn't mounted. */
export function useFinanceSenders() {
  return useQuery({
    queryKey: ["finance-senders"],
    queryFn: getFinanceSenders,
    retry: false,
    refetchInterval: 60_000,
  });
}

export function useProposeSender() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (domain: string) => proposeFinanceSender(domain),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["actions"] });
      qc.invalidateQueries({ queryKey: ["finance-senders"] });
    },
  });
}

export function useCreateTask() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: createTask,
    // The same two the status mutation refreshes: the Chat panel reads the
    // first and the sidebar/tab badge reads the second, so a new task shows
    // in both without a reload.
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["chat-outstanding-items"] });
      qc.invalidateQueries({ queryKey: ["actions"] });
    },
  });
}

export function useUpdateTaskStatus() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, status }: { id: string; status: "open" | "doing" | "done" | "dropped" }) =>
      updateTaskStatus(id, status),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["chat-outstanding-items"] });
      qc.invalidateQueries({ queryKey: ["actions"] });
    },
  });
}

export function useUpdateDueStatus() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, status }: { id: string; status: "open" | "resolved" | "dismissed" }) =>
      updateFinanceDueStatus(id, status),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["chat-outstanding-items"] }),
  });
}

/* Surface feedback (issue 0028) — mark a pending action "not useful" so IRIS
 * suppresses similar ones. Refetches actions so the suppressed item disappears. */
export function useSubmitSurfaceFeedback() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ ref, verdict }: { ref: string; verdict: "not_useful" | "useful" }) =>
      submitSurfaceFeedback(ref, verdict),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["actions"] }),
  });
}

/* Per-agent console (ADR-0074) — read-only. */
export function useAgents() {
  return useQuery({ queryKey: ["agents"], queryFn: getAgents, staleTime: 60_000 });
}

/* Plugin inventory — read-only; the registry only changes on restart. */
export function usePlugins() {
  return useQuery({ queryKey: ["plugins"], queryFn: getPlugins, staleTime: 60_000 });
}

export function usePlugin(name: string) {
  return useQuery({
    queryKey: ["plugin", name],
    queryFn: () => getPlugin(name),
    enabled: Boolean(name),
    staleTime: 60_000,
  });
}

export function useAgentDashboard(name: string) {
  return useQuery({
    queryKey: ["agent", name],
    queryFn: () => getAgentDashboard(name),
    enabled: Boolean(name),
    refetchInterval: 30_000,
  });
}

export function useAgentMetrics(name: string) {
  return useQuery({
    queryKey: ["agent-metrics", name],
    queryFn: () => getAgentMetrics(name),
    enabled: Boolean(name),
    staleTime: 60_000,
  });
}

export function useSetAgentToggle(name: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ key, enabled }: { key: string; enabled: boolean }) =>
      setAgentToggle(name, key, enabled),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["agent", name] }),
  });
}

/* Runtime inventory (#5) + self-learning experiments (#4) — read-only. */
export function useRuntimeInventory() {
  return useQuery({
    queryKey: ["runtime-inventory"],
    queryFn: getRuntimeInventory,
    staleTime: 60_000,
  });
}

export function useExperiments() {
  return useQuery({ queryKey: ["learning-experiments"], queryFn: getExperiments });
}

export function useLearningIntelligence() {
  return useQuery({
    queryKey: ["learning-intelligence"],
    queryFn: getLearningIntelligence,
    staleTime: 30_000,
  });
}

export function useLearningAnalysis() {
  return useQuery({
    queryKey: ["learning-analysis"],
    queryFn: getLearningAnalysis,
    staleTime: 60_000,
  });
}

export function usePromoteRecommendation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (index: number) => promoteRecommendation(index),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["learning-experiments"] }),
  });
}

export function useMemory(sessionId: string) {
  return useQuery({
    queryKey: ["memory", sessionId],
    queryFn: () => getMemory(sessionId),
    enabled: Boolean(sessionId),
  });
}

/* Digital-twin review surfaces (HITL). Reads below; the approve/reject/dismiss
 * mutations each invalidate the list they affect. */

export function useBehaviors(status = "pending") {
  return useQuery({ queryKey: ["twin-behaviors", status], queryFn: () => getBehaviors(status) });
}

export function useSignals(kind?: string) {
  return useQuery({ queryKey: ["twin-signals", kind ?? "all"], queryFn: () => getSignals(kind) });
}

export function useIntentions(status = "proposed") {
  return useQuery({ queryKey: ["twin-intentions", status], queryFn: () => getIntentions(status) });
}

export function useApproveBehavior() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => approveBehavior(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["twin-behaviors"] }),
  });
}

export function useRejectBehavior() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => rejectBehavior(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["twin-behaviors"] }),
  });
}

export function useApproveIntention() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => approveIntention(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["twin-intentions"] }),
  });
}

export function useDismissIntention() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => dismissIntention(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["twin-intentions"] }),
  });
}

/* Control-surface WRITES (Phase 7) — each invalidates the read it affects. */

export function useUpdateRoutineStatus() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, status }: { id: string; status: RoutineStatus }) =>
      updateRoutineStatus(id, status),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["routines"] }),
  });
}

export function usePatchHeartbeat() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, patch }: { name: string; patch: HeartbeatPatch }) =>
      patchHeartbeat(name, patch),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["heartbeats"] }),
  });
}

export function useResetHeartbeat() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: resetHeartbeat,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["heartbeats"] }),
  });
}

export function useTriggerHeartbeat() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: triggerHeartbeat,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["heartbeat-runs"] }),
  });
}

export function usePinLlmMode() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (mode: string) => pinLlmMode(mode),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["llm-mode"] }),
  });
}

export function useUnpinLlmMode() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: unpinLlmMode,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["llm-mode"] }),
  });
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: deleteRagDocument,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rag-documents"] }),
  });
}

// ---- Playground ---------------------------------------------------------

export function usePlaygroundSuites() {
  return useQuery({ queryKey: ["playground-suites"], queryFn: getPlaygroundSuites });
}

export function usePlaygroundDrift() {
  return useQuery({ queryKey: ["playground-drift"], queryFn: getPlaygroundDrift });
}

export function useRunPlaygroundSuite() {
  return useMutation({ mutationFn: (suite: string) => runPlaygroundSuite(suite) });
}

/* ── Memory screen: profile, facts, review, sessions, context, housekeeping ── */

export function useCuratedProfile() {
  return useQuery({ queryKey: ["memory-profile"], queryFn: () => getCuratedProfile() });
}

export function useWriteCuratedProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (profile: string) => writeCuratedProfile(profile),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["memory-profile"] });
    },
  });
}

export function useStoredFacts(confirmedOnly = true) {
  return useQuery({
    queryKey: ["memory-facts", confirmedOnly],
    queryFn: () => getStoredFacts(confirmedOnly),
  });
}

export function useMemoryReview() {
  return useQuery({ queryKey: ["memory-review"], queryFn: () => getMemoryReview() });
}

export function useLookalikes() {
  return useQuery({ queryKey: ["memory-lookalikes"], queryFn: () => getLookalikes() });
}

export function useLearnedTerms() {
  return useQuery({ queryKey: ["memory-terms"], queryFn: () => getLearnedTerms() });
}

export function useLessons() {
  return useQuery({ queryKey: ["memory-lessons"], queryFn: () => getLessons() });
}

/** Every review mutation invalidates the queue, the fact list it feeds, and the
 * Action Center row that counts it. */
function useReviewMutation<T>(fn: (arg: T) => Promise<unknown>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["memory-review"] });
      void qc.invalidateQueries({ queryKey: ["memory-lessons"] });
      void qc.invalidateQueries({ queryKey: ["memory-lookalikes"] });
      void qc.invalidateQueries({ queryKey: ["memory-terms"] });
      void qc.invalidateQueries({ queryKey: ["memory-graph"] });
      void qc.invalidateQueries({ queryKey: ["memory-facts"] });
      void qc.invalidateQueries({ queryKey: ["actions"] });
    },
  });
}

export const useApproveProposal = () => useReviewMutation(approveProposal);
export const useRejectProposal = () => useReviewMutation(rejectProposal);
export const useConfirmFact = () => useReviewMutation(confirmFact);
export const useMarkEntitiesSame = () => useReviewMutation(markEntitiesSame);
export const useMarkEntitiesDistinct = () => useReviewMutation(markEntitiesDistinct);
export const useRejectTerm = () => useReviewMutation(rejectTerm);
export const useActivateTerm = () => useReviewMutation(activateTerm);
export const useForgetFact = () => useReviewMutation(forgetFact);
export const useApproveLesson = () => useReviewMutation(approveLesson);
export const useRejectLesson = () => useReviewMutation(rejectLesson);
export const useCorrectFact = () =>
  useReviewMutation(({ key, value, id }: { key: string; value: string; id?: string | null }) =>
    correctFact(key, value, id),
  );

export function useStoredSessions() {
  return useQuery({ queryKey: ["memory-sessions"], queryFn: () => getStoredSessions() });
}

export function useContextInspection(sessionId: string) {
  return useQuery({
    queryKey: ["memory-context", sessionId],
    queryFn: () => getContextInspection(sessionId),
    enabled: Boolean(sessionId),
  });
}

export function useCompactSession() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (sessionId: string) => compactSession(sessionId),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["memory-context"] });
      void qc.invalidateQueries({ queryKey: ["memory-sessions"] });
    },
  });
}

export function useForgetMatching() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ needle, confirm }: { needle: string; confirm: boolean }) =>
      forgetMatching(needle, confirm),
    onSuccess: (_data, vars) => {
      if (vars.confirm) {
        void qc.invalidateQueries({ queryKey: ["memory-sessions"] });
        void qc.invalidateQueries({ queryKey: ["memory-facts"] });
      }
    },
  });
}

export function useHousekeepingRuns() {
  return useQuery({ queryKey: ["memory-housekeeping"], queryFn: () => getHousekeepingRuns() });
}

export function useRunHousekeeping() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (dryRun: boolean) => runHousekeeping(dryRun),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ["memory-housekeeping"] });
      void qc.invalidateQueries({ queryKey: ["memory-sessions"] });
    },
  });
}

/* Removal (ADR-0119). A removal reaches the Map, recall and the chat list together, and
 * an entity's removal forgets facts that About you and the Context tab show, so every
 * change invalidates all of them plus the review surfaces a restored fact returns to. */
/** The one-time "Looks like a test session" review (ADR-0119, cleanup decision 7). */
export function useTestSessions() {
  return useQuery({ queryKey: ["memory-test-sessions"], queryFn: getTestSessions });
}

export function useRemovedItems() {
  return useQuery({ queryKey: ["memory-removed"], queryFn: getRemoved });
}

function useRemovalMutation<T, R>(fn: (arg: T) => Promise<R>) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => {
      for (const key of [
        "memory-removed",
        "memory-test-sessions",
        "memory",
        "memory-context",
        "memory-graph",
        "memory-facts",
        "memory-review",
        "memory-sessions",
        "sessions",
      ]) {
        void qc.invalidateQueries({ queryKey: [key] });
      }
    },
  });
}

/** A read, sent as a POST because it carries a target list; it changes nothing. */
export const usePreviewRemoval = () => useMutation({ mutationFn: previewRemoval });
export const useRemoveFromMemory = () => useRemovalMutation(removeFromMemory);
export const useRestoreRemoved = () => useRemovalMutation(restoreRemoved);
export const useDeleteRemoved = () => useRemovalMutation(deleteRemoved);

export function useMemoryGraph(opts: {
  focus: string | null;
  kinds: string[];
  confirmedOnly: boolean;
}) {
  return useQuery({
    queryKey: ["memory-graph", opts.focus, [...opts.kinds].sort().join(","), opts.confirmedOnly],
    queryFn: () => getMemoryGraph(opts),
  });
}

/* Paired devices (ADR-0117). Not behind IRIS_WEBUI_ALLOW_WRITES: pairing and
 * revoking are authentication administration, scoped per route by the server. */
export function useDevices() {
  return useQuery({ queryKey: ["devices"], queryFn: listDevices, refetchInterval: 30_000 });
}

/** Who this browser is to the API — service (dev proxy) or a paired device + scope. */
export function usePrincipal() {
  return useQuery({ queryKey: ["devices-me"], queryFn: getPrincipal, staleTime: 60_000 });
}

export function useStartPairing() {
  return useMutation({ mutationFn: (scope: DeviceScope) => startPairing(scope) });
}

export function useRevokeDevice() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (deviceId: string) => revokeDevice(deviceId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["devices"] }),
  });
}
