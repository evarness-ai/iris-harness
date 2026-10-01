/* Self-learning / Intelligence screen (fast-follow #4) — read-only. Surfaces the
 * EXISTING telemetry as the baseline: signal health (from /observability/llm-metrics)
 * and the experiments ledger (hypothesis -> baseline vs current -> status). New
 * measures (accuracy, pattern mining, an agentic recommender) are a later phase. */
import { Tag } from "@/components/Tag";
import { Kpi } from "@/components/Card";
import { Button } from "@/components/ui/button";
import { Grid, Section } from "@/components/layout";
import { Notice, QueryState, fmtDateTime } from "@/components/control/parts";
import {
  useExperiments,
  useLearningAnalysis,
  useLearningFlags,
  useLearningIntelligence,
  useLearningMetrics,
  useProposalQuality,
  usePromoteRecommendation,
  useRunLearningNow,
  useSetLearningFlag,
  useWritesEnabled,
} from "@/lib/queries";
import {
  isMetricsDisabled,
  type Experiment,
  type LearningHealth,
  type LearningRecommendation,
  type OutcomeCell,
} from "@/lib/control";

function ExperimentStatusTag({ status }: { status: string }) {
  const tone =
    status === "kept"
      ? "ok"
      : status === "discarded" || status === "failed"
        ? "bad"
        : status === "evaluating"
          ? "warn"
          : "info"; // pending / running
  return <Tag kind={tone}>{status}</Tag>;
}

function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

function SignalHealth() {
  const { data, isLoading, isError } = useLearningMetrics();
  if (isMetricsDisabled(data)) {
    return (
      <Section title="Signal health">
        <Notice>
          Observability metrics are off. Set IRIS_OBSERVABILITY_METRICS_ENABLED=1 (Phoenix on) to
          populate self-learning signal health.
        </Notice>
      </Section>
    );
  }
  const learning = (data && !isMetricsDisabled(data) ? data.summary.learning : undefined) as
    | LearningHealth
    | undefined;
  const volume = learning?.signal_volume ?? {};
  return (
    <Section title="Signal health">
      <QueryState
        loading={isLoading}
        error={isError}
        empty={!learning}
        emptyText="No learning signals yet."
      >
        {learning && (
          <div className="space-y-3">
            <Grid cols={3}>
              <Kpi value={`${learning.signals_recorded_total ?? 0}`} label="signals recorded" />
              <Kpi value={`${learning.signals_dropped_total ?? 0}`} label="signals dropped" />
              <Kpi
                value={pct(learning.drop_rate ?? 0)}
                label="drop rate"
                tone={(learning.drop_rate ?? 0) > 0.05 ? "bad" : "ok"}
              />
            </Grid>
            {Object.keys(volume).length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(volume).map(([metric, count]) => (
                  <span
                    key={metric}
                    className="rounded-lg border border-border bg-surface px-2 py-1 text-[11px] text-fg"
                  >
                    {metric} <span className="font-mono text-fg-subtle">{count}</span>
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </QueryState>
    </Section>
  );
}

function OutcomeMatrix({ cells }: { cells: OutcomeCell[] }) {
  // Most-trafficked first (already sorted by the backend). Pure measurement —
  // no thresholds tint a cell "bad"; the reader judges from the numbers.
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-[11px]">
        <thead className="text-fg-subtle">
          <tr className="border-b border-border">
            <th className="py-1 pr-3 font-medium">intent @ tier</th>
            <th className="py-1 pr-3 font-medium">n</th>
            <th className="py-1 pr-3 font-medium">completion</th>
            <th className="py-1 pr-3 font-medium">correction</th>
            <th className="py-1 pr-3 font-medium">reuse</th>
            <th className="py-1 font-medium">avg tok</th>
          </tr>
        </thead>
        <tbody className="font-mono text-fg">
          {cells.map((c) => (
            <tr key={`${c.intent}@${c.tier}`} className="border-b border-border/50">
              <td className="py-1 pr-3">
                {c.intent} <span className="text-fg-subtle">@ {c.tier}</span>
              </td>
              <td className="py-1 pr-3 text-fg-subtle">{c.samples}</td>
              <td className="py-1 pr-3">{pct(c.completion_rate)}</td>
              <td className="py-1 pr-3">
                {c.correction_rate == null ? "—" : pct(c.correction_rate)}
              </td>
              <td className="py-1 pr-3">{c.reuse_count}</td>
              <td className="py-1">{c.avg_tokens == null ? "—" : Math.round(c.avg_tokens)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Intelligence() {
  const { data, isLoading, isError } = useLearningIntelligence();
  const acc = data?.available ? data.accuracy : undefined;
  const matrix = data?.available ? (data.matrix ?? []) : [];
  const windowH = data?.window_hours;
  return (
    <Section
      title="Intelligence"
      actions={
        windowH ? (
          <span className="font-mono text-[11px] text-fg-subtle">last {Math.round(windowH)}h</span>
        ) : undefined
      }
    >
      <QueryState
        loading={isLoading}
        error={isError}
        empty={!data?.available}
        emptyText="Learning intelligence unavailable (no learning store)."
      >
        {acc && (
          <div className="space-y-4">
            <Grid cols={3}>
              <Kpi
                value={acc.escalation_precision == null ? "—" : pct(acc.escalation_precision)}
                label="escalation-judge precision"
              />
              <Kpi
                value={pct(acc.escalation_would_act_rate)}
                label={`would-escalate (${acc.escalation_would_act}/${acc.escalation_shadow_total})`}
              />
              <Kpi
                value={pct(acc.drop_rate)}
                label="signal drop rate"
                tone={acc.drop_rate > 0.05 ? "bad" : "ok"}
              />
            </Grid>
            {matrix.length > 0 ? (
              <OutcomeMatrix cells={matrix} />
            ) : (
              <Notice>No measured turns in the window yet.</Notice>
            )}
          </div>
        )}
      </QueryState>
    </Section>
  );
}

function Feedback() {
  // Explicit user feedback (ADR-0072) — the satisfaction signal that feeds the
  // analyst. Read from the same intelligence report; hidden until any exists.
  const { data } = useLearningIntelligence();
  const fb = data?.available ? data.feedback : undefined;
  if (!fb || fb.total === 0) return null;
  const worst = [...fb.by_intent].sort((a, b) => a.satisfaction - b.satisfaction)[0];
  return (
    <Section title="User feedback">
      <div className="space-y-4">
        <Grid cols={3}>
          <Kpi
            value={fb.satisfaction == null ? "—" : pct(fb.satisfaction)}
            label="satisfaction"
            tone={fb.satisfaction != null && fb.satisfaction < 0.6 ? "bad" : "ok"}
          />
          <Kpi value={`${fb.positive}`} label="thumbs up" />
          <Kpi value={`${fb.negative}`} label="thumbs down" />
        </Grid>
        {fb.by_intent.length > 0 && (
          <div className="space-y-1">
            {fb.by_intent.map((f) => (
              <div
                key={f.intent}
                className="flex items-center justify-between rounded-md px-2.5 py-1.5 text-[13px] hover:bg-surface"
              >
                <span className="text-fg">{f.intent}</span>
                <span className="flex items-center gap-3 font-mono text-[11px] text-fg-subtle">
                  <span>{pct(f.satisfaction)}</span>
                  <span className="text-success">+{f.positive}</span>
                  <span className="text-danger">-{f.negative}</span>
                </span>
              </div>
            ))}
          </div>
        )}
        {worst && worst.satisfaction < 0.5 && (
          <Notice>
            Lowest satisfaction: <span className="font-medium">{worst.intent}</span> at{" "}
            {pct(worst.satisfaction)} — worth a closer look.
          </Notice>
        )}
      </div>
    </Section>
  );
}

function ConfidenceTag({ confidence }: { confidence: LearningRecommendation["confidence"] }) {
  const tone = confidence === "high" ? "ok" : confidence === "low" ? "warn" : "info";
  return <Tag kind={tone}>{confidence}</Tag>;
}

function RecommendationCard({
  r,
  index,
  canWrite,
  onTrack,
  tracking,
}: {
  r: LearningRecommendation;
  index: number;
  canWrite: boolean;
  onTrack: (index: number) => void;
  tracking: boolean;
}) {
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <ConfidenceTag confidence={r.confidence} />
        <span className="text-xs font-medium text-fg">{r.title}</span>
        {canWrite && (
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="ml-auto"
            disabled={tracking}
            onClick={() => onTrack(index)}
            title="Capture a baseline and measure this over time (does not apply the change)"
          >
            Track as experiment
          </Button>
        )}
      </div>
      {r.finding && <p className="mt-1.5 text-xs text-fg-muted">{r.finding}</p>}
      {r.action && (
        <p className="mt-1 text-xs text-fg">
          <span className="text-fg-subtle">Try:</span> {r.action}
        </p>
      )}
      {r.evidence.length > 0 && (
        <div className="mt-1.5 flex flex-wrap gap-1.5">
          {r.evidence.map((e, i) => (
            <span
              key={i}
              className="rounded-lg border border-border bg-bg px-2 py-0.5 font-mono text-[11px] text-fg-subtle"
            >
              {e}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function Recommendations() {
  const { data, isLoading, isError } = useLearningAnalysis();
  const canWrite = useWritesEnabled();
  const promote = usePromoteRecommendation();
  const recs = data?.available ? (data.recommendations ?? []) : [];
  return (
    <Section
      title="Recommendations"
      actions={
        data?.available && data.generated_at ? (
          <span className="font-mono text-[11px] text-fg-subtle">
            {data.model} · {fmtDateTime(data.generated_at)}
          </span>
        ) : undefined
      }
    >
      <QueryState
        loading={isLoading}
        error={isError}
        empty={!data?.available}
        emptyText="No recommendations yet. The learning analyst is opt-in (IRIS_LEARNING_ANALYST) and runs on a schedule; its advisory proposals appear here once it has analyzed the measured report."
      >
        <div className="space-y-3">
          {data?.summary && <p className="text-xs text-fg-muted">{data.summary}</p>}
          {recs.length > 0 ? (
            <div className="space-y-2">
              {recs.map((r, i) => (
                <RecommendationCard
                  key={i}
                  r={r}
                  index={i + 1}
                  canWrite={canWrite}
                  onTrack={(idx) => promote.mutate(idx)}
                  tracking={promote.isPending}
                />
              ))}
            </div>
          ) : (
            <Notice>The latest run found nothing worth acting on.</Notice>
          )}
        </div>
      </QueryState>
    </Section>
  );
}

function SandboxTag({ s }: { s: NonNullable<Experiment["sandbox"]> }) {
  const tone = s.verdict === "better" ? "ok" : s.verdict === "worse" ? "bad" : "info";
  const pct =
    s.improvement_pct != null ? ` ${s.improvement_pct > 0 ? "+" : ""}${s.improvement_pct}%` : "";
  return (
    <span title={`Sandbox pre-flight (${s.metric}): ${s.note}`}>
      <Tag kind={tone}>
        sandbox: {s.verdict}
        {pct}
      </Tag>
    </span>
  );
}

function ExperimentRow({ e }: { e: Experiment }) {
  const delta = e.current_metric != null ? e.current_metric - e.baseline_metric : null;
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <ExperimentStatusTag status={e.status} />
        <span className="text-xs font-medium text-fg">{e.hypothesis}</span>
        <Tag kind="res">{e.domain}</Tag>
        {e.sandbox && <SandboxTag s={e.sandbox} />}
        <span className="ml-auto font-mono text-[11px] text-fg-subtle">
          {fmtDateTime(e.created_at)}
        </span>
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-3 font-mono text-[11px] text-fg-subtle">
        <span>
          baseline {e.baseline_metric.toFixed(3)} →{" "}
          {e.current_metric != null ? e.current_metric.toFixed(3) : "—"}
        </span>
        {delta != null && (
          <span className={delta >= 0 ? "text-success" : "text-danger"}>
            {delta >= 0 ? "+" : ""}
            {delta.toFixed(3)}
          </span>
        )}
      </div>
      {e.variant_description && (
        <p className="mt-1 text-xs text-fg-muted">{e.variant_description}</p>
      )}
    </div>
  );
}

function Experiments() {
  const { data, isLoading, isError } = useExperiments();
  const experiments = data?.experiments ?? [];
  const counts = data?.status_counts ?? {};
  return (
    <Section
      title="Experiments"
      actions={
        Object.keys(counts).length > 0 ? (
          <div className="flex flex-wrap items-center gap-2">
            {Object.entries(counts).map(([status, n]) => (
              <span key={status} className="inline-flex items-center gap-1">
                <ExperimentStatusTag status={status} />
                <span className="font-mono text-[11px] text-fg-subtle">{n}</span>
              </span>
            ))}
          </div>
        ) : undefined
      }
    >
      <QueryState
        loading={isLoading}
        error={isError}
        empty={experiments.length === 0}
        emptyText="No experiments yet — the learning loop proposes them once enough signals accrue."
      >
        <div className="space-y-2">
          {experiments.map((e) => (
            <ExperimentRow key={e.id} e={e} />
          ))}
        </div>
      </QueryState>
    </Section>
  );
}

/* Experiment console (ADR-0083): hot-toggle the learning capabilities (no restart),
 * run them on demand, and watch the proposal-acceptance measurement — the tweak loop.
 * Toggles/run are write-gated (IRIS_WEBUI_ALLOW_WRITES); the state + measurement read
 * openly. Toggles are process-scoped: env_default is what a restart reverts to. */
const FLAG_LABELS: Record<string, string> = {
  behavior_miner: "Behavior miner (L1 habits)",
  intention_rollup: "Intention rollup (L3 goals)",
  learning_analyst: "Learning analyst",
  self_management: "Agent self-management (ADR-0086)",
};

// Miners can be run on demand; the self-management flag is a per-turn behaviour switch
// with nothing to "run now".
const RUNNABLE = new Set(["behavior_miner", "intention_rollup", "learning_analyst"]);

function ExperimentConsole() {
  const { data, isLoading, isError } = useLearningFlags();
  const quality = useProposalQuality();
  const canWrite = useWritesEnabled();
  const setFlag = useSetLearningFlag();
  const runNow = useRunLearningNow();
  const flags = data?.available ? data.flags : {};

  return (
    <Section title="Experiment console">
      {!canWrite && (
        <Notice>Read-only — set IRIS_WEBUI_ALLOW_WRITES=1 to flip flags and run miners.</Notice>
      )}
      <QueryState
        loading={isLoading}
        error={isError}
        empty={Object.keys(flags).length === 0}
        emptyText="Runtime not ready — no learning flags to show."
      >
        <div className="space-y-2">
          {Object.entries(flags).map(([name, f]) => (
            <div
              key={name}
              className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2"
            >
              <Tag kind={f.enabled ? "ok" : "info"}>{f.enabled ? "on" : "off"}</Tag>
              <span className="text-xs font-medium text-fg">{FLAG_LABELS[name] ?? name}</span>
              {f.enabled !== f.env_default && (
                <span className="font-mono text-[11px] text-fg-subtle">
                  (env: {f.env_default ? "on" : "off"} — resets on restart)
                </span>
              )}
              {canWrite && (
                <div className="ml-auto flex gap-1.5">
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={setFlag.isPending}
                    onClick={() => setFlag.mutate({ name, enabled: !f.enabled })}
                  >
                    {f.enabled ? "Turn off" : "Turn on"}
                  </Button>
                  {RUNNABLE.has(name) && (
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      disabled={!f.enabled || runNow.isPending}
                      onClick={() => runNow.mutate(name)}
                      title="Run now instead of waiting for the heartbeat"
                    >
                      Run now
                    </Button>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </QueryState>
      {quality.data?.available && (
        <div className="mt-2 flex flex-wrap gap-4 text-[11px] text-fg-muted">
          {[quality.data.behaviors, quality.data.intentions]
            .filter((q): q is NonNullable<typeof q> => Boolean(q))
            .map((q) => (
              <span key={q.subsystem}>
                <span className="text-fg-subtle">{q.subsystem}</span> acceptance{" "}
                <span className="text-fg">{pct(q.acceptance_rate)}</span> ({q.accepted}/{q.reviewed}),{" "}
                {q.awaiting} awaiting
              </span>
            ))}
        </div>
      )}
    </Section>
  );
}

export function LearningScreen() {
  return (
    <div className="space-y-6">
      <ExperimentConsole />
      <SignalHealth />
      <Intelligence />
      <Feedback />
      <Recommendations />
      <Experiments />
    </div>
  );
}
