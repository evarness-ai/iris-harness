/* System Check screen (issue #67's decision-support ask, OSS plan R5) — the install
 * preflight `iris doctor` prints, read-only, over GET /health/doctor: Python, platform,
 * RAM, disk, the vault key, Ollama and the starter model, each with a verdict. A new
 * user picking hardware/models gets this instead of only a terminal.
 *
 * Every fix stays the CLI's (`iris doctor --fix`) except one: pulling a missing starter
 * model is just a download, so it is this screen's one write, gated like any other
 * (IRIS_WEBUI_ALLOW_WRITES) and run through the Activity spine with a live progress bar
 * — the same spine email setup's fetch/judge steps use. */
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { Section } from "@/components/layout";
import { Tag } from "@/components/Tag";
import { Button } from "@/components/ui/button";
import { ConfirmDialog, type ConfirmState } from "@/components/ConfirmDialog";
import { HealthStateTag, Notice, QueryState } from "@/components/control/parts";
import { ProgressBar } from "@/screens/ActivityFeed";
import { useActivities, useDoctorReport, usePullStarterModel, useWritesEnabled } from "@/lib/queries";
import { useQueryClient } from "@tanstack/react-query";
import type { DoctorCheck, DoctorReport, StarterModel } from "@/lib/control";

const VERDICT_TAG: Record<DoctorReport["verdict"], { kind: "ok" | "warn" | "bad"; text: string }> = {
  ready: { kind: "ok", text: "Ready" },
  demo_only: { kind: "warn", text: "Ready for the demo only" },
  not_ready: { kind: "bad", text: "Not ready" },
};

function CheckRow({ c }: { c: DoctorCheck }) {
  return (
    <div
      className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-lg border border-border bg-surface px-3 py-2"
      data-testid="doctor-check"
    >
      <HealthStateTag state={c.state} />
      <span className="text-xs font-medium text-fg">{c.name}</span>
      <span className="text-xs text-fg-muted">{c.detail}</span>
      {c.fix && c.status !== "pass" && (
        <code className="w-full break-all font-mono text-[11px] text-fg-subtle sm:ml-auto sm:w-auto sm:truncate">
          fix: {c.fix}
        </code>
      )}
    </div>
  );
}

/** One missing starter model: its size, a Pull button, and — once submitted — the
 * same live progress bar the Activity Feed uses, found in the already-polling
 * activities list by the id this screen's own mutation returned. */
function MissingModelRow({ model, canWrite }: { model: StarterModel; canWrite: boolean }) {
  const pull = usePullStarterModel();
  const { data: activities } = useActivities();
  const qc = useQueryClient();
  const [activityId, setActivityId] = useState<string | null>(null);
  const [confirm, setConfirm] = useState<ConfirmState | null>(null);
  const notified = useRef(false);

  const activity = (activities?.activities ?? []).find((a) => a.id === activityId) ?? null;
  const running = activity?.status === "queued" || activity?.status === "running";

  useEffect(() => {
    if (!activity || !activity.status || activity.status === "queued" || activity.status === "running") {
      return;
    }
    if (notified.current) return;
    notified.current = true;
    if (activity.status === "completed") {
      toast.success(`${model.name} pulled`);
    } else {
      toast.error(`${model.name}: ${activity.error || activity.status}`);
    }
    qc.invalidateQueries({ queryKey: ["doctor"] });
  }, [activity, model.name, qc]);

  const askPull = () =>
    setConfirm({
      title: `Pull ${model.name}?`,
      description: `Downloads roughly ${model.size_gb.toFixed(1)} GB from Ollama. Runs in the
        background; you can leave this page while it finishes.`,
      confirmLabel: "Pull",
      run: async () => {
        notified.current = false;
        const res = await pull.mutateAsync(model.name);
        setActivityId(res.activity_id);
      },
    });

  return (
    <div className="rounded-lg border border-border bg-surface px-3 py-2" data-testid="missing-model">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-xs text-fg">{model.name}</span>
        <span className="text-[11px] text-fg-subtle">~{model.size_gb.toFixed(1)} GB</span>
        {canWrite && !running && (
          <Button
            type="button"
            size="sm"
            className="ml-auto min-h-[36px]"
            disabled={pull.isPending}
            onClick={askPull}
          >
            {pull.isPending ? "Starting…" : "Pull"}
          </Button>
        )}
      </div>
      {running && activity && <ProgressBar frac={activity.progress} message={activity.progress_message} />}
      <ConfirmDialog state={confirm} onClose={() => setConfirm(null)} />
    </div>
  );
}

export function SystemCheckScreen() {
  const { data, isLoading, isError } = useDoctorReport();
  const canWrite = useWritesEnabled();
  const verdict = data ? VERDICT_TAG[data.verdict] : null;

  return (
    <div className="space-y-6">
      <Section title="Verdict">
        <QueryState
          loading={isLoading}
          error={isError}
          empty={!data}
          emptyText="No preflight report."
        >
          {data && verdict && (
            <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2">
              <Tag kind={verdict.kind}>{verdict.text}</Tag>
              <span className="text-xs text-fg-muted">
                {data.verdict === "demo_only"
                  ? "The synthetic demo runs; real use needs what's below fixed."
                  : data.verdict === "not_ready"
                    ? "Fix what's below first."
                    : "Nothing stands in the way."}
              </span>
            </div>
          )}
        </QueryState>
      </Section>

      {data && (
        <Section title="Checks">
          <div className="space-y-1.5">
            {data.checks.map((c, i) => (
              <CheckRow key={`${c.name}:${i}`} c={c} />
            ))}
          </div>
        </Section>
      )}

      {data && data.missing_models.length > 0 && (
        <Section title="Starter model">
          <div className="space-y-1.5">
            {data.missing_models.map((m) => (
              <MissingModelRow key={m.name} model={m} canWrite={canWrite} />
            ))}
            {!canWrite && (
              <Notice tone="muted">
                Read-only here — pull it from a terminal with `iris doctor --fix`, or open this
                console from a control-paired device.
              </Notice>
            )}
          </div>
        </Section>
      )}

      {data && (
        <Section title="Vault key">
          <div className="rounded-lg border border-border bg-surface px-3 py-2 text-xs">
            <span className="text-fg-muted">{data.key.detail || data.key.source}</span>
          </div>
        </Section>
      )}
    </div>
  );
}
