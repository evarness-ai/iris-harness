import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/Card";
import { Tag } from "@/components/Tag";
import { Section } from "@/components/layout";
import { ConfirmDialog, type ConfirmState } from "@/components/ConfirmDialog";
import {
  PENDING_STATUSES,
  QueryState,
  RoutineStatusTag,
  fmtDateTime,
} from "@/components/control/parts";
import { useRoutines, useUpdateRoutineStatus, useWritesEnabled } from "@/lib/queries";
import type { Routine, RoutineStatus } from "@/lib/control";

function Stat({ label, value, tone }: { label: string; value: number; tone?: "ok" | "bad" }) {
  const color = tone === "bad" ? "text-danger" : tone === "ok" ? "text-success" : "text-fg";
  return (
    <span className="font-mono text-[11px] text-fg-subtle">
      {label} <span className={color}>{value}</span>
    </span>
  );
}

type Act = { status: RoutineStatus; verb: string };

function rowActions(r: Routine): Act[] {
  if (PENDING_STATUSES.has(r.approval_status)) return [{ status: "approved", verb: "Approve" }];
  if (r.approval_status === "approved" || r.approval_status === "scheduled")
    return [{ status: "paused", verb: "Pause" }];
  if (r.approval_status === "paused") return [{ status: "scheduled", verb: "Resume" }];
  return [];
}

function RoutineRow({
  r,
  onAct,
  canWrite,
}: {
  r: Routine;
  onAct: (r: Routine, act: Act) => void;
  canWrite: boolean;
}) {
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-fg">{r.title}</span>
        <RoutineStatusTag status={r.approval_status} />
        {r.promotion_candidate && <Tag kind="opp">promotion candidate</Tag>}
        <span className="ml-auto font-mono text-[11px] text-fg-subtle">{r.schedule}</span>
        {canWrite &&
          rowActions(r).map((a) => (
            <Button
              key={a.status}
              type="button"
              size="sm"
              variant={a.verb === "Pause" ? "outline" : "default"}
              onClick={() => onAct(r, a)}
            >
              {a.verb}
            </Button>
          ))}
      </div>
      {r.goal && r.goal !== r.title && (
        <p className="mt-1 text-xs text-fg-muted">{r.goal}</p>
      )}
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="font-mono text-[11px] text-fg-subtle">
          template <span className="text-fg-muted">{r.template}</span>
        </span>
        <span className="font-mono text-[11px] text-fg-subtle">
          via <span className="text-fg-muted">{r.delivery_channel}</span>
        </span>
        <Stat label="runs" value={r.run_count} />
        <Stat label="ok" value={r.success_count} tone="ok" />
        <Stat label="fail" value={r.failure_count} tone={r.failure_count > 0 ? "bad" : undefined} />
        <span className="font-mono text-[11px] text-fg-subtle">
          last {fmtDateTime(r.last_run_at)}
        </span>
      </div>
    </div>
  );
}

export function RoutinesScreen() {
  const { data, isLoading, isError } = useRoutines();
  const update = useUpdateRoutineStatus();
  const canWrite = useWritesEnabled();
  const [confirm, setConfirm] = useState<ConfirmState | null>(null);
  const routines = data ?? [];
  const pending = routines.filter((r) => PENDING_STATUSES.has(r.approval_status));
  const live = routines.filter((r) => !PENDING_STATUSES.has(r.approval_status));

  const onAct = (r: Routine, act: Act) =>
    setConfirm({
      title: `${act.verb} routine?`,
      description: `"${r.title}" will move to "${act.status}".`,
      confirmLabel: act.verb,
      run: async () => {
        try {
          await update.mutateAsync({ id: r.id, status: act.status });
          toast.success(`${r.title} → ${act.status}`);
        } catch (e) {
          toast.error(e instanceof Error ? e.message : "update failed");
          throw e;
        }
      },
    });

  return (
    <div className="space-y-6">
      <QueryState
        loading={isLoading}
        error={isError}
        empty={routines.length === 0}
        emptyText="No routines yet. Author one in chat, or via the routines CLI."
      >
        {pending.length > 0 && (
          <Section title={`Proposals & drafts — awaiting your review (${pending.length})`}>
            <div className="space-y-2 rounded-lg border-l-2 border-l-primary bg-primary/5 p-2">
              {pending.map((r) => (
                <RoutineRow key={r.id} r={r} onAct={onAct} canWrite={canWrite} />
              ))}
            </div>
            <p className="mt-2 text-[11px] text-fg-subtle">
              The reflection loop proposes routines here. Approval is human-gated —{" "}
              {canWrite
                ? "use the actions above, or via chat / the CLI."
                : "enable editing (IRIS_WEBUI_ALLOW_WRITES=1) to act here, or approve via chat / the CLI."}
            </p>
          </Section>
        )}

        <Section title={`Routines (${live.length})`}>
          <div className="space-y-2">
            {live.length === 0 ? (
              <Card>
                <span className="text-xs text-fg-subtle">No active routines.</span>
              </Card>
            ) : (
              live.map((r) => <RoutineRow key={r.id} r={r} onAct={onAct} canWrite={canWrite} />)
            )}
          </div>
        </Section>
      </QueryState>

      <ConfirmDialog state={confirm} onClose={() => setConfirm(null)} />
    </div>
  );
}
