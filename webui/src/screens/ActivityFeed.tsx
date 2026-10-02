import { Tag } from "@/components/Tag";
import { CopyBlock } from "@/components/CopyBlock";
import { Section } from "@/components/layout";
import { QueryState, fmtDateTime } from "@/components/control/parts";
import { useActivities } from "@/lib/queries";
import type { Activity } from "@/lib/control";

function statusTone(status: string): "ok" | "bad" | "warn" | "info" {
  switch (status) {
    case "completed":
      return "ok";
    case "failed":
    case "cancelled":
      return "bad";
    case "running":
    case "queued":
      return "warn";
    default:
      return "info";
  }
}

export function ProgressBar({ frac, message }: { frac: number; message: string }) {
  const pct = Math.round(Math.max(0, Math.min(1, frac)) * 100);
  return (
    <div className="mt-2">
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-border">
        {/* eslint-disable-next-line react/forbid-dom-props -- dynamic width requires inline style */}
        <div className="h-full rounded-full bg-primary transition-all" style={{ width: `${pct}%` }} />
      </div>
      <div className="mt-1 flex items-center justify-between text-[11px] text-fg-subtle">
        <span>{message || "working…"}</span>
        <span className="font-mono">{pct}%</span>
      </div>
    </div>
  );
}

function ActivityRow({ a }: { a: Activity }) {
  const running = a.status === "running" || a.status === "queued";
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[13px] text-fg">{a.title}</span>
        <Tag kind={statusTone(a.status)}>{a.status}</Tag>
        <span className="font-mono text-[11px] text-fg-subtle">{a.kind}</span>
        <span className="ml-auto font-mono text-[11px] text-fg-subtle">
          {fmtDateTime(a.finished_at ?? a.started_at ?? a.created_at)}
        </span>
      </div>
      {running && <ProgressBar frac={a.progress} message={a.progress_message} />}
      {a.status === "failed" && a.error ? (
        <CopyBlock label="error" text={a.error} tone="text-danger" />
      ) : a.result_summary ? (
        <CopyBlock label="result" text={a.result_summary} />
      ) : null}
      {a.undo_ref && a.status === "completed" && (
        <p className="mt-1 font-mono text-[11px] text-fg-subtle">
          undo: iris files organize undo {a.undo_ref}
        </p>
      )}
    </div>
  );
}

export function ActivityFeedScreen() {
  const { data, isLoading, isError } = useActivities();
  const activities = data?.activities ?? [];
  const running = data?.running ?? 0;
  return (
    <div className="space-y-6">
      <Section
        title={`Activity (${activities.length}${running ? ` · ${running} running` : ""})`}
      >
        <QueryState
          loading={isLoading}
          error={isError}
          empty={activities.length === 0}
          emptyText="No background activity yet. Long FileManager jobs (categorize / cleanup / organize) show up here while they run."
        >
          <div className="space-y-2">
            {activities.map((a) => (
              <ActivityRow key={a.id} a={a} />
            ))}
          </div>
        </QueryState>
      </Section>
    </div>
  );
}
