import { Button } from "@/components/ui/button";
import { Tag } from "@/components/Tag";
import { Notice, QueryState } from "@/components/control/parts";
import {
  useApproveBehavior,
  useApproveIntention,
  useBehaviors,
  useDismissIntention,
  useIntentions,
  useRejectBehavior,
  useSignals,
  useWritesEnabled,
} from "@/lib/queries";

const CONFIDENCE_TONE: Record<string, "ok" | "warn" | "info"> = {
  high: "ok",
  medium: "info",
  low: "warn",
};

/** Layer 1 — mined recurring habits awaiting review. Approve appends the habit to
 * durable episodic memory; reject drops it. Both are write-gated. */
function Behaviors() {
  const { data, isLoading, isError } = useBehaviors();
  const approve = useApproveBehavior();
  const reject = useRejectBehavior();
  const canWrite = useWritesEnabled();
  const rows = data?.behaviors ?? [];
  const busy = approve.isPending || reject.isPending;

  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-sm font-semibold text-fg">Behaviors</h2>
        <p className="text-[11px] text-fg-subtle">
          Mined habits. Facts and lessons are reviewed in <a className="underline" href="/memory">Memory → Review</a>.
        </p>
        <p className="text-[11px] text-fg-subtle">
          {data ? `${data.count} pending` : "review queue"} — recurring habits IRIS mined from your
          activity. Approve to add to episodic memory; reject to discard.
        </p>
      </div>

      <QueryState
        loading={isLoading}
        error={isError}
        empty={rows.length === 0}
        emptyText="No mined behaviors pending. Enable IRIS_BEHAVIOR_MINER to populate this queue."
      >
        <div className="space-y-1.5">
          {rows.map((b) => (
            <div
              key={b.pattern_id}
              className="flex items-start justify-between gap-2 rounded-lg border border-border bg-surface p-2 text-xs"
            >
              <span className="flex-1">
                <span className="text-fg">{b.text}</span>{" "}
                <Tag kind={CONFIDENCE_TONE[b.confidence] ?? "info"}>{b.confidence}</Tag>
                {b.evidence.length > 0 && (
                  <span className="ml-1 text-[10px] text-fg-subtle">
                    {b.evidence.slice(0, 3).join(" · ")}
                  </span>
                )}
              </span>
              {canWrite && (
                <span className="flex shrink-0 gap-1">
                  <Button
                    type="button"
                    disabled={busy}
                    onClick={() => approve.mutate(b.pattern_id)}
                  >
                    Approve
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    disabled={busy}
                    onClick={() => reject.mutate(b.pattern_id)}
                  >
                    Reject
                  </Button>
                </span>
              )}
            </div>
          ))}
        </div>
      </QueryState>
    </section>
  );
}

/** Layer 2 — how the user steers the assistant. Read-only ground truth: corrections,
 * dismissals, confirmations. The substrate the intention rollup draws on. */
function Signals() {
  const { data, isLoading, isError } = useSignals();
  const rows = data?.signals ?? [];
  const summary = data?.summary ?? {};

  return (
    <section className="space-y-3 border-t border-border pt-4">
      <div>
        <h2 className="text-sm font-semibold text-fg">Steering Signals</h2>
        <p className="text-[11px] text-fg-subtle">
          How you've steered IRIS — corrections, dismissals, confirmations. Read-only ground truth.
        </p>
      </div>

      {Object.keys(summary).length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {Object.entries(summary).map(([kind, count]) => (
            <Tag key={kind} kind="res">
              {kind} {count}
            </Tag>
          ))}
        </div>
      )}

      <QueryState
        loading={isLoading}
        error={isError}
        empty={rows.length === 0}
        emptyText="No steering signals yet — they accrue as you correct facts and review proposals."
      >
        <div className="space-y-1.5">
          {rows.map((s) => (
            <div
              key={s.id}
              className="flex items-start justify-between gap-2 rounded-lg border border-border bg-surface p-2 text-xs"
            >
              <span className="flex-1">
                <Tag kind="info">{s.kind}</Tag>{" "}
                <span className="text-fg">{s.subject}</span>
                {s.detail && <span className="ml-1 text-fg-subtle">({s.detail})</span>}
              </span>
              <span className="shrink-0 text-[10px] text-fg-subtle">
                {s.created_at.slice(0, 10)}
              </span>
            </div>
          ))}
        </div>
      </QueryState>
    </section>
  );
}

/** Layer 3 — rolled-up longitudinal goals ("what you're working toward"). Approve
 * writes the goal to the ACTIVE identity layer (injected into context); dismiss drops
 * it. Both write-gated. The capstone that closes the loop. */
function Intentions() {
  const { data, isLoading, isError } = useIntentions();
  const approve = useApproveIntention();
  const dismiss = useDismissIntention();
  const canWrite = useWritesEnabled();
  const rows = data?.intentions ?? [];
  const busy = approve.isPending || dismiss.isPending;

  return (
    <section className="space-y-3 border-t border-border pt-4">
      <div>
        <h2 className="text-sm font-semibold text-fg">Intentions</h2>
        <p className="text-[11px] text-fg-subtle">
          {data ? `${data.count} proposed` : "review queue"} — higher-level goals you're working
          toward. Approve to adopt as an active goal (the agent works toward it); dismiss to drop.
        </p>
      </div>

      <QueryState
        loading={isLoading}
        error={isError}
        empty={rows.length === 0}
        emptyText="No proposed intentions. Enable IRIS_INTENTION_ROLLUP to roll your activity into goals."
      >
        <div className="space-y-1.5">
          {rows.map((i) => (
            <div
              key={i.intention_id}
              className="rounded-lg border border-border bg-surface p-2.5 text-xs"
            >
              <div className="flex items-start justify-between gap-2">
                <span className="flex-1 font-semibold text-fg">{i.title}</span>
                {canWrite && (
                  <span className="flex shrink-0 gap-1">
                    <Button
                      type="button"
                      disabled={busy}
                      onClick={() => approve.mutate(i.intention_id)}
                    >
                      Approve
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      disabled={busy}
                      onClick={() => dismiss.mutate(i.intention_id)}
                    >
                      Dismiss
                    </Button>
                  </span>
                )}
              </div>
              {i.summary && <p className="mt-1 text-fg-subtle">{i.summary}</p>}
              {i.supporting.length > 0 && (
                <p className="mt-1 text-[10px] text-fg-subtle">
                  draws on: {i.supporting.join(" · ")}
                </p>
              )}
            </div>
          ))}
        </div>
      </QueryState>
    </section>
  );
}

/** Digital Twin — the three propose-only layers IRIS builds about the user over time:
 * mined habits, how the user steers, and rolled-up goals. All HITL: nothing becomes
 * durable without explicit approval here (or via the `iris behaviors` / `iris intentions`
 * CLIs). Reads are always open; approve/reject/dismiss need IRIS_WEBUI_ALLOW_WRITES. */
export function DigitalTwinScreen() {
  const canWrite = useWritesEnabled();
  return (
    <div className="space-y-4">
      <Behaviors />
      <Signals />
      <Intentions />
      {!canWrite && (
        <Notice>
          Review only — set IRIS_WEBUI_ALLOW_WRITES=1 to approve/reject here (or use the `iris
          behaviors` / `iris intentions` CLIs).
        </Notice>
      )}
    </div>
  );
}
