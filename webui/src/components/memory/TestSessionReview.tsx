/** "Looks like a test session" — a one-time bulk review (ADR-0119, cleanup decision 7).
 *
 * Old test runs were named by hand (cal6-verify, cascade, clr-probe …) and match none of
 * retention.yaml's test prefixes, so they sat in the Map, recall and the chat list as
 * the owner's conversations. The server lists every session whose id is not shaped like
 * a real conversation's (`test_session_review` in retention.yaml); each is ticked by
 * default, and removing goes through the same preview-fed confirmation as the Map, so
 * every one lands in the Removed tab and can be restored. CLI twin:
 * `iris memory test-sessions --remove`. */
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { fmtDateTime } from "@/components/control/parts";
import { RemoveDialog, type PickedTarget } from "@/components/memory/removal";
import { useTestSessions, useWritesEnabled } from "@/lib/queries";

export function TestSessionReview() {
  const canWrite = useWritesEnabled();
  const { data } = useTestSessions();
  // Ticked by default: remember only what the owner unticked.
  const [unticked, setUnticked] = useState<Set<string>>(new Set());
  const [removing, setRemoving] = useState<PickedTarget[]>([]);
  const sessions = data?.sessions ?? [];
  if (sessions.length === 0) return null;

  const ticked = sessions.filter((s) => !unticked.has(s.session_id));
  const toggle = (id: string) =>
    setUnticked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const shapes = data?.real_shapes ?? [];

  return (
    <section className="space-y-1.5" data-testid="test-session-review">
      <h3 className="text-xs font-semibold text-fg">Looks like a test session</h3>
      <p className="text-[10px] text-fg-subtle">
        {sessions.length} conversation(s) whose id is not shaped like a real one
        {shapes.length > 0 ? ` (${shapes.join("; ")})` : ""}. Untick any you want to keep;
        removed ones leave the Map, recall and the chat list, and can be restored from the
        Removed tab.
      </p>
      {canWrite && (
        <div className="flex flex-wrap items-center gap-2">
          <Button
            size="sm"
            variant="destructive"
            disabled={!ticked.length}
            onClick={() =>
              setRemoving(
                ticked.map((s) => ({ kind: "session", id: s.session_id, label: s.session_id })),
              )
            }
          >
            Remove ticked ({ticked.length})
          </Button>
          <Button size="sm" variant="ghost" onClick={() => setUnticked(new Set())}>
            Tick all
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setUnticked(new Set(sessions.map((s) => s.session_id)))}
          >
            Untick all
          </Button>
        </div>
      )}
      <ul className="max-h-[60vh] space-y-1 overflow-y-auto" data-testid="test-session-list">
        {sessions.map((s) => (
          <li
            key={s.session_id}
            className="flex items-start gap-2 rounded-lg border border-border bg-surface p-2 text-xs"
          >
            {canWrite && (
              <input
                type="checkbox"
                className="mt-0.5 h-4 w-4 shrink-0"
                aria-label={`Keep ticked to remove ${s.session_id}`}
                checked={!unticked.has(s.session_id)}
                onChange={() => toggle(s.session_id)}
              />
            )}
            <div className="min-w-0 flex-1 space-y-0.5">
              <p className="break-all font-mono text-fg">{s.session_id}</p>
              {s.summary_goal && <p className="break-words text-fg-subtle">{s.summary_goal}</p>}
              <p className="break-words text-[10px] text-fg-subtle">
                {s.reasons.join(" · ")} · {s.turns} turn(s) · {fmtDateTime(s.last_activity)}
              </p>
            </div>
          </li>
        ))}
      </ul>
      <RemoveDialog
        targets={removing}
        onClose={() => setRemoving([])}
        onRemoved={() => {
          setRemoving([]);
          setUnticked(new Set());
        }}
      />
    </section>
  );
}
