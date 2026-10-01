import type { GovernanceEvent } from '../lib/types';
import { fmtMs } from '../lib/nodeMeta';
import { Tag } from './Tag';
import { ClassificationTag, DecisionTag, LocalityTag } from './control/parts';

/* Every hook decision of one turn, in the order the kernel recorded them:
 * PRE_TURN, PRE_LLM_CALL per step, PRE/POST_TOOL_USE, PRE_RESPONSE. The graph above
 * folds a hook point's rows into one node per host; this keeps each row with who called,
 * the label, where the model ran and whether a deterministic handler answered. The
 * server sends the documented payload fields only and masks addresses in reasons. */
export function GovernanceTimeline({ events }: { events: GovernanceEvent[] }) {
  if (events.length === 0) {
    return (
      <p className="text-xs text-fg-subtle">
        No governance rows for this turn (no ledger, or the kernel was off).
      </p>
    );
  }
  return (
    <ol className="space-y-2" data-testid="governance-timeline">
      {events.map((e, i) => {
        const called = e.tool_name ?? (e.capability ? `${e.capability}.${e.method ?? '?'}` : undefined);
        return (
          <li
            key={e.id ?? i}
            className="rounded-lg border border-border bg-bg p-2.5"
            data-testid="governance-event"
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-[10px] text-fg-subtle">{i + 1}</span>
              <span className="font-mono text-xs font-semibold text-fg">{e.hook_point.toUpperCase()}</span>
              {e.step_id != null && (
                <span className="font-mono text-[10px] text-fg-subtle">step {e.step_id}</span>
              )}
              <DecisionTag decision={e.decision} />
              <span className="font-mono text-[11px] text-fg-muted">{e.plugin}</span>
              {e.classification && <ClassificationTag value={e.classification} />}
              <LocalityTag locality={e.locality} tier={e.tier ?? null} />
              {e.deterministic && (
                <Tag kind="opp">deterministic{e.handler ? ` · ${e.handler}` : ''}</Tag>
              )}
              <span className="ml-auto font-mono text-[10px] text-fg-subtle">+{fmtMs(e.t_offset_ms)}</span>
            </div>
            {(e.caller || called || e.digest_alg) && (
              <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[11px]">
                {e.caller && <span className="break-all text-fg">caller {e.caller}</span>}
                {called && <span className="break-all text-fg-muted">{called}</span>}
                {e.digest_alg && (
                  <span className="text-fg-subtle" title={e.digest_alg}>
                    keyed digest
                  </span>
                )}
              </div>
            )}
            {e.reason && <p className="mt-1 break-words text-[11px] text-fg-muted">{e.reason}</p>}
          </li>
        );
      })}
    </ol>
  );
}
