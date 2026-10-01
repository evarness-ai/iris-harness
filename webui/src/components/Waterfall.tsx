import { KIND_META, fmtMs } from "../lib/nodeMeta";
import type { Trace } from "../lib/types";

// Compact waterfall: one horizontal bar per node, positioned by t_offset_ms and
// sized by duration_ms — at-a-glance "how long each step took". Bar position +
// width are runtime values, so inline style is unavoidable; colors are tokens.
export function Waterfall({
  trace,
  selectedId,
  onSelect,
}: {
  trace: Trace;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const span = Math.max(trace.total_duration_ms, 1);
  const rows = [...trace.nodes].sort((a, b) => a.t_offset_ms - b.t_offset_ms);
  return (
    <div className="space-y-1">
      {rows.map((n) => {
        const meta = KIND_META[n.kind];
        // Clamp so a bar never extends past the track (left + width <= 100); a
        // late step with a long duration must not spill into the panel beside it.
        const left = Math.max(0, Math.min(99.2, (n.t_offset_ms / span) * 100));
        const width = Math.min(100 - left, Math.max((n.duration_ms / span) * 100, 0.8));
        const sel = n.id === selectedId;
        return (
          <button
            key={n.id}
            type="button"
            onClick={() => onSelect(n.id)}
            className="flex w-full items-center gap-2 rounded px-1 py-0.5 text-left hover:bg-surface-raised"
          >
            <span className="w-24 shrink-0 truncate font-mono text-[10.5px] text-fg-muted sm:w-40">
              {n.label}
            </span>
            <span className="relative h-3 flex-1 overflow-hidden rounded bg-bg">
              <span
                className="absolute top-0 h-3 rounded"
                style={{
                  left: `${left}%`,
                  width: `${width}%`,
                  background: n.status === "error" ? "rgb(var(--danger))" : meta.color,
                  opacity: sel ? 1 : 0.7,
                  outline: sel ? "1px solid rgb(var(--fg))" : undefined,
                }}
              />
            </span>
            <span className="w-14 shrink-0 text-right font-mono text-[10px] text-fg-subtle">
              {fmtMs(n.duration_ms)}
            </span>
          </button>
        );
      })}
    </div>
  );
}
