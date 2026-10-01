import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import type { IRISNodeData } from "../lib/layout";
import { KIND_META, fmtMs, fmtTokens } from "../lib/nodeMeta";

// Custom React Flow node: one card per IRIS component in the trace. Colors come
// from design tokens (rgb(var(--...))); per-node sizing/opacity/selection are
// runtime values, so inline style is unavoidable here (React Flow node).
export function IRISNode({ data, selected }: NodeProps<Node<IRISNodeData>>) {
  const n = data.trace;
  const meta = KIND_META[n.kind];
  const gov = n.kind === "governance";
  const isErr = n.status === "error";
  const border = isErr ? "rgb(var(--danger))" : meta.color;
  const horizontal = data.dir === "LR";

  const denied = n.governance && n.governance.decision !== "allow";

  return (
    <div
      className={`rounded-lg border bg-surface ${data.active ? "iris-node-active" : ""}`}
      style={{
        width: gov ? 160 : 210,
        borderColor: selected ? "rgb(var(--fg))" : border,
        borderWidth: selected ? 2 : 1.5,
        opacity: data.revealed ? 1 : 0.28,
        padding: gov ? "6px 9px" : "8px 11px",
        boxShadow: selected ? "0 0 0 2px rgb(var(--ring) / 0.4)" : undefined,
      }}
    >
      <Handle
        type="target"
        position={horizontal ? Position.Left : Position.Top}
        style={{ background: border, width: 7, height: 7 }}
      />

      <div className="flex items-center gap-2">
        <span style={{ color: border }} className="text-sm leading-none">
          {meta.glyph}
        </span>
        <span className="truncate text-[12.5px] font-semibold text-fg">{n.label}</span>
      </div>

      {gov && n.governance ? (
        <div className="mt-1 flex items-center gap-1.5">
          <span
            className="rounded px-1.5 py-0.5 font-mono text-[9.5px] font-semibold"
            style={{
              color: denied ? "rgb(var(--danger))" : "rgb(var(--success))",
              background: denied ? "rgb(var(--danger) / 0.15)" : "rgb(var(--success) / 0.15)",
            }}
          >
            {n.governance.decision}
          </span>
          <span className="font-mono text-[9.5px] text-fg-subtle">{n.governance.hook}</span>
        </div>
      ) : (
        <div className="mt-1.5 flex flex-wrap items-center gap-1.5 font-mono text-[10px] text-fg-muted">
          <span className="rounded bg-bg px-1.5 py-0.5">{fmtMs(n.duration_ms)}</span>
          {n.tokens && (
            <span className="rounded bg-bg px-1.5 py-0.5 text-node-llm">
              {fmtTokens(n.tokens.total)} tok
            </span>
          )}
          {n.model && <span className="truncate text-fg-subtle">{n.model}</span>}
          {isErr && <span className="text-danger">error</span>}
        </div>
      )}

      {data.verbose && (
        <div className="mt-1.5 border-t border-border pt-1 font-mono text-[9px] leading-tight text-fg-subtle">
          {n.method && <div className="truncate text-info">{n.method}</div>}
          <div className="truncate">{n.component_path}</div>
        </div>
      )}

      <Handle
        type="source"
        position={horizontal ? Position.Right : Position.Bottom}
        style={{ background: border, width: 7, height: 7 }}
      />
    </div>
  );
}
