import { Handle, Position, type NodeProps } from "@xyflow/react";
import { cn } from "@/lib/utils";
import type { KGKind } from "@/lib/knowledge";

// Per-kind token classes (no inline style, no raw hex) — same node palette as
// the Call Trace graph.
export const KG_KIND_CLASS: Record<KGKind, string> = {
  wiki: "border-node-cognition/50 bg-node-cognition/10 text-node-cognition",
  document: "border-node-action/50 bg-node-action/10 text-node-action",
  tag: "border-node-llm/50 bg-node-llm/15 text-node-llm",
  fact: "border-node-perception/50 bg-node-perception/10 text-node-perception",
  signal: "border-node-runtime/50 bg-node-runtime/10 text-node-runtime",
};

const HANDLE = "!h-1 !w-1 !min-w-0 !border-0 !bg-transparent";

export function KGNode({ data, selected }: NodeProps) {
  const d = data as unknown as { kind: KGKind; label: string };
  return (
    <div
      className={cn(
        "flex items-center rounded-full border px-3 py-1 text-xs font-medium shadow-e1",
        KG_KIND_CLASS[d.kind],
        selected && "ring-2 ring-ring",
      )}
    >
      <Handle type="target" position={Position.Left} className={HANDLE} />
      <span className="max-w-[150px] truncate">{d.label}</span>
      <Handle type="source" position={Position.Right} className={HANDLE} />
    </div>
  );
}
