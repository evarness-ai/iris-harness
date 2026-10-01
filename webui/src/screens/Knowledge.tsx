import { useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  useEdgesState,
  useNodesState,
  type Edge,
  type Node,
  type NodeMouseHandler,
  type ReactFlowInstance,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import { Card } from "@/components/Card";
import { Notice } from "@/components/control/parts";
import { cn } from "@/lib/utils";
import { useKnowledgeGraph } from "@/lib/queries";
import { KGNode, KG_KIND_CLASS } from "@/components/knowledge/KGNode";
import { layoutKnowledge, type KGRFData } from "@/lib/kgLayout";
import { KG_KINDS, KG_KIND_LABEL, type KGKind, type KGNode as KGNodeT } from "@/lib/knowledge";

const nodeTypes = { kg: KGNode };

function FilterPill({
  kind,
  active,
  count,
  onToggle,
}: {
  kind: KGKind;
  active: boolean;
  count: number;
  onToggle: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onToggle}
      className={cn(
        "rounded-full border px-2.5 py-1 text-xs font-medium transition-colors",
        active ? KG_KIND_CLASS[kind] : "border-border bg-bg text-fg-subtle hover:bg-surface",
      )}
    >
      {KG_KIND_LABEL[kind]} <span className="font-mono">{count}</span>
    </button>
  );
}

function NodeDetail({ node }: { node: KGNodeT | null }) {
  if (!node) {
    return (
      <Notice>Click a node to inspect it. Tag nodes bridge wiki notes and documents.</Notice>
    );
  }
  const entries = Object.entries(node.meta).filter(([, v]) => v !== null && v !== "");
  return (
    <Card title={`${node.kind} · degree ${node.degree}`}>
      <div className="mb-2 text-sm font-medium text-fg">{node.label}</div>
      <div className="space-y-1 text-xs">
        {entries.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-3">
            <span className="text-fg-subtle">{k}</span>
            <span className="break-all text-right font-mono text-fg">{String(v)}</span>
          </div>
        ))}
        {entries.length === 0 && <span className="text-fg-subtle">No extra metadata.</span>}
      </div>
    </Card>
  );
}

export function KnowledgeScreen() {
  const { data, isLoading, isError } = useKnowledgeGraph();
  const [visible, setVisible] = useState<Set<KGKind>>(() => new Set(KG_KINDS));
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [rfNodes, setRfNodes, onNodesChange] = useNodesState<Node<KGRFData>>([]);
  const [rfEdges, setRfEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const rfi = useRef<ReactFlowInstance<Node<KGRFData>, Edge> | null>(null);

  const byId = useMemo(() => new Map((data?.nodes ?? []).map((n) => [n.id, n])), [data]);

  useEffect(() => {
    if (!data) {
      setRfNodes([]);
      setRfEdges([]);
      return;
    }
    const { nodes, edges } = layoutKnowledge(data, visible);
    setRfNodes(nodes);
    setRfEdges(edges);
    const t = window.setTimeout(() => rfi.current?.fitView({ padding: 0.15, duration: 300 }), 30);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, visible]);

  const onNodeClick: NodeMouseHandler = (_e, n) => setSelectedId(n.id);

  const toggle = (kind: KGKind) =>
    setVisible((prev) => {
      const next = new Set(prev);
      if (next.has(kind)) next.delete(kind);
      else next.add(kind);
      return next;
    });

  const stats = data?.stats;
  const selected = selectedId ? (byId.get(selectedId) ?? null) : null;

  if (isLoading) return <Notice>Building the context map…</Notice>;
  if (isError || !data)
    return <Notice tone="danger">API unavailable — is the IRIS API running? Start it with <code>iris serve</code> (port 8003 by default).</Notice>;

  const empty = data.nodes.length === 0;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        {KG_KINDS.map((k) => (
          <FilterPill
            key={k}
            kind={k}
            active={visible.has(k)}
            count={stats ? (stats[k as keyof typeof stats] as number) : 0}
            onToggle={() => toggle(k)}
          />
        ))}
        <span className="ml-auto font-mono text-[11px] text-fg-subtle">
          {stats?.edges ?? 0} edges
          {stats && stats.facts_unlinked + stats.signals_unlinked > 0 && (
            <> · {stats.facts_unlinked + stats.signals_unlinked} memory items unlinked</>
          )}
        </span>
      </div>

      {empty ? (
        <Notice>
          The context map is empty. Add wiki notes or upload documents (with #tags and [[links]]) to
          grow the graph.
        </Notice>
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_320px]">
          <div className="relative h-[460px] rounded-lg border border-border bg-bg md:h-[640px]">
            <ReactFlow
              nodes={rfNodes}
              edges={rfEdges}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onInit={(i) => (rfi.current = i)}
              nodeTypes={nodeTypes}
              onNodeClick={onNodeClick}
              fitView
              fitViewOptions={{ padding: 0.15 }}
              proOptions={{ hideAttribution: true }}
              nodesConnectable={false}
              minZoom={0.1}
            >
              <Background color="rgb(var(--surface-raised))" gap={18} />
              <Controls showInteractive={false} />
              <MiniMap pannable zoomable className="!bg-surface" />
            </ReactFlow>
          </div>
          <NodeDetail node={selected} />
        </div>
      )}
    </div>
  );
}
