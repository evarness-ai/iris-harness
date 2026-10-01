/** The Map tab: the memory graph, opened small and expanded by clicking.
 *
 * It draws what is already stored — confirmed facts, session summaries, lessons,
 * patterns — and holds no copy of any of it. The wiki's Context Map laid out all 2,505
 * page nodes at once and struggled; this asks the server for "You" plus one hop, and a
 * click asks for one node's neighbours.
 *
 * Remove (ADR-0119): the detail panel removes one node; shift-click picks several for
 * one confirmation. Lessons and patterns carry no `ref` — they are files the owner
 * edits, so they have no Remove. */
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  Controls,
  ReactFlow,
  useEdgesState,
  useNodesState,
  type Edge,
  type Node,
  type ReactFlowInstance,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import dagre from "@dagrejs/dagre";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Button } from "@/components/ui/button";
import { Tag } from "@/components/Tag";
import { QueryState } from "@/components/control/parts";
import { RemoveDialog, type PickedTarget } from "@/components/memory/removal";
import { WriteGate } from "@/components/memory/tabs";
import { cn } from "@/lib/utils";
import { useMemoryGraph, useWritesEnabled } from "@/lib/queries";
import type { MemoryGraph, MemoryGraphKind, MemoryGraphNode } from "@/lib/control";

const KIND_CLASS: Record<MemoryGraphKind, string> = {
  you: "border-node-llm/60 bg-node-llm/20 text-node-llm",
  entity: "border-node-cognition/50 bg-node-cognition/10 text-node-cognition",
  fact: "border-node-perception/50 bg-node-perception/10 text-node-perception",
  session: "border-node-action/50 bg-node-action/10 text-node-action",
  lesson: "border-node-runtime/50 bg-node-runtime/10 text-node-runtime",
  pattern: "border-node-runtime/40 bg-node-runtime/5 text-node-runtime",
  more: "border-border bg-surface text-fg-subtle",
};

const KINDS: MemoryGraphKind[] = ["entity", "fact", "session", "lesson", "pattern"];
const HANDLE = "!h-1 !w-1 !min-w-0 !border-0 !bg-transparent";

interface MemoryNodeData extends Record<string, unknown> {
  kind: MemoryGraphKind;
  label: string;
  confirmed: boolean;
  /** The ontology class ("fin:FinancialInstitution"), when the node is a typed thing. */
  cls?: string;
  /** Shift-clicked into the removal selection. */
  picked?: boolean;
  previouslyRemoved?: boolean;
}

/** "fin:FinancialInstitution" → "FinancialInstitution"; the prefix is noise on a map. */
function localName(qualified: string | undefined): string | undefined {
  return qualified ? qualified.slice(qualified.indexOf(":") + 1) : undefined;
}

function MemoryNode({ data, selected }: NodeProps) {
  const d = data as unknown as MemoryNodeData;
  return (
    <div
      className={cn(
        "flex items-center rounded-full border px-3 py-1 text-xs font-medium shadow-e1",
        KIND_CLASS[d.kind],
        !d.confirmed && "opacity-50 border-dashed",
        d.previouslyRemoved && "border-2 border-dashed",
        selected && "ring-2 ring-ring",
        d.picked && "ring-2 ring-warning",
      )}
      data-picked={d.picked ? "true" : undefined}
      title={
        [
          localName(d.cls),
          d.confirmed ? undefined : "not confirmed — not used in prompts",
          d.previouslyRemoved ? "previously removed — a confirmed fact brought it back" : undefined,
        ]
          .filter(Boolean)
          .join(" · ") || undefined
      }
    >
      <Handle type="target" position={Position.Left} className={HANDLE} />
      <span className="max-w-[160px] truncate">{d.label}</span>
      <Handle type="source" position={Position.Right} className={HANDLE} />
    </div>
  );
}

const nodeTypes = { memory: MemoryNode };
const NODE_W = 180;
const NODE_H = 38;

function layout(
  graph: MemoryGraph,
  picked: ReadonlyMap<string, unknown>,
): { nodes: Node<MemoryNodeData>[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", nodesep: 24, ranksep: 90, marginx: 20, marginy: 20 });
  g.setDefaultEdgeLabel(() => ({}));
  for (const n of graph.nodes) g.setNode(n.id, { width: NODE_W, height: NODE_H });
  for (const e of graph.edges) g.setEdge(e.source, e.target);
  dagre.layout(g);

  return {
    nodes: graph.nodes.map((n) => {
      const pos = g.node(n.id);
      return {
        id: n.id,
        type: "memory",
        position: { x: (pos?.x ?? 0) - NODE_W / 2, y: (pos?.y ?? 0) - NODE_H / 2 },
        data: {
          kind: n.kind,
          label: n.label,
          confirmed: n.confirmed,
          cls: typeof n.meta?.class === "string" ? n.meta.class : undefined,
          picked: picked.has(n.id),
          previouslyRemoved: n.previously_removed === true,
        },
      };
    }),
    edges: graph.edges.map((e) => ({
      id: e.id,
      source: e.source,
      target: e.target,
      label: e.label,
      labelStyle: { fontSize: 9 },
    })),
  };
}

export function MemoryMapTab() {
  const [focus, setFocus] = useState<string | null>(null);
  const [visible, setVisible] = useState<Set<MemoryGraphKind>>(() => new Set(KINDS));
  const [confirmedOnly, setConfirmedOnly] = useState(false);
  const { data, isLoading, isError } = useMemoryGraph({
    focus,
    kinds: [...visible],
    confirmedOnly,
  });
  const [rfNodes, setRfNodes, onNodesChange] = useNodesState<Node<MemoryNodeData>>([]);
  const [rfEdges, setRfEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const rfi = useRef<ReactFlowInstance<Node<MemoryNodeData>, Edge> | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const canWrite = useWritesEnabled();
  // Kept as whole nodes, not ids: a focus click refetches the graph, and a picked node
  // may not be in the next one.
  const [picked, setPicked] = useState<Map<string, MemoryGraphNode>>(() => new Map());
  const [removing, setRemoving] = useState<PickedTarget[]>([]);

  useEffect(() => {
    if (!data) {
      setRfNodes([]);
      setRfEdges([]);
      return;
    }
    const laid = layout(data, picked);
    setRfNodes(laid.nodes);
    setRfEdges(laid.edges);
    window.setTimeout(() => rfi.current?.fitView({ padding: 0.2 }), 0);
    // Picking re-marks nodes below without a relayout, so it is not a dependency here.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, setRfNodes, setRfEdges]);

  useEffect(() => {
    setRfNodes((nodes) =>
      nodes.map((n) =>
        n.data.picked === picked.has(n.id) ? n : { ...n, data: { ...n.data, picked: picked.has(n.id) } },
      ),
    );
  }, [picked, setRfNodes]);

  const togglePick = (node: MemoryGraphNode) =>
    setPicked((prev) => {
      const next = new Map(prev);
      if (next.has(node.id)) next.delete(node.id);
      else next.set(node.id, node);
      return next;
    });

  const askRemove = (nodes: MemoryGraphNode[]) =>
    setRemoving(
      nodes
        .filter((n) => n.ref)
        .map((n) => ({ kind: n.ref!.kind, id: n.ref!.id, label: n.label })),
    );
  const pickedNodes = [...picked.values()];
  const pickedRemovable = pickedNodes.filter((n) => n.ref);

  const detail = useMemo(
    () => data?.nodes.find((n) => n.id === selected) ?? null,
    [data, selected],
  );

  return (
    <div className="space-y-3">
      <div>
        <h2 className="text-sm font-semibold text-fg">Map</h2>
        <p className="text-[11px] text-fg-subtle">
          Drawn from what is already stored — confirmed facts, session summaries, lessons
          and patterns. Nothing extra is extracted or kept. Click a node to expand it;
          shift-click to select several to remove.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-1">
        {KINDS.map((k) => (
          <button
            key={k}
            type="button"
            onClick={() =>
              setVisible((prev) => {
                const next = new Set(prev);
                if (next.has(k)) next.delete(k);
                else next.add(k);
                return next;
              })
            }
            className={`rounded-full border px-2 py-0.5 text-[11px] ${
              visible.has(k) ? KIND_CLASS[k] : "border-border text-fg-subtle"
            }`}
          >
            {k}
          </button>
        ))}
        <button
          type="button"
          onClick={() => setConfirmedOnly((v) => !v)}
          className={`rounded-full border px-2 py-0.5 text-[11px] ${
            confirmedOnly ? KIND_CLASS.you : "border-border text-fg-subtle"
          }`}
        >
          confirmed only
        </button>
        {focus && (
          <Button size="sm" variant="ghost" onClick={() => setFocus(null)}>
            Back to you
          </Button>
        )}
        {data && (
          <span className="ml-auto text-[11px] text-fg-subtle">
            {data.stats.shown} of {data.stats.total_nodes} nodes
          </span>
        )}
      </div>

      <QueryState
        loading={isLoading}
        error={isError}
        empty={(data?.nodes.length ?? 0) <= 1}
        emptyText="Nothing to draw yet — confirm a fact or let a session summarize, and it lands here."
      >
        <div className="h-[420px] rounded-lg border border-border bg-surface">
          <ReactFlow
            nodes={rfNodes}
            edges={rfEdges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onInit={(instance) => {
              rfi.current = instance;
            }}
            onNodeClick={(e, node) => {
              const hit = data?.nodes.find((n) => n.id === node.id);
              if (e.shiftKey) {
                if (hit?.ref) togglePick(hit);
                return;
              }
              setSelected(node.id);
              if (node.id !== "more") setFocus(node.id === focus ? null : node.id);
            }}
            multiSelectionKeyCode={null}
            fitView
            proOptions={{ hideAttribution: true }}
          >
            <Background />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>
      </QueryState>

      {picked.size > 0 && (
        <div
          className="flex flex-wrap items-center gap-2 rounded-lg border border-warning/50 bg-surface p-2 text-xs"
          data-testid="map-selection"
        >
          <span className="text-fg">
            <b>{picked.size}</b> selected — shift-click to add or drop
          </span>
          {canWrite && (
            <Button
              size="sm"
              variant="destructive"
              disabled={!pickedRemovable.length}
              onClick={() => askRemove(pickedRemovable)}
            >
              Remove selected…
            </Button>
          )}
          <Button size="sm" variant="ghost" onClick={() => setPicked(new Map())}>
            Clear
          </Button>
        </div>
      )}

      {detail && (
        <div className="space-y-1 rounded-lg border border-border bg-surface p-2 text-xs">
          <p className="flex flex-wrap items-center gap-1">
            <Tag kind="info">{detail.kind}</Tag>
            <span className="break-words text-fg">{detail.label}</span>
            {!detail.confirmed && <Tag kind="warn">not confirmed</Tag>}
          </p>
          {detail.previously_removed && (
            <p className="text-[11px] text-fg-subtle">
              Previously removed — a confirmed fact named it again, so it is back.
            </p>
          )}
          {Object.entries(detail.meta).map(([k, v]) => (
            <p key={k} className="break-words text-[10px] text-fg-subtle">
              {k}: {String(v).slice(0, 200)}
            </p>
          ))}
          {detail.ref ? (
            canWrite ? (
              <div className="pt-1">
                <Button size="sm" variant="destructive" onClick={() => askRemove([detail])}>
                  Remove from memory…
                </Button>
                <p className="mt-1 text-[10px] text-fg-subtle">
                  Reversible, from the Removed tab.
                </p>
              </div>
            ) : (
              <WriteGate canWrite={canWrite} />
            )
          ) : (
            (detail.kind === "lesson" || detail.kind === "pattern") && (
              <p className="break-words text-[11px] text-fg-subtle">
                {detail.kind === "lesson" ? "Lessons" : "Patterns"} are files you curate, so
                they have no Remove here — edit{" "}
                {typeof detail.meta.file === "string" && detail.meta.file ? (
                  <code className="text-fg">{detail.meta.file}</code>
                ) : (
                  "the file"
                )}{" "}
                to change this one, or hide the kind with the toggle above.
              </p>
            )
          )}
        </div>
      )}

      <RemoveDialog
        targets={removing}
        onClose={() => setRemoving([])}
        onRemoved={() => {
          setRemoving([]);
          setPicked(new Map());
          setSelected(null);
        }}
      />
    </div>
  );
}
