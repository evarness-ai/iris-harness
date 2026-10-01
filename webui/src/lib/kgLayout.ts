import dagre from "@dagrejs/dagre";
import type { Edge, Node } from "@xyflow/react";
import type { KGKind, KnowledgeGraph } from "./knowledge";

export interface KGRFData extends Record<string, unknown> {
  kind: KGKind;
  label: string;
  degree: number;
  meta: Record<string, unknown>;
}

const NODE_W = 170;
const NODE_H = 38;

/**
 * Lay the (filtered) knowledge graph out with dagre. Same deterministic
 * approach as the Call Trace graph; colors stay in the node component (token
 * classes), edge colors come from design tokens here.
 */
export function layoutKnowledge(
  graph: KnowledgeGraph,
  visible: Set<KGKind>,
): { nodes: Node<KGRFData>[]; edges: Edge[] } {
  const keep = new Set(graph.nodes.filter((n) => visible.has(n.kind)).map((n) => n.id));

  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", nodesep: 24, ranksep: 80, marginx: 20, marginy: 20 });
  g.setDefaultEdgeLabel(() => ({}));

  for (const n of graph.nodes) {
    if (keep.has(n.id)) g.setNode(n.id, { width: NODE_W, height: NODE_H });
  }
  const edges = graph.edges.filter((e) => keep.has(e.source) && keep.has(e.target));
  for (const e of edges) g.setEdge(e.source, e.target);

  dagre.layout(g);

  const nodes: Node<KGRFData>[] = graph.nodes
    .filter((n) => keep.has(n.id))
    .map((n) => {
      const p = g.node(n.id);
      return {
        id: n.id,
        type: "kg",
        position: { x: p.x - NODE_W / 2, y: p.y - NODE_H / 2 },
        data: { kind: n.kind, label: n.label, degree: n.degree, meta: n.meta },
      };
    });

  const rfEdges: Edge[] = edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    style: {
      // tag edges (the cross-corpus bridge) = teal dashed; link edges = neutral.
      stroke: e.kind === "tag" ? "rgb(var(--node-llm))" : "rgb(var(--border-strong))",
      strokeDasharray: e.kind === "tag" ? "4 3" : undefined,
      opacity: 0.7,
    },
  }));

  return { nodes, edges: rfEdges };
}
