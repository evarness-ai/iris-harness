import dagre from '@dagrejs/dagre';
import type { Edge, Node } from '@xyflow/react';
import type { Trace } from './types';

export type Direction = 'TB' | 'LR';

const NODE_W = 210;
const GOV_W = 160;

function nodeH(verbose: boolean): number {
  return verbose ? 104 : 76;
}
function govH(verbose: boolean): number {
  return verbose ? 80 : 58;
}

export interface IRISNodeData extends Record<string, unknown> {
  trace: Trace['nodes'][number];
  active: boolean; // currently firing (replay)
  revealed: boolean; // already reached by the playhead (replay)
  replaying: boolean;
  verbose: boolean;
  dir: Direction;
}

export interface BuildOpts {
  active: string | null;
  revealedIds: Set<string>;
  replaying: boolean;
  dir: Direction;
  verbose: boolean;
}

/** "3 · llm" — the step number first, so the order is readable at a glance. */
export function edgeLabel(e: Trace['edges'][number]): string | undefined {
  if (e.seq === undefined) return e.label;
  return e.label ? `${e.seq} · ${e.label}` : `${e.seq}`;
}

/**
 * Lay the trace out with dagre. Direction TB (top→bottom) or LR (left→right);
 * deterministic — same (trace, dir, verbose) -> same coordinates.
 */
export function buildGraph(trace: Trace, opts: BuildOpts): { nodes: Node<IRISNodeData>[]; edges: Edge[] } {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: opts.dir, nodesep: 48, ranksep: opts.dir === 'LR' ? 90 : 64, marginx: 24, marginy: 24 });
  g.setDefaultEdgeLabel(() => ({}));

  const h = nodeH(opts.verbose);
  const gh = govH(opts.verbose);
  // Insert in execution order: dagre seeds sibling order from insertion order, so a
  // node's children read left→right (top→bottom in LR) in the order they ran.
  const bySeq = [...trace.edges].sort((a, b) => (a.seq ?? Infinity) - (b.seq ?? Infinity));
  for (const n of [...trace.nodes].sort((a, b) => a.t_offset_ms - b.t_offset_ms)) {
    const gov = n.kind === 'governance';
    g.setNode(n.id, { width: gov ? GOV_W : NODE_W, height: gov ? gh : h });
  }
  for (const e of bySeq) g.setEdge(e.source, e.target);

  dagre.layout(g);

  const nodes: Node<IRISNodeData>[] = trace.nodes.map((n) => {
    const pos = g.node(n.id);
    const gov = n.kind === 'governance';
    const w = gov ? GOV_W : NODE_W;
    const hh = gov ? gh : h;
    return {
      id: n.id,
      type: 'iris',
      position: { x: pos.x - w / 2, y: pos.y - hh / 2 },
      data: {
        trace: n,
        active: opts.active === n.id,
        revealed: !opts.replaying || opts.revealedIds.has(n.id),
        replaying: opts.replaying,
        verbose: opts.verbose,
        dir: opts.dir,
      },
    };
  });

  const edges: Edge[] = trace.edges.map((e) => {
    const targetRevealed = !opts.replaying || opts.revealedIds.has(e.target);
    const isData = e.kind === 'data';
    return {
      id: e.id,
      source: e.source,
      target: e.target,
      label: edgeLabel(e),
      animated: isData && (!opts.replaying || opts.active === e.target),
      style: {
        // control edges = governance gating (danger hue); data edges = neutral.
        stroke: e.kind === 'control' ? 'rgb(var(--danger))' : 'rgb(var(--border-strong))',
        strokeDasharray: e.kind === 'control' ? '4 3' : undefined,
        opacity: targetRevealed ? 1 : 0.18,
      },
      labelStyle: { fill: 'rgb(var(--fg-muted))', fontSize: 10, fontFamily: 'monospace' },
      labelBgStyle: { fill: 'rgb(var(--bg))' },
    };
  });

  return { nodes, edges };
}
