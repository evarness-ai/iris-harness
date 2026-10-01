import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Background,
  Controls,
  ReactFlow,
  useEdgesState,
  useNodesState,
  type Edge,
  type Node,
  type NodeMouseHandler,
  type ReactFlowInstance,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { Link, useNavigate, useParams } from 'react-router-dom';
import { Card, Kpi } from '../components/Card';
import { Tag } from '../components/Tag';
import { IRISNode } from '../components/IRISNode';
import { GovernanceTimeline } from '../components/GovernanceTimeline';
import { NodeDetail } from '../components/NodeDetail';
import { Reasoning } from '../components/Reasoning';
import { Replay } from '../components/Replay';
import { Waterfall } from '../components/Waterfall';
import { buildGraph, type Direction, type IRISNodeData } from '../lib/layout';
import { fmtMs } from '../lib/nodeMeta';
import { getTrace, listTraces, type Source, type TraceSummary } from '../lib/client';
import type { Trace } from '../lib/types';

const nodeTypes = { iris: IRISNode };

function Toggle({
  options,
  value,
  onChange,
}: {
  options: { label: string; value: string }[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="inline-flex overflow-hidden rounded-lg border border-border">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={`min-h-[44px] px-2.5 py-1 text-xs font-medium transition-colors sm:min-h-0 ${
            o.value === value
              ? 'bg-primary/15 text-primary'
              : 'bg-bg text-fg-muted hover:bg-surface hover:text-fg'
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function CallTraceScreen() {
  // The selected trace is the URL: /calltrace/:traceId (deep-linkable).
  const { traceId } = useParams<{ traceId?: string }>();
  const navigate = useNavigate();
  const tid = traceId ?? '';
  const [summaries, setSummaries] = useState<TraceSummary[]>([]);
  const [source, setSource] = useState<Source>('mock');
  const [listed, setListed] = useState(false);
  const [trace, setTrace] = useState<Trace | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  // View controls
  const [view, setView] = useState<'graph' | 'reasoning'>('graph');
  const [dir, setDir] = useState<Direction>('TB');
  const [verbose, setVerbose] = useState(false);

  // Replay state
  const [replayMode, setReplayMode] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [step, setStep] = useState(0);
  const [speed, setSpeed] = useState(1.5);

  const [rfNodes, setRfNodes, onNodesChange] = useNodesState<Node<IRISNodeData>>([]);
  const [rfEdges, setRfEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const rfi = useRef<ReactFlowInstance<Node<IRISNodeData>, Edge> | null>(null);

  // Load the list of recent traces (live API, or mock fallback).
  useEffect(() => {
    listTraces().then(({ source: src, traces }) => {
      setSource(src);
      setSummaries(traces);
      setListed(true);
      // No trace in the URL → redirect to the newest (deep-linkable default).
      if (!traceId && traces[0]) navigate(`/calltrace/${traces[0].trace_id}`, { replace: true });
    }).catch(() => {
      /* 401 — lib/http.ts is already routing to /pair; never fall back to mock here */
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch the selected trace's full graph.
  useEffect(() => {
    if (!tid) return;
    let alive = true;
    setTrace(null);
    getTrace(tid).then((t) => {
      if (alive) setTrace(t ?? null);
    }).catch(() => {
      /* 401 — routed to /pair */
    });
    return () => {
      alive = false;
    };
  }, [tid]);

  const orderedIds = useMemo(
    () => (trace ? [...trace.nodes].sort((a, b) => a.t_offset_ms - b.t_offset_ms).map((n) => n.id) : []),
    [trace],
  );
  const total = orderedIds.length;
  const edgeKind = useMemo(() => new Map((trace?.edges ?? []).map((e) => [e.id, e.kind])), [trace]);

  const revealedIds = useMemo(
    () => (replayMode ? new Set(orderedIds.slice(0, step)) : new Set(orderedIds)),
    [replayMode, step, orderedIds],
  );
  const activeId = replayMode ? (orderedIds[step - 1] ?? null) : null;

  // Reset replay + selection when switching traces.
  useEffect(() => {
    setReplayMode(false);
    setPlaying(false);
    setStep(total);
    setSelectedNode(null);
  }, [tid, total]);

  // Structural (re)layout — runs only when the graph shape/size changes.
  useEffect(() => {
    if (!trace) {
      setRfNodes([]);
      setRfEdges([]);
      return;
    }
    const { nodes, edges } = buildGraph(trace, {
      active: activeId,
      revealedIds,
      replaying: replayMode,
      dir,
      verbose,
    });
    setRfNodes(nodes);
    setRfEdges(edges);
    const id = window.setTimeout(() => rfi.current?.fitView({ padding: 0.15, duration: 300 }), 30);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace, dir, verbose]);

  // Overlay replay + selection without disturbing dragged positions.
  useEffect(() => {
    setRfNodes((ns) =>
      ns.map((n) => ({
        ...n,
        selected: n.id === selectedNode,
        data: {
          ...n.data,
          active: activeId === n.id,
          revealed: !replayMode || revealedIds.has(n.id),
          replaying: replayMode,
        },
      })),
    );
    setRfEdges((es) =>
      es.map((e) => {
        const isData = edgeKind.get(e.id) === 'data';
        const targetRevealed = !replayMode || revealedIds.has(e.target);
        return {
          ...e,
          animated: isData && (!replayMode || activeId === e.target),
          style: { ...(e.style ?? {}), opacity: targetRevealed ? 1 : 0.18 },
        };
      }),
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, replayMode, activeId, selectedNode, revealedIds, edgeKind]);

  // Replay timer.
  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setStep((s) => {
        if (s >= total) {
          setPlaying(false);
          return s;
        }
        return s + 1;
      });
    }, 750 / speed);
    return () => window.clearInterval(id);
  }, [playing, speed, total]);

  // While playing, follow the active node in the detail panel.
  useEffect(() => {
    if (playing && activeId) setSelectedNode(activeId);
  }, [playing, activeId]);

  const onNodeClick: NodeMouseHandler = (_e, n) => setSelectedNode(n.id);

  const selected = trace?.nodes.find((n) => n.id === selectedNode) ?? null;
  const llmCount = trace?.nodes.filter((n) => n.kind === 'llm').length ?? 0;

  const onPlayPause = () => {
    if (!replayMode || step >= total) {
      setReplayMode(true);
      setStep(0);
      setPlaying(true);
    } else {
      setPlaying((p) => !p);
    }
  };
  const onStepBtn = (d: 1 | -1) => {
    setReplayMode(true);
    setPlaying(false);
    setStep((s) => Math.max(0, Math.min(total, s + d)));
  };
  const onReset = () => {
    setReplayMode(true);
    setPlaying(false);
    setStep(0);
  };

  // A fresh install: the API is up and has traced no turn yet. Without this the screen
  // would wait on "Loading trace…" for a trace that does not exist.
  if (listed && source === 'live' && summaries.length === 0 && !tid) {
    return (
      <div data-testid="calltrace-empty">
        <Card title="No turns traced yet">
          <p className="text-sm text-fg-muted">
            Every chat turn is traced here: each model call, tool call and governance
            decision, in order, with where it ran. Ask IRIS something in Chat, then come back.
          </p>
          <Link
            to="/chat"
            className="mt-4 inline-flex min-h-[44px] items-center rounded-lg border border-border px-3 text-sm text-fg hover:bg-surface-raised"
          >
            Open Chat
          </Link>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Trace selector + view controls */}
      <div className="flex flex-wrap items-center gap-3">
        <select
          value={tid}
          onChange={(e) => navigate(`/calltrace/${e.target.value}`)}
          className="max-w-[520px] rounded-lg border border-border bg-surface px-3 py-2 text-sm text-fg"
        >
          {summaries.map((t) => (
            <option key={t.trace_id} value={t.trace_id}>
              “{t.request || '(empty)'}” · {fmtMs(t.total_duration_ms)}
            </option>
          ))}
        </select>
        <Tag kind={source === 'live' ? 'ok' : 'warn'}>{source === 'live' ? 'live logs' : 'mock data'}</Tag>
        {trace && <span className="font-mono text-[11px] text-fg-subtle">session {trace.session_id}</span>}
        <div className="ml-auto flex items-center gap-2">
          <span className="text-[11px] text-fg-subtle">layout</span>
          <Toggle
            options={[
              { label: '↓ Vertical', value: 'TB' },
              { label: '→ Horizontal', value: 'LR' },
            ]}
            value={dir}
            onChange={(v) => setDir(v as Direction)}
          />
          <button
            type="button"
            onClick={() => setVerbose((v) => !v)}
            className={`min-h-[44px] rounded-lg border px-2.5 py-1 text-xs font-medium sm:min-h-0 ${
              verbose ? 'border-warning/50 bg-warning/15 text-warning' : 'border-border bg-bg text-fg-muted hover:bg-surface'
            }`}
          >
            {verbose ? '● Verbose' : '○ Verbose'}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Kpi value={trace ? fmtMs(trace.total_duration_ms) : '—'} label="total latency" />
        <Kpi value={trace ? `${trace.total_tokens}` : '—'} label="total tokens" />
        <Kpi value={`${llmCount}`} label="LLM calls" />
        <Kpi value={`${trace?.nodes.length ?? 0}`} label="components" />
      </div>

      <Toggle
        options={[
          { label: 'Graph', value: 'graph' },
          { label: 'Reasoning', value: 'reasoning' },
        ]}
        value={view}
        onChange={(v) => setView(v as 'graph' | 'reasoning')}
      />

      {view === 'reasoning' ? (
        <Reasoning steps={trace?.steps} />
      ) : (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_360px]">
          <div className="space-y-4">
            <Card>
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <Replay
                playing={playing}
                step={step}
                total={total}
                speed={speed}
                onPlayPause={onPlayPause}
                onStep={onStepBtn}
                onReset={onReset}
                onSpeed={setSpeed}
              />
              <span className="font-mono text-[10px] text-fg-subtle">drag nodes · scroll to zoom</span>
            </div>
            <div className="relative h-[440px] rounded-lg border border-border bg-bg md:h-[620px]">
              {!trace && (
                <div className="absolute inset-0 z-10 flex items-center justify-center text-sm text-fg-subtle">
                  Loading trace…
                </div>
              )}
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
                nodesDraggable
                nodesConnectable={false}
                elementsSelectable={false}
                minZoom={0.2}
              >
                <Background color="rgb(var(--surface-raised))" gap={18} />
                <Controls showInteractive={false} />
              </ReactFlow>
            </div>
          </Card>

          {trace && (
            <Card title="Waterfall — per-step duration">
              <Waterfall trace={trace} selectedId={selectedNode} onSelect={setSelectedNode} />
            </Card>
          )}
        </div>

          <NodeDetail node={selected} verbose={verbose} onClose={() => setSelectedNode(null)} />
        </div>
      )}

      {trace?.governance && (
        <Card title={`Governance — ${trace.governance.length} hook decisions, in order`}>
          <GovernanceTimeline events={trace.governance} />
        </Card>
      )}
    </div>
  );
}
