import { Card } from "./Card";
import { CopyBlock } from "./CopyBlock";
import { Tag } from "./Tag";
import { KIND_META, fmtMs } from "../lib/nodeMeta";
import type { Resources, TraceNode } from "../lib/types";

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-3 py-0.5">
      <span className="shrink-0 text-fg-subtle">{k}</span>
      <span className="break-all text-right font-mono text-fg">{v}</span>
    </div>
  );
}

function ResourceBar({ label, pct, tone }: { label: string; pct: number | null; tone?: string }) {
  return (
    <div className="mb-1.5">
      <div className="mb-0.5 flex justify-between text-[10.5px]">
        <span className="text-fg-subtle">{label}</span>
        <span className="font-mono text-fg-muted">{pct == null ? "n/a" : `${Math.round(pct)}%`}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded bg-bg">
        <div
          className="h-full rounded"
          style={{ width: `${pct ?? 0}%`, background: tone ?? "rgb(var(--success))" }}
        />
      </div>
    </div>
  );
}

function ResourcePanel({ r }: { r: Resources }) {
  const ramUsedPct = ((r.ram_total_gb - r.ram_free_gb) / r.ram_total_gb) * 100;
  return (
    <Card title="Resources">
      <ResourceBar label="CPU" pct={r.cpu_percent} tone="rgb(var(--node-perception))" />
      <ResourceBar label="GPU" pct={r.gpu_percent} tone="rgb(var(--node-llm))" />
      <ResourceBar
        label={`RAM used (${(r.ram_total_gb - r.ram_free_gb).toFixed(1)}/${r.ram_total_gb} GB)`}
        pct={ramUsedPct}
        tone="rgb(var(--node-action))"
      />
      <div className="mt-1.5 flex items-center gap-2 text-[10.5px]">
        {r.thermal_throttled ? <Tag kind="warn">thermal-throttled</Tag> : <Tag kind="ok">nominal</Tag>}
        {r.gpu_percent == null && <span className="text-fg-subtle">GPU% not reported on this host</span>}
      </div>
    </Card>
  );
}

export function NodeDetail({
  node,
  verbose,
  onClose,
}: {
  node: TraceNode | null;
  verbose: boolean;
  onClose: () => void;
}) {
  if (!node) {
    return (
      <div className="rounded-lg border border-dashed border-border bg-surface p-6 text-center text-xs text-fg-subtle">
        Click any node to inspect its inputs, outputs, timing, tokens and resource usage.
      </div>
    );
  }
  const meta = KIND_META[node.kind];
  const isTool = node.kind === "tool";

  return (
    <div className="space-y-3">
      <Card>
        <div className="flex items-start justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span style={{ color: meta.color }}>{meta.glyph}</span>
              <span className="text-sm font-semibold text-fg">{node.label}</span>
            </div>
            <div className="mt-1 break-all font-mono text-[11px] text-fg-subtle">
              {node.component_path}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close detail panel"
            className="ml-2 shrink-0 text-fg-subtle hover:text-fg-muted"
          >
            ✕
          </button>
        </div>
        <div className="mt-3 text-xs">
          <Row k="kind" v={node.kind} />
          <Row
            k="status"
            v={
              <Tag kind={node.status === "ok" ? "ok" : node.status === "error" ? "bad" : "info"}>
                {node.status}
              </Tag>
            }
          />
          <Row k="t+offset" v={fmtMs(node.t_offset_ms)} />
          <Row k="duration" v={fmtMs(node.duration_ms)} />
        </div>
      </Card>

      {verbose && (
        <Card title="Trace (module · file · method)" accent="amber">
          <div className="text-xs">
            {node.module && <Row k="module" v={node.module} />}
            <Row k="file" v={node.component_path} />
            {node.method && <Row k="method" v={node.method} />}
          </div>
        </Card>
      )}

      {node.kind === "llm" && (
        <Card title="LLM call">
          <div className="text-xs">
            <Row k="model" v={node.model} />
            <Row k="provider" v={node.provider} />
            {node.tier && <Row k="tier" v={node.tier} />}
            {node.agent && <Row k="called by" v={node.agent} />}
            {node.tokens && (
              <>
                <Row k="prompt tokens" v={node.tokens.prompt} />
                <Row k="completion tokens" v={node.tokens.completion} />
                <Row k="total tokens" v={<span className="text-node-llm">{node.tokens.total}</span>} />
              </>
            )}
          </div>
        </Card>
      )}

      {node.kind === "governance" && node.governance && (
        <Card
          title="Governance decision"
          accent={node.governance.decision !== "allow" ? "amber" : undefined}
        >
          <div className="text-xs">
            <Row k="hook" v={node.governance.hook} />
            <Row
              k="decision"
              v={
                <Tag
                  kind={
                    node.governance.decision === "allow"
                      ? "ok"
                      : node.governance.decision === "deny"
                        ? "bad"
                        : "warn"
                  }
                >
                  {node.governance.decision}
                </Tag>
              }
            />
          </div>
          {node.governance.reason && (
            <div className="mt-2 text-[11px] text-fg-muted">{node.governance.reason}</div>
          )}
        </Card>
      )}

      {/* Full, copyable input/output for EVERY stage. */}
      <Card title={node.kind === "llm" ? "Prompt / completion" : isTool ? "Tool I/O" : "Input / output"}>
        {isTool && node.tool && (
          <div className="mb-1 text-xs">
            <Row
              k="exit code"
              v={
                <span className={node.tool.exit_code === 0 ? "text-success" : "text-danger"}>
                  {node.tool.exit_code}
                </span>
              }
            />
          </div>
        )}
        {node.input != null && (
          <CopyBlock label={node.kind === "llm" ? "prompt (input)" : "input"} text={node.input} />
        )}
        {node.output != null && (
          <CopyBlock
            label={node.kind === "llm" ? "completion (output)" : "output"}
            text={node.output}
            tone={node.status === "error" || node.status === "skipped" ? "text-danger" : "text-success"}
          />
        )}
        {isTool && node.tool?.stderr && (
          <CopyBlock label="stderr" text={node.tool.stderr} tone="text-danger" />
        )}
        {node.input == null && node.output == null && (
          <div className="text-[11px] text-fg-subtle">No input/output captured for this stage.</div>
        )}
      </Card>

      {node.resources && <ResourcePanel r={node.resources} />}
    </div>
  );
}
