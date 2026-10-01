import { useState } from "react";
import { Card } from "./Card";
import { CopyBlock } from "./CopyBlock";
import { fmtMs } from "../lib/nodeMeta";
import type { ReasoningStep, StepType } from "../lib/types";

// Step accent colors come from design tokens (rgb(var(--node-*)) — same values
// as the graph node palette).
const STEP_META: Record<StepType, { color: string; glyph: string }> = {
  request: { color: "rgb(var(--node-perception))", glyph: "✎" },
  intent: { color: "rgb(var(--node-cognition))", glyph: "⌖" },
  memory: { color: "rgb(var(--node-action))", glyph: "❖" },
  plan: { color: "rgb(var(--node-cognition))", glyph: "☰" },
  llm: { color: "rgb(var(--node-llm))", glyph: "✦" },
  tool: { color: "rgb(var(--node-action))", glyph: "⚒" },
  curator: { color: "rgb(var(--node-perception))", glyph: "✓" },
  handler: { color: "rgb(var(--node-action))", glyph: "◇" },
  guard: { color: "rgb(var(--node-governance))", glyph: "⊡" },
  stop: { color: "rgb(var(--node-governance))", glyph: "■" },
  response: { color: "rgb(var(--node-perception))", glyph: "➜" },
  error: { color: "rgb(var(--danger))", glyph: "!" },
};

function StepRow({ step }: { step: ReasoningStep }) {
  const [open, setOpen] = useState(false);
  const meta = STEP_META[step.type] ?? STEP_META.llm;
  const err = step.status === "error";
  const border = err ? "rgb(var(--danger))" : meta.color;
  const fieldEntries = Object.entries(step.fields ?? {}).filter(
    ([, v]) => v !== null && v !== undefined && v !== "",
  );

  return (
    <div className="relative pl-8">
      <span
        className="absolute left-[10px] top-3 h-3 w-3 -translate-x-1/2 rounded-full ring-2 ring-surface"
        style={{ background: border }}
      />
      <div
        className="rounded-lg border bg-surface p-2.5"
        style={{ borderColor: err ? "rgb(var(--danger) / 0.45)" : "rgb(var(--border))" }}
      >
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="flex w-full items-start gap-2 text-left"
          disabled={!step.detail}
        >
          <span style={{ color: border }} className="mt-0.5 text-sm leading-none">
            {meta.glyph}
          </span>
          <span className="min-w-0 flex-1">
            <span className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-[9.5px] uppercase tracking-wide text-fg-subtle">
                {step.type}
              </span>
              {step.iteration != null && (
                <span className="rounded bg-bg px-1.5 py-0.5 font-mono text-[9.5px] text-fg-muted">
                  iter {step.iteration}
                </span>
              )}
              <span className={`text-[13px] ${err ? "text-danger" : "text-fg"}`}>{step.title}</span>
            </span>
            <span className="mt-1 flex flex-wrap items-center gap-1.5 font-mono text-[10px] text-fg-subtle">
              <span>t+{fmtMs(step.t_offset_ms)}</span>
              {step.duration_ms ? <span>· {fmtMs(step.duration_ms)}</span> : null}
              {fieldEntries.map(([k, v]) => (
                <span key={k} className="rounded bg-bg px-1.5 py-0.5 text-fg-muted">
                  {k}: {String(v)}
                </span>
              ))}
            </span>
          </span>
          {step.detail ? (
            <span className="shrink-0 text-[10px] text-fg-subtle">{open ? "▾" : "▸"}</span>
          ) : null}
        </button>
        {open && step.detail ? (
          <CopyBlock label="detail" text={step.detail} tone={err ? "text-danger" : undefined} />
        ) : null}
      </div>
    </div>
  );
}

export function Reasoning({ steps }: { steps: ReasoningStep[] | undefined }) {
  if (!steps || steps.length === 0) {
    return (
      <div className="rounded-lg border border-dashed border-border bg-surface p-6 text-center text-xs text-fg-subtle">
        No reasoning steps for this trace.
      </div>
    );
  }
  return (
    <Card title={`Reasoning — ${steps.length} steps (tap a step to expand)`}>
      <div className="relative space-y-2">
        <span className="pointer-events-none absolute bottom-3 left-[10px] top-3 w-px bg-border" />
        {steps.map((s) => (
          <StepRow key={s.id} step={s} />
        ))}
      </div>
    </Card>
  );
}
