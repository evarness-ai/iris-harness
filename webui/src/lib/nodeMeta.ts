import type { NodeKind } from "./types";

// Per-kind visual metadata: accent color + glyph. Colors are design tokens
// (rgb(var(--node-*)) — same values as before, now themeable via tokens.css)
// so graph nodes stay consistent with the rest of the console and light/dark.
export const KIND_META: Record<NodeKind, { color: string; glyph: string; title: string }> = {
  gateway: { color: "rgb(var(--node-perception))", glyph: "⇆", title: "Channel Gateway" },
  api: { color: "rgb(var(--node-perception))", glyph: "◷", title: "IRIS API" },
  runtime: { color: "rgb(var(--node-runtime))", glyph: "◆", title: "Runtime" },
  intent_router: { color: "rgb(var(--node-cognition))", glyph: "⌖", title: "Intent Router" },
  memory: { color: "rgb(var(--node-action))", glyph: "❖", title: "Memory Retriever" },
  task_planner: { color: "rgb(var(--node-cognition))", glyph: "☰", title: "Task Planner" },
  agent_executor: { color: "rgb(var(--node-runtime))", glyph: "⚙", title: "Agent Executor" },
  agent: { color: "rgb(var(--node-action))", glyph: "◈", title: "Agent" },
  llm: { color: "rgb(var(--node-llm))", glyph: "✦", title: "LLM Call" },
  tool: { color: "rgb(var(--node-action))", glyph: "⚒", title: "Tool" },
  governance: { color: "rgb(var(--node-governance))", glyph: "⛨", title: "Governance" },
  response_curator: { color: "rgb(var(--node-perception))", glyph: "✓", title: "Response Curator" },
  handler: { color: "rgb(var(--node-action))", glyph: "◇", title: "Deterministic Handler" },
  guard: { color: "rgb(var(--node-governance))", glyph: "⊡", title: "Response Check" },
};

export function fmtMs(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function fmtTokens(n: number): string {
  if (n < 1000) return `${n}`;
  return `${(n / 1000).toFixed(1)}k`;
}
