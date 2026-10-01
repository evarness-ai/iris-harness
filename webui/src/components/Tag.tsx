import type { ReactNode } from "react";

// Token-driven status/kind badge. Opacity modifiers (/15) work because the
// tokens are channel-form (tokens.css).
const styles: Record<string, string> = {
  res: "bg-node-runtime/15 text-node-runtime",
  opp: "bg-node-cognition/15 text-node-cognition",
  ok: "bg-success/15 text-success",
  bad: "bg-danger/15 text-danger",
  warn: "bg-warning/15 text-warning",
  info: "bg-info/15 text-info",
};

export function Tag({ kind, children }: { kind: keyof typeof styles; children: ReactNode }) {
  return (
    <span
      className={`rounded-full px-2 py-0.5 font-mono text-[10.5px] font-semibold ${styles[kind]}`}
    >
      {children}
    </span>
  );
}
