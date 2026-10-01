import type { ReactNode } from "react";
import { Tag } from "../Tag";
import type { RoutineStatus } from "@/lib/control";

/** Bordered inline notice — neutral by default, danger tone for failures. */
export function Notice({
  tone = "muted",
  children,
}: {
  tone?: "muted" | "danger";
  children: ReactNode;
}) {
  return (
    <div
      className={`rounded-lg border border-dashed p-6 text-center text-xs ${
        tone === "danger" ? "border-danger/40 text-danger" : "border-border text-fg-subtle"
      }`}
    >
      {children}
    </div>
  );
}

/** Standard loading / error / empty handling for a read-only control query. */
export function QueryState({
  loading,
  error,
  empty,
  emptyText,
  children,
}: {
  loading: boolean;
  error: boolean;
  empty: boolean;
  emptyText: string;
  children: ReactNode;
}) {
  if (loading) return <Notice>Loading…</Notice>;
  if (error) return <Notice tone="danger">API unavailable — is the IRIS API running? Start it with <code>iris serve</code> (port 8003 by default).</Notice>;
  if (empty) return <Notice>{emptyText}</Notice>;
  return <>{children}</>;
}

const ROUTINE_TONE: Record<RoutineStatus, "ok" | "info" | "warn" | "bad"> = {
  scheduled: "ok",
  approved: "ok",
  draft: "info",
  clarify: "info",
  template: "info",
  paused: "warn",
  retired: "bad",
};

export function RoutineStatusTag({ status }: { status: RoutineStatus }) {
  return <Tag kind={ROUTINE_TONE[status] ?? "info"}>{status}</Tag>;
}

/** Routines awaiting human review (the reflection loop proposes into these). */
export const PENDING_STATUSES: ReadonlySet<RoutineStatus> = new Set([
  "draft",
  "clarify",
  "template",
]);

/** Data classification badge — secret/personal stay local; surfaced honestly. */
export function ClassificationTag({ value }: { value: string }) {
  const v = value.toLowerCase();
  const kind = v === "secret" ? "bad" : v === "personal" ? "warn" : "info";
  return <Tag kind={kind}>{value}</Tag>;
}

/** Health-state badge — green/yellow/red colored; grey ("not configured") is a
 * neutral pill so an unconnected integration never reads as a problem. */
const HEALTH_TONE: Record<string, "ok" | "warn" | "bad"> = {
  green: "ok",
  yellow: "warn",
  red: "bad",
};

export function HealthStateTag({ state }: { state: string }) {
  if (state === "grey") {
    return (
      <span className="shrink-0 rounded-full border border-border px-2 py-0.5 text-[10.5px] text-fg-subtle">
        not set
      </span>
    );
  }
  return <Tag kind={HEALTH_TONE[state] ?? "info"}>{state}</Tag>;
}

/** Governance decision badge — allow/deny/transform/require_approval. */
export function DecisionTag({ decision }: { decision: string }) {
  const d = decision.toLowerCase();
  const kind = d === "allow" ? "ok" : d === "deny" ? "bad" : "warn";
  return <Tag kind={kind}>{decision}</Tag>;
}

/** Where a governed call's tier runs (server-derived: tier_3 is cloud, any other local). */
export function LocalityTag({ locality, tier }: { locality?: string | null; tier: string | null }) {
  if (!tier) return null;
  if (!locality) return <Tag kind="res">{tier}</Tag>;
  return (
    <Tag kind={locality === "cloud" ? "warn" : "res"}>
      {locality} · {tier}
    </Tag>
  );
}

/** On/off flag grid — reused by Governance (posture) and Settings (runtime). */
export function FlagGrid({ flags }: { flags: { key: string; label: string; on: boolean }[] }) {
  return (
    <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
      {flags.map((f) => (
        <div
          key={f.key}
          className="flex items-center justify-between gap-2 rounded-lg border border-border bg-surface px-3 py-2"
        >
          <div className="min-w-0">
            <div className="truncate text-xs font-medium text-fg">{f.label}</div>
            <div className="truncate font-mono text-[10px] text-fg-subtle">{f.key}</div>
          </div>
          {f.on ? (
            <Tag kind="ok">on</Tag>
          ) : (
            <span className="shrink-0 rounded-full border border-border px-2 py-0.5 text-[10.5px] text-fg-subtle">
              off
            </span>
          )}
        </div>
      ))}
    </div>
  );
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

export function fmtDateTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}
