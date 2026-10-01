import type { ReactNode } from "react";

export function Card({
  title,
  accent,
  action,
  children,
}: {
  title?: string;
  accent?: "amber";
  action?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div
      className={`rounded-lg border bg-surface p-4 ${
        accent === "amber" ? "border-warning/40" : "border-border"
      }`}
    >
      {(title || action) && (
        <div className="mb-3 flex items-center justify-between gap-2">
          {title && (
            <h3
              className={`text-xs font-semibold uppercase tracking-wide ${
                accent === "amber" ? "text-warning" : "text-fg-muted"
              }`}
            >
              {title}
            </h3>
          )}
          {action}
        </div>
      )}
      {children}
    </div>
  );
}

export function Kpi({
  value,
  label,
  tone,
}: {
  value: string;
  label: string;
  tone?: "ok" | "bad";
}) {
  return (
    <div className="rounded-lg border border-border border-l-2 border-l-primary bg-surface p-4">
      <div
        className={`font-mono text-2xl font-semibold ${
          tone === "bad" ? "text-danger" : tone === "ok" ? "text-success" : "text-primary"
        }`}
      >
        {value}
      </div>
      <div className="mt-1 text-xs font-medium uppercase tracking-wide text-fg-subtle">{label}</div>
    </div>
  );
}
