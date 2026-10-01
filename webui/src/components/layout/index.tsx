/* Reusable layout primitives. Screens compose these — never bespoke flex/grid
 * markup per screen — so density + rhythm stay consistent and new screens are
 * composition, not copy-paste. Token classes only; no inline CSS, no raw hex. */
import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

const GAP: Record<string, string> = {
  "1": "gap-1",
  "2": "gap-2",
  "3": "gap-3",
  "4": "gap-4",
  "6": "gap-6",
  "8": "gap-8",
};

type Gap = keyof typeof GAP;

// Literal class maps so Tailwind's content scanner sees them (template-literal
// class names like `items-${x}` get purged).
const ALIGN = {
  start: "items-start",
  center: "items-center",
  end: "items-end",
  stretch: "items-stretch",
} as const;
const JUSTIFY = {
  start: "justify-start",
  center: "justify-center",
  end: "justify-end",
  between: "justify-between",
} as const;

/** Page shell: optional title/description header + content column. */
export function Page({
  title,
  description,
  actions,
  children,
  className,
}: {
  title?: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("mx-auto w-full max-w-7xl", className)}>
      {(title || actions) && (
        <div className="mb-5 flex items-start justify-between gap-4">
          <div className="min-w-0">
            {title && <h2 className="truncate text-xl font-semibold text-fg">{title}</h2>}
            {description && <p className="mt-1 text-sm text-fg-muted">{description}</p>}
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </div>
      )}
      {children}
    </div>
  );
}

/** A titled content section within a page. */
export function Section({
  title,
  actions,
  children,
  className,
}: {
  title?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("mb-6", className)}>
      {(title || actions) && (
        <div className="mb-3 flex flex-wrap items-center justify-between gap-x-3 gap-y-2">
          {title && (
            <h3 className="text-xs font-medium uppercase tracking-wider text-fg-subtle">
              {title}
            </h3>
          )}
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

/** Flex stack — vertical by default. */
export function Stack({
  direction = "col",
  gap = "3",
  align,
  justify,
  wrap,
  children,
  className,
}: {
  direction?: "row" | "col";
  gap?: Gap;
  align?: "start" | "center" | "end" | "stretch";
  justify?: "start" | "center" | "end" | "between";
  wrap?: boolean;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex",
        direction === "row" ? "flex-row" : "flex-col",
        GAP[gap],
        align && ALIGN[align],
        justify && JUSTIFY[justify],
        wrap && "flex-wrap",
        className,
      )}
    >
      {children}
    </div>
  );
}

/** Responsive grid — 1 col on mobile up to `cols` on large screens. */
export function Grid({
  cols = 3,
  gap = "4",
  children,
  className,
}: {
  cols?: 2 | 3 | 4;
  gap?: Gap;
  children: ReactNode;
  className?: string;
}) {
  // Every count starts at one column. `4` used to start at two, which on a
  // 390px phone left ~170px of card: Control's KPI labels clipped to "TOTA",
  // "ACTIV" and "PROPOSA", and its buttons wrapped one per line.
  const colsClass: Record<number, string> = {
    2: "grid-cols-1 sm:grid-cols-2",
    3: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3",
    4: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-4",
  };
  return <div className={cn("grid", colsClass[cols], GAP[gap], className)}>{children}</div>;
}

/** Horizontal action/filter bar. */
export function Toolbar({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn("flex flex-wrap items-center gap-2", className)}>{children}</div>
  );
}
