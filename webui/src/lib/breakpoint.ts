/* Tailwind's breakpoints, readable from React.
 *
 * Almost everything responsive here is CSS, which is where it belongs. This is
 * for the handful of cases CSS cannot reach: text that must differ (a
 * placeholder is an attribute, not a node) and a route that should not exist at
 * a given width. Reach for a `sm:` class first. */
import { useEffect, useState } from "react";

/** Matches Tailwind's own `sm` and `md` (tailwind.config.ts defaults). */
export const BREAKPOINTS = { sm: 640, md: 768 } as const;

/** True while the viewport is at least `min` px wide. SSR-safe and live. */
export function useMinWidth(min: number): boolean {
  const [matches, setMatches] = useState(() =>
    typeof window === "undefined" ? true : window.matchMedia(`(min-width: ${min}px)`).matches,
  );
  useEffect(() => {
    const mq = window.matchMedia(`(min-width: ${min}px)`);
    const on = () => setMatches(mq.matches);
    on();
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [min]);
  return matches;
}

/** Below Tailwind's `sm` — a phone held upright. */
export function useIsPhone(): boolean {
  return !useMinWidth(BREAKPOINTS.sm);
}

/** At or above Tailwind's `md` — where the sidebar replaces the bottom bar. */
export function useIsDesktop(): boolean {
  return useMinWidth(BREAKPOINTS.md);
}
