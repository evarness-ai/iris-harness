/* The console's navigation, as the API gives it (OSS plan R17).
 *
 * GET /api/v1/webui/nav is the core's screens (config/webui/nav.yaml) plus those of
 * every MOUNTED plugin (its manifest's `webui.screens`), grouped and ordered by the
 * harness (runtime/plugin_host/nav.py). The sidebar, the phone's bar and More all
 * render from it; nothing here decides which screen exists. A plugin that is not
 * mounted is absent from `groups`/`off_nav` and, when it is installed, listed under
 * `unavailable` with why, so a direct link to its screen can say so. */
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "./http";

export const NAV_PATH = "/api/v1/webui/nav";

export interface NavScreen {
  id: string;
  label: string;
  /** One path segment; the screen owns its sub-routes (/agents/:name). */
  route: string;
  title: string;
  subtitle: string;
  /** A lucide icon name; routes.tsx maps the ones the bundle carries. */
  icon: string;
  group: string;
  order: number;
  nav: boolean;
  /** The plugin that owns the screen; null for the core's own. */
  plugin: string | null;
  /** The owning plugin's status (loaded | degraded); null for the core. */
  status: string | null;
  mobile_tab?: { label: string; order: number };
}

export interface NavGroup {
  id: string;
  /** Null for the pinned top group (Chat), which renders without a header. */
  label: string | null;
  items: NavScreen[];
}

export interface UnavailableScreen {
  id: string;
  label: string;
  route: string;
  plugin: string;
  reason: string;
}

export interface WebNav {
  groups: NavGroup[];
  off_nav: NavScreen[];
  mobile_tabs: { route: string; label: string }[];
  unavailable: UnavailableScreen[];
  problems: string[];
}

function isWebNav(value: unknown): value is WebNav {
  const v = value as Partial<WebNav> | null;
  return !!v && Array.isArray(v.groups) && Array.isArray(v.off_nav) && Array.isArray(v.mobile_tabs);
}

export async function getNav(): Promise<WebNav> {
  const r = await apiFetch(NAV_PATH, { headers: { accept: "application/json" } });
  if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`.trim());
  const body: unknown = await r.json();
  if (!isWebNav(body)) throw new Error("navigation response has an unexpected shape");
  return body;
}

/** Profiles change on a restart, not mid-session: fetch once, keep it. */
export function useNav() {
  return useQuery({ queryKey: ["webui-nav"], queryFn: getNav, staleTime: 5 * 60_000 });
}

/** `/agents/plugins/x` -> `/agents`: the segment a screen is declared by. */
export function routeOf(pathname: string): string {
  const first = pathname.split("/").filter(Boolean)[0];
  return first ? `/${first}` : "/";
}

/** Every screen this install renders: nav entries and off-nav pages. */
export function allScreens(nav: WebNav): NavScreen[] {
  return [...nav.groups.flatMap((g) => g.items), ...nav.off_nav];
}

/** The screen and group a pathname belongs to, when the nav has it. */
export function findScreen(
  nav: WebNav,
  pathname: string,
): { screen: NavScreen; group: NavGroup | undefined } | undefined {
  const route = routeOf(pathname);
  const screen = allScreens(nav).find((s) => s.route === route);
  if (!screen) return undefined;
  return { screen, group: nav.groups.find((g) => g.id === screen.group) };
}

/** Why a pathname has no screen here: its plugin is installed but not mounted
 * (named, with the reason), or nothing in this install declares it at all. */
export function unavailableFor(nav: WebNav, pathname: string): UnavailableScreen | undefined {
  const route = routeOf(pathname);
  return nav.unavailable.find((u) => u.route === route);
}
