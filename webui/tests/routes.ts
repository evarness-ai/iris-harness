/* The route list under test: the nav the API returns, as the console receives it.
 *
 * The console draws its sidebar, phone bar and More page from GET /api/v1/webui/nav
 * (OSS plan R17). The two JSON files beside this one are that endpoint's output for
 * real profiles, written by the Python suite
 * (tests/unit/iris_harness/runtime/test_plugin_host/test_nav.py), which fails when
 * config/webui/nav.yaml or a plugin manifest changes without regenerating them. The
 * fixtures serve them to the app, and `nav parity` in viewport.spec.ts counts the
 * links the running app renders, so the three can never disagree quietly.
 *
 * `personal-assistant` mounts every plugin, so the viewport smoke covers every
 * screen; `core-email` is the release-1 shape (the default profile + the email
 * slice), used to prove the private domains' screens gate off. */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));

export interface NavRoute {
  path: string;
  label: string;
}

export interface NavFixture {
  groups: { id: string; label: string | null; items: { route: string; label: string }[] }[];
  off_nav: { route: string; label: string }[];
  mobile_tabs: { route: string; label: string }[];
  unavailable: { route: string; label: string; plugin: string; reason: string }[];
  problems: string[];
}

export function navFixture(profile: "personal-assistant" | "core-email"): NavFixture {
  return JSON.parse(readFileSync(path.join(here, `nav.${profile}.json`), "utf8")) as NavFixture;
}

/** The nav entries (sidebar + More) of a profile, in order. */
export function navRoutes(profile: "personal-assistant" | "core-email" = "personal-assistant"): NavRoute[] {
  const out = navFixture(profile).groups.flatMap((g) =>
    g.items.map((i) => ({ path: i.route, label: i.label })),
  );
  if (out.length === 0) throw new Error(`no routes in nav.${profile}.json`);
  return out;
}
