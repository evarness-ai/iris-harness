/* Reconnecting an external account from the console (Settings > Connections).
 *
 * The console names no provider. A credential row in GET /health (or
 * /health/connectors) that its plugin made reconnectable carries a `reconnect`
 * descriptor: the route to POST, the setup route to GET, and the group (one
 * identity provider, e.g. "Google") its rows gather under. A build without those
 * plugins has no such rows, and so shows no buttons.
 *
 * Tokens never reach this code: the start route answers with the provider's
 * consent URL, the provider redirects the browser to the server, and the server
 * redirects back to /settings?connect=<result>#connections. */
import { apiFetch } from "@/lib/http";
import type { HealthCheck, Reconnect } from "@/lib/control";

export interface ConnectionSetup {
  configured: boolean;
  public_url_set: boolean;
  redirect_uri: string | null;
  group: string;
  group_label: string;
  providers: { provider: string; label: string }[];
  add_provider: string | null;
  /** Only on the upload's answer: whether the client lists the redirect URI. */
  redirect_uri_listed?: boolean | null;
}

async function send<T>(path: string, method: string, body?: unknown): Promise<T> {
  const r = await apiFetch(path, {
    method,
    headers: { "content-type": "application/json", accept: "application/json" },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try {
      const j = (await r.json()) as { detail?: unknown };
      if (typeof j?.detail === "string") msg = j.detail;
    } catch {
      /* non-JSON */
    }
    throw new Error(msg);
  }
  return (await r.json()) as T;
}

export const getConnectionSetup = (route: string) => send<ConnectionSetup>(route, "GET");

export const uploadConnectionClient = (route: string, clientJson: string) =>
  send<ConnectionSetup>(route, "PUT", { client_json: clientJson });

/** Ask the server for a consent URL, then leave for it. Resolves only on failure. */
export async function beginReconnect(target: Reconnect): Promise<void> {
  const { auth_url } = await send<{ auth_url: string }>(target.route, "POST", {
    provider: target.provider,
    account: target.account,
  });
  window.location.assign(auth_url);
}

/** A group of reconnectable rows: one identity provider, one card per account. */
export interface ConnectionGroup {
  group: string;
  label: string;
  route: string;
  setupRoute: string;
  /** Every account any row names, in first-seen order. */
  accounts: string[];
  rows: HealthCheck[];
}

export function connectionGroups(checks: HealthCheck[]): ConnectionGroup[] {
  const out = new Map<string, ConnectionGroup>();
  for (const c of checks) {
    const r = c.reconnect;
    if (!r) continue;
    let g = out.get(r.group);
    if (!g) {
      g = {
        group: r.group,
        label: r.group_label,
        route: r.route,
        setupRoute: r.setup_route,
        accounts: [],
        rows: [],
      };
      out.set(r.group, g);
    }
    g.rows.push(c);
    if (r.account && !g.accounts.includes(r.account)) g.accounts.push(r.account);
  }
  return [...out.values()];
}

export type ConnState = "ok" | "revoked" | "warn" | "none";

/** One provider's state for one account: its row if there is one, else "none". */
export function stateOf(row: HealthCheck | undefined): ConnState {
  if (!row || row.state === "grey") return "none";
  if (row.state === "green") return "ok";
  if (row.state === "red") return "revoked";
  return "warn";
}

export const STATE_TEXT: Record<ConnState, string> = {
  ok: "connected",
  revoked: "access revoked — reconnect",
  warn: "needs attention — reconnect",
  none: "not connected",
};

/** What the server's redirect said, read once from the landing URL. */
export interface ConnectResult {
  result: "connected" | "cancelled" | "wrong_account" | "expired" | "failed";
  provider: string | null;
  account: string | null;
  approved: string | null;
  reason: string | null;
}

const RESULTS = new Set(["connected", "cancelled", "wrong_account", "expired", "failed"]);

export function readConnectResult(search: string): ConnectResult | null {
  const q = new URLSearchParams(search);
  const result = q.get("connect");
  if (!result || !RESULTS.has(result)) return null;
  return {
    result: result as ConnectResult["result"],
    provider: q.get("provider"),
    account: q.get("account"),
    approved: q.get("approved"),
    reason: q.get("reason"),
  };
}
