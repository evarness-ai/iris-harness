/* A fresh install, as the console sees it: the IRIS API is up and nothing has run yet.
 *
 * FRESH holds what the API answers on an empty IRIS_HOME for the endpoints whose empty
 * shape a screen depends on (the ledger, the proof window, the trace and session lists).
 * Every other endpoint gets `emptied(fixtureFor(path))`: the viewport fixture's shape
 * with every list emptied, which is how the API's collections read before the first
 * chat. The nav, the capabilities and this device stay as they are; they describe the
 * install, not anything it has recorded. */
import { fixtureFor } from "./fixtures";

export const FRESH: Record<string, unknown> = {
  "/governance/state": {
    enabled: true,
    audit_db: "/home/owner/.iris/governance/audit.db",
    audit_count: 0,
    flags: [{ key: "IRIS_GOVERNANCE_ENABLED", label: "Kernel enabled", on: true }],
  },
  "/governance/audit": {
    count: 0,
    total: 0,
    audit_db: "/home/owner/.iris/governance/audit.db",
    callers: [],
    entries: [],
  },
  "/governance/pii-shadow": {
    mode: "off",
    since: "2026-09-24T10:00:00+00:00",
    rows: 0,
    checked: {},
    unobserved: {},
    cells: [],
  },
  "/governance/proof-bundle/check": {
    since: "2026-09-24T10:00:00+00:00",
    until: null,
    ledger_rows: 0,
    ok: true,
    integrity: [],
    invariants: [
      { id: "no-private-to-cloud", statement: "No private call to a cloud tier.", ok: true, violation_count: 0, violations: [], evidence: { model_call_rows: 0 } },
      { id: "mailbox-write-approved", statement: "No mailbox write without an approval.", ok: true, violation_count: 0, violations: [], evidence: { approval_rows: 0, writes_observed: 0 } },
      { id: "every-call-and-answer-audited", statement: "Every call and answer audited.", ok: true, violation_count: 0, violations: [], evidence: { model_calls_observed: 0, answers_observed: 0 } },
    ],
  },
  "/api/traces": [],
  "/api/sessions": [],
};

/** Endpoints that describe the install itself; a fresh one answers them in full. */
const INSTALL = ["/api/v1/webui/nav", "/capabilities", "/api/v1/devices/me"];

/** `body` with every list (at any depth) emptied. */
export function emptied(body: unknown): unknown {
  if (Array.isArray(body)) return [];
  if (body && typeof body === "object") {
    return Object.fromEntries(Object.entries(body).map(([k, v]) => [k, emptied(v)]));
  }
  return body;
}

/** The fresh-install body for `pathname`. */
export function freshFor(pathname: string): unknown {
  const fresh = Object.keys(FRESH).find((p) => pathname.endsWith(p));
  if (fresh) return FRESH[fresh];
  if (INSTALL.includes(pathname)) return fixtureFor(pathname);
  return emptied(fixtureFor(pathname));
}
