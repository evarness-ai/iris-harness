/* Stored digests (loop-proof plan PR 2, graph §7 "web: full, stored").
 *
 * The harness stores every brief it renders; the push notification's tap lands
 * on /digest/<id>, and this is the one rendering that shows the whole thing,
 * including the in-app actions the other channels drop:
 *
 *   [👎](iris:not-useful/<url-quoted sender>)  →  a button that POSTs
 *   {"sender": "<sender>"} to /api/digest/not-useful, so that sender leaves
 *   tomorrow's Focus list (D17). */
import { apiFetch } from "./http";

export interface FailedSection {
  name: string;
  title: string;
  reason: string;
}

export interface StoredDigest {
  id: string;
  created_at: string;
  heartbeat: string;
  skill_id: string;
  subject: string;
  body: string;
  failed_sections: FailedSection[];
}

export interface DigestSummary {
  id: string;
  created_at: string;
  subject: string;
  skill_id: string;
  failed_sections: number;
}

/** A 404 is "no digest yet" (or an old link), not an outage. */
export class DigestNotFound extends Error {
  constructor() {
    super("digest not found");
    this.name = "DigestNotFound";
  }
}

async function getJson<T>(url: string): Promise<T> {
  const r = await apiFetch(url);
  if (r.status === 404) throw new DigestNotFound();
  if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`.trim());
  return (await r.json()) as T;
}

/** The newest digest, or the one a push notification linked to. */
export function fetchDigest(id?: string): Promise<StoredDigest> {
  return getJson<StoredDigest>(id ? `/api/digest/${encodeURIComponent(id)}` : "/api/digest/latest");
}

export async function fetchDigestList(limit = 14): Promise<DigestSummary[]> {
  const body = await getJson<{ digests: DigestSummary[] }>(`/api/digest?limit=${limit}`);
  return body.digests;
}

export const NOT_USEFUL_SCHEME = "iris:not-useful/";

/** The sender an `iris:not-useful/<quoted>` link names, or null for any other link. */
export function notUsefulSender(href: string | undefined | null): string | null {
  if (!href || !href.startsWith(NOT_USEFUL_SCHEME)) return null;
  const raw = href.slice(NOT_USEFUL_SCHEME.length);
  if (!raw) return null;
  try {
    return decodeURIComponent(raw);
  } catch {
    return null;
  }
}

/** Mark a Focus sender "not useful": it is hidden from tomorrow's digest. */
export async function markNotUseful(sender: string): Promise<void> {
  const r = await apiFetch("/api/digest/not-useful", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sender }),
  });
  if (!r.ok) {
    let msg = `HTTP ${r.status} ${r.statusText}`.trim();
    try {
      const j = (await r.json()) as { detail?: unknown };
      if (typeof j?.detail === "string") msg = j.detail;
    } catch {
      /* keep the status line */
    }
    throw new Error(msg);
  }
}
