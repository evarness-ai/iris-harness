/* The one door to the IRIS API. Every lib module calls `apiFetch`, never `fetch`
 * (tests/unit/test_webui/test_webui_pairing.py holds that line).
 *
 * A paired browser authenticates with the HttpOnly `iris_device` cookie (ADR-0117).
 * JS never sees that token and must never store one — no localStorage, no header.
 * The cookie rides along on same-origin requests; when the server answers 401
 * (never paired, or the device was revoked) the console goes to /pair, once, and
 * the caller still gets its Response so its own error handling runs unchanged. */

export const PAIR_PATH = "/pair";
const CLAIM_PATH = "/api/v1/devices/pair/claim";

/** Thrown by callers that would otherwise hide a 401 behind a fallback (mock data). */
export class UnauthorizedError extends Error {
  constructor() {
    super("HTTP 401 Unauthorized");
    this.name = "UnauthorizedError";
  }
}

/** Open-redirect guard for `?next=`: a same-origin path only. It starts with a
 * single "/" — "//host" and "/\host" are other origins to a browser — and is
 * never /pair itself (that would loop). Anything else becomes "/". */
export function safeNext(raw: string | null | undefined): string {
  if (!raw || !raw.startsWith("/") || raw.startsWith("//") || raw.includes("\\")) return "/";
  if (raw === PAIR_PATH || raw.startsWith(`${PAIR_PATH}?`) || raw.startsWith(`${PAIR_PATH}/`)) {
    return "/";
  }
  return raw;
}

type Navigator = (to: string) => void;

let navigator_: Navigator | null = null;
let redirecting = false;

/** The router registers itself here (main.tsx) so a 401 is a client-side
 * navigation. The lib modules are plain TS with no React, hence the hook;
 * without one we fall back to a full page load. Returns an unsubscribe. */
export function onUnauthorized(navigate: Navigator): () => void {
  navigator_ = navigate;
  return () => {
    if (navigator_ === navigate) navigator_ = null;
  };
}

/** The /pair screen calls this on mount: the redirect has landed, so a later
 * 401 (the user backs out unpaired, or is revoked after pairing) redirects again. */
export function pairScreenReached(): void {
  redirecting = false;
}

function pathOf(input: RequestInfo | URL): string {
  const raw = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
  try {
    return new URL(raw, window.location.origin).pathname;
  } catch {
    return "";
  }
}

function goPair(): void {
  const { pathname, search } = window.location;
  if (redirecting || pathname === PAIR_PATH) return;
  redirecting = true;
  const to = `${PAIR_PATH}?next=${encodeURIComponent(safeNext(pathname + search))}`;
  if (navigator_) navigator_(to);
  else window.location.assign(to);
}

/** `fetch`, with the device cookie and the 401 → /pair rule. Same signature and
 * the untouched Response, so streaming readers (lib/chat.ts) work as before. */
export async function apiFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const res = await fetch(input, { credentials: "same-origin", ...init });
  // A refused pairing code is the claim's own answer, not a lost session.
  if (res.status === 401 && pathOf(input) !== CLAIM_PATH) goPair();
  return res;
}
