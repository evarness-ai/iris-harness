/* Paired-device client (ADR-0117, mobile-cloud-ui-plan track 1 PR 3).
 *
 * Wraps /api/v1/devices: pair/start, pair/claim, list, me, revoke. A browser's
 * token arrives as an HttpOnly cookie on claim and is never in the body, so
 * nothing here (or anywhere in the console) holds, reads or stores a token.
 * Device administration is NOT behind IRIS_WEBUI_ALLOW_WRITES — scope is
 * enforced per route by the server; the UI only mirrors it to avoid dead buttons. */
import { apiFetch } from "./http";

export type DeviceKind = "app" | "browser";
export type DeviceScope = "read" | "control";

export interface Device {
  device_id: string;
  name: string;
  kind: DeviceKind;
  scope: DeviceScope;
  created_at: string;
  last_seen_at: string | null;
  revoked_at: string | null;
  /** True on the row that is the calling device. */
  current: boolean;
}

export interface PairingCode {
  code: string;
  scope: DeviceScope;
  expires_at: string;
}

/** GET /api/v1/devices/me — `device` is null for the service principal. */
export interface Principal {
  kind: "service" | "device";
  scope: DeviceScope;
  via: "bearer" | "cookie";
  device: Device | null;
}

/** An API refusal with its status, so screens can tell 400 / 403 / 422 / 429 apart. */
export class ApiError extends Error {
  readonly status: number;
  /** Seconds from a `Retry-After` header (429), else null. */
  readonly retryAfter: number | null;

  constructor(message: string, status: number, retryAfter: number | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.retryAfter = retryAfter;
  }
}

async function fail(r: Response): Promise<never> {
  let msg = `HTTP ${r.status} ${r.statusText}`.trim();
  try {
    const j = (await r.json()) as { detail?: unknown };
    if (typeof j?.detail === "string") msg = j.detail;
    // FastAPI's own 422 body is a list of {msg}; the contract's is a string.
    else if (Array.isArray(j?.detail)) {
      const parts = j.detail.map((d) => (d as { msg?: string })?.msg).filter(Boolean);
      if (parts.length > 0) msg = parts.join("; ");
    }
  } catch {
    /* non-JSON */
  }
  const retry = Number.parseInt(r.headers.get("retry-after") ?? "", 10);
  throw new ApiError(msg, r.status, Number.isFinite(retry) ? retry : null);
}

async function send<T>(path: string, method: string, body?: unknown): Promise<T> {
  const r = await apiFetch(path, {
    method,
    headers:
      body === undefined
        ? { accept: "application/json" }
        : { accept: "application/json", "content-type": "application/json" },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  if (!r.ok) return fail(r);
  return (await r.json()) as T;
}

export async function listDevices(): Promise<Device[]> {
  const data = await send<{ devices: Device[] }>("/api/v1/devices", "GET");
  return data.devices ?? [];
}

export const getPrincipal = () => send<Principal>("/api/v1/devices/me", "GET");

/** Start a pairing: a short-lived code the new device types into /pair. */
export const startPairing = (scope: DeviceScope) =>
  send<PairingCode>("/api/v1/devices/pair/start", "POST", { scope });

/** Claim a code for THIS browser. The token comes back as an HttpOnly cookie. */
export async function claimPairing(code: string, name: string): Promise<Device> {
  const data = await send<{ device: Device }>("/api/v1/devices/pair/claim", "POST", {
    code,
    name,
    kind: "browser",
  });
  return data.device;
}

/** Revoke a device. Revoking the current one is "sign out" (the server clears the cookie). */
export async function revokeDevice(deviceId: string): Promise<Device> {
  const data = await send<{ device: Device }>(
    `/api/v1/devices/${encodeURIComponent(deviceId)}`,
    "DELETE",
  );
  return data.device;
}

// ---- pairing-code + device-name helpers (pure; used by the /pair screen) ----

/** The server's code alphabet: no 0/O, 1/I/L or U (kernel/governance/devices). */
const CODE_ALPHABET = /^[ABCDEFGHJKMNPQRSTVWXYZ23456789]*$/;
export const CODE_LENGTH = 8;
export const MAX_DEVICE_NAME = 64;

/** Uppercase and drop the dash/spaces: "abcd efgh" → "ABCDEFGH" (at most 8 chars). */
export function normalizeCode(raw: string): string {
  return raw
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, "")
    .slice(0, CODE_LENGTH);
}

/** "ABCDEFGH" → "ABCD-EFGH", as the code is shown and sent. */
export function formatCode(raw: string): string {
  const c = normalizeCode(raw);
  return c.length > 4 ? `${c.slice(0, 4)}-${c.slice(4)}` : c;
}

/** Why a typed code cannot be right yet, or null when it is ready to send. */
export function codeProblem(raw: string): string | null {
  const c = normalizeCode(raw);
  if (!CODE_ALPHABET.test(c)) return "Codes never contain 0, O, 1, I, L or U — check the code.";
  if (c.length < CODE_LENGTH) return `The code has ${CODE_LENGTH} characters.`;
  return null;
}

/** A readable default device name from the user agent, e.g. "Safari on iPhone". */
export function defaultDeviceName(ua: string = navigator.userAgent): string {
  // Order matters: Edge and Chrome both say "Safari"; Chrome on iOS says "CriOS".
  const browser = /Edg(e|A|iOS)?\//.test(ua)
    ? "Edge"
    : /Firefox\/|FxiOS\//.test(ua)
      ? "Firefox"
      : /Chrome\/|CriOS\//.test(ua)
        ? "Chrome"
        : /Safari\//.test(ua)
          ? "Safari"
          : "Browser";
  const os = /iPhone/.test(ua)
    ? "iPhone"
    : /iPad/.test(ua)
      ? "iPad"
      : /Android/.test(ua)
        ? "Android"
        : /Macintosh|Mac OS X/.test(ua)
          ? "Mac"
          : /Windows/.test(ua)
            ? "Windows"
            : /Linux/.test(ua)
              ? "Linux"
              : "";
  return (os ? `${browser} on ${os}` : browser).slice(0, MAX_DEVICE_NAME);
}
