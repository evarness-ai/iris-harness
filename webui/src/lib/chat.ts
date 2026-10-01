/* Streaming chat client over POST /chat/stream.
 *
 * The IRIS API streams NDJSON (one JSON object per line, media type
 * application/x-ndjson) — see src/iris_harness/server/iris_api/main.py::chat_stream. Event
 * shapes:
 *   { event: "token",    text }                       incremental answer text
 *   { event: "activity", text }                       status line (thinking, …)
 *   { event: "trace",    text, payload? }             debug breadcrumbs
 *   { event: "done",     response, intent, agent_type,
 *                        sources, has_errors, error_summary, metadata }
 *   { event: "error",    error }
 *
 * We read the body as a stream and split on newlines so tokens render live. */
import { apiFetch } from "./http";

export interface ChatDone {
  response: string;
  intent: string;
  agent_type: string;
  sources: string[];
  has_errors: boolean;
  error_summary: string | null;
  metadata: Record<string, unknown>;
}

export interface ChatHandlers {
  onToken?: (text: string) => void;
  onActivity?: (text: string) => void;
  onTrace?: (text: string, payload?: unknown) => void;
  onDone?: (done: ChatDone) => void;
  onError?: (error: string) => void;
}

export interface ChatStreamRequest {
  message: string;
  sessionId: string;
  channel?: string;
  signal?: AbortSignal;
}

interface RawEvent {
  event?: string;
  text?: string;
  payload?: unknown;
  error?: string;
  response?: string;
  intent?: string;
  agent_type?: string;
  sources?: unknown;
  has_errors?: boolean;
  error_summary?: string | null;
  metadata?: unknown;
}

const SESSION_KEY = "iris-chat-session";

function freshId(): string {
  // crypto.randomUUID is available in all evergreen browsers (secure context).
  const uuid = typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : "";
  return `web-${uuid ? uuid.slice(0, 8) : Math.floor(performance.now()).toString(36)}`;
}

/** The persisted chat session id (one per browser, until "New chat"). */
export function getSessionId(): string {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = freshId();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

/** Start a new session and return its id. */
export function newSession(): string {
  const id = freshId();
  localStorage.setItem(SESSION_KEY, id);
  return id;
}

/** Resume an existing session: make it the active (persisted) one. */
export function setSession(id: string): string {
  localStorage.setItem(SESSION_KEY, id);
  return id;
}

/** The first-chat welcome (ADR-0127), as POST /chat/welcome answers it. */
export interface Welcome {
  session_id: string;
  /** True only for the call that ran it; later calls get the same welcome back. */
  created: boolean;
  response: string;
  trace_id: string;
}

let welcomeRequest: Promise<Welcome | null> | null = null;

/**
 * Ask the harness for the first-chat welcome. The harness decides whether it is due
 * (once per install) and runs it; this only asks and hands back the answer. One request
 * per page load, shared by every caller, so a remounted Chat never asks twice. Best
 * effort: null when the API cannot answer, and the chat works as before.
 */
export function requestWelcome(): Promise<Welcome | null> {
  welcomeRequest ??= (async () => {
    try {
      const res = await apiFetch("/chat/welcome", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ channel: "web" }),
      });
      if (!res.ok) return null;
      const body = (await res.json()) as Partial<Welcome> | null;
      if (!body || typeof body.session_id !== "string" || typeof body.response !== "string") {
        return null;
      }
      return {
        session_id: body.session_id,
        created: body.created === true,
        response: body.response,
        trace_id: typeof body.trace_id === "string" ? body.trace_id : "",
      };
    } catch {
      return null;
    }
  })();
  return welcomeRequest;
}

/** One user reaction to an answer (ADR-0072). Best-effort: never throws. */
export interface FeedbackInput {
  sentiment: "up" | "down";
  sessionId?: string;
  traceId?: string;
  rating?: number;
  note?: string;
  intent?: string;
  agentType?: string;
}

export async function sendFeedback(input: FeedbackInput): Promise<boolean> {
  try {
    const res = await apiFetch("/api/feedback", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        sentiment: input.sentiment,
        session_id: input.sessionId,
        trace_id: input.traceId,
        rating: input.rating,
        note: input.note,
        intent: input.intent,
        agent_type: input.agentType,
      }),
    });
    return res.ok;
  } catch {
    return false; // feedback is best-effort; never disrupt the chat
  }
}

/**
 * Stop the session's running turn on the server. Closing the stream no longer does:
 * the turn outlives its connection, so a phone that backgrounds the app does not
 * lose it. Best-effort — the caller has already stopped listening.
 */
export async function cancelChat(sessionId: string): Promise<void> {
  try {
    await apiFetch("/chat/cancel", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
  } catch {
    /* nothing to add: the stop already happened here */
  }
}

function dispatch(evt: RawEvent, h: ChatHandlers): void {
  switch (evt.event) {
    case "token":
      h.onToken?.(evt.text ?? "");
      break;
    case "activity":
      h.onActivity?.(evt.text ?? "");
      break;
    case "trace":
      h.onTrace?.(evt.text ?? "", evt.payload);
      break;
    case "done":
      h.onDone?.({
        response: evt.response ?? "",
        intent: evt.intent ?? "",
        agent_type: evt.agent_type ?? "",
        sources: Array.isArray(evt.sources) ? evt.sources.map(String) : [],
        has_errors: Boolean(evt.has_errors),
        error_summary: evt.error_summary ?? null,
        metadata:
          evt.metadata && typeof evt.metadata === "object"
            ? (evt.metadata as Record<string, unknown>)
            : {},
      });
      break;
    case "error":
      h.onError?.(evt.error ?? "unknown error");
      break;
    default:
      // Includes the server's keep-alive ("ping") during a long model call.
      break;
  }
}

/**
 * POST a message and drive `handlers` from the NDJSON stream. Resolves when the
 * stream ends. Pass `signal` (AbortController) to cancel an in-flight turn —
 * aborting rejects, so callers should treat AbortError as a clean stop.
 */
export async function streamChat(req: ChatStreamRequest, handlers: ChatHandlers): Promise<void> {
  const res = await apiFetch("/chat/stream", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      message: req.message,
      session_id: req.sessionId,
      channel: req.channel ?? "web",
    }),
    signal: req.signal,
  });

  if (!res.ok || !res.body) {
    handlers.onError?.(`HTTP ${res.status} ${res.statusText}`.trim());
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  const drain = (flush: boolean) => {
    let nl = buf.indexOf("\n");
    while (nl >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) {
        try {
          dispatch(JSON.parse(line) as RawEvent, handlers);
        } catch {
          /* skip malformed line */
        }
      }
      nl = buf.indexOf("\n");
    }
    if (flush && buf.trim()) {
      try {
        dispatch(JSON.parse(buf.trim()) as RawEvent, handlers);
      } catch {
        /* ignore trailing garbage */
      }
      buf = "";
    }
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    drain(false);
  }
  drain(true);
}
