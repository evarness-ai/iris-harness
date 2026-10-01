import { apiFetch } from './http';
import type { Trace } from './types';

// Data source for the trace and session screens: the live IRIS API
// (GET /api/traces, /api/traces/:id, /api/sessions — served by src/iris_harness/server/iris_api
// from session logs, via the Vite /api proxy). There is no canned fallback: an API that
// cannot be reached (or answers 401 — lib/http.ts routes that to /pair) is an error the
// React Query hook surfaces, and the screen shows the "API unavailable" notice. A live
// API with no logs yet (a fresh install) answers [] and the screen says what to do next.

export interface TraceSummary {
  session_id: string;
  trace_id: string;
  request: string;
  started_at: string;
  total_duration_ms: number;
  total_tokens: number;
}

export interface SessionSummary {
  session_id: string;
  title: string;
  turn_count: number;
  started_at: string;
  last_at: string;
  total_tokens: number;
  total_duration_ms: number;
  turns: TraceSummary[];
}

/** One replayed message from a past session (GET /api/sessions/:id/messages). */
export interface SessionMessage {
  role: "user" | "assistant";
  text: string;
  ts: string;
  trace_id?: string;
}

export interface SlashCommandSummary {
  name: string;
  args: string;
  description: string;
}

export interface ChatStatus {
  session_id: string;
  iris_version: string;
  provider: string;
  provider_label: string;
  model: string;
  /** A turn still running for this session, e.g. one whose stream the app lost
   * when the phone put it in the background. */
  turn_in_progress?: boolean;
  window: {
    budget_tokens: number;
    current_tokens: number;
    fill_pct: number;
  } | null;
}

export interface SlashDispatchResult {
  ok: boolean;
  executed: boolean;
  command: string;
  output: string;
  action?: "new_session" | "clear_messages";
  provider?: string;
  provider_label?: string;
  model?: string;
}

async function getJSON<T>(url: string): Promise<T> {
  const r = await apiFetch(url, { headers: { accept: 'application/json' } });
  if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`.trim());
  return (await r.json()) as T;
}

/** Newest-first summaries of recent turns; [] on a fresh install. */
export async function listTraces(): Promise<TraceSummary[]> {
  const data = await getJSON<TraceSummary[]>('/api/traces');
  if (!Array.isArray(data)) throw new Error('GET /api/traces: expected a list');
  return data;
}

/** Recent conversation sessions; [] on a fresh install. */
export async function listSessions(): Promise<SessionSummary[]> {
  const data = await getJSON<SessionSummary[]>('/api/sessions');
  if (!Array.isArray(data)) throw new Error('GET /api/sessions: expected a list');
  return data;
}

/** Replay a past session's user/assistant messages (empty if none / API down). */
export async function listSessionMessages(sessionId: string): Promise<SessionMessage[]> {
  try {
    const r = await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}/messages`);
    if (r.ok) {
      const data = (await r.json()) as SessionMessage[];
      if (Array.isArray(data)) return data;
    }
  } catch {
    /* API down — no history to replay */
  }
  return [];
}

/** One turn's full graph, or null when the API has no such trace (404). */
export async function getTrace(traceId: string): Promise<Trace | null> {
  const r = await apiFetch(`/api/traces/${encodeURIComponent(traceId)}`, {
    headers: { accept: 'application/json' },
  });
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`HTTP ${r.status} ${r.statusText}`.trim());
  return (await r.json()) as Trace;
}

export async function listSlashCommands(): Promise<SlashCommandSummary[]> {
  try {
    const r = await apiFetch("/api/slash-commands");
    if (r.ok) {
      const data = (await r.json()) as { commands?: SlashCommandSummary[] };
      if (Array.isArray(data.commands)) return data.commands;
    }
  } catch {
    /* API down — no slash command suggestions */
  }
  return [];
}

export async function getChatStatus(sessionId: string): Promise<ChatStatus | undefined> {
  try {
    const r = await apiFetch(`/api/chat-status?session_id=${encodeURIComponent(sessionId)}`);
    if (r.ok) return (await r.json()) as ChatStatus;
  } catch {
    /* API down */
  }
  return undefined;
}

export async function dispatchSlashCommand(
  command: string,
  sessionId: string,
): Promise<SlashDispatchResult> {
  const r = await apiFetch("/api/slash-dispatch", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ command, session_id: sessionId }),
  });

  const data = (await r.json().catch(() => ({}))) as {
    detail?: string;
    message?: string;
  } & Partial<SlashDispatchResult>;

  if (!r.ok) {
    throw new Error(data.detail || data.message || `HTTP ${r.status} ${r.statusText}`);
  }

  return {
    ok: Boolean(data.ok),
    executed: Boolean(data.executed),
    command: String(data.command || command),
    output: String(data.output || "Command executed."),
    action:
      data.action === "new_session" || data.action === "clear_messages"
        ? data.action
        : undefined,
    provider: typeof data.provider === "string" ? data.provider : undefined,
    provider_label: typeof data.provider_label === "string" ? data.provider_label : undefined,
    model: typeof data.model === "string" ? data.model : undefined,
  };
}
