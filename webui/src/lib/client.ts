import { MOCK_TRACES } from '../mock/traces';
import { apiFetch, UnauthorizedError } from './http';
import type { Trace } from './types';

// Data source for the trace screen. Hits the live IRIS API
// (GET /api/traces, /api/traces/:id — served by src/iris_harness/server/iris_api from session
// logs, via the Vite /api proxy). If the API is unreachable, it falls back to the
// canned mock traces so the prototype always renders. (Sessions still also falls
// back when the API has no logs yet; traces do not, see listTraces.)
//
// A 401 is neither: the API is up and this browser is not paired (or was revoked).
// Mock traces under a "mock data" badge would hide that — and CallTrace would
// navigate to a mock trace, racing the redirect to /pair — so the mock-backed
// reads rethrow it instead of falling back.

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

/** Where the current list came from — surfaced in the UI as a small badge. */
export type Source = 'live' | 'mock';

function refuseUnauthorized(r: Response): void {
  if (r.status === 401) throw new UnauthorizedError();
}

const mockSummaries = (): TraceSummary[] =>
  MOCK_TRACES.map((t) => ({
    session_id: t.session_id,
    trace_id: t.trace_id,
    request: t.request,
    started_at: t.started_at,
    total_duration_ms: t.total_duration_ms,
    total_tokens: t.total_tokens,
  }));

// A live API with no logs yet (a fresh install) is live and empty, not "no API": mock
// traces there would show a builder someone else's turns under a small "mock data"
// badge, and Call Trace says what to do instead. Only an unreachable API falls back.
export async function listTraces(): Promise<{ source: Source; traces: TraceSummary[] }> {
  try {
    const r = await apiFetch('/api/traces');
    refuseUnauthorized(r);
    if (r.ok) {
      const data = (await r.json()) as TraceSummary[];
      if (Array.isArray(data)) return { source: 'live', traces: data };
    }
  } catch (e) {
    if (e instanceof UnauthorizedError) throw e;
    /* API down — fall back to mock */
  }
  return { source: 'mock', traces: mockSummaries() };
}

export async function listSessions(): Promise<{ source: Source; sessions: SessionSummary[] }> {
  try {
    const r = await apiFetch('/api/sessions');
    refuseUnauthorized(r);
    if (r.ok) {
      const data = (await r.json()) as SessionSummary[];
      if (Array.isArray(data) && data.length > 0) return { source: 'live', sessions: data };
    }
  } catch (e) {
    if (e instanceof UnauthorizedError) throw e;
    /* API down — fall back to mock */
  }
  // Mock: each canned trace is a single-turn session.
  const sessions: SessionSummary[] = mockSummaries().map((t) => ({
    session_id: t.session_id,
    title: t.request,
    turn_count: 1,
    started_at: t.started_at,
    last_at: t.started_at,
    total_tokens: t.total_tokens,
    total_duration_ms: t.total_duration_ms,
    turns: [t],
  }));
  return { source: 'mock', sessions };
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

export async function getTrace(traceId: string): Promise<Trace | undefined> {
  try {
    const r = await apiFetch(`/api/traces/${encodeURIComponent(traceId)}`);
    refuseUnauthorized(r);
    if (r.ok) return (await r.json()) as Trace;
  } catch (e) {
    if (e instanceof UnauthorizedError) throw e;
    /* fall through to mock */
  }
  return MOCK_TRACES.find((t) => t.trace_id === traceId);
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
