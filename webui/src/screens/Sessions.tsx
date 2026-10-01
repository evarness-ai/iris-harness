import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Card, Kpi } from "../components/Card";
import { Tag } from "../components/Tag";
import { fmtMs, fmtTokens } from "../lib/nodeMeta";
import { useSessions } from "../lib/queries";

function fmtTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function SessionsScreen() {
  const navigate = useNavigate();
  const { data, isLoading } = useSessions();
  const source = data?.source ?? "mock";
  const sessions = useMemo(() => data?.sessions ?? [], [data]);
  const [sel, setSel] = useState<string>("");

  const selected = useMemo(
    () => sessions.find((s) => s.session_id === sel) ?? sessions[0] ?? null,
    [sessions, sel],
  );
  const totalTurns = sessions.reduce((n, s) => n + s.turn_count, 0);
  const totalTokens = sessions.reduce((n, s) => n + s.total_tokens, 0);

  if (isLoading) {
    return <div className="p-8 text-center text-sm text-fg-muted">Loading sessions…</div>;
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Tag kind={source === "live" ? "ok" : "warn"}>
          {source === "live" ? "live logs" : "mock data"}
        </Tag>
        <span className="text-xs text-fg-subtle">
          recent conversation sessions — open any turn in the Call Trace
        </span>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Kpi value={`${sessions.length}`} label="sessions" />
        <Kpi value={`${totalTurns}`} label="turns" />
        <Kpi value={fmtTokens(totalTokens)} label="total tokens" />
        <Kpi value={selected ? `${selected.turn_count}` : "—"} label="selected turns" />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[320px_1fr]">
        <Card title="Sessions">
          <div className="max-h-[560px] space-y-1 overflow-auto">
            {sessions.map((s) => (
              <button
                key={s.session_id}
                type="button"
                onClick={() => setSel(s.session_id)}
                className={`w-full rounded-lg border-l-2 px-2.5 py-2 text-left transition-colors ${
                  s.session_id === (selected?.session_id ?? "")
                    ? "border-l-primary bg-primary/10"
                    : "border-l-transparent hover:bg-surface"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="truncate font-mono text-[12px] text-fg">{s.session_id}</span>
                  <span className="shrink-0 text-[10px] text-fg-subtle">{fmtTime(s.last_at)}</span>
                </div>
                <div className="mt-0.5 flex items-center gap-2 font-mono text-[10px] text-fg-subtle">
                  <span>
                    {s.turn_count} turn{s.turn_count !== 1 ? "s" : ""}
                  </span>
                  <span className="text-node-runtime">{fmtTokens(s.total_tokens)} tok</span>
                  <span>{fmtMs(s.total_duration_ms)}</span>
                </div>
              </button>
            ))}
            {sessions.length === 0 && (
              <div className="p-4 text-center text-xs text-fg-subtle">No sessions found.</div>
            )}
          </div>
        </Card>

        <Card
          title={selected ? `Turns · ${selected.session_id}` : "Turns"}
          action={
            selected ? (
              <button
                type="button"
                onClick={() => navigate(`/chat/${selected.session_id}`)}
                className="rounded-md bg-primary px-2 py-1 text-[11px] font-semibold text-primary-fg hover:opacity-90"
              >
                open in chat →
              </button>
            ) : undefined
          }
        >
          {!selected ? (
            <div className="text-xs text-fg-subtle">Select a session.</div>
          ) : (
            <div className="space-y-1">
              {selected.turns.map((t) => (
                <div
                  key={t.trace_id}
                  className="flex items-center gap-3 rounded-lg px-2.5 py-2 hover:bg-surface"
                >
                  <span className="min-w-0 flex-1 truncate text-[13px] text-fg">
                    “{t.request || "(empty)"}”
                  </span>
                  <span className="hidden shrink-0 font-mono text-[10px] text-fg-subtle sm:inline">
                    {fmtTime(t.started_at)}
                  </span>
                  <span className="hidden shrink-0 font-mono text-[10px] text-node-runtime sm:inline">
                    {fmtTokens(t.total_tokens)} tok
                  </span>
                  <span className="hidden shrink-0 font-mono text-[10px] text-fg-muted sm:inline">
                    {fmtMs(t.total_duration_ms)}
                  </span>
                  <button
                    type="button"
                    onClick={() => navigate(`/calltrace/${t.trace_id}`)}
                    className="shrink-0 rounded-md bg-primary px-2 py-1 text-[11px] font-semibold text-primary-fg hover:opacity-90"
                  >
                    view trace →
                  </button>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
