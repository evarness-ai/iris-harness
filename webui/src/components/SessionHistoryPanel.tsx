/* Past chat sessions, newest day first -- click to resume (Claude-style). Lives
 * nested under the "Chat" nav entry (App.tsx's NavList), not as its own column
 * in the Chat screen: that freed the conversation + composer the full content
 * width instead of splitting it with a dedicated 256px history rail. */
import { useEffect, useState } from "react";
import { Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useSessions } from "@/lib/queries";

function fmtWhen(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function dayKey(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "Unknown";
  return d.toISOString().slice(0, 10);
}

function dayLabel(key: string): string {
  if (key === "Unknown") return key;
  const target = new Date(`${key}T00:00:00`);
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const diffDays = Math.floor((today.getTime() - target.getTime()) / 86_400_000);
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  return target.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
}

export function SessionHistoryPanel({
  active,
  onSelect,
  onNew,
}: {
  active: string;
  onSelect: (id: string) => void;
  onNew: () => void;
}) {
  const { data, isLoading, isError } = useSessions();
  const sessions = data ?? [];
  const [collapsedByDay, setCollapsedByDay] = useState<Record<string, boolean>>({});

  const grouped = sessions.reduce<Record<string, typeof sessions>>((acc, s) => {
    const key = dayKey(s.last_at || s.started_at);
    if (!acc[key]) acc[key] = [];
    acc[key].push(s);
    return acc;
  }, {});
  const orderedDayKeys = Object.keys(grouped).sort((a, b) => (a < b ? 1 : -1));

  useEffect(() => {
    setCollapsedByDay((prev) => {
      const next: Record<string, boolean> = {};
      for (const key of orderedDayKeys) {
        next[key] = prev[key] ?? false;
      }
      return next;
    });
  }, [sessions.length, orderedDayKeys.join("|")]);

  return (
    <div className="mb-1 ml-1 flex flex-col gap-1.5 border-l border-border pb-1 pl-2.5">
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="w-full justify-start"
        onClick={onNew}
      >
        <Plus /> New chat
      </Button>
      <div className="max-h-[60vh] space-y-1 overflow-y-auto rounded-lg border border-border bg-bg p-1.5">
        {isLoading && (
          <div className="p-3 text-center text-xs text-fg-subtle">Loading history…</div>
        )}
        {isError && (
          <div className="p-3 text-center text-xs text-danger">
            API unavailable — start it with <code>iris serve</code>.
          </div>
        )}
        {!isLoading && !isError && sessions.length === 0 && (
          <div className="p-3 text-center text-xs text-fg-subtle">No conversations yet.</div>
        )}
        {orderedDayKeys.map((key) => {
          const items = grouped[key] ?? [];
          const collapsed = collapsedByDay[key] ?? false;
          return (
            <div key={key} className="space-y-1">
              <button
                type="button"
                onClick={() =>
                  setCollapsedByDay((prev) => ({
                    ...prev,
                    [key]: !collapsed,
                  }))
                }
                className="flex w-full items-center justify-between rounded px-1.5 py-1 text-left text-[11px] font-medium uppercase tracking-wide text-fg-subtle hover:bg-surface"
              >
                <span>{dayLabel(key)}</span>
                <span className="font-mono text-[10px]">{collapsed ? "+" : "-"}</span>
              </button>
              {!collapsed &&
                items.map((s) => (
                  <button
                    key={s.session_id}
                    type="button"
                    onClick={() => onSelect(s.session_id)}
                    title={s.title}
                    className={`w-full rounded-md border-l-2 px-2.5 py-2 text-left transition-colors ${
                      s.session_id === active
                        ? "border-l-primary bg-primary/10"
                        : "border-l-transparent hover:bg-surface"
                    }`}
                  >
                    <div className="truncate text-[13px] text-fg">{s.title || "(untitled)"}</div>
                    <div className="mt-0.5 flex items-center justify-between gap-2 font-mono text-[10px] text-fg-subtle">
                      <span>{fmtWhen(s.last_at)}</span>
                      <span>
                        {s.turn_count} turn{s.turn_count !== 1 ? "s" : ""}
                      </span>
                    </div>
                  </button>
                ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
