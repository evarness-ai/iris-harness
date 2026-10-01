import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Plus, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  ChatMessage,
  type ChatMsgView,
  type PendingConfirmation,
} from "@/components/chat/ChatMessage";
import { Composer } from "@/components/chat/Composer";
import {
  dispatchSlashCommand,
  getChatStatus,
  listSessionMessages,
  type SessionMessage,
} from "@/lib/client";
import {
  cancelChat,
  getSessionId,
  newSession,
  setSession as persistSession,
  streamChat,
} from "@/lib/chat";
import {
  useActivities,
  useChatStatus,
  useOutstandingItems,
  useSessionMessages,
  useSlashCommands,
  useSessions,
  useUpdateDueStatus,
  useCreateTask,
  useUpdateTaskStatus,
  useReminderAct,
  useReminderUndo,
} from "@/lib/queries";
import type { ReminderItem, SnoozeFor } from "@/lib/control";

let seq = 0;
const nextId = () => `m${++seq}`;

// A turn runs on the server whether or not the app is still listening: a phone that
// backgrounds the app drops the stream, not the turn. When the stream is lost, or the
// app comes back to a session whose turn is still running, the chat waits for the
// turn to end and then shows the session as the server recorded it.
const FOLLOW_POLL_MS = 2500;
const FOLLOW_LIMIT_MS = 15 * 60_000;

function historyToViews(history: SessionMessage[]): ChatMsgView[] {
  return history.map((m, i) => ({
    id: `h${i}`,
    role: m.role,
    text: m.text,
    status: "done" as const,
    meta: m.role === "assistant" && m.trace_id ? { traceId: m.trace_id } : undefined,
  }));
}

/** Read a harness `pending_confirmation` (ADR-0076) off a done event's metadata. */
function pendingConfirmationFrom(
  metadata: Record<string, unknown>,
): PendingConfirmation | undefined {
  const pc = metadata.pending_confirmation;
  if (!pc || typeof pc !== "object") return undefined;
  const obj = pc as Record<string, unknown>;
  const id = typeof obj.id === "string" ? obj.id : "";
  const options = Array.isArray(obj.options)
    ? obj.options.filter((o): o is string => typeof o === "string")
    : ["approve", "reject"];
  if (!id || options.length === 0) return undefined;
  return {
    id,
    kind: typeof obj.kind === "string" ? obj.kind : undefined,
    summary: typeof obj.summary === "string" ? obj.summary : undefined,
    detail: typeof obj.detail === "string" ? obj.detail : undefined,
    options,
  };
}

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

function EmptyState() {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center">
      <div className="text-base font-semibold text-fg">Talk to IRIS</div>
      <p className="max-w-md text-sm text-fg-muted">
        Streaming, local-first. Ask a question, request a task, or check on your
        day. Your messages are processed on this machine — nothing leaves unless
        a tool is explicitly egress-approved.
      </p>
    </div>
  );
}

function shortDate(isoLike: string): string {
  const d = new Date(isoLike);
  if (Number.isNaN(d.getTime())) return isoLike;
  return d.toLocaleDateString([], { month: "short", day: "numeric" });
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

/** Left rail: past sessions, newest-first — click to resume (Claude-style). */
const SNOOZES: { action: SnoozeFor; label: string }[] = [
  { action: "10m", label: "10 min" },
  { action: "1h", label: "1 hour" },
  { action: "tomorrow_9am", label: "Tomorrow 9am" },
];

/** One Active reminder: its text opens the sheet (/reminders/<id>); Done and a Snooze
 * menu answer it in place, with Undo on the confirmation toast (loop-proof PR 3b). A
 * failed one keeps its red line and gets the same buttons, because acting on it is
 * how the owner closes it. */
function ReminderRow({ r }: { r: ReminderItem }) {
  const act = useReminderAct();
  const undo = useReminderUndo();
  const [snoozing, setSnoozing] = useState(false);
  const busy = act.isPending;

  const run = (action: "done" | SnoozeFor) => {
    setSnoozing(false);
    act.mutate(
      { id: r.id, action, source: "chat_panel" },
      {
        onSuccess: (res) => {
          const msg =
            action === "done"
              ? `Done: ${r.text}`
              : `Snoozed until ${res.reminder.remind_at_local}`;
          toast.success(msg, {
            action: {
              label: "Undo",
              onClick: () =>
                undo.mutate(
                  { id: r.id, undo: res.undo },
                  {
                    onError: (err) =>
                      toast.error(err instanceof Error ? err.message : "could not undo that"),
                  },
                ),
            },
          });
        },
        onError: (err) =>
          toast.error(err instanceof Error ? err.message : "could not update the reminder"),
      },
    );
  };

  return (
    <div className="rounded border border-border bg-bg px-2.5 py-2" data-testid="active-reminder">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Link
          to={`/reminders/${encodeURIComponent(r.id)}`}
          className="min-w-0 break-words text-sm text-fg hover:underline"
        >
          {r.text}
        </Link>
        <span className="font-mono text-[10px] text-fg-subtle">
          {r.remind_at_local}
          {r.recurrence_label ? ` · ${r.recurrence_label}` : ""}
        </span>
      </div>
      {r.status === "failed" && <p className="mt-1 text-xs text-danger">Couldn't be delivered</p>}
      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        <Button type="button" size="sm" variant="outline" disabled={busy} onClick={() => run("done")}>
          Done
        </Button>
        <Button
          type="button"
          size="sm"
          variant="ghost"
          disabled={busy}
          aria-expanded={snoozing}
          onClick={() => setSnoozing((v) => !v)}
        >
          Snooze ▾
        </Button>
        {snoozing &&
          SNOOZES.map((s) => (
            <Button
              key={s.action}
              type="button"
              size="sm"
              variant="ghost"
              disabled={busy}
              onClick={() => run(s.action)}
            >
              {s.label}
            </Button>
          ))}
      </div>
    </div>
  );
}

function HistorySidebar({
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
    <aside className="hidden w-64 shrink-0 flex-col gap-2 md:flex">
      <Button
        type="button"
        variant="outline"
        size="sm"
        className="w-full justify-start"
        onClick={onNew}
      >
        <Plus /> New chat
      </Button>
      <div className="flex-1 space-y-1 overflow-y-auto rounded-lg border border-border bg-bg p-1.5">
        {isLoading && (
          <div className="p-3 text-center text-xs text-fg-subtle">
            Loading history…
          </div>
        )}
        {isError && (
          <div className="p-3 text-center text-xs text-danger">
            API unavailable — start it with <code>iris serve</code>.
          </div>
        )}
        {!isLoading && !isError && sessions.length === 0 && (
          <div className="p-3 text-center text-xs text-fg-subtle">
            No conversations yet.
          </div>
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
    </aside>
  );
}

export function ChatScreen() {
  const params = useParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [messages, setMessages] = useState<ChatMsgView[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  // Waiting on a turn this view lost the stream of (see FOLLOW_POLL_MS).
  const [following, setFollowing] = useState(false);
  // When this view last saw its own turn end: a status read from before then that
  // says "in progress" is about that turn, not a lost one.
  const lastTurnEndRef = useRef(0);
  const [sentHistory, setSentHistory] = useState<string[]>([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [draftBeforeHistory, setDraftBeforeHistory] = useState("");
  const [session, setActiveSession] = useState(
    () => params.sessionId || getSessionId(),
  );
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  // The session id whose history we've already loaded into `messages`. Prevents a
  // background refetch from clobbering live (just-sent) messages.
  const hydratedRef = useRef<string | null>(null);
  // Async-Activity completion → in-chat notice. A long FileManager job (categorize/
  // cleanup/organize) submitted from this session posts its result back "here" when
  // it finishes. We watch the polled activity feed and append the result_summary of
  // any newly-terminal activity for this session — deduped by id, and primed per
  // session so activities already reflected in hydrated history don't double-post.
  const shownActivityRef = useRef<Set<string>>(new Set());
  const activityPrimedRef = useRef<string | null>(null);
  const [searchParams, setSearchParams] = useSearchParams();
  const [duePromptShown, setDuePromptShown] = useState(false);
  const [showDueChooser, setShowDueChooser] = useState(false);
  const [showOutstandingDetails, setShowOutstandingDetails] = useState(false);
  const [showResolvedDues, setShowResolvedDues] = useState(false);
  const [selectedDueIds, setSelectedDueIds] = useState<string[]>([]);
  const { data: outstanding } = useOutstandingItems(showResolvedDues);
  const { data: outstandingWithResolved } = useOutstandingItems(true);
  const { data: chatStatus, dataUpdatedAt: chatStatusAt } = useChatStatus(session);
  const { data: slashCommands = [] } = useSlashCommands();
  const setTaskStatus = useUpdateTaskStatus();
  const setDueStatus = useUpdateDueStatus();
  const createTask = useCreateTask();
  const [addingTask, setAddingTask] = useState(false);
  const [newTaskTitle, setNewTaskTitle] = useState("");

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  // Follow deep links / back-forward: when the URL's :sessionId changes, switch.
  useEffect(() => {
    const target = params.sessionId;
    if (target && target !== session) {
      setActiveSession(target);
      persistSession(target);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.sessionId]);

  // Replay the active session's past turns once (until we switch sessions).
  const { data: history } = useSessionMessages(session);
  useEffect(() => {
    if (!history || hydratedRef.current === session) return;
    hydratedRef.current = session;
    setMessages(historyToViews(history));
  }, [history, session]);

  // Follow a turn this view lost the stream of, until the server says it ended; then
  // show the session as recorded, which now holds the answer.
  useEffect(() => {
    if (!following) return;
    let stopped = false;
    let timer: number | undefined;
    const started = Date.now();
    const tick = async () => {
      const status = await getChatStatus(session);
      if (stopped) return;
      if (status && !status.turn_in_progress) {
        const past = await listSessionMessages(session);
        if (stopped) return;
        hydratedRef.current = session;
        setMessages(historyToViews(past));
        setFollowing(false);
        setBusy(false);
        lastTurnEndRef.current = Date.now();
        qc.invalidateQueries({ queryKey: ["sessions"] });
        qc.invalidateQueries({ queryKey: ["session-messages", session] });
        qc.invalidateQueries({ queryKey: ["chat-status", session] });
        return;
      }
      if (Date.now() - started > FOLLOW_LIMIT_MS) {
        setFollowing(false);
        setBusy(false);
        setMessages((ms) => [
          ...ms,
          {
            id: nextId(),
            role: "assistant",
            text: "That is taking too long to finish. Check Activity, or ask again.",
            status: "error",
          },
        ]);
        return;
      }
      // An undefined status is the API out of reach (the phone may still be waking
      // up): keep trying.
      timer = window.setTimeout(() => void tick(), FOLLOW_POLL_MS);
    };
    void tick();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
    };
  }, [following, session, qc]);

  // Came back to a session whose turn is still running (the app was reloaded, or
  // woke from the background): wait for it rather than show a half-finished chat.
  useEffect(() => {
    if (!chatStatus?.turn_in_progress || busy || following) return;
    if (chatStatusAt <= lastTurnEndRef.current) return;
    setBusy(true);
    setFollowing(true);
  }, [chatStatus, chatStatusAt, busy, following]);

  // Post async-Activity completions into this session's chat (see refs above).
  const { data: activityData } = useActivities();
  useEffect(() => {
    if (!activityData) return;
    const mine = (activityData.activities ?? []).filter(
      (a) => a.origin === `chat:${session}`,
    );
    // First pass for a session: everything already terminal is assumed present in
    // the hydrated history — mark seen, don't re-post.
    if (activityPrimedRef.current !== session) {
      for (const a of mine) shownActivityRef.current.add(a.id);
      activityPrimedRef.current = session;
      return;
    }
    for (const a of mine) {
      const terminal = a.status === "completed" || a.status === "failed";
      if (!terminal || shownActivityRef.current.has(a.id)) continue;
      shownActivityRef.current.add(a.id);
      const text =
        a.status === "completed"
          ? a.result_summary || `${a.title} — done.`
          : `I couldn't finish "${a.title}": ${a.error || "unknown error"}`;
      setMessages((ms) => [...ms, { id: `act-${a.id}`, role: "assistant", text, status: "done" }]);
    }
  }, [activityData, session]);

  // Prefill the composer from a deep link (e.g. the health banner's
  // "Re-authenticate"); the user reviews and sends, so the agent runs the
  // remediation under the normal flow. Consume the param once.
  useEffect(() => {
    const ask = searchParams.get("ask");
    if (!ask) return;
    setInput(ask);
    const nextParams = new URLSearchParams(searchParams);
    nextParams.delete("ask");
    setSearchParams(nextParams, { replace: true });
  }, [searchParams, setSearchParams]);

  const patch = useCallback(
    (id: string, fn: (m: ChatMsgView) => ChatMsgView) => {
      setMessages((ms) => ms.map((m) => (m.id === id ? fn(m) : m)));
    },
    [],
  );

  const send = useCallback(
    async (override?: string) => {
      const text = (override ?? input).trim();
      if (!text || busy) return;

      if (override === undefined) {
        setSentHistory((items) => (items[items.length - 1] === text ? items : [...items, text]));
        setHistoryIndex(-1);
        setDraftBeforeHistory("");
      }

      if (text.startsWith("/")) {
        const botId = nextId();
        setMessages((ms) => [
          ...ms,
          { id: nextId(), role: "user", text },
          { id: botId, role: "assistant", text: "", status: "streaming" },
        ]);
        if (override === undefined) setInput("");
        setBusy(true);

        try {
          const result = await dispatchSlashCommand(text, session);
          if (result.action === "clear_messages") {
            setMessages([]);
            return;
          }
          if (result.action === "new_session") {
            const id = newSession();
            hydratedRef.current = id;
            setActiveSession(id);
            setMessages([]);
            setInput("");
            setHistoryIndex(-1);
            setDraftBeforeHistory("");
            navigate(`/chat/${id}`);
            return;
          }
          patch(botId, (m) => ({
            ...m,
            text: result.output,
            status: "done",
            activity: undefined,
            meta: {
              intent: "slash_command",
              agentType: "command",
            },
          }));
          qc.invalidateQueries({ queryKey: ["sessions"] });
          qc.invalidateQueries({ queryKey: ["chat-status", session] });
        } catch (e) {
          patch(botId, (m) => ({
            ...m,
            status: "error",
            activity: undefined,
            text: `Command failed: ${String(e)}`,
          }));
        } finally {
          setBusy(false);
          abortRef.current = null;
        }
        return;
      }

      const botId = nextId();
      setMessages((ms) => [
        ...ms,
        { id: nextId(), role: "user", text },
        { id: botId, role: "assistant", text: "", status: "streaming" },
      ]);
      if (override === undefined) setInput("");
      setBusy(true);

      const ctrl = new AbortController();
      abortRef.current = ctrl;
      let acc = "";
      // Set by "done" or "error": a stream that ends without either was cut off.
      let ended = false;
      let lost = false;
      const loseStream = () => {
        // The server keeps going; wait for it (see FOLLOW_POLL_MS). The live bubble
        // goes: the follower shows its own, then the recorded answer.
        lost = true;
        setMessages((ms) => ms.filter((m) => m.id !== botId));
        setFollowing(true);
      };

      try {
        await streamChat(
          { message: text, sessionId: session, signal: ctrl.signal },
          {
            onToken: (t) => {
              acc += t;
              patch(botId, (m) => ({ ...m, text: acc }));
            },
            onActivity: (t) => patch(botId, (m) => ({ ...m, activity: t })),
            onDone: (d) => {
              ended = true;
              patch(botId, (m) => ({
                ...m,
                text: d.response || acc,
                status: d.has_errors ? "error" : "done",
                activity: undefined,
                meta: {
                  intent: d.intent || undefined,
                  agentType: d.agent_type || undefined,
                  sources: d.sources,
                  traceId:
                    typeof d.metadata.trace_id === "string"
                      ? d.metadata.trace_id
                      : undefined,
                  pendingConfirmation: pendingConfirmationFrom(d.metadata),
                },
              }));
              // The turn is now in the logs — refresh the history rail.
              qc.invalidateQueries({ queryKey: ["sessions"] });
            },
            onError: (e) => {
              ended = true;
              patch(botId, (m) => ({
                ...m,
                status: "error",
                activity: undefined,
                text: m.text || `Error: ${e}`,
              }));
            },
          },
        );
        if (!ended && !ctrl.signal.aborted) loseStream();
      } catch (e) {
        if (ctrl.signal.aborted) {
          patch(botId, (m) => ({
            ...m,
            status: "stopped",
            activity: undefined,
          }));
        } else {
          // A network failure mid-turn ("TypeError: Load failed" when iOS suspends
          // the app) is a lost stream, not a failed turn.
          loseStream();
        }
      } finally {
        abortRef.current = null;
        if (!lost) {
          setBusy(false);
          lastTurnEndRef.current = Date.now();
        }
      }
    },
    [input, busy, session, patch, qc, navigate],
  );

  const recallHistory = useCallback(
    (direction: "up" | "down") => {
      if (sentHistory.length === 0) return false;
      if (direction === "up") {
        if (historyIndex === -1) {
          setDraftBeforeHistory(input);
          setHistoryIndex(sentHistory.length - 1);
          setInput(sentHistory[sentHistory.length - 1] ?? "");
          return true;
        }
        if (historyIndex > 0) {
          const nextIndex = historyIndex - 1;
          setHistoryIndex(nextIndex);
          setInput(sentHistory[nextIndex] ?? "");
          return true;
        }
        return false;
      }

      if (historyIndex === -1) return false;
      if (historyIndex < sentHistory.length - 1) {
        const nextIndex = historyIndex + 1;
        setHistoryIndex(nextIndex);
        setInput(sentHistory[nextIndex] ?? "");
        return true;
      }
      setHistoryIndex(-1);
      setInput(draftBeforeHistory);
      return true;
    },
    [draftBeforeHistory, historyIndex, input, sentHistory],
  );

  const dueItems = outstanding?.dues ?? [];
  const hasResolvedDues = (outstandingWithResolved?.dues ?? []).some((d) => d.status !== "open");
  const openDueItems = dueItems.filter((d) => d.status === "open");
  const taskItems = outstanding?.tasks ?? [];
  const reminderItems = outstanding?.reminders ?? [];
  const hasOutstanding = (outstanding?.count ?? 0) > 0;

  useEffect(() => {
    if (dueItems.length === 0) {
      if (showDueChooser) setShowDueChooser(false);
      if (selectedDueIds.length > 0) setSelectedDueIds([]);
      return;
    }
    const next = selectedDueIds.filter((id) => dueItems.some((d) => d.id === id));
    if (next.length === selectedDueIds.length && next.every((id, i) => id === selectedDueIds[i])) {
      return;
    }
    setSelectedDueIds(next);
  }, [dueItems, selectedDueIds, showDueChooser]);

  useEffect(() => {
    if (!showOutstandingDetails) setShowDueChooser(false);
  }, [showOutstandingDetails]);

  const submitDueReminderSelection = useCallback(() => {
    const picked = dueItems.filter((d) => selectedDueIds.includes(d.id));
    if (picked.length === 0) return;
    const lines = picked.map((d) => {
      const amount = d.amount ? `${d.currency} ${d.amount}` : `${d.currency} amount unknown`;
      const due = d.due_date ? ` (due ${shortDate(d.due_date)})` : "";
      const overdue = d.overdue ? " (OVERDUE)" : "";
      return `- ${d.label}: ${amount}${due}${overdue}`;
    });
    setDuePromptShown(true);
    setShowDueChooser(false);
    void send(
      [
        "Set reminders for these outstanding dues:",
        ...lines,
        "Ask me the date/time for each if missing, then remember this preference for future dues.",
      ].join("\n"),
    );
  }, [dueItems, selectedDueIds, send]);

  // Stop means stop the turn: closing the stream alone no longer does.
  const stop = useCallback(() => {
    abortRef.current?.abort();
    void cancelChat(session);
    if (following) {
      setFollowing(false);
      setBusy(false);
      lastTurnEndRef.current = Date.now();
    }
  }, [session, following]);

  // Leaving a session only stops listening; its turn still finishes and is recorded.
  const startNew = useCallback(() => {
    abortRef.current?.abort();
    setFollowing(false);
    setBusy(false);
    const id = newSession();
    hydratedRef.current = id; // a fresh session has no history to replay
    setActiveSession(id);
    setMessages([]);
    setInput("");
    navigate(`/chat/${id}`);
  }, [navigate]);

  const selectSession = useCallback(
    (id: string) => {
      if (id === session) return;
      abortRef.current?.abort();
      setFollowing(false);
      setBusy(false);
      hydratedRef.current = null; // force a re-hydrate from the chosen session
      setActiveSession(id);
      persistSession(id);
      setMessages([]);
      setHistoryIndex(-1);
      setDraftBeforeHistory("");
      navigate(`/chat/${id}`);
    },
    [session, navigate],
  );

  return (
    /* Fills exactly what the shell has left — the history list and the message
       list inside are `flex-1 overflow-y-auto`, so they need a parent whose
       height is definite, not merely bounded below. The fixed `100dvh - 7.5rem`
       this replaced guessed the chrome above it and was 28px short once the
       target badge and the health banner joined the shell. */
    <div className="mx-auto flex h-full min-h-0 w-full max-w-6xl gap-3">
      <HistorySidebar
        active={session}
        onSelect={selectSession}
        onNew={startNew}
      />

      <div className="flex min-w-0 flex-1 flex-col gap-3">
        <div className="flex items-center justify-between gap-2">
          <span className="font-mono text-[11px] text-fg-subtle">
            session {session}
          </span>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={startNew}
            disabled={!messages.length}
          >
            <RotateCcw /> New chat
          </Button>
        </div>

        {/* One line on a phone, where it wrapped to three and cost 62px.
            `session` is dropped from it below `md`: the row directly above
            already shows the id, so the old layout printed it twice on the
            same screen. Scrolls sideways in its own box rather than wrapping,
            so a long model name cannot grow it back. */}
        <div className="rounded-lg border border-border bg-surface px-2.5 py-1 md:px-3 md:py-2">
          <div className="flex items-center gap-x-2 overflow-x-auto whitespace-nowrap font-mono text-[10.5px] text-fg-subtle [scrollbar-width:none] md:flex-wrap md:gap-x-4 md:gap-y-1 md:overflow-visible md:whitespace-normal md:text-[11px]">
            <span>IRIS v{chatStatus?.iris_version ?? "-"}</span>
            <span className="hidden md:inline">session {session}</span>
            <span>{chatStatus?.provider ?? "-"}</span>
            <span>{chatStatus?.model ?? "-"}</span>
            <span>
              context {chatStatus?.window?.current_tokens ?? 0}/{chatStatus?.window?.budget_tokens ?? 0}
            </span>
            <span>({Math.round(chatStatus?.window?.fill_pct ?? 0)}%)</span>
          </div>
        </div>

        {/* Always rendered, unlike before: the panel used to appear only when
            something was outstanding, which would have hidden "Add task"
            exactly when the list was empty and you most wanted it. With
            nothing outstanding it collapses to the one control. */}
        {/* On a phone this keeps only its controls: the LIST moved to the
            header bell, which counts these together with health alerts — one
            number for "what wants me" rather than two badges disagreeing.
            Add task stays, because the bell is a read surface and this is the
            only place on the phone that creates a task. */}
        <section className="rounded-lg border border-border bg-surface p-2.5 md:p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="hidden md:block">
                <p className="text-sm font-semibold text-fg">
                  {hasOutstanding
                    ? `Outstanding items (${outstanding?.count ?? 0})`
                    : "Nothing outstanding"}
                </p>
                <p className="text-xs text-fg-subtle">
                  Track dues, follow-ups, and reminders here.
                </p>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  type="button"
                  size="sm"
                  variant={hasOutstanding ? "ghost" : "outline"}
                  onClick={() => setAddingTask((v) => !v)}
                >
                  {addingTask ? "Cancel" : "Add task"}
                </Button>
                {hasOutstanding && (
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="hidden md:inline-flex"
                  onClick={() => setShowOutstandingDetails((v) => !v)}
                >
                  {showOutstandingDetails ? "Hide details" : "Show details"}
                </Button>
                )}
                {!duePromptShown && dueItems.length > 0 && showOutstandingDetails && (
                  <>
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      setShowDueChooser(true);
                      if (selectedDueIds.length === 0) {
                        setSelectedDueIds(openDueItems.slice(0, 3).map((d) => d.id));
                      }
                    }}
                    disabled={openDueItems.length === 0}
                  >
                    Choose dues for reminder
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    onClick={() => {
                      setDuePromptShown(true);
                      void send("No reminder for dues right now.");
                    }}
                  >
                    Not now
                  </Button>
                  </>
                )}
                {showOutstandingDetails && (dueItems.length > 0 || hasResolvedDues) && (
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    onClick={() => setShowResolvedDues((v) => !v)}
                  >
                    {showResolvedDues ? "Hide resolved" : "Show resolved"}
                  </Button>
                )}
              </div>
            </div>

            {addingTask && (
              <form
                className="mt-3 flex flex-wrap items-center gap-2"
                onSubmit={(e) => {
                  e.preventDefault();
                  const title = newTaskTitle.trim();
                  if (!title) return;
                  createTask.mutate(
                    { title },
                    {
                      onSuccess: () => {
                        setNewTaskTitle("");
                        setAddingTask(false);
                        toast.success("Task added");
                      },
                      onError: (err) =>
                        toast.error(err instanceof Error ? err.message : "could not add the task"),
                    },
                  );
                }}
              >
                <input
                  autoFocus
                  value={newTaskTitle}
                  onChange={(e) => setNewTaskTitle(e.target.value)}
                  placeholder="What needs doing?"
                  maxLength={500}
                  aria-label="New task"
                  className="min-h-[44px] min-w-0 flex-1 rounded-md border border-border-strong bg-bg px-3 text-sm text-fg placeholder:text-fg-subtle focus:outline-none focus:ring-1 focus:ring-ring sm:min-h-[36px]"
                />
                <Button type="submit" size="sm" disabled={!newTaskTitle.trim() || createTask.isPending}>
                  {createTask.isPending ? "Adding…" : "Add"}
                </Button>
              </form>
            )}

            {hasOutstanding && showOutstandingDetails && (
              // Desktop only; the phone reads the same items in the bell sheet.
              <div className="mt-3 hidden max-h-64 space-y-3 overflow-y-auto pr-1 md:block">
                {showDueChooser && openDueItems.length > 0 && (
                  <div className="rounded border border-border bg-bg p-2.5">
                <p className="text-xs font-medium text-fg">Select due items for reminder setup</p>
                <div className="mt-2 space-y-1.5">
                  {openDueItems.slice(0, 8).map((d) => {
                    const checked = selectedDueIds.includes(d.id);
                    return (
                      <label
                        key={d.id}
                        className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 hover:bg-surface"
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={(e) => {
                            const on = e.currentTarget.checked;
                            setSelectedDueIds((ids) =>
                              on ? [...ids, d.id] : ids.filter((id) => id !== d.id),
                            );
                          }}
                        />
                        <span className="text-xs text-fg">
                          {d.label}
                          {d.due_date ? (
                            <span className="ml-1 font-mono text-[10px] text-fg-subtle">
                              ({d.currency} {d.amount ?? "amount unknown"} · due {shortDate(d.due_date)})
                            </span>
                          ) : null}
                        </span>
                      </label>
                    );
                  })}
                </div>
                <div className="mt-2 flex flex-wrap items-center gap-2">
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    disabled={selectedDueIds.length === 0}
                    onClick={submitDueReminderSelection}
                  >
                    Ask IRIS for selected ({selectedDueIds.length})
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    onClick={() => setSelectedDueIds(openDueItems.map((d) => d.id))}
                  >
                    Select all
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    onClick={() => setSelectedDueIds([])}
                  >
                    Clear
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="ghost"
                    onClick={() => setShowDueChooser(false)}
                  >
                    Cancel
                  </Button>
                </div>
                  </div>
                )}

                {dueItems.length > 0 && (
                  <div className="space-y-2">
                    {dueItems.slice(0, 6).map((d) => (
                  <div
                    key={d.id}
                    className="rounded border border-border bg-bg px-2.5 py-2"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p className="text-sm text-fg">{d.label}</p>
                      <span className="font-mono text-[10px] text-fg-subtle">
                        {d.currency} {d.amount ?? "amount unknown"}
                        {d.due_date ? ` · due ${shortDate(d.due_date)}` : ""}
                        {d.overdue ? " · overdue" : ""}
                        {d.status !== "open" ? ` · ${d.status}` : ""}
                      </span>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        disabled={setDueStatus.isPending || d.status !== "open"}
                        onClick={() =>
                          setDueStatus.mutate({ id: d.id, status: "resolved" })
                        }
                      >
                        Completed
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        disabled={setDueStatus.isPending || d.status !== "open"}
                        onClick={() =>
                          setDueStatus.mutate({ id: d.id, status: "dismissed" })
                        }
                      >
                        Dismiss
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        disabled={d.status !== "open"}
                        onClick={() =>
                          void send(
                            `Set a reminder for this outstanding due: ${d.label}${d.due_date ? ` (due ${d.due_date})` : ""}. Ask me for date and time if missing.`,
                          )
                        }
                      >
                        Set reminder
                      </Button>
                    </div>
                  </div>
                    ))}
                  </div>
                )}

                {taskItems.length > 0 && (
                  <div className="space-y-2">
                    {taskItems.slice(0, 6).map((t) => (
                  <div
                    key={t.id}
                    className="rounded border border-border bg-bg px-2.5 py-2"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p className="text-sm text-fg">{t.title}</p>
                      <span className="font-mono text-[10px] text-fg-subtle">
                        {t.source_kind ?? "task"}
                        {t.due_at ? ` · due ${shortDate(t.due_at)}` : ""}
                      </span>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2">
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        disabled={setTaskStatus.isPending}
                        onClick={() => setTaskStatus.mutate({ id: t.id, status: "done" })}
                      >
                        Completed
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        disabled={setTaskStatus.isPending}
                        onClick={() => setTaskStatus.mutate({ id: t.id, status: "doing" })}
                      >
                        Not completed
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        onClick={() =>
                          void send(
                            `Set a reminder for this outstanding item: ${t.title}. Ask me for date and time if missing.`,
                          )
                        }
                      >
                        Set reminder
                      </Button>
                    </div>
                  </div>
                    ))}
                  </div>
                )}

                {reminderItems.length > 0 && (
                  <div className="space-y-2 border-t border-border pt-3">
                    <p className="text-xs font-medium uppercase tracking-wide text-fg-subtle">
                      Active reminders
                    </p>
                    {reminderItems.slice(0, 6).map((r) => (
                      <ReminderRow key={r.id} r={r} />
                    ))}
                  </div>
                )}
              </div>
            )}
        </section>

        <div className="flex-1 space-y-3 overflow-y-auto rounded-lg border border-border bg-bg p-3">
          {messages.length === 0 ? (
            <EmptyState />
          ) : (
            messages.map((m) => (
              <ChatMessage
                key={m.id}
                msg={m}
                sessionId={session}
                onAction={(t) => {
                  patch(m.id, (x) => ({
                    ...x,
                    meta: { ...x.meta, pendingConfirmation: undefined },
                  }));
                  void send(t);
                }}
              />
            ))
          )}
          {following && (
            <ChatMessage
              msg={{
                id: "following",
                role: "assistant",
                text: "",
                status: "streaming",
                activity: "Still working on it. The answer will appear here.",
              }}
              sessionId={session}
              onAction={() => undefined}
            />
          )}
          <div ref={endRef} />
        </div>

        <Composer
          value={input}
          onChange={(next) => {
            setInput(next);
            setHistoryIndex(-1);
          }}
          onSend={() => void send()}
          onStop={stop}
          busy={busy}
          slashCommands={slashCommands}
          onRecallHistory={recallHistory}
        />
      </div>
    </div>
  );
}
