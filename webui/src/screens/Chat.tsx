import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  ChatMessage,
  type ChatMsgView,
  type PendingConfirmation,
} from "@/components/chat/ChatMessage";
import { Composer } from "@/components/chat/Composer";
import { ChatMoreMenu } from "@/components/chat/ChatMoreMenu";
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
  requestWelcome,
  setSession as persistSession,
  streamChat,
} from "@/lib/chat";
import {
  useActivities,
  useChatStatus,
  useOutstandingItems,
  useSessionMessages,
  useSlashCommands,
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

  // The first chat on this install opens on IRIS's welcome (ADR-0127). The harness
  // decides whether it is due and runs it; Chat only asks when it opens and, when this
  // was the call that ran it, shows the welcome's session, so the first question
  // continues it.
  useEffect(() => {
    let current = true;
    void requestWelcome().then((welcome) => {
      if (!current || !welcome?.created) return;
      persistSession(welcome.session_id);
      hydratedRef.current = welcome.session_id;
      setActiveSession(welcome.session_id);
      setMessages([
        {
          id: nextId(),
          role: "assistant",
          text: welcome.response,
          status: "done",
          meta: welcome.trace_id ? { traceId: welcome.trace_id } : undefined,
        },
      ]);
      qc.invalidateQueries({ queryKey: ["sessions"] });
      navigate(`/chat/${welcome.session_id}`, { replace: true });
    });
    return () => {
      current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Follow deep links, back-forward, and the nav sidebar's session list (which
  // only navigates -- it has no access to this screen's own state): when the
  // URL's :sessionId changes to a different session, switch to it the same way
  // an in-screen control used to (this absorbed what was a separate
  // `selectSession` callback, so every caller of a session switch is now just
  // "change the URL" and gets the same reset).
  useEffect(() => {
    const target = params.sessionId;
    if (!target || target === session) return;
    abortRef.current?.abort();
    setFollowing(false);
    setBusy(false);
    hydratedRef.current = null; // force a re-hydrate from the chosen session
    setActiveSession(target);
    persistSession(target);
    setMessages([]);
    setHistoryIndex(-1);
    setDraftBeforeHistory("");
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

  return (
    /* Fills exactly what the shell has left — the history list and the message
       list inside are `flex-1 overflow-y-auto`, so they need a parent whose
       height is definite, not merely bounded below. The fixed `100dvh - 7.5rem`
       this replaced guessed the chrome above it and was 28px short once the
       target badge and the health banner joined the shell. */
    <div className="mx-auto flex h-full min-h-0 w-full max-w-6xl flex-col gap-3">
        {/* Phone-only: everything below (session id, New chat, version/context,
            Add task) moved here so the message list gets the screen. Desktop
            keeps all of it inline below -- there's room, and a wide screen
            reads a row faster than it reads a tap. Session history itself
            moved further still, into the "Chat" nav entry (App.tsx's
            NavList) -- not a phone/desktop split, just not this screen's
            column to own once it's reachable from the sidebar on any width. */}
        <div className="flex items-center justify-between gap-2 md:hidden">
          <span className="min-w-0 truncate font-mono text-[11px] text-fg-subtle">
            IRIS v{chatStatus?.iris_version ?? "-"} · {chatStatus?.model ?? "-"}
          </span>
          <ChatMoreMenu
            session={session}
            chatStatus={chatStatus}
            onNewChat={startNew}
            newChatDisabled={!messages.length}
            addingTask={addingTask}
            onToggleAddingTask={() => setAddingTask((v) => !v)}
            newTaskTitle={newTaskTitle}
            onNewTaskTitleChange={setNewTaskTitle}
            onSubmitTask={(title) =>
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
              )
            }
            creatingTask={createTask.isPending}
          />
        </div>

        <div className="hidden items-center justify-between gap-2 md:flex">
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

        <div className="hidden rounded-lg border border-border bg-surface px-3 py-2 md:block">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mono text-[11px] text-fg-subtle">
            <span>IRIS v{chatStatus?.iris_version ?? "-"}</span>
            <span>session {session}</span>
            <span>{chatStatus?.provider ?? "-"}</span>
            <span>{chatStatus?.model ?? "-"}</span>
            <span>
              context {chatStatus?.window?.current_tokens ?? 0}/{chatStatus?.window?.budget_tokens ?? 0}
            </span>
            <span>({Math.round(chatStatus?.window?.fill_pct ?? 0)}%)</span>
          </div>
        </div>

        {/* Desktop only now: the phone's "Add task" and the outstanding list
            both live in ChatMoreMenu above (the list via the existing global
            AttentionBell, which already covers this on a phone -- see its own
            comment -- not duplicated here). */}
        <section className="hidden rounded-lg border border-border bg-surface p-2.5 md:block md:p-3">
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
  );
}
