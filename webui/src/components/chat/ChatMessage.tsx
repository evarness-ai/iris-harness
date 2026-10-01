import { useState } from "react";
import { Link } from "react-router-dom";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import { cn } from "@/lib/utils";
import { sendFeedback } from "@/lib/chat";
import { useFeedbackEnabled } from "@/lib/queries";
import { Markdown } from "./Markdown";

export interface PendingConfirmation {
  id: string;
  kind?: string;
  summary?: string;
  detail?: string;
  options: string[];
}

export interface ChatMeta {
  intent?: string;
  agentType?: string;
  sources?: string[];
  traceId?: string;
  pendingConfirmation?: PendingConfirmation;
}

export type ChatStatus = "streaming" | "done" | "error" | "stopped";

export interface ChatMsgView {
  id: string;
  role: "user" | "assistant";
  text: string;
  status?: ChatStatus;
  activity?: string;
  meta?: ChatMeta;
}

/** Approve/reject control for an in-chat confirmation (ADR-0076). Channel-agnostic:
 * each button just sends the plain decision text the harness already understands. */
function ConfirmationControls({
  pc,
  onAction,
}: {
  pc: PendingConfirmation;
  onAction: (text: string) => void;
}) {
  const opts = pc.options.length ? pc.options : ["approve", "reject"];
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-border pt-2">
      {opts.map((opt) => {
        const approve = opt.toLowerCase() === "approve";
        return (
          <button
            key={opt}
            type="button"
            onClick={() => onAction(opt)}
            className={cn(
              "rounded px-2.5 py-1 text-xs font-medium",
              approve
                ? "bg-primary text-white hover:bg-primary/90"
                : "border border-border bg-bg text-fg hover:bg-surface",
            )}
          >
            {opt.charAt(0).toUpperCase() + opt.slice(1)}
          </button>
        );
      })}
    </div>
  );
}

function Pill({ children }: { children: React.ReactNode }) {
  return (
    <span className="rounded bg-primary/10 px-1.5 py-0.5 font-medium text-primary">
      {children}
    </span>
  );
}

function MessageFooter({ meta }: { meta: ChatMeta }) {
  const sources = meta.sources ?? [];
  const has =
    meta.intent || meta.agentType || sources.length > 0 || meta.traceId;
  if (!has) return null;
  return (
    <div className="mt-2 flex flex-wrap items-center gap-1.5 border-t border-border pt-2 text-[11px]">
      {meta.intent && <Pill>{meta.intent}</Pill>}
      {meta.agentType && meta.agentType !== meta.intent && (
        <Pill>{meta.agentType}</Pill>
      )}
      {sources.map((s) => (
        <span
          key={s}
          className="rounded bg-bg px-1.5 py-0.5 font-mono text-fg-subtle"
        >
          {s}
        </span>
      ))}
      {meta.traceId && (
        <Link
          to={`/calltrace/${meta.traceId}`}
          className="ml-auto font-medium text-primary hover:underline"
        >
          view trace →
        </Link>
      )}
    </div>
  );
}

/** Non-blocking answer feedback (ADR-0072): thumbs + "not what I meant". Only shown
 * when IRIS_FEEDBACK_CAPTURE is on; one reaction per message, never required. */
function MessageFeedback({
  msg,
  sessionId,
}: {
  msg: ChatMsgView;
  sessionId?: string;
}) {
  const enabled = useFeedbackEnabled();
  const [given, setGiven] = useState<"up" | "down" | null>(null);
  if (!enabled) return null;

  const submit = (sentiment: "up" | "down", note?: string) => {
    setGiven(sentiment);
    void sendFeedback({
      sentiment,
      note,
      sessionId,
      traceId: msg.meta?.traceId,
      intent: msg.meta?.intent,
      agentType: msg.meta?.agentType,
    });
  };

  if (given) {
    return (
      <div className="mt-2 text-[11px] text-fg-subtle">Thanks — noted.</div>
    );
  }
  return (
    <div className="mt-2 flex items-center gap-2 text-[11px] text-fg-subtle">
      <span>Was this helpful?</span>
      <button
        type="button"
        aria-label="Helpful"
        onClick={() => submit("up")}
        className="rounded p-1 hover:bg-bg hover:text-fg"
      >
        <ThumbsUp className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        aria-label="Not helpful"
        onClick={() => submit("down")}
        className="rounded p-1 hover:bg-bg hover:text-fg"
      >
        <ThumbsDown className="h-3.5 w-3.5" />
      </button>
      <button
        type="button"
        onClick={() => submit("down", "not what I meant")}
        className="hover:text-fg hover:underline"
      >
        Not what I meant
      </button>
    </div>
  );
}

/** A single chat turn — teal-tinted for the user, surface card for IRIS. */
export function ChatMessage({
  msg,
  sessionId,
  onAction,
}: {
  msg: ChatMsgView;
  sessionId?: string;
  onAction?: (text: string) => void;
}) {
  const isUser = msg.role === "user";
  const err = msg.status === "error";
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[85%] rounded-lg border px-3.5 py-2.5 text-sm leading-relaxed",
          isUser
            ? "border-primary/30 bg-primary/10 text-fg"
            : err
              ? "border-danger/40 bg-danger/10 text-fg"
              : "border-border bg-surface text-fg",
        )}
      >
        {isUser ? (
          <div className="whitespace-pre-wrap break-words">{msg.text}</div>
        ) : (
          <div>
            {msg.text && <Markdown text={msg.text} />}
            {msg.status === "streaming" && (
              <span className="ml-0.5 inline-block animate-pulse text-primary">
                ▍
              </span>
            )}
          </div>
        )}
        {msg.status === "streaming" && msg.activity && (
          <div className="mt-1.5 font-mono text-[11px] text-fg-subtle">
            · {msg.activity}
          </div>
        )}
        {msg.status === "stopped" && (
          <div className="mt-1.5 text-[11px] text-fg-subtle">stopped</div>
        )}
        {!isUser && msg.status === "done" && (
          <>
            {msg.meta?.pendingConfirmation && onAction && (
              <ConfirmationControls
                pc={msg.meta.pendingConfirmation}
                onAction={onAction}
              />
            )}
            {msg.meta && <MessageFooter meta={msg.meta} />}
            <MessageFeedback msg={msg} sessionId={sessionId} />
          </>
        )}
      </div>
    </div>
  );
}
