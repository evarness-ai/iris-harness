/* Action Center (ADR-0073) — the unified pending-actions inbox. Persisted agent
 * actions (finance, …) unioned with live Health items, over GET /actions. A thin
 * renderer; raising/resolving happens in the harness. Display-only actions show a
 * copyable command (secrets stay local CLI); safe actions get a button when
 * writes are enabled. */
import { useState } from "react";
import { Link } from "react-router-dom";
import { RotateCcw, ThumbsDown } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Tag } from "@/components/Tag";
import { CopyBlock } from "@/components/CopyBlock";
import { Section } from "@/components/layout";
import { ConfirmDialog, type ConfirmState } from "@/components/ConfirmDialog";
import { QueryState, fmtDateTime } from "@/components/control/parts";
import {
  useActions,
  useFinanceSenders,
  useInvokeAction,
  usePendingApprovals,
  useProposeSender,
  useRespondToApproval,
  useSubmitSurfaceFeedback,
  useWritesEnabled,
} from "@/lib/queries";
import type {
  ActionAnswer,
  ApprovalCard,
  PendingAction,
  PendingApproval,
} from "@/lib/control";

function originTone(origin: PendingAction["origin"]): "info" | "warn" {
  return origin === "health" ? "warn" : "info";
}

/* "Not useful" feedback (issue 0028) — records a learning signal so IRIS stops
 * surfacing items like this one. Not write-gated (telemetry only); on success the
 * actions list refetches and the now-suppressed item drops out. */
function NotUsefulButton({ a }: { a: PendingAction }) {
  const feedback = useSubmitSurfaceFeedback();
  if (!a.feedback_ref) return null;
  const ref = a.feedback_ref;
  const onClick = () =>
    feedback.mutate(
      { ref, verdict: "not_useful" },
      {
        onSuccess: () => toast.success("Got it — I won't surface this again."),
        onError: (e) => toast.error(e instanceof Error ? e.message : "feedback failed"),
      },
    );
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={feedback.isPending}
      aria-label="Not useful — stop surfacing items like this"
      className="inline-flex min-h-[44px] items-center gap-1 rounded px-2 py-0.5 text-[11px] text-fg-subtle hover:bg-bg hover:text-fg disabled:opacity-50 sm:min-h-0 sm:px-1.5"
    >
      <ThumbsDown className="h-3 w-3" />
      Not useful
    </button>
  );
}

/* A run the evaluator halted, waiting on a human (HITL approval queue). It also
 * appears in the pending-actions list below as a display-only item, because every
 * channel reads that — the difference here is the buttons, which go through
 * /governance/approvals/{id}/respond. Approving continues the run from its
 * checkpoint, so the answer comes back in `detail` and lands in the chat session the
 * run belongs to. Write-gated like every other server-side mutation in this UI. */
function ApprovalRow({
  a,
  canWrite,
  onRespond,
}: {
  a: PendingApproval;
  canWrite: boolean;
  onRespond: (a: PendingApproval, status: "approved" | "rejected") => void;
}) {
  return (
    <div className="rounded-lg border border-warning/40 bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Tag kind="warn">approval</Tag>
        <span className="text-sm font-medium text-fg">{a.signal}</span>
        <span className="font-mono text-[10.5px] text-fg-subtle">run {a.run_id.slice(0, 8)}…</span>
        {/* `sm:ml-auto`, not `ml-auto`: when the row wraps on a phone the
            pushed-right timestamp lands alone on its own line, reading as a
            separate item. Wrapped inline it stays part of the header. */}
        <span className="font-mono text-[11px] text-fg-subtle sm:ml-auto">
          {a.overdue ? "expired" : "times out"} {fmtDateTime(a.timeout_at)}
        </span>
      </div>
      <p className="mt-1 text-xs text-fg-muted">{a.context_summary}</p>
      <p className="mt-1 text-[11px] text-fg-subtle">
        {a.overdue
          ? // Its window has closed and the next sweep will retire it. Saying so beats
            // offering buttons that are about to start failing.
            "The window closed with no answer — this is about to be retired unanswered."
          : a.resumable
            ? "Approving continues this run from where it stopped."
            : "No checkpoint is linked, so approving records the decision only."}
      </p>
      {canWrite ? (
        <div className="mt-2 flex gap-2">
          <Button type="button" size="sm" onClick={() => onRespond(a, "approved")}>
            Approve
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => onRespond(a, "rejected")}
          >
            Reject
          </Button>
        </div>
      ) : (
        <div className="mt-2">
          <CopyBlock label="run locally" text={`iris approvals approve ${a.approval_id}`} />
        </div>
      )}
    </div>
  );
}

/** "answer within 58 min" / "answer within 2 h" — the window left, not a timestamp. */
function timeLeft(timeoutAt: string): string {
  const minutes = Math.max(0, Math.round((new Date(timeoutAt).getTime() - Date.now()) / 60000));
  return minutes >= 120 ? `${Math.round(minutes / 60)} h` : `${minutes} min`;
}

/** Whether it can be undone, in words (ADR-0118 decision 3). */
function undoSentence(card: ApprovalCard): string {
  if (!card.undo_tool) return "This cannot be undone.";
  const window = card.undo_window_days ? ` for ${card.undo_window_days} days` : "";
  return `Reversible${window}. Say "undo that" to restore.`;
}

/* A destructive tool call waiting on the owner (ADR-0118 step 4, prototype signed off
 * 2026-09-21). It deletes or overwrites data, so the card says what, in the plugin's
 * words, whether it can be undone, and what the owner asked for — and approving is a
 * red button naming the action, behind a confirm sheet. The exact call is one tap
 * away, because that, not the wording, is what approving runs. */
function DestructiveApprovalRow({
  a,
  canWrite,
  onRespond,
}: {
  a: PendingApproval;
  canWrite: boolean;
  onRespond: (a: PendingApproval, status: "approved" | "rejected") => void;
}) {
  const [showCall, setShowCall] = useState(false);
  const card = a.card ?? { title: a.signal, lines: [], undo_tool: null, undo_window_days: null, asked: null };
  const reversible = Boolean(card.undo_tool);
  return (
    <div className="space-y-3 rounded-xl border border-danger/45 bg-surface p-3.5">
      <div className="flex items-center gap-2">
        <Tag kind="bad">{card.effect === "write" ? "acts for you" : "deletes data"}</Tag>
        <span className="ml-auto text-xs text-fg-subtle">
          {a.overdue ? "expired" : `answer within ${timeLeft(a.timeout_at)}`}
        </span>
      </div>
      <div className="text-[19px] font-semibold leading-tight text-fg">{card.title}</div>
      {card.lines.length > 0 && (
        <ul className="divide-y divide-border overflow-hidden rounded-lg border border-border">
          {card.lines.map((line, i) => (
            <li key={i} className="break-words px-3 py-2.5 text-sm text-fg">
              {line}
            </li>
          ))}
        </ul>
      )}
      <div
        className={`flex items-start gap-2.5 rounded-lg px-3 py-2.5 text-[13px] leading-snug ${
          reversible ? "bg-success/10 text-fg-muted" : "bg-danger/10 text-danger"
        }`}
      >
        <RotateCcw className={`mt-0.5 h-4 w-4 shrink-0 ${reversible ? "text-success" : ""}`} />
        <span>{undoSentence(card)}</span>
      </div>
      {card.asked && (
        <p className="text-[13px] text-fg-muted">
          You asked: <i className="break-words">"{card.asked}"</i> ·{" "}
          <Link to="/chat" className="text-info hover:underline">
            open chat
          </Link>
        </p>
      )}
      <button
        type="button"
        onClick={() => setShowCall((v) => !v)}
        className="min-h-[44px] text-[13px] text-info hover:underline"
        aria-expanded={showCall}
      >
        {showCall ? "Hide the exact call" : "Show the exact call"}
      </button>
      {showCall && (
        <pre className="whitespace-pre-wrap break-all rounded-md border border-border bg-bg p-2.5 font-mono text-[11px] text-fg-muted">
          {a.items.map((i) => `${i.tool} ${JSON.stringify(i.args)}`).join("\n")}
        </pre>
      )}
      {a.overdue ? (
        <p className="text-[11px] text-fg-subtle">
          The window closed with no answer. Nothing was changed, and this is about to be retired.
        </p>
      ) : canWrite ? (
        <div className="flex gap-2.5">
          <Button
            type="button"
            variant="outline"
            className="h-12 flex-1 text-[15px]"
            onClick={() => onRespond(a, "rejected")}
          >
            Reject
          </Button>
          <Button
            type="button"
            variant="destructive"
            className="h-12 flex-1 text-[15px] font-semibold"
            onClick={() => onRespond(a, "approved")}
          >
            {card.title}
          </Button>
        </div>
      ) : (
        <CopyBlock label="run locally" text={`iris approvals approve ${a.approval_id}`} />
      )}
    </div>
  );
}

/* A choice card (ADR-0121, prototype signed off 2026-09-24): one card per question
 * ("Is this your Woodgrove bank account?") with what IRIS read, the evidence, an option the
 * owner can change (the account type), and the answers. Answering is one tap — Ignore
 * is undoable from "Not proposed", and Yes adds a record rather than deleting one. */
function ChoiceCardRow({
  a,
  canWrite,
  busy,
  onAnswer,
}: {
  a: PendingAction;
  canWrite: boolean;
  busy: boolean;
  onAnswer: (a: PendingAction, answer: ActionAnswer) => void;
}) {
  const { action } = a;
  const options = action.options ?? null;
  const [picked, setPicked] = useState<string | null>(options?.default ?? null);
  const card = action.card ?? null;
  const choices = action.choices ?? [];
  return (
    <div className="space-y-3 rounded-xl border border-info/45 bg-surface p-3.5">
      <div className="flex flex-wrap items-center gap-2">
        {card?.tag && <Tag kind="info">{card.tag}</Tag>}
        {a.created_at && (
          <span className="font-mono text-[11px] text-fg-subtle sm:ml-auto">
            {fmtDateTime(a.created_at)}
          </span>
        )}
      </div>
      <div className="text-[18px] font-semibold leading-tight text-fg">{a.title}</div>
      {a.description && <p className="text-xs text-fg-muted">{a.description}</p>}
      {(card?.facts.length || options) && (
        <div className="divide-y divide-border overflow-hidden rounded-lg border border-border">
          {card?.facts.map((f) => (
            <div key={f.label} className="grid grid-cols-[96px_1fr] gap-2 px-3 py-2 text-[13px]">
              <span className="text-fg-subtle">{f.label}</span>
              <span className="break-words text-fg">{f.value}</span>
            </div>
          ))}
          {options && (
            <div className="grid grid-cols-[96px_1fr] gap-2 px-3 py-2 text-[13px]">
              <span className="text-fg-subtle">{options.label}</span>
              <div className="flex flex-wrap gap-1.5" role="group" aria-label={options.label}>
                {options.values.map((v) => (
                  <button
                    key={v.value}
                    type="button"
                    aria-pressed={picked === v.value}
                    onClick={() => setPicked(v.value)}
                    className={`min-h-[32px] rounded-full border px-2.5 text-[12.5px] ${
                      picked === v.value
                        ? "border-primary bg-primary font-semibold text-primary-fg"
                        : "border-border-strong bg-surface text-fg hover:bg-bg"
                    }`}
                  >
                    {v.label}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      {card && card.evidence.length > 0 && (
        <div>
          <div className="mb-1 text-[12px] text-fg-subtle">{card.evidence_label}</div>
          <ul className="divide-y divide-dashed divide-border">
            {card.evidence.map((e, i) => (
              <li key={i} className="break-words py-1.5 text-[12.5px] text-fg">
                {e.when && (
                  <span className="mr-1.5 font-mono text-[11px] text-fg-subtle">{e.when}</span>
                )}
                {e.text}
              </li>
            ))}
          </ul>
        </div>
      )}
      {card?.note && (
        <p className="rounded-lg bg-success/10 px-3 py-2.5 text-[12.5px] leading-snug text-fg-muted">
          {card.note}
        </p>
      )}
      {canWrite ? (
        <div className="flex gap-2.5">
          {choices.map((c) => {
            const blocked = c.needs_option && !picked;
            return (
              <Button
                key={c.value}
                type="button"
                variant={c.primary ? "default" : "outline"}
                className={`h-12 flex-1 text-[15px] ${c.primary ? "font-semibold" : ""}`}
                disabled={busy || blocked}
                title={blocked ? `Pick a ${options?.label.toLowerCase() ?? "value"} first` : undefined}
                onClick={() =>
                  onAnswer(a, { choice: c.value, option: c.needs_option ? picked : null })
                }
              >
                {c.label}
              </Button>
            );
          })}
        </div>
      ) : (
        <span className="text-[11px] text-fg-subtle">
          Enable writes (IRIS_WEBUI_ALLOW_WRITES=1) to answer this here, or answer it in chat.
        </span>
      )}
    </div>
  );
}

/* Senders the finance sweep saw but did not propose, and ones the owner ignored
 * (ADR-0121). "Add anyway" / "Undo ignore" opens the sender's card now. Hidden when the
 * finance plugin is not mounted (the route 404s). */
function SkippedSenders({ canWrite }: { canWrite: boolean }) {
  const { data, isError } = useFinanceSenders();
  const propose = useProposeSender();
  // Defensive on shape: an older server (or none) must never take the Action Center down.
  const skipped = Array.isArray(data?.not_proposed) ? data.not_proposed : [];
  const waiting = typeof data?.waiting === "number" ? data.waiting : 0;
  if (isError || skipped.length === 0) return null;
  const add = (domain: string) =>
    propose.mutate(domain, {
      onSuccess: (r) => toast.success(r.result),
      onError: (e) => toast.error(e instanceof Error ? e.message : "could not add it"),
    });
  return (
    <Section title={`Senders not proposed (${skipped.length})`}>
      <details className="rounded-lg border border-border bg-surface">
        <summary className="min-h-[44px] cursor-pointer px-3 py-2.5 text-[13px] text-fg-muted">
          Senders IRIS saw in your mail but didn't ask about, and ones you ignored
          {waiting > 0 ? ` · ${waiting} more waiting to be asked` : ""}
        </summary>
        <ul className="divide-y divide-border border-t border-border">
          {skipped.map((s) => (
            <li key={s.domain} className="flex items-start gap-3 px-3 py-2.5">
              <div className="min-w-0 flex-1">
                <div className="break-all font-mono text-[12px] text-fg">{s.domain}</div>
                <div className="text-[12px] text-fg-muted">
                  {s.email_count} email{s.email_count === 1 ? "" : "s"} ·{" "}
                  {s.status === "ignored" ? "you chose Ignore" : s.reason || s.verdict}
                </div>
              </div>
              {canWrite && (
                <button
                  type="button"
                  onClick={() => add(s.domain)}
                  disabled={propose.isPending}
                  className="min-h-[36px] shrink-0 text-[12.5px] text-info hover:underline disabled:opacity-50"
                >
                  {s.status === "ignored" ? "Undo ignore" : "Add anyway"}
                </button>
              )}
            </li>
          ))}
        </ul>
      </details>
    </Section>
  );
}

function ActionRow({
  a,
  canWrite,
  onInvoke,
}: {
  a: PendingAction;
  canWrite: boolean;
  onInvoke: (a: PendingAction) => void;
}) {
  const { action } = a;
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Tag kind={originTone(a.origin)}>{a.origin}</Tag>
        <span className="text-sm font-medium text-fg">{a.title}</span>
        <span className="font-mono text-[10.5px] text-fg-subtle">{a.source_kind}</span>
        {a.created_at && (
          <span className="font-mono text-[11px] text-fg-subtle sm:ml-auto">
            {fmtDateTime(a.created_at)}
          </span>
        )}
      </div>
      {a.description && <p className="mt-1 text-xs text-fg-muted">{a.description}</p>}

      {action.kind === "copy_command" && action.command ? (
        // Display-only: the user runs it locally (secrets never cross the web UI).
        <CopyBlock label="run locally" text={action.command} />
      ) : action.safe ? (
        <div className="mt-2">
          {canWrite ? (
            <Button type="button" size="sm" variant="outline" onClick={() => onInvoke(a)}>
              {action.label}
            </Button>
          ) : (
            <span className="text-[11px] text-fg-subtle">
              {action.label} — enable writes (IRIS_WEBUI_ALLOW_WRITES=1) to run this
            </span>
          )}
        </div>
      ) : null}

      {a.feedback_ref && (
        <div className="mt-2 flex justify-end">
          <NotUsefulButton a={a} />
        </div>
      )}
    </div>
  );
}

export function ActionCenterScreen() {
  const { data, isLoading, isError } = useActions();
  const approvalsQuery = usePendingApprovals();
  const canWrite = useWritesEnabled();
  const invoke = useInvokeAction();
  const respond = useRespondToApproval();
  const [confirm, setConfirm] = useState<ConfirmState | null>(null);
  const approvals = approvalsQuery.data?.approvals ?? [];
  // Approvals reach GET /actions too — deliberately, so chat, the CLI and the
  // pending_actions tool all see that a run is waiting. This screen renders them in
  // its own section above, with buttons, so they are dropped from the inbox below
  // rather than listed twice.
  const actions = (data?.actions ?? []).filter((a) => a.origin !== "approval");

  const answer = async (a: PendingApproval, status: "approved" | "rejected") => {
    const title = a.card?.title ?? a.signal;
    try {
      const outcome = await respond.mutateAsync({ id: a.approval_id, status });
      toast.success(
        status === "approved"
          ? outcome.resumed
            ? `${title}: done. IRIS replied in the chat.`
            : `${title}: approved. ${outcome.detail}`
          : "Rejected. Nothing was changed.",
      );
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "could not answer the approval");
      throw e;
    }
  };

  const onRespond = (a: PendingApproval, status: "approved" | "rejected") => {
    if (a.kind === "destructive") {
      // Rejecting changes nothing, so it needs no second tap; approving deletes, so it
      // is restated in a red sheet that names exactly what will run.
      if (status === "rejected") {
        void answer(a, status).catch(() => undefined);
        return;
      }
      const title = a.card?.title ?? a.signal;
      setConfirm({
        title: `${title}?`,
        description: `IRIS will run exactly the call on the card and nothing else. ${
          a.card ? undoSentence(a.card) : ""
        }`.trim(),
        confirmLabel: title,
        destructive: true,
        run: () => answer(a, status),
      });
      return;
    }
    setConfirm({
      title: status === "approved" ? "Approve this run?" : "Reject this run?",
      description:
        status === "approved" && a.resumable
          ? `${a.context_summary} — the run will continue from where it stopped.`
          : a.context_summary,
      confirmLabel: status === "approved" ? "Approve" : "Reject",
      run: async () => {
        try {
          const outcome = await respond.mutateAsync({ id: a.approval_id, status });
          // The answer of a resumed run can be long, and it is also in the chat
          // session — so the toast reports what happened, not the whole thing.
          toast.success(
            outcome.resumed ? "Approved — the run continued." : `${status}: ${outcome.detail}`,
          );
        } catch (e) {
          toast.error(e instanceof Error ? e.message : "could not answer the approval");
          throw e;
        }
      },
    });
  };

  const onAnswer = (a: PendingAction, answer: ActionAnswer) =>
    invoke.mutate(
      { id: a.id, answer },
      {
        onSuccess: (res) => toast.success(res.result),
        onError: (e) => toast.error(e instanceof Error ? e.message : "could not answer"),
      },
    );

  const onInvoke = (a: PendingAction) =>
    setConfirm({
      title: a.action.label + "?",
      description: a.title,
      confirmLabel: a.action.label,
      run: async () => {
        try {
          const res = await invoke.mutateAsync(a.id);
          toast.success(res.result);
        } catch (e) {
          toast.error(e instanceof Error ? e.message : "action failed");
          throw e;
        }
      },
    });

  return (
    <div className="space-y-6">
      {/* Above the inbox: a halted run is more urgent than anything the harness
       * merely noticed, and it is the only item here that something is waiting on. */}
      {approvals.length > 0 && (
        <Section title={`Waiting on you (${approvals.length})`}>
          <div className="space-y-2">
            {approvals.map((a) =>
              a.kind === "destructive" ? (
                <DestructiveApprovalRow
                  key={a.approval_id}
                  a={a}
                  canWrite={canWrite}
                  onRespond={onRespond}
                />
              ) : (
                <ApprovalRow
                  key={a.approval_id}
                  a={a}
                  canWrite={canWrite}
                  onRespond={onRespond}
                />
              ),
            )}
          </div>
        </Section>
      )}
      <Section title={`Pending actions (${actions.length})`}>
        <QueryState
          loading={isLoading}
          error={isError}
          empty={actions.length === 0}
          emptyText="Nothing needs your attention — background ticks will surface blockers here."
        >
          <div className="space-y-2">
            {actions.map((a) =>
              a.action.choices && a.action.choices.length > 0 ? (
                <ChoiceCardRow
                  key={a.id}
                  a={a}
                  canWrite={canWrite}
                  busy={invoke.isPending}
                  onAnswer={onAnswer}
                />
              ) : (
                <ActionRow key={a.id} a={a} canWrite={canWrite} onInvoke={onInvoke} />
              ),
            )}
          </div>
        </QueryState>
      </Section>
      <SkippedSenders canWrite={canWrite} />
      <ConfirmDialog state={confirm} onClose={() => setConfirm(null)} />
    </div>
  );
}
