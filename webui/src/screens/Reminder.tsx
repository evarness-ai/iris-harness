/* One reminder, with Done and Snooze (loop-proof PR 3b, D14; prototype "iPhone sheet").
 *
 * Where a reminder's push notification lands (/reminders/<id>). iOS cannot show
 * buttons on a web push, so on the owner's iPhone a tap opens this and the buttons
 * live here; Chrome and Android get Done / Snooze 1h on the notification itself
 * (public/sw.js) and can still open this for the other choices.
 *
 * Works from a READ-ONLY paired device: the server allows exactly these POSTs
 * without control scope, so nothing here is hidden behind the writes flag.
 *
 * The buttons come from the reminder's own `actions`: a closed, expired or
 * cancelled reminder has none, and shows its state instead. After an action the
 * status line says what happened and Undo hands the server's undo token back.
 *
 * A bill's reminder (loop-proof PR 4, prototype "Push + sheet"): "💳 Discover" over
 * "$35.00 min due Mon Oct 13 · statement $1,284.50", ✅ Paid in place of Done, Not
 * yet on a "Did you pay?" question, then Remind me in 1 hour / Tomorrow 9:00 AM. Once
 * the bill is closed as paid — here, on Telegram, or by a payment email — it says
 * "✓ Paid — <who>" with a "Not paid — reopen" button. */
import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { toast } from "sonner";
import {
  ReminderGone,
  type ReminderAction,
  type ReminderActionResult,
  type ReminderBill,
  type ReminderDetail,
} from "@/lib/control";
import { useReminder, useReminderAct, useReminderUndo } from "@/lib/queries";

/** Order and wording of the sheet's buttons (the prototype's). */
const BUTTONS: { action: ReminderAction; label: string }[] = [
  { action: "done", label: "✅ Done" },
  { action: "10m", label: "Snooze 10 min" },
  { action: "1h", label: "Snooze 1 hour" },
  { action: "tomorrow_9am", label: "Tomorrow 9:00 AM" },
];

/** A bill's buttons (the PR 4 prototype's sheet), shown as its `actions` allow. */
const BILL_BUTTONS: { action: ReminderAction; label: string }[] = [
  { action: "paid", label: "✅ Paid" },
  { action: "not_yet", label: "Not yet" },
  { action: "1h", label: "Remind me in 1 hour" },
  { action: "tomorrow_9am", label: "Tomorrow 9:00 AM" },
];

/** "$35.00 min due Mon Oct 13 · statement $1,284.50". */
function billWhen(bill: ReminderBill): string {
  const owed = bill.amount
    ? `${bill.amount} ${bill.statement ? "min due" : "due"} ${bill.due_local}`
    : `due ${bill.due_local}`;
  return [owed.trim(), bill.statement ? `statement ${bill.statement}` : ""]
    .filter(Boolean)
    .join(" · ");
}

/** The heading: "💳 Discover", or the confirmation's own "✅ Discover marked paid". */
function billTitle(bill: ReminderBill): string {
  return bill.step === "paid" ? bill.headline : `💳 ${bill.entity}`;
}

/** A closed bill's line: who paid it, or that the owner said not yet. */
function billClosedLine(r: ReminderDetail, bill: ReminderBill): string {
  if (bill.paid_by) return `✓ Paid — ${bill.paid_by}`;
  if (bill.step === "paid") return "✓ Paid";
  if (bill.not_yet) return "Not paid yet — noted";
  return closedLine(r);
}

const LINK_CLASS = "inline-flex min-h-11 items-center text-primary underline-offset-2 hover:underline";

/** "8:03 AM" in the device's zone, for "Done at …". */
function clockNow(): string {
  return new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

/** What a reminder with nothing left to do says instead of buttons. */
function closedLine(r: ReminderDetail): string {
  switch (r.status) {
    case "done":
      return "✓ Done";
    case "cancelled":
      return "Cancelled";
    case "expired":
      return "Expired: it was missed and is no longer active";
    case "sent":
      return "Delivered";
    default:
      return "Nothing to do for this reminder";
  }
}

interface Acted {
  action: ReminderAction;
  line: string;
  result: ReminderActionResult;
}

export function ReminderScreen() {
  const { reminderId } = useParams();
  const { data, isLoading, error } = useReminder(reminderId);
  const act = useReminderAct();
  const undo = useReminderUndo();
  const [acted, setActed] = useState<Acted | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  if (isLoading) {
    return <p className="mx-auto max-w-md text-sm text-fg-muted">Loading…</p>;
  }
  if (error instanceof ReminderGone || !reminderId) {
    return (
      <div className="mx-auto max-w-md rounded-2xl border border-border bg-surface p-5">
        <h2 className="text-lg font-semibold text-fg">This reminder is gone</h2>
        <p className="mt-1 text-sm text-fg-muted">
          It was deleted, or the link is from somewhere else. Nothing needs doing here.
        </p>
        <Link to="/chat" className={LINK_CLASS}>
          Back to chat
        </Link>
      </div>
    );
  }
  if (error || !data) {
    return (
      <p className="mx-auto max-w-md text-sm text-danger">
        Couldn't load this reminder{error instanceof Error ? `: ${error.message}` : ""}. Try again
        in a moment.
      </p>
    );
  }

  const reminder = data;
  const bill = reminder.bill;
  const actions = new Set(reminder.actions ?? []);
  const choices = bill ? BILL_BUTTONS : BUTTONS;
  const buttons = acted ? [] : choices.filter((b) => actions.has(b.action));
  const busy = act.isPending || undo.isPending;
  const reopenId = !acted && bill ? (bill.reopen_id ?? null) : null;

  const lineFor = (action: ReminderAction, result: ReminderActionResult): string => {
    if (action === "done") return `✓ Done at ${clockNow()}`;
    if (action === "paid" && result.already_paid)
      return `✓ ${result.reminder.bill?.entity || "This bill"} is already marked paid`;
    if (action === "paid") return `✓ Paid — ${result.reminder.bill?.paid_by ?? "you"}`;
    if (action === "not_yet") {
      const next = result.next?.remind_at_local;
      return next
        ? `Noted — not paid yet. I'll ask again ${next}.`
        : "Noted — not paid yet. That was the last ask: it stays in the digest.";
    }
    return `⏰ Snoozed until ${result.reminder.remind_at_local}`;
  };

  const onAct = (action: ReminderAction) => {
    setFailure(null);
    // Paid is the reminder's Done; the rest are the snooze call's choices.
    const call = action === "paid" || action === "done" ? "done" : action;
    act.mutate(
      { id: reminder.id, action: call, source: "sheet" },
      {
        onSuccess: (result) => setActed({ action, line: lineFor(action, result), result }),
        onError: (err) => {
          const msg = err instanceof Error ? err.message : "could not update the reminder";
          setFailure(msg);
          toast.error(msg);
        },
      },
    );
  };

  const onReopen = () => {
    if (!reopenId) return;
    setFailure(null);
    undo.mutate(
      { id: reopenId, undo: { kind: "done", source: "sheet" } },
      {
        onSuccess: () => toast.success("Reopened — IRIS will remind you about this bill again."),
        onError: (err) => {
          const msg = err instanceof Error ? err.message : "could not reopen the bill";
          setFailure(msg);
          toast.error(msg);
        },
      },
    );
  };

  const onUndo = () => {
    if (!acted) return;
    setFailure(null);
    undo.mutate(
      { id: reminder.id, undo: acted.result.undo },
      {
        onSuccess: () => setActed(null),
        onError: (err) => {
          const msg = err instanceof Error ? err.message : "could not undo that";
          setFailure(msg);
          toast.error(msg);
        },
      },
    );
  };

  return (
    <div className="mx-auto w-full max-w-md">
      <article className="rounded-2xl border border-border bg-surface p-5" aria-live="polite">
        <p className="text-xs text-fg-subtle">{bill ? "IRIS · Bill" : "IRIS · Reminder"}</p>
        <h2 className="mt-1 break-words text-xl font-semibold text-fg">
          {bill ? billTitle(bill) : `⏰ ${reminder.text}`}
        </h2>
        <p className="mt-1 text-sm text-fg-muted">
          {bill ? (
            bill.step === "paid" ? bill.line : billWhen(bill)
          ) : (
            <>
              Due {reminder.remind_at_local}
              {reminder.recurrence_label ? ` · repeats ${reminder.recurrence_label}` : ""}
            </>
          )}
        </p>
        {reminder.status === "failed" && !acted && (
          <p className="mt-1 text-sm text-danger">Couldn't be delivered</p>
        )}

        {buttons.length > 0 && (
          <div className="mt-4 grid gap-2">
            {buttons.map((b) => (
              <button
                key={b.action}
                type="button"
                disabled={busy}
                onClick={() => onAct(b.action)}
                className={
                  b.action === "done" || b.action === "paid"
                    ? "min-h-12 rounded-xl border border-success bg-success px-4 text-left text-base font-semibold text-bg disabled:opacity-50"
                    : "min-h-12 rounded-xl border border-border bg-bg px-4 text-left text-base text-fg hover:bg-accent disabled:opacity-50"
                }
              >
                {b.label}
              </button>
            ))}
          </div>
        )}

        {acted ? (
          <div className="mt-4 flex flex-wrap items-center gap-3">
            <p
              data-testid="reminder-status"
              className={`text-sm ${
                acted.action === "done" || acted.action === "paid" ? "text-success" : "text-primary"
              }`}
            >
              {acted.line}
            </p>
            <button
              type="button"
              disabled={busy}
              onClick={onUndo}
              className="min-h-11 rounded-lg border border-border px-4 text-sm text-fg hover:bg-accent disabled:opacity-50"
            >
              Undo
            </button>
          </div>
        ) : (
          buttons.length === 0 && (
            <div className="mt-4 flex flex-wrap items-center gap-3">
              <p
                data-testid="reminder-status"
                className={`text-sm ${bill?.paid_by || bill?.step === "paid" ? "text-success" : "text-fg-muted"}`}
              >
                {bill ? billClosedLine(reminder, bill) : closedLine(reminder)}
              </p>
              {reopenId && (
                <button
                  type="button"
                  disabled={busy}
                  onClick={onReopen}
                  className="min-h-11 rounded-lg border border-border px-4 text-sm text-fg hover:bg-accent disabled:opacity-50"
                >
                  Not paid — reopen
                </button>
              )}
            </div>
          )
        )}

        {failure && <p className="mt-2 text-sm text-danger">{failure}</p>}
      </article>
      <p className="mt-3 text-xs text-fg-subtle">
        {bill ? "Paid and Not yet" : "Done and Snooze"} work from a read-only paired phone too.
      </p>
    </div>
  );
}
