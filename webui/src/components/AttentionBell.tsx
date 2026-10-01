/* One count for everything wanting the owner (phone only).
 *
 * On a phone the health banner and the outstanding-items panel cost ~140px of
 * a 664px screen between them, on the screen where vertical room matters most.
 * They also competed: two badges, neither saying which mattered more.
 *
 * So both collapse into one bell in the header. The count is health alerts
 * plus outstanding items, because from the owner's side a revoked credential
 * and an unpaid bill are the same question — what wants me? The sheet keeps
 * them apart, alerts first, since only one of those can be IRIS failing.
 *
 * Desktop keeps the banner and the panel: there is room, and a wide screen
 * reads a row faster than it reads a tap.
 */
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { Link } from "react-router-dom";
import { AlertTriangle, Bell } from "lucide-react";
import { useHealth, useOutstandingItems } from "@/lib/queries";

/** More than this and the badge would widen the header. */
const MAX_BADGE = 99;

interface Row {
  key: string;
  title: string;
  detail: string;
  alert: boolean;
}

function useAttention(): { rows: Row[]; alerts: number } {
  const alerts = useHealth().data?.alerts ?? [];
  const outstanding = useOutstandingItems();

  const rows: Row[] = alerts.map((a, i) => ({
    key: `alert:${a.target}:${i}`,
    title: a.target,
    detail: a.detail,
    alert: true,
  }));

  const data = outstanding.data;
  for (const due of data?.dues ?? []) {
    rows.push({
      key: `due:${due.id}`,
      title: due.label,
      detail: due.due_date ? `due ${due.due_date}` : "due",
      alert: false,
    });
  }
  for (const task of data?.tasks ?? []) {
    rows.push({ key: `task:${task.id}`, title: task.title, detail: "task", alert: false });
  }
  for (const reminder of data?.reminders ?? []) {
    rows.push({
      key: `reminder:${reminder.id}`,
      title: reminder.text,
      detail:
        reminder.status === "failed"
          ? `due ${reminder.remind_at_local}, couldn't be delivered`
          : reminder.remind_at_local,
      alert: false,
    });
  }
  return { rows, alerts: alerts.length };
}

export function AttentionBell() {
  const [open, setOpen] = useState(false);
  const { rows, alerts } = useAttention();
  const count = rows.length;

  // Escape closes it, like every other dismissible surface in the console.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  if (count === 0) return null;

  // Red only when IRIS is actually failing. An unpaid bill in red would teach
  // the owner to ignore red, and then a revoked credential reads as routine.
  const urgent = alerts > 0;
  const shown = rows.slice(0, 6);

  return (
    <div className="md:hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label={`${count} ${count === 1 ? "thing wants" : "things want"} your attention`}
        className={`relative inline-flex h-11 w-11 items-center justify-center rounded-md border transition-colors ${
          urgent
            ? "border-danger/50 text-danger hover:bg-danger/10"
            : "border-border-strong text-fg-muted hover:bg-surface-raised"
        }`}
      >
        <Bell size={16} aria-hidden />
        <span
          aria-hidden
          className={`absolute -right-1 -top-1 min-w-[17px] rounded-full border-2 border-bg px-1 text-center text-[9.5px] font-bold leading-[17px] text-white ${
            urgent ? "bg-danger" : "bg-warning"
          }`}
        >
          {count > MAX_BADGE ? `${MAX_BADGE}+` : count}
        </span>
      </button>

      {/* Portalled to <body>, and not by preference.
       *
       * The header sets `backdrop-blur`, and `backdrop-filter` makes an
       * element a containing block for fixed-position descendants — so a
       * `fixed bottom-0` sheet rendered in place anchors to the HEADER and
       * lands at the top of the screen, over the content. */}
      {open &&
        createPortal(
          <>
            <button
              type="button"
              aria-label="Close"
              className="fixed inset-0 z-40 bg-overlay"
              onClick={() => setOpen(false)}
            />
            {/* A bottom sheet, not a dropdown under the header: this is a
                phone, and the thumb is at the bottom of it. */}
            <div className="fixed inset-x-0 bottom-0 z-50 rounded-t-2xl border-t border-border-strong bg-surface p-4 pb-[calc(1rem+env(safe-area-inset-bottom,0px))] shadow-e3">
              <div className="mb-2 flex items-center gap-2">
                <h2 className="text-sm font-semibold text-fg">
                  {count} {count === 1 ? "wants" : "want"} you
                </h2>
                {urgent && (
                  <span className="inline-flex items-center gap-1 text-[11px] font-medium text-danger">
                    <AlertTriangle size={12} aria-hidden /> {alerts} needs fixing
                  </span>
                )}
              </div>

              <ul className="max-h-[45dvh] space-y-0 overflow-y-auto">
                {shown.map((row) => (
                  <li
                    key={row.key}
                    className="flex items-baseline justify-between gap-3 border-t border-border py-2 first:border-t-0"
                  >
                    <span className="min-w-0 flex-[2] truncate text-xs text-fg">
                      {row.alert && <span className="mr-1.5 text-danger">●</span>}
                      {row.title}
                    </span>
                    {/* Bounded, not `shrink-0`: a long detail ("token revoked
                        — refresh returned invalid_grant") took the whole row
                        and crushed the target it described to "go…". The
                        name is the part that has to survive. */}
                    <span className="min-w-0 max-w-[45%] truncate font-mono text-[10.5px] text-fg-subtle">
                      {row.detail}
                    </span>
                  </li>
                ))}
              </ul>

              <div className="mt-3 flex gap-2">
                <Link
                  to="/actions"
                  onClick={() => setOpen(false)}
                  className="inline-flex min-h-[44px] flex-1 items-center justify-center rounded-md bg-primary px-3 text-xs font-medium text-primary-foreground"
                >
                  Open Activity
                </Link>
                {urgent && (
                  <Link
                    to="/health"
                    onClick={() => setOpen(false)}
                    className="inline-flex min-h-[44px] flex-1 items-center justify-center rounded-md border border-border-strong px-3 text-xs font-medium text-fg"
                  >
                    Open Pulse
                  </Link>
                )}
              </div>
            </div>
          </>,
          document.body,
        )}
    </div>
  );
}
