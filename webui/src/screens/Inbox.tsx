/* Inbox → Judged (loop-proof PR 5): every email the judge sorted, and the place to
 * change one.
 *
 * Each row: the sender, the subject, its bucket, how sure the judge was, and the
 * figures it read. "Change" lists the other buckets (and Promo); a pick moves the row
 * at once, posts to /api/v1/email/judgments/<id>/bucket, and a toast offers Undo,
 * which posts the previous bucket back. The same server path the Action Center card,
 * chat and a Gmail relabel use: the Gmail label moves and the judge learns.
 *
 * The bucket names and their order come from the API (judge.yaml), never from here;
 * the badge colours follow that order. */
import { useState } from "react";
import { toast } from "sonner";
import { Tag } from "@/components/Tag";
import { ApiUnavailable } from "@/components/control/parts";
import { isUnreachable } from "@/lib/http";
import {
  useJudgments,
  useSetJudgmentBucket,
  type BucketOption,
  type JudgedEmail,
} from "@/lib/judgments";

type TagKind = "res" | "opp" | "ok" | "bad" | "warn" | "info";
/** Badge colours by the bucket's position in the API's list. */
const PALETTE: TagKind[] = ["warn", "info", "bad", "ok", "opp", "res"];

const CHIP =
  "min-h-11 rounded-full border px-4 text-sm transition-colors disabled:opacity-50";
const BUTTON =
  "min-h-11 rounded-lg border border-border bg-bg px-4 text-sm text-fg hover:bg-accent disabled:opacity-50";

function badgeKind(buckets: BucketOption[], key: string | null): TagKind {
  const i = buckets.findIndex((b) => b.key === key);
  return i < 0 ? "info" : PALETTE[i % PALETTE.length];
}

function nameOf(buckets: BucketOption[], key: string | null): string {
  return buckets.find((b) => b.key === key)?.name ?? key ?? "—";
}

/** "min due 35.00 · due 2026-10-12". */
function figuresText(figures: Record<string, unknown>): string {
  return Object.entries(figures ?? {})
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `${k.replace(/_/g, " ")} ${typeof v === "object" ? JSON.stringify(v) : String(v)}`)
    .join(" · ");
}

function Row({
  row,
  buckets,
  onChange,
  busy,
}: {
  row: JudgedEmail;
  buckets: BucketOption[];
  onChange: (row: JudgedEmail, bucket: BucketOption) => void;
  busy: boolean;
}) {
  const [open, setOpen] = useState(false);
  const figures = figuresText(row.figures);
  const corrected = row.owner_bucket && row.judge_bucket && row.owner_bucket !== row.judge_bucket;
  return (
    <li className="rounded-xl border border-border bg-surface p-3" data-testid="judged-row">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold text-fg">{row.sender || row.from_address}</p>
          <p className="break-words text-sm text-fg-muted">{row.subject}</p>
        </div>
        <span className="shrink-0" data-testid="bucket">
          <Tag kind={badgeKind(buckets, row.bucket)}>{row.bucket_name}</Tag>
        </span>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-fg-subtle">
        {row.confidence !== null && <span>confidence {row.confidence.toFixed(2)}</span>}
        {corrected && <span>judge said {nameOf(buckets, row.judge_bucket)}</span>}
        {figures && <span className="min-w-0 break-all">{figures}</span>}
        <button
          type="button"
          className={`${BUTTON} ml-auto`}
          aria-haspopup="menu"
          aria-expanded={open}
          disabled={busy}
          onClick={() => setOpen((v) => !v)}
        >
          Change
        </button>
      </div>
      {open && (
        <div role="menu" aria-label={`Move “${row.subject}” to`} className="mt-2 flex flex-wrap gap-2">
          {buckets
            .filter((b) => b.key !== row.bucket)
            .map((b) => (
              <button
                key={b.key}
                type="button"
                role="menuitem"
                className={BUTTON}
                onClick={() => {
                  setOpen(false);
                  onChange(row, b);
                }}
              >
                {b.name}
              </button>
            ))}
        </div>
      )}
    </li>
  );
}

export function InboxScreen() {
  const [filter, setFilter] = useState<string | null>(null);
  const { data, isLoading, error } = useJudgments(filter);
  const move = useSetJudgmentBucket();
  const buckets = data?.buckets ?? [];
  const rows = data?.judgments ?? [];

  /** Move one email. `undoTo` is where Undo puts it back; null for an Undo itself. */
  const send = (messageId: string, bucket: BucketOption, undoTo: BucketOption | null) => {
    move.mutate(
      { messageId, bucket: bucket.key, name: bucket.name },
      {
        onSuccess: () => {
          if (!undoTo) {
            toast.success(`Moved back to ${bucket.name}`);
            return;
          }
          toast.success(`Moved to ${bucket.name} — its Gmail label follows`, {
            action: { label: "Undo", onClick: () => send(messageId, undoTo, null) },
          });
        },
        onError: (err) =>
          toast.error(err instanceof Error ? err.message : "Couldn't change that email"),
      },
    );
  };

  const onChange = (row: JudgedEmail, bucket: BucketOption) =>
    send(row.message_id, bucket, row.bucket ? { key: row.bucket, name: row.bucket_name } : null);

  return (
    <div className="mx-auto w-full max-w-3xl">
      <div role="tablist" aria-label="Inbox" className="mb-3 flex gap-2 border-b border-border">
        <span
          role="tab"
          aria-selected="true"
          className="-mb-px inline-flex min-h-11 items-center border-b-2 border-primary px-3 text-sm font-semibold text-fg"
        >
          Judged
        </span>
      </div>
      <p className="mb-3 text-sm text-fg-muted">
        Every email IRIS sorted, newest first. Change one and its Gmail label moves, and
        IRIS judges mail like it from that sender the same way next time.
      </p>
      <div className="mb-4 flex flex-wrap gap-2" aria-label="Filter by bucket">
        {[{ key: "", name: "All" }, ...buckets].map((b) => {
          const on = (filter ?? "") === b.key;
          return (
            <button
              key={b.key || "all"}
              type="button"
              aria-pressed={on}
              onClick={() => setFilter(b.key || null)}
              className={`${CHIP} ${
                on ? "border-primary bg-primary text-bg" : "border-border bg-bg text-fg hover:bg-accent"
              }`}
            >
              {b.name}
            </button>
          );
        })}
      </div>
      {isLoading ? (
        <p className="text-sm text-fg-muted">Loading…</p>
      ) : isUnreachable(error) ? (
        <ApiUnavailable />
      ) : error ? (
        <p className="text-sm text-danger">
          Couldn't load the judged emails{error instanceof Error ? `: ${error.message}` : ""}.
        </p>
      ) : rows.length === 0 ? (
        <p className="text-sm text-fg-muted" data-testid="judged-empty">
          {filter ? `Nothing in ${nameOf(buckets, filter)}.` : "Nothing judged yet."}
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {rows.map((row) => (
            <Row
              key={row.message_id}
              row={row}
              buckets={buckets}
              onChange={onChange}
              busy={move.isPending}
            />
          ))}
        </ul>
      )}
    </div>
  );
}
