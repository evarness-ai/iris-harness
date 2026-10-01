import { useMemo, useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Switch } from "@/components/Switch";
import { Tag } from "@/components/Tag";
import { CopyBlock } from "@/components/CopyBlock";
import { Section } from "@/components/layout";
import { ConfirmDialog, type ConfirmState } from "@/components/ConfirmDialog";
import { QueryState, fmtDateTime } from "@/components/control/parts";
import {
  useHeartbeatRuns,
  useHeartbeats,
  usePatchHeartbeat,
  useResetHeartbeat,
  useTriggerHeartbeat,
  useWritesEnabled,
} from "@/lib/queries";
import type {
  Heartbeat,
  HeartbeatJobStatus,
  HeartbeatRun,
} from "@/lib/control";

function runTone(status: string): "ok" | "bad" | "warn" | "info" {
  const s = status.toLowerCase();
  if (s.includes("ok") || s.includes("success") || s.includes("complete"))
    return "ok";
  if (s.includes("error") || s.includes("fail")) return "bad";
  if (s.includes("skip") || s.includes("pending") || s.includes("running"))
    return "warn";
  return "info";
}

// Loop-proof D13: did the most recent scheduled slot run? The server words it
// ("Missed 12:15 — last success 06:15"); the tag says it at a glance.
function jobTag(job: HeartbeatJobStatus): {
  kind: "ok" | "bad" | "warn" | "info";
  text: string;
} {
  if (job.missed) return { kind: "bad", text: "missed" };
  if (job.state === "red") return { kind: "bad", text: "failed" };
  if (job.state === "yellow") return { kind: "warn", text: "skipped" };
  if (job.state === "green") return { kind: "ok", text: "ran" };
  return { kind: "info", text: "waiting" };
}

function JobLine({ beat }: { beat: Heartbeat }) {
  const job = beat.job;
  if (!job || !beat.enabled) return null;
  const tag = jobTag(job);
  return (
    <div
      className="mt-2 flex flex-wrap items-center gap-2"
      data-testid={`beat-job-${beat.name}`}
    >
      <Tag kind={tag.kind}>{tag.text}</Tag>
      <span
        className={`min-w-0 text-xs [word-break:break-word] ${
          job.state === "red" ? "text-danger" : "text-fg"
        }`}
      >
        {job.detail}
      </span>
      {beat.watched && (
        <span className="text-[11px] text-fg-subtle">· alerts when missed</span>
      )}
    </div>
  );
}

// ── Schedules in words ───────────────────────────────────────────────────────
// The server's describe_schedule is the record (`schedule_text`); this mirror only
// previews an edit before it is saved. The server validates again on save.
const MIN_INTERVAL_SECONDS = 30;
const DAILY = /^(\d{1,2}) (\d{1,2}) \* \* \*$/;

type Mode = "every" | "daily" | "cron";
type Unit = "s" | "min" | "h";
const UNIT_SECONDS: Record<Unit, number> = { s: 1, min: 60, h: 3600 };

function describe(schedule: string): string {
  if (schedule.startsWith("interval:")) {
    const n = Number(schedule.slice(9));
    if (!Number.isInteger(n)) return schedule;
    if (n % 86400 === 0)
      return n === 86400 ? "every day" : `every ${n / 86400} days`;
    if (n % 3600 === 0)
      return n === 3600 ? "every hour" : `every ${n / 3600} hours`;
    if (n % 60 === 0) return n === 60 ? "every minute" : `every ${n / 60} min`;
    return `every ${n} s`;
  }
  const daily = schedule.match(DAILY);
  if (daily)
    return `daily at ${daily[2].padStart(2, "0")}:${daily[1].padStart(2, "0")}`;
  return `cron ${schedule}`;
}

interface Draft {
  mode: Mode;
  amount: number;
  unit: Unit;
  time: string;
  cron: string;
}

function draftFrom(schedule: string): Draft {
  const base: Draft = {
    mode: "every",
    amount: 10,
    unit: "min",
    time: "08:00",
    cron: schedule,
  };
  if (schedule.startsWith("interval:")) {
    const n = Number(schedule.slice(9));
    if (n % 3600 === 0) return { ...base, amount: n / 3600, unit: "h" };
    if (n % 60 === 0) return { ...base, amount: n / 60, unit: "min" };
    return { ...base, amount: n, unit: "s" };
  }
  const daily = schedule.match(DAILY);
  if (daily) {
    return {
      ...base,
      mode: "daily",
      time: `${daily[2].padStart(2, "0")}:${daily[1].padStart(2, "0")}`,
    };
  }
  return { ...base, mode: "cron" };
}

function scheduleFrom(d: Draft): { schedule: string; error: string | null } {
  if (d.mode === "every") {
    const seconds = Math.round(d.amount * UNIT_SECONDS[d.unit]);
    if (!Number.isFinite(seconds) || seconds < MIN_INTERVAL_SECONDS) {
      return {
        schedule: "",
        error: `Shortest allowed is ${MIN_INTERVAL_SECONDS} seconds.`,
      };
    }
    return { schedule: `interval:${seconds}`, error: null };
  }
  if (d.mode === "daily") {
    const [h, m] = d.time.split(":").map(Number);
    if (!Number.isInteger(h) || !Number.isInteger(m)) {
      return { schedule: "", error: "Pick a time." };
    }
    return { schedule: `${m} ${h} * * *`, error: null };
  }
  const cron = d.cron.trim().split(/\s+/).join(" ");
  if (cron.split(" ").length !== 5) {
    return {
      schedule: "",
      error: "A cron needs 5 fields: min hour day month weekday.",
    };
  }
  return { schedule: cron, error: null };
}

// ── Pieces ───────────────────────────────────────────────────────────────────

function Segmented<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { label: string; value: T }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex overflow-hidden rounded-lg border border-border">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={`flex-1 px-3 py-2 text-sm transition-colors ${
            o.value === value
              ? "bg-primary/15 font-semibold text-primary"
              : "text-fg-muted"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

function EditSchedule({
  beat,
  timezone,
  onClose,
}: {
  beat: Heartbeat | null;
  timezone: string | null;
  onClose: () => void;
}) {
  const patch = usePatchHeartbeat();
  const [draft, setDraft] = useState<Draft | null>(null);
  const [serverError, setServerError] = useState<string | null>(null);
  const current = beat ? (draft ?? draftFrom(beat.schedule)) : null;
  const built = current ? scheduleFrom(current) : null;

  const close = () => {
    setDraft(null);
    setServerError(null);
    onClose();
  };
  const set = (change: Partial<Draft>) => {
    if (!current) return;
    setServerError(null);
    setDraft({ ...current, ...change });
  };
  const save = async () => {
    if (!beat || !built || built.error) return;
    if (built.schedule === beat.schedule) return close();
    try {
      const saved = await patch.mutateAsync({
        name: beat.name,
        patch: { schedule: built.schedule },
      });
      toast.success(
        `${beat.name}: ${saved.schedule_text ?? describe(saved.schedule)}`,
      );
      close();
    } catch (e) {
      setServerError(e instanceof Error ? e.message : "save failed");
    }
  };

  return (
    <Dialog
      open={beat !== null}
      onOpenChange={(open) => !open && !patch.isPending && close()}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="font-mono">{beat?.name}</DialogTitle>
          {beat?.description && (
            <DialogDescription>{beat.description}</DialogDescription>
          )}
        </DialogHeader>
        {current && built && (
          <div className="space-y-3">
            <Segmented<Mode>
              value={current.mode}
              onChange={(mode) => set({ mode })}
              options={[
                { label: "Every", value: "every" },
                { label: "Daily at", value: "daily" },
                { label: "Custom", value: "cron" },
              ]}
            />
            {current.mode === "every" && (
              <div className="flex gap-2">
                <Input
                  type="number"
                  min={1}
                  inputMode="numeric"
                  aria-label="Run every"
                  value={Number.isFinite(current.amount) ? current.amount : ""}
                  onChange={(e) => set({ amount: Number(e.target.value) })}
                />
                <select
                  aria-label="Unit"
                  value={current.unit}
                  onChange={(e) => set({ unit: e.target.value as Unit })}
                  className="h-9 rounded-md border border-border bg-bg px-2 text-sm text-fg"
                >
                  <option value="s">seconds</option>
                  <option value="min">minutes</option>
                  <option value="h">hours</option>
                </select>
              </div>
            )}
            {current.mode === "daily" && (
              <label className="block text-xs text-fg-subtle">
                Time on the harness clock{timezone ? ` (${timezone})` : ""}
                <Input
                  type="time"
                  className="mt-1"
                  value={current.time}
                  onChange={(e) => set({ time: e.target.value })}
                />
              </label>
            )}
            {current.mode === "cron" && (
              <label className="block text-xs text-fg-subtle">
                Cron: min hour day month weekday
                {timezone ? ` (${timezone})` : ""}
                <Input
                  className="mt-1 font-mono"
                  value={current.cron}
                  onChange={(e) => set({ cron: e.target.value })}
                />
              </label>
            )}
            <p
              className={`rounded-md px-3 py-2 text-xs ${
                built.error || serverError
                  ? "bg-danger/10 text-danger"
                  : "bg-surface text-fg-muted"
              }`}
            >
              {serverError ??
                built.error ??
                `Will run ${describe(built.schedule)}. Applies at once, no restart.`}
              {!built.error && !serverError && beat?.default_schedule && (
                <>
                  <br />
                  Shipped default: {describe(beat.default_schedule)}.
                </>
              )}
            </p>
          </div>
        )}
        <DialogFooter>
          <Button
            type="button"
            variant="outline"
            onClick={close}
            disabled={patch.isPending}
          >
            Cancel
          </Button>
          <Button
            type="button"
            onClick={save}
            disabled={patch.isPending || !built || built.error !== null}
          >
            {patch.isPending ? "Saving…" : "Save"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function BeatCard({
  beat,
  canWrite,
  onEdit,
  onTrigger,
}: {
  beat: Heartbeat;
  canWrite: boolean;
  onEdit: (beat: Heartbeat) => void;
  onTrigger: (name: string) => void;
}) {
  const patch = usePatchHeartbeat();
  const reset = useResetHeartbeat();
  const runnable = beat.runnable !== false;
  const busy = patch.isPending || reset.isPending;

  const toggle = async (enabled: boolean) => {
    try {
      await patch.mutateAsync({ name: beat.name, patch: { enabled } });
      toast.success(`${beat.name} ${enabled ? "on" : "off"}. Applies now.`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "change failed");
    }
  };
  const onReset = async () => {
    try {
      const back = await reset.mutateAsync(beat.name);
      toast.success(
        `${beat.name} back to ${back.enabled ? (back.schedule_text ?? back.schedule) : "off"}`,
      );
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "reset failed");
    }
  };

  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm text-fg">{beat.name}</span>
        {beat.overridden && <Tag kind="info">changed</Tag>}
        {!runnable && <Tag kind="warn">can&apos;t run here</Tag>}
        <span className="ml-auto text-[13px] text-fg">
          {beat.enabled ? (
            (beat.schedule_text ?? beat.schedule)
          ) : (
            <span className="text-fg-subtle">off</span>
          )}
        </span>
        <Switch
          checked={beat.enabled}
          label={`${beat.name} on or off`}
          disabled={!canWrite || busy || (!runnable && !beat.enabled)}
          onChange={toggle}
        />
      </div>
      {beat.description && (
        <p className="mt-1 text-xs text-fg-muted">{beat.description}</p>
      )}
      <JobLine beat={beat} />
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <span className="text-xs text-fg-subtle">
          {beat.enabled
            ? beat.next_run_at
              ? `Next: ${fmtDateTime(beat.next_run_at)}`
              : "Scheduled"
            : runnable
              ? "Won't run until turned on"
              : `Can't run here: ${beat.unavailable_reason ?? "no handler on this harness"}`}
        </span>
        <span className="ml-auto" />
        {canWrite && beat.overridden && (
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={onReset}
            disabled={busy}
          >
            Reset
          </Button>
        )}
        {canWrite && (
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => onEdit(beat)}
          >
            Edit schedule
          </Button>
        )}
        {canWrite && beat.enabled && (
          <Button
            type="button"
            size="sm"
            variant="outline"
            onClick={() => onTrigger(beat.name)}
          >
            Run now
          </Button>
        )}
      </div>
    </div>
  );
}

function Definitions({ onTrigger }: { onTrigger: (name: string) => void }) {
  const { data, isLoading, isError } = useHeartbeats();
  const canWrite = useWritesEnabled();
  const [query, setQuery] = useState("");
  const [editing, setEditing] = useState<Heartbeat | null>(null);
  const shown = useMemo(() => {
    const beats = data?.heartbeats ?? [];
    const q = query.trim().toLowerCase();
    return q
      ? beats.filter(
          (b) =>
            b.name.toLowerCase().includes(q) ||
            b.description.toLowerCase().includes(q),
        )
      : beats;
  }, [data, query]);
  const beats = data?.heartbeats ?? [];
  const changed = beats.filter((b) => b.overridden).length;

  return (
    <Section title={`Definitions (${beats.length})`}>
      <p className="-mt-2 mb-3 text-xs text-fg-subtle">
        {changed > 0 ? `${changed} changed from the shipped schedule · ` : ""}
        {canWrite
          ? "Changes apply at once and survive restarts and deploys."
          : "This device is read-only; a control device can change schedules."}
      </p>
      <Input
        placeholder="Filter heartbeats"
        aria-label="Filter heartbeats"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="mb-3"
      />
      <QueryState
        loading={isLoading}
        error={isError}
        empty={shown.length === 0}
        emptyText={
          beats.length === 0
            ? "No heartbeats defined."
            : "No heartbeat matches."
        }
      >
        <div className="space-y-2">
          {shown.map((b) => (
            <BeatCard
              key={b.name}
              beat={b}
              canWrite={canWrite}
              onEdit={setEditing}
              onTrigger={onTrigger}
            />
          ))}
        </div>
      </QueryState>
      <EditSchedule
        beat={editing}
        timezone={data?.timezone ?? null}
        onClose={() => setEditing(null)}
      />
    </Section>
  );
}

function RunRow({ run }: { run: HeartbeatRun }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-[13px] text-fg">{run.name}</span>
        <Tag kind={runTone(run.status)}>{run.status}</Tag>
        <span className="ml-auto font-mono text-[11px] text-fg-subtle">
          {fmtDateTime(run.finished_at ?? run.started_at)}
        </span>
      </div>
      {run.error ? (
        <CopyBlock label="error" text={run.error} tone="text-danger" />
      ) : run.output ? (
        <CopyBlock label="output" text={run.output} />
      ) : null}
    </div>
  );
}

function RecentRuns() {
  const { data, isLoading, isError } = useHeartbeatRuns(50);
  const runs = (data ?? []).slice().reverse(); // newest first
  return (
    <Section title={`Recent runs (${runs.length})`}>
      <QueryState
        loading={isLoading}
        error={isError}
        empty={runs.length === 0}
        emptyText="No heartbeat runs recorded yet."
      >
        <div className="space-y-2">
          {runs.map((r, i) => (
            <RunRow key={`${r.name}-${r.started_at}-${i}`} run={r} />
          ))}
        </div>
      </QueryState>
    </Section>
  );
}

export function HeartbeatsScreen() {
  const trigger = useTriggerHeartbeat();
  const [confirm, setConfirm] = useState<ConfirmState | null>(null);

  const onTrigger = (name: string) =>
    setConfirm({
      title: "Run heartbeat now?",
      description: `Triggers "${name}" immediately, outside its schedule.`,
      confirmLabel: "Run now",
      run: async () => {
        try {
          const res = await trigger.mutateAsync(name);
          if (res.error) toast.error(`${name}: ${res.error}`);
          else toast.success(`${name}: ${res.status}`);
        } catch (e) {
          toast.error(e instanceof Error ? e.message : "trigger failed");
          throw e;
        }
      },
    });

  return (
    <div className="space-y-6">
      <Definitions onTrigger={onTrigger} />
      <RecentRuns />
      <ConfirmDialog state={confirm} onClose={() => setConfirm(null)} />
    </div>
  );
}
