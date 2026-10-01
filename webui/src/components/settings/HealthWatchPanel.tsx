import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/Switch";
import { Tag } from "@/components/Tag";
import { Section } from "@/components/layout";
import { QueryState } from "@/components/control/parts";
import {
  usePatchWatchConfig,
  useResetWatchConfig,
  useWatchConfig,
} from "@/lib/queries";
import type { WatchField } from "@/lib/control";

const LABELS: Record<string, [string, string]> = {
  confirm_ticks: [
    "Confirm after (ticks)",
    "Bad this many minutes in a row before an incident.",
  ],
  renotify_hours: [
    "Re-alert every (hours)",
    "Remind about an incident still open.",
  ],
  max_attempts: ["Repair attempts", "Token refresh / retry / restart tries."],
  recurring_days: [
    "Recurring window (days)",
    "The same fault again within this is recurring.",
  ],
  notify_recovered: [
    "Tell me when it recovers",
    "Send a message when an incident closes.",
  ],
  notify_self_healed: [
    "Tell me about self-repairs",
    "Also message when a repair fixed it.",
  ],
  heartbeat_retry: [
    "Retry failed heartbeats",
    "Re-run a heartbeat that errored.",
  ],
  diagnose_credentials: [
    "Diagnose credentials",
    "Tell a revoked token from a network blip.",
  ],
};

function NumberRow({ f, canWrite }: { f: WatchField; canWrite: boolean }) {
  const patch = usePatchWatchConfig();
  const [draft, setDraft] = useState<string | null>(null);
  const [label, help] = LABELS[f.name] ?? [f.name, ""];
  const save = async () => {
    try {
      await patch.mutateAsync({ [f.name]: Number(draft) });
      setDraft(null);
      toast.success(`${label}: ${draft} · from the next tick`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "save failed");
    }
  };
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-fg">{label}</span>
        {f.changed && <Tag kind="info">changed</Tag>}
        <span className="ml-auto font-mono text-[11px] text-fg-subtle">
          file: {String(f.file)}
        </span>
      </div>
      <p className="mt-1 text-xs text-fg-muted">{help}</p>
      <form
        className="mt-2 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void save();
        }}
      >
        <Input
          type="number"
          aria-label={label}
          min={f.min ?? undefined}
          max={f.max ?? undefined}
          step={f.kind === "float" ? "0.5" : "1"}
          value={draft ?? String(f.value)}
          disabled={!canWrite}
          onChange={(e) => setDraft(e.target.value)}
          className="h-8 w-28 text-xs"
        />
        {canWrite && (
          <Button
            type="submit"
            size="sm"
            variant="outline"
            disabled={draft === null || patch.isPending}
          >
            Save
          </Button>
        )}
      </form>
    </div>
  );
}

function SwitchRow({ f, canWrite }: { f: WatchField; canWrite: boolean }) {
  const patch = usePatchWatchConfig();
  const [label, help] = LABELS[f.name] ?? [f.name, ""];
  return (
    <div className="flex items-center gap-2 rounded-lg border border-border bg-surface p-3">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2 text-sm text-fg">
          {label} {f.changed && <Tag kind="info">changed</Tag>}
        </div>
        <p className="text-xs text-fg-muted">{help}</p>
      </div>
      <Switch
        checked={Boolean(f.value)}
        label={label}
        disabled={!canWrite || patch.isPending}
        onChange={(next) =>
          void patch
            .mutateAsync({ [f.name]: next })
            .then(() =>
              toast.success(
                `${label}: ${next ? "on" : "off"} · from the next tick`,
              ),
            )
            .catch((e: unknown) =>
              toast.error(e instanceof Error ? e.message : "save failed"),
            )
        }
      />
    </div>
  );
}

/** Settings → Health watch: health_watch.yaml, editable (ADR-0120). */
export function HealthWatchPanel({ canWrite }: { canWrite: boolean }) {
  const { data, isLoading, isError } = useWatchConfig();
  const reset = useResetWatchConfig();
  const fields = data?.fields ?? [];
  const anyChanged = fields.some((f) => f.changed);
  return (
    <Section title="Health watch">
      <p className="-mt-2 mb-3 text-xs text-fg-subtle">
        Applies at the next health tick (within a minute). On/off is the "Health
        watch" setting below.
        {data && !data.running
          ? " The watch isn't running in this process."
          : ""}
      </p>
      <QueryState
        loading={isLoading}
        error={isError}
        empty={fields.length === 0}
        emptyText="—"
      >
        <div className="space-y-2">
          {fields.map((f) =>
            f.kind === "bool" ? (
              <SwitchRow key={f.name} f={f} canWrite={canWrite} />
            ) : (
              <NumberRow key={f.name} f={f} canWrite={canWrite} />
            ),
          )}
          {canWrite && anyChanged && (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={() =>
                reset
                  .mutateAsync(undefined)
                  .then(() => toast.success("Back to health_watch.yaml"))
              }
            >
              Reset all to the file
            </Button>
          )}
        </div>
      </QueryState>
    </Section>
  );
}
