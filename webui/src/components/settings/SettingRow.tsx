import { useState } from "react";
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
import { usePatchSetting, useResetSetting } from "@/lib/queries";
import type { CatalogSetting } from "@/lib/control";

const APPLIES: Record<CatalogSetting["applies"], string> = {
  now: "applies now",
  next_run: "next run",
  restart: "after restart",
};
const TRUE = new Set(["1", "true", "yes", "on"]);
const HIDDEN = new Set(["secret", "path", "url"]);

export function isOn(value: unknown): boolean {
  return TRUE.has(String(value ?? "").trim().toLowerCase());
}

/** The value in effect: what the process has, else the code's default. */
export function effectiveValue(s: CatalogSetting): string {
  if (s.value !== null && s.value !== undefined) return s.value;
  return s.default === null || s.default === undefined ? "" : String(s.default);
}

function shown(s: CatalogSetting, value: string): string {
  if (HIDDEN.has(s.kind)) return s.is_set ? "set (hidden)" : "not set";
  if (s.kind === "bool") return isOn(value) ? "on" : "off";
  return value === "" ? "—" : value;
}

export interface PendingChange {
  setting: CatalogSetting;
  value: string | boolean | number;
  from: string;
  to: string;
}

/**
 * The owner's yes to a guarded change (ADR-0120: "editable with confirm"). The API
 * refuses a guarded change without ``confirm``, so this is the only way through.
 */
export function GuardConfirm({
  pending,
  onClose,
  onConfirm,
}: {
  pending: PendingChange | null;
  onClose: () => void;
  onConfirm: (p: PendingChange) => Promise<void>;
}) {
  const [ack, setAck] = useState(false);
  const [busy, setBusy] = useState(false);
  const close = () => {
    setAck(false);
    onClose();
  };
  return (
    <Dialog open={pending !== null} onOpenChange={(open) => !open && !busy && close()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Confirm guarded change</DialogTitle>
          <DialogDescription>{pending?.setting.label}</DialogDescription>
        </DialogHeader>
        {pending && (
          <div className="space-y-3 text-sm">
            <p className="font-mono text-xs">
              <span className="text-danger line-through">{pending.from}</span> →{" "}
              <span className="font-semibold text-primary">{pending.to}</span>
            </p>
            <p className="rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-xs text-fg">
              {pending.setting.guard_reason}
            </p>
            <label className="flex items-start gap-2 text-xs">
              <input
                type="checkbox"
                checked={ack}
                onChange={(e) => setAck(e.target.checked)}
                className="mt-0.5"
              />
              I understand what this changes.
            </label>
            <p className="text-[11px] text-fg-subtle">
              Recorded in History with this device's name.{" "}
              {pending.setting.applies === "restart"
                ? "Takes effect after the next restart."
                : "Takes effect now."}
            </p>
          </div>
        )}
        <DialogFooter>
          <Button type="button" variant="outline" onClick={close} disabled={busy}>
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            disabled={!ack || busy}
            onClick={async () => {
              if (!pending) return;
              setBusy(true);
              try {
                await onConfirm(pending);
                close();
              } finally {
                setBusy(false);
              }
            }}
          >
            {busy ? "Changing…" : "Change it"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** Save a setting, asking for the guard confirm first when it is guarded. */
export function useSaveSetting() {
  const patch = usePatchSetting();
  const [pending, setPending] = useState<PendingChange | null>(null);

  const send = async (s: CatalogSetting, value: string | boolean | number, confirm: boolean) => {
    try {
      const r = await patch.mutateAsync({ name: s.name, value, confirm });
      toast.success(
        `${s.label}: ${shown(r, r.value ?? "")}${r.restart_required ? " · applies after restart" : ""}`,
      );
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "change failed");
      throw e;
    }
  };

  const save = async (s: CatalogSetting, value: string | boolean | number) => {
    if (s.guarded) {
      setPending({
        setting: s,
        value,
        from: shown(s, effectiveValue(s)),
        to: shown(s, typeof value === "boolean" ? (value ? "1" : "0") : String(value)),
      });
      return;
    }
    await send(s, value, false).catch(() => undefined);
  };

  const dialog = (
    <GuardConfirm
      pending={pending}
      onClose={() => setPending(null)}
      onConfirm={(p) => send(p.setting, p.value, true)}
    />
  );
  return { save, dialog, busy: patch.isPending };
}

export function SettingRow({ s, canWrite }: { s: CatalogSetting; canWrite: boolean }) {
  const { save, dialog, busy } = useSaveSetting();
  const reset = useResetSetting();
  const current = effectiveValue(s);
  const [draft, setDraft] = useState<string | null>(null);
  const editable = s.editable && canWrite;
  const text = draft ?? current;

  const onReset = async () => {
    try {
      const r = await reset.mutateAsync(s.name);
      toast.success(`${s.label} back to ${shown(r, effectiveValue(r))}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "reset failed");
    }
  };

  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-fg">{s.label}</span>
        {s.guarded && <Tag kind="bad">guarded</Tag>}
        {s.editable && (
          <Tag kind={s.applies === "restart" ? "warn" : "ok"}>{APPLIES[s.applies]}</Tag>
        )}
        {s.overridden && <Tag kind="info">changed</Tag>}
        {!s.editable && <Tag kind="info">read-only</Tag>}
        <span className="ml-auto" />
        {s.kind === "bool" && s.editable ? (
          <Switch
            checked={isOn(current)}
            label={s.label}
            disabled={!editable || busy}
            onChange={(next) => void save(s, next)}
          />
        ) : (
          <span className="break-all font-mono text-[11px] text-fg">{shown(s, current)}</span>
        )}
      </div>
      <p className="mt-1 text-xs text-fg-muted">{s.description}</p>
      {s.kind !== "bool" && editable && (
        <form
          className="mt-2 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void save(s, text).then(() => setDraft(null));
          }}
        >
          <Input
            aria-label={s.label}
            value={text}
            inputMode={s.kind === "int" || s.kind === "float" ? "decimal" : undefined}
            onChange={(e) => setDraft(e.target.value)}
            className="h-8 font-mono text-xs"
          />
          <Button type="submit" size="sm" variant="outline" disabled={busy || draft === null}>
            Save
          </Button>
        </form>
      )}
      <div className="mt-1 flex flex-wrap items-center gap-2">
        <span className="font-mono text-[10.5px] text-fg-subtle">
          {s.name} · {s.owner}
          {!s.editable && s.not_editable_reason ? ` · ${s.not_editable_reason}` : ""}
          {s.overridden && s.deploy_value !== undefined
            ? ` · deploy value ${s.deploy_value === null ? "unset" : shown(s, s.deploy_value)}`
            : ""}
        </span>
        {s.overridden && editable && (
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className="ml-auto"
            onClick={onReset}
            disabled={reset.isPending}
          >
            Reset
          </Button>
        )}
      </div>
      {dialog}
    </div>
  );
}
