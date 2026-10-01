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
import { Tag } from "@/components/Tag";
import { Section } from "@/components/layout";
import { QueryState } from "@/components/control/parts";
import {
  useModels,
  useMoveIntent,
  usePatchTier,
  useResetIntent,
  useResetTier,
} from "@/lib/queries";
import type { ModelTier, ModelsView, TierFields } from "@/lib/control";

type NumField = "max_tokens" | "temperature" | "timeout_seconds";
const NUMBERS: { key: NumField; label: string; step?: string }[] = [
  { key: "timeout_seconds", label: "Timeout (s)" },
  { key: "max_tokens", label: "Max tokens" },
  { key: "temperature", label: "Temperature", step: "0.1" },
];

interface Pending {
  title: string;
  from: string;
  to: string;
  why: string;
  run: () => Promise<void>;
}

/** The guarded-change confirm, as for guarded settings: a checkbox, then "Change it". */
function RouteConfirm({ pending, onClose }: { pending: Pending | null; onClose: () => void }) {
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
          <DialogDescription>{pending?.title}</DialogDescription>
        </DialogHeader>
        {pending && (
          <div className="space-y-3 text-sm">
            <p className="font-mono text-xs">
              <span className="text-danger line-through">{pending.from}</span> →{" "}
              <span className="font-semibold text-primary">{pending.to}</span>
            </p>
            <p className="rounded-md border border-danger/40 bg-danger/10 px-3 py-2 text-xs">
              {pending.why}
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
                await pending.run();
                close();
              } catch (e) {
                toast.error(e instanceof Error ? e.message : "change failed");
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

function TierCard({
  tier,
  view,
  canWrite,
  onMove,
  confirmRoute,
}: {
  tier: ModelTier;
  view: ModelsView;
  canWrite: boolean;
  onMove: (intent: string) => void;
  confirmRoute: (p: Pending) => void;
}) {
  const patch = usePatchTier();
  const reset = useResetTier();
  const [draft, setDraft] = useState<Partial<TierFields>>({});
  const value = <K extends keyof TierFields>(k: K) => (draft[k] ?? tier[k]) as TierFields[K];
  const dirty = Object.keys(draft).length > 0;

  const save = async () => {
    try {
      await patch.mutateAsync({ name: tier.name, fields: draft });
      setDraft({});
      toast.success(`${tier.name}: saved · applies from the next turn`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "save failed");
    }
  };

  const changeRoute = (provider: string) => {
    if (provider === tier.provider) return;
    confirmRoute({
      title: `Route of tier ${tier.name}`,
      from: tier.provider,
      to: provider,
      why:
        `Every turn on ${tier.name} (${tier.use_for.join(", ") || "nothing mapped"}) ` +
        `would go to the ${provider} route instead. On the VM the routes differ in where ` +
        "prompts can end up (e.g. Mac only vs. Mac first with a cloud fallback).",
      run: async () => {
        await patch.mutateAsync({ name: tier.name, fields: { provider }, confirm: true });
        toast.success(`${tier.name} now uses ${provider}`);
      },
    });
  };

  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-sm font-semibold text-fg">{tier.name}</span>
        <span className="text-xs text-fg-subtle">{tier.title}</span>
        {tier.changed.length > 0 && <Tag kind="info">changed</Tag>}
        <span className="ml-auto" />
        {canWrite && tier.changed.length > 0 && (
          <Button
            type="button"
            size="sm"
            variant="ghost"
            disabled={reset.isPending}
            onClick={() =>
              reset
                .mutateAsync(tier.name)
                .then(() => toast.success(`${tier.name} back to llm_tiers.yaml`))
            }
          >
            Reset
          </Button>
        )}
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-5">
        <label className="col-span-2 text-[11px] text-fg-subtle">
          Model
          <Input
            className={`mt-0.5 h-8 font-mono text-xs ${tier.changed.includes("model") ? "border-primary" : ""}`}
            value={value("model")}
            disabled={!canWrite}
            onChange={(e) => setDraft({ ...draft, model: e.target.value })}
          />
        </label>
        {NUMBERS.map((n) => (
          <label key={n.key} className="text-[11px] text-fg-subtle">
            {n.label}
            <Input
              type="number"
              step={n.step}
              className={`mt-0.5 h-8 text-xs ${tier.changed.includes(n.key) ? "border-primary" : ""}`}
              value={value(n.key)}
              disabled={!canWrite}
              onChange={(e) => setDraft({ ...draft, [n.key]: Number(e.target.value) })}
            />
          </label>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <label className="flex items-center gap-2 text-[11px] text-fg-subtle">
          Route <Tag kind="bad">guarded</Tag>
          <select
            aria-label={`${tier.name} route`}
            className="h-8 rounded-md border border-border bg-bg px-2 text-xs text-fg"
            value={tier.provider}
            disabled={!canWrite}
            onChange={(e) => changeRoute(e.target.value)}
          >
            {view.providers.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
        <span className="ml-auto" />
        {canWrite && dirty && (
          <>
            <Button type="button" size="sm" variant="ghost" onClick={() => setDraft({})}>
              Cancel
            </Button>
            <Button type="button" size="sm" onClick={save} disabled={patch.isPending}>
              Save
            </Button>
          </>
        )}
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {tier.use_for.length === 0 && (
          <span className="text-[11px] text-fg-subtle">no intents: catches unmapped ones</span>
        )}
        {tier.use_for.map((intent) => (
          <button
            key={intent}
            type="button"
            disabled={!canWrite}
            onClick={() => onMove(intent)}
            className={`rounded-md border px-2 py-0.5 font-mono text-[11px] ${
              view.moved_intents.includes(intent)
                ? "border-primary text-primary"
                : "border-border text-fg"
            }`}
          >
            {intent}
          </button>
        ))}
      </div>
    </div>
  );
}

function MoveIntent({
  intent,
  view,
  onClose,
  confirmRoute,
}: {
  intent: string | null;
  view: ModelsView;
  onClose: () => void;
  confirmRoute: (p: Pending) => void;
}) {
  const move = useMoveIntent();
  const reset = useResetIntent();
  const [target, setTarget] = useState<string>("");
  const from = intent ? view.intents[intent] : "";
  const fromTier = view.tiers.find((t) => t.name === from);
  const chosen = target || from;
  const toTier = view.tiers.find((t) => t.name === chosen);
  const crosses = Boolean(fromTier && toTier && fromTier.provider !== toTier.provider);
  const close = () => {
    setTarget("");
    onClose();
  };

  const go = async () => {
    if (!intent || chosen === from) return close();
    if (crosses && fromTier && toTier) {
      close();
      confirmRoute({
        title: `Run ${intent} on ${chosen}`,
        from: `${from} (${fromTier.provider})`,
        to: `${chosen} (${toTier.provider})`,
        why: `${intent} turns would go to the ${toTier.provider} route instead of ${fromTier.provider}.`,
        run: async () => {
          await move.mutateAsync({ intent, tier: chosen, confirm: true });
          toast.success(`${intent} → ${chosen}`);
        },
      });
      return;
    }
    try {
      await move.mutateAsync({ intent, tier: chosen });
      toast.success(`${intent} → ${chosen} · from the next turn`);
      close();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "move failed");
    }
  };

  return (
    <Dialog open={intent !== null} onOpenChange={(open) => !open && close()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="font-mono">{intent}</DialogTitle>
          <DialogDescription>
            Runs on <b>{from}</b> ({fromTier?.provider}). Pick another tier.
          </DialogDescription>
        </DialogHeader>
        <select
          aria-label="Run on tier"
          className="h-9 w-full rounded-md border border-border bg-bg px-2 text-sm text-fg"
          value={chosen}
          onChange={(e) => setTarget(e.target.value)}
        >
          {view.tiers.map((t) => (
            <option key={t.name} value={t.name}>
              {t.name} — {t.model} ({t.provider})
            </option>
          ))}
        </select>
        <p className="text-xs text-fg-muted">
          {crosses
            ? "This changes the route, so you'll be asked to confirm."
            : "Applies from the next turn."}
        </p>
        <DialogFooter>
          {intent && view.moved_intents.includes(intent) && (
            <Button
              type="button"
              variant="ghost"
              onClick={() =>
                reset.mutateAsync(intent).then(() => {
                  toast.success(`${intent} back to llm_tiers.yaml`);
                  close();
                })
              }
            >
              Reset
            </Button>
          )}
          <Button type="button" variant="outline" onClick={close}>
            Cancel
          </Button>
          <Button type="button" onClick={go} disabled={move.isPending}>
            Move
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

/** The Models tab: llm_tiers.yaml, editable, saved on the data volume (ADR-0120). */
export function ModelsPanel({ canWrite }: { canWrite: boolean }) {
  const { data, isLoading, isError } = useModels();
  const [moving, setMoving] = useState<string | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  return (
    <Section title={`Model tiers (${data?.tiers.length ?? 0})`}>
      <p className="-mt-2 mb-3 text-xs text-fg-subtle">
        Edits apply from the next turn; helpers built at startup pick them up after a restart.
        Tap an intent to run it on another tier.
      </p>
      <QueryState
        loading={isLoading}
        error={isError}
        empty={(data?.tiers.length ?? 0) === 0}
        emptyText="No tiers."
      >
        {data && (
          <div className="space-y-2">
            {data.tiers.map((t) => (
              <TierCard
                key={t.name}
                tier={t}
                view={data}
                canWrite={canWrite}
                onMove={setMoving}
                confirmRoute={setPending}
              />
            ))}
            <MoveIntent
              intent={moving}
              view={data}
              onClose={() => setMoving(null)}
              confirmRoute={setPending}
            />
          </div>
        )}
      </QueryState>
      <RouteConfirm pending={pending} onClose={() => setPending(null)} />
    </Section>
  );
}
