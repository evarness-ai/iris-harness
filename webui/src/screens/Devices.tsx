import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { Laptop, Smartphone } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { ConfirmDialog, type ConfirmState } from "@/components/ConfirmDialog";
import { Tag } from "@/components/Tag";
import { Section } from "@/components/layout";
import { QueryState, fmtDateTime } from "@/components/control/parts";
import { revokeDevice, type Device, type DeviceScope, type PairingCode } from "@/lib/devices";
import { PAIR_PATH } from "@/lib/http";
import { useDevices, usePrincipal, useRevokeDevice, useStartPairing } from "@/lib/queries";
import { isSubscribed, pushSupport, subscribeToPush, unsubscribeFromPush } from "@/lib/push";

const SCOPES: { value: DeviceScope; label: string; hint: string }[] = [
  { value: "control", label: "control", hint: "can read, approve and change things" },
  { value: "read", label: "read", hint: "can look, never act" },
];

/** "3 min ago" for the last-seen column; falls back to the date past a week. */
function fmtAgo(iso: string | null, now: number): string {
  if (!iso) return "never";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return iso;
  const s = Math.max(0, Math.round((now - t) / 1000));
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86_400) return `${Math.floor(s / 3600)} h ago`;
  if (s < 7 * 86_400) return `${Math.floor(s / 86_400)} d ago`;
  return fmtDateTime(iso);
}

/** A ticking clock, only while something on screen is counting. */
function useNow(active: boolean, everyMs = 1000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const id = window.setInterval(() => setNow(Date.now()), everyMs);
    return () => window.clearInterval(id);
  }, [active, everyMs]);
  return now;
}

function ScopeTag({ scope }: { scope: DeviceScope }) {
  return <Tag kind={scope === "control" ? "ok" : "info"}>{scope}</Tag>;
}

/** The code, large, with a live countdown. Mounted per code (keyed), so the
 * clock starts fresh with each one. */
function PairingCard({ pairing }: { pairing: PairingCode }) {
  const now = useNow(true);
  const left = Math.max(0, Math.ceil((new Date(pairing.expires_at).getTime() - now) / 1000));
  const expired = left <= 0;
  return (
    <div
      className={`rounded-lg border p-4 text-center ${
        expired ? "border-border opacity-50" : "border-primary/40 bg-primary/5"
      }`}
    >
      <div
        aria-label={`Pairing code ${pairing.code.split("").join(" ")}`}
        className="select-all break-all font-mono text-3xl font-semibold tracking-[0.15em] text-fg sm:text-4xl"
      >
        {pairing.code}
      </div>
      <div className="mt-2 flex flex-wrap items-center justify-center gap-2 text-xs text-fg-muted">
        <ScopeTag scope={pairing.scope} />
        <span className="font-mono">
          {expired
            ? "expired — get a new code"
            : `expires in ${Math.floor(left / 60)}:${String(left % 60).padStart(2, "0")}`}
        </span>
      </div>
      <p className="mt-3 text-xs text-fg-subtle">
        On the new device, open this console ({window.location.host}) — it asks for the code. It
        works once, and stops working after 5 wrong tries.
      </p>
    </div>
  );
}

function PairNew({ canPair }: { canPair: boolean }) {
  const start = useStartPairing();
  const [scope, setScope] = useState<DeviceScope>("control");
  const [pairing, setPairing] = useState<PairingCode | null>(null);

  const onStart = async () => {
    try {
      setPairing(await start.mutateAsync(scope));
    } catch (e) {
      // A read-only device gets the server's 403 detail here.
      toast.error(e instanceof Error ? e.message : "could not start pairing");
    }
  };

  return (
    <Section title="Pair a new device">
      <div className="rounded-lg border border-border bg-surface p-4">
        {!canPair ? (
          <p className="text-xs text-fg-subtle">
            This device is paired read-only, so it cannot pair others. Use a control device, or run{" "}
            <code className="font-mono text-fg-muted">iris device pair</code> on the machine running
            IRIS.
          </p>
        ) : (
          <div className="space-y-3">
            <div className="flex flex-wrap items-center gap-3">
              <div
                role="radiogroup"
                aria-label="Scope for the new device"
                className="inline-flex overflow-hidden rounded-lg border border-border"
              >
                {SCOPES.map((s) => (
                  <button
                    key={s.value}
                    type="button"
                    role="radio"
                    aria-checked={s.value === scope}
                    onClick={() => setScope(s.value)}
                    className={`px-3 py-1.5 text-xs font-medium transition-colors ${
                      s.value === scope
                        ? "bg-primary/15 text-primary"
                        : "bg-bg text-fg-muted hover:bg-surface hover:text-fg"
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
              <span className="text-xs text-fg-subtle">
                {SCOPES.find((s) => s.value === scope)?.hint}
              </span>
              <Button
                type="button"
                size="sm"
                className="ml-auto"
                onClick={() => void onStart()}
                disabled={start.isPending}
              >
                {start.isPending ? "Working…" : pairing ? "New code" : "Get a pairing code"}
              </Button>
            </div>

            {pairing && <PairingCard key={pairing.code} pairing={pairing} />}
          </div>
        )}
      </div>
    </Section>
  );
}

function DeviceRow({
  d,
  now,
  canRevoke,
  onRevoke,
}: {
  d: Device;
  now: number;
  canRevoke: boolean;
  onRevoke: (d: Device) => void;
}) {
  const revoked = d.revoked_at !== null;
  const Icon = d.kind === "app" ? Smartphone : Laptop;
  return (
    <div
      className={`rounded-lg border border-border bg-surface p-3 ${revoked ? "opacity-50" : ""}`}
    >
      <div className="flex flex-wrap items-center gap-2">
        <Icon size={15} className="shrink-0 text-fg-subtle" aria-hidden />
        <span className={`min-w-0 truncate text-sm text-fg ${revoked ? "line-through" : ""}`}>
          {d.name}
        </span>
        {d.current && <Tag kind="opp">this device</Tag>}
        <Tag kind="res">{d.kind}</Tag>
        <ScopeTag scope={d.scope} />
        {revoked && <Tag kind="bad">revoked</Tag>}
        {!revoked && canRevoke && (
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="ml-auto"
            onClick={() => onRevoke(d)}
          >
            {d.current ? "Sign out" : "Revoke"}
          </Button>
        )}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 font-mono text-[11px] text-fg-subtle">
        <span>paired {fmtDateTime(d.created_at)}</span>
        <span>last seen {fmtAgo(d.last_seen_at, now)}</span>
        {revoked && <span>revoked {fmtDateTime(d.revoked_at)}</span>}
      </div>
    </div>
  );
}

/* Notifications on this browser (Track 2b PR 9).
 *
 * Lives on Devices because that is what it is: a property of this paired
 * browser, revoked with it. The interesting state is the iOS one — in Safari
 * proper the push API is simply absent, `requestPermission` resolves "denied"
 * with no prompt, and the only fix is Share -> Add to Home Screen. Saying so
 * is the difference between a working feature and a mystery. */
function NotificationsSection() {
  const [subscribed, setSubscribed] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const support = pushSupport();

  useEffect(() => {
    void isSubscribed().then(setSubscribed).catch(() => setSubscribed(false));
  }, []);

  const enable = async () => {
    setBusy(true);
    try {
      await subscribeToPush(navigator.userAgent.slice(0, 80));
      setSubscribed(true);
      toast.success("Notifications on for this device");
    } catch (err) {
      const reason = err instanceof Error ? err.message : "could not subscribe";
      toast.error(
        reason === "denied"
          ? "Notifications are blocked for this site in your browser settings."
          : reason,
      );
    } finally {
      setBusy(false);
    }
  };

  const disable = async () => {
    setBusy(true);
    try {
      await unsubscribeFromPush();
      setSubscribed(false);
      toast.success("Notifications off for this device");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "could not unsubscribe");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Section title="Notifications">
      <div className="space-y-2 rounded-lg border border-border bg-surface px-3 py-2.5">
        {!support.ok && support.reason === "needs-home-screen" ? (
          <p className="text-xs text-fg-muted">
            iOS only delivers notifications to an installed app. Tap{" "}
            <span className="font-medium text-fg">Share</span> then{" "}
            <span className="font-medium text-fg">Add to Home Screen</span>, open IRIS from
            there, and this button will work.
          </p>
        ) : !support.ok && support.reason === "denied" ? (
          <p className="text-xs text-fg-muted">
            Notifications are blocked for this site. Allow them in your browser settings for
            this site, then come back.
          </p>
        ) : !support.ok ? (
          <p className="text-xs text-fg-muted">This browser cannot receive push notifications.</p>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-2">
              <Tag kind={subscribed ? "ok" : "res"}>{subscribed ? "on" : "off"}</Tag>
              <span className="text-xs text-fg-muted">
                {subscribed
                  ? "Approvals and incidents reach this device."
                  : "Nothing is sent to this device."}
              </span>
            </div>
            <Button
              type="button"
              size="sm"
              variant={subscribed ? "outline" : "default"}
              disabled={busy || subscribed === null}
              onClick={() => void (subscribed ? disable() : enable())}
            >
              {busy ? "Working…" : subscribed ? "Turn off" : "Turn on notifications"}
            </Button>
          </>
        )}
      </div>
    </Section>
  );
}

export function DevicesScreen() {
  const { data, isLoading, isError } = useDevices();
  const me = usePrincipal();
  const revoke = useRevokeDevice();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [confirm, setConfirm] = useState<ConfirmState | null>(null);
  const now = useNow(true, 30_000); // keeps "last seen" honest without a re-render a second
  const devices = data ?? [];
  const live = devices.filter((d) => d.revoked_at === null).length;

  // Mirror of the server's rule, only to avoid dead buttons: a read device may
  // not pair or revoke OTHERS; any device may revoke itself. Unknown → allow,
  // and let the server's 403 speak.
  const readOnly = me.data?.scope === "read";

  const onRevoke = (d: Device) =>
    setConfirm({
      title: d.current ? "Sign out this device?" : `Revoke "${d.name}"?`,
      description: d.current
        ? "This browser is unpaired right away. To use the console here again you will need a new pairing code."
        : "It is signed out right away and cannot reach IRIS again until it is paired with a new code.",
      confirmLabel: d.current ? "Sign out" : "Revoke",
      destructive: true,
      run: async () => {
        try {
          // Signing out skips the mutation hook: its list refresh would only 401.
          if (d.current) await revokeDevice(d.device_id);
          else await revoke.mutateAsync(d.device_id);
        } catch (e) {
          toast.error(e instanceof Error ? e.message : "revoke failed");
          throw e;
        }
        if (d.current) {
          qc.clear();
          navigate(PAIR_PATH, { replace: true });
        } else {
          toast.success(`${d.name} revoked`);
        }
      },
    });

  return (
    <div className="space-y-6">
      {me.data && (
        <p className="text-xs text-fg-subtle">
          {me.data.kind === "service"
            ? "You are using the service secret (local dev proxy), not a paired device."
            : `This browser is paired as "${me.data.device?.name ?? "a device"}" with ${me.data.scope} scope.`}
        </p>
      )}

      <PairNew canPair={!readOnly} />

      <NotificationsSection />

      <Section title={`Devices${data ? ` (${live} active of ${devices.length})` : ""}`}>
        <QueryState
          loading={isLoading}
          error={isError}
          empty={devices.length === 0}
          emptyText="No devices paired yet."
        >
          <div className="space-y-2">
            {devices.map((d) => (
              <DeviceRow
                key={d.device_id}
                d={d}
                now={now}
                canRevoke={!readOnly || d.current}
                onRevoke={onRevoke}
              />
            ))}
          </div>
        </QueryState>
      </Section>

      <ConfirmDialog state={confirm} onClose={() => setConfirm(null)} />
    </div>
  );
}
