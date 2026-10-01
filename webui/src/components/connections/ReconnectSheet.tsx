/* The confirm sheet before leaving for the provider's consent page (Reconnect
 * Google prototype, "confirm"). Three steps, so the owner knows the phone will
 * leave the app and come back, and that the token never lands on it. Shared by
 * Health (the Reconnect button on a revoked row) and Settings > Connections. */
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { beginReconnect } from "@/lib/connections";
import type { Reconnect } from "@/lib/control";

export interface SheetTarget {
  reconnect: Reconnect;
  /** "Reconnect" a revoked credential, "Connect" one that never was. */
  verb: "Reconnect" | "Connect";
}

export function ReconnectSheet({
  target,
  onClose,
}: {
  target: SheetTarget | null;
  onClose: () => void;
}) {
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const continueRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    setPending(false);
    setError(null);
    if (target) continueRef.current?.focus();
  }, [target]);

  useEffect(() => {
    if (!target) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !pending) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [target, pending, onClose]);

  if (!target) return null;
  const r = target.reconnect;
  const provider = r.group_label;

  const go = async () => {
    setPending(true);
    setError(null);
    try {
      await beginReconnect(r); // navigates away on success
    } catch (e) {
      setError(e instanceof Error ? e.message : "could not start");
      setPending(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/60 sm:items-center">
      <button
        type="button"
        aria-label="Close"
        className="absolute inset-0 cursor-default"
        onClick={() => !pending && onClose()}
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="reconnect-title"
        data-testid="reconnect-sheet"
        className="relative w-full max-w-lg space-y-4 rounded-t-2xl border border-border bg-surface px-5 pb-[calc(1.75rem+env(safe-area-inset-bottom,0px))] pt-6 shadow-lg sm:rounded-2xl"
      >
        <div className="space-y-1">
          <h2 id="reconnect-title" className="text-lg font-semibold text-fg">
            {target.verb} {r.label}
          </h2>
          <p className="break-all text-sm text-fg-muted">{r.account ?? "a new account"}</p>
        </div>
        <ol className="space-y-2.5 text-sm text-fg">
          <li className="flex gap-2.5">
            <span className="font-semibold text-primary">1</span>
            <span>You sign in on {provider}'s own page and approve access.</span>
          </li>
          <li className="flex gap-2.5">
            <span className="font-semibold text-primary">2</span>
            <span>{provider} sends you straight back here.</span>
          </li>
          <li className="flex gap-2.5">
            <span className="font-semibold text-primary">3</span>
            <span>
              The server stores the new token and checks it live. It never reaches this phone.
            </span>
          </li>
        </ol>
        {error && (
          <p role="alert" className="text-xs text-danger">
            {error}
          </p>
        )}
        <div className="flex flex-col gap-2">
          <Button
            ref={continueRef}
            type="button"
            className="min-h-[48px] text-[15px] font-semibold"
            disabled={pending}
            onClick={go}
          >
            {pending ? "Opening…" : `Continue to ${provider}`}
          </Button>
          <Button
            type="button"
            variant="ghost"
            className="min-h-[44px]"
            disabled={pending}
            onClick={onClose}
          >
            Cancel
          </Button>
        </div>
      </div>
    </div>
  );
}
