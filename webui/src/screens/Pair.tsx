import { useEffect, useState, type FormEvent } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useQueryClient } from "@tanstack/react-query";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ThemeToggle } from "@/components/ThemeToggle";
import {
  ApiError,
  MAX_DEVICE_NAME,
  claimPairing,
  codeProblem,
  defaultDeviceName,
  formatCode,
  normalizeCode,
} from "@/lib/devices";
import { pairScreenReached, safeNext } from "@/lib/http";

/** What to tell the user for each way a claim can fail. A 400 is one uniform
 * answer by design (wrong, expired, used and voided codes look the same). */
function explain(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 429) {
      const wait = e.retryAfter ? `${e.retryAfter} seconds` : "a minute";
      return `Too many pairing attempts. Try again in ${wait}.`;
    }
    if (e.status === 400) {
      return "That code was not accepted. Codes last 5 minutes and stop working after 5 wrong tries — run `iris device pair` again for a fresh one.";
    }
    if (e.status === 422) return `Check the code and the device name. (${e.message})`;
    return e.message;
  }
  return "Could not reach IRIS. Check the connection and try again.";
}

/** Pair this browser (ADR-0117). Lives OUTSIDE the app shell: the shell's own
 * data calls would 401 here. The token arrives as an HttpOnly cookie — this
 * screen never sees it. */
export function PairScreen() {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const next = safeNext(params.get("next"));
  const [code, setCode] = useState("");
  const [name, setName] = useState(defaultDeviceName);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(pairScreenReached, []);

  const typed = normalizeCode(code);
  const problem = codeProblem(code);
  const ready = !problem && name.trim().length > 0;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!ready || busy) return;
    setBusy(true);
    setError(null);
    try {
      await claimPairing(formatCode(code), name.trim());
      qc.clear(); // drop the 401 errors cached before pairing
      navigate(next, { replace: true });
    } catch (err) {
      setError(explain(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-bg text-fg">
      <header className="flex items-center justify-between px-4 py-3">
        <div>
          <div className="text-base font-bold tracking-tight text-primary">IRIS</div>
          <div className="text-[10px] uppercase tracking-widest text-fg-subtle">Console</div>
        </div>
        <ThemeToggle />
      </header>

      <main className="mx-auto w-full max-w-sm flex-1 px-4 pb-10 pt-6">
        <h1 className="text-lg font-semibold text-fg">Pair this device</h1>
        <p className="mt-1 text-sm text-fg-muted">
          This browser is not paired with IRIS yet. Get a pairing code, then enter it here.
        </p>

        <ol className="mt-4 space-y-2 rounded-lg border border-border bg-surface p-3 text-xs text-fg-muted">
          <li>
            On the machine running IRIS:{" "}
            <code className="rounded bg-bg px-1.5 py-0.5 font-mono text-fg">iris device pair</code>
          </li>
          <li>
            Or, on a device that is already paired: <span className="text-fg">Devices</span> →{" "}
            <span className="text-fg">Pair a new device</span>.
          </li>
        </ol>

        <form onSubmit={submit} className="mt-5 space-y-4" noValidate>
          <div>
            <label htmlFor="pair-code" className="text-xs font-medium text-fg-muted">
              Pairing code
            </label>
            <Input
              id="pair-code"
              value={formatCode(code)}
              onChange={(e) => setCode(e.target.value)}
              placeholder="ABCD-EFGH"
              autoFocus
              autoComplete="one-time-code"
              autoCapitalize="characters"
              autoCorrect="off"
              spellCheck={false}
              inputMode="text"
              aria-describedby="pair-code-hint"
              className="mt-1 h-12 text-center font-mono text-xl tracking-[0.2em] text-fg"
            />
            <p id="pair-code-hint" className="mt-1 text-[11px] text-fg-subtle">
              {typed.length > 0 && problem
                ? problem
                : "8 characters, shown as ABCD-EFGH. The dash and letter case do not matter."}
            </p>
          </div>

          <div>
            <label htmlFor="pair-name" className="text-xs font-medium text-fg-muted">
              Device name
            </label>
            <Input
              id="pair-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              maxLength={MAX_DEVICE_NAME}
              autoComplete="off"
              className="mt-1 text-fg"
            />
            <p className="mt-1 text-[11px] text-fg-subtle">
              How this browser appears in the Devices list, so you can revoke it later.
            </p>
          </div>

          {error && (
            <div
              role="alert"
              className="rounded-lg border border-danger/40 bg-danger/10 p-3 text-xs text-danger"
            >
              {error}
            </div>
          )}

          <Button type="submit" className="h-11 w-full" disabled={!ready || busy}>
            {busy ? "Pairing…" : "Pair this device"}
          </Button>
        </form>
      </main>
    </div>
  );
}
