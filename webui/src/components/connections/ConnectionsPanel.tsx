/* Settings > Connections (Reconnect Google prototype, signed off 2026-09-26).
 *
 * One card per account; a row per service with its state and a Reconnect (revoked)
 * or Connect (not connected) button; "Add a <provider> account"; a setup banner
 * while the server cannot run the flow; the Web client upload (the prototype's
 * setup step 3). After the provider sends the browser back, the server redirects
 * here with `?connect=<result>` and this tab shows the result screen once.
 *
 * Generic: everything provider-specific comes from the rows' `reconnect`
 * descriptors and the group's setup route, so a build without the plugins that
 * declare them shows an empty tab, not dead buttons. */
import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Check, TriangleAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import { QueryState } from "@/components/control/parts";
import { ReconnectSheet, type SheetTarget } from "@/components/connections/ReconnectSheet";
import {
  STATE_TEXT,
  beginReconnect,
  connectionGroups,
  getConnectionSetup,
  readConnectResult,
  stateOf,
  uploadConnectionClient,
  type ConnState,
  type ConnectResult,
  type ConnectionGroup,
  type ConnectionSetup,
} from "@/lib/connections";
import { useHealth, useWritesEnabled } from "@/lib/queries";
import type { HealthCheck, Reconnect } from "@/lib/control";

const DOT: Record<ConnState, string> = {
  ok: "bg-success",
  revoked: "bg-danger",
  warn: "bg-warning",
  none: "bg-fg-subtle",
};

function useSetup(route: string) {
  return useQuery({
    queryKey: ["connection-setup", route],
    queryFn: () => getConnectionSetup(route),
    staleTime: 30_000,
  });
}

export function ConnectionsTab() {
  const { data, isLoading, isError } = useHealth();
  const [result, setResult] = useState<ConnectResult | null>(() =>
    readConnectResult(window.location.search),
  );
  // The server's redirect carries the outcome in the query; read it once, then drop
  // it, so a reload does not show the same result again.
  useEffect(() => {
    if (window.location.search) window.history.replaceState(null, "", "/settings#connections");
  }, []);
  const [justConnected, setJustConnected] = useState<string[]>([]);
  const groups = useMemo(() => connectionGroups(data?.checks ?? []), [data]);
  const qc = useQueryClient();

  if (result) {
    return (
      <ResultScreen
        result={result}
        checks={data?.checks ?? []}
        onDone={() => {
          if (result.result === "connected" && result.provider && result.account) {
            setJustConnected((k) => [...k, `${result.provider}:${result.account}`]);
          }
          setResult(null);
          void qc.invalidateQueries({ queryKey: ["health"] });
        }}
      />
    );
  }

  return (
    <QueryState loading={isLoading} error={isError} empty={!data} emptyText="No health data.">
      <div className="space-y-6" data-testid="connections">
        {groups.length === 0 && (
          <p className="rounded-lg border border-border bg-surface px-3 py-2 text-xs text-fg-muted">
            Nothing on this server can be reconnected from here.
          </p>
        )}
        {groups.map((g) => (
          <GroupSection key={g.group} group={g} justConnected={justConnected} />
        ))}
      </div>
    </QueryState>
  );
}

function GroupSection({
  group,
  justConnected,
}: {
  group: ConnectionGroup;
  justConnected: string[];
}) {
  const canWrite = useWritesEnabled();
  const { data: setup } = useSetup(group.setupRoute);
  const [sheet, setSheet] = useState<SheetTarget | null>(null);
  const ready = !!setup && setup.configured && setup.public_url_set;
  const providers =
    setup?.providers ??
    [...new Map(group.rows.map((c) => [c.reconnect!.provider, c.reconnect!.label])).entries()].map(
      ([provider, label]) => ({ provider, label }),
    );

  const target = (provider: string, label: string, account: string | null): Reconnect => ({
    route: group.route,
    setup_route: group.setupRoute,
    group: group.group,
    group_label: group.label,
    provider,
    label,
    account,
  });
  const addProvider = providers.find((p) => p.provider === setup?.add_provider) ?? providers[0];

  return (
    <section className="space-y-4">
      {setup && !ready && <SetupBanner setup={setup} />}
      {group.accounts.map((account) => (
        <div
          key={account}
          className="overflow-hidden rounded-xl border border-border bg-surface"
          data-testid="connection-account"
        >
          <div className="break-all border-b border-border px-3.5 py-3 text-sm font-semibold text-fg">
            {account}
          </div>
          {providers.map((p) => {
            const row = group.rows.find(
              (c) => c.reconnect?.provider === p.provider && c.reconnect?.account === account,
            );
            const fresh = justConnected.includes(`${p.provider}:${account}`);
            const state: ConnState = fresh ? "ok" : stateOf(row);
            const verb = state === "none" ? "Connect" : "Reconnect";
            return (
              <div
                key={p.provider}
                className="flex items-center gap-2.5 px-3.5 py-2.5"
                data-testid="connection-row"
              >
                <span className={`h-2.5 w-2.5 shrink-0 rounded-full ${DOT[state]}`} />
                <div className="flex min-w-0 flex-1 flex-col">
                  <span className="text-sm text-fg">{p.label}</span>
                  <span className="text-xs text-fg-muted">
                    {fresh ? "connected — health updates on the next check" : STATE_TEXT[state]}
                  </span>
                </div>
                {state !== "ok" && canWrite && (
                  <Button
                    type="button"
                    size="sm"
                    variant={state === "none" ? "outline" : "default"}
                    className="min-h-[44px] px-3.5 font-semibold"
                    disabled={!ready}
                    onClick={() => setSheet({ reconnect: target(p.provider, p.label, account), verb })}
                  >
                    {verb}
                  </Button>
                )}
              </div>
            );
          })}
        </div>
      ))}
      {canWrite && addProvider && (
        <Button
          type="button"
          variant="outline"
          className="min-h-[44px] w-full border-dashed text-fg-muted"
          disabled={!ready}
          onClick={() =>
            setSheet({
              reconnect: target(addProvider.provider, addProvider.label, null),
              verb: "Connect",
            })
          }
        >
          Add a {group.label} account
        </Button>
      )}
      {canWrite && setup && <ClientUpload route={group.setupRoute} setup={setup} />}
      <p className="text-xs text-fg-subtle">
        Only a paired control device (or the owner's secret) can reconnect. Tokens stay on the
        server; this phone never sees them.
      </p>
      <ReconnectSheet target={sheet} onClose={() => setSheet(null)} />
    </section>
  );
}

function SetupBanner({ setup }: { setup: ConnectionSetup }) {
  return (
    <div
      className="space-y-1 rounded-lg border border-warning/40 bg-warning/10 px-3.5 py-3 text-[13px] text-fg"
      data-testid="connections-setup-banner"
    >
      {!setup.configured && (
        <p>
          One-time setup needed: this server has no {setup.group_label} “Web application” client
          yet, so Reconnect is off. The setup takes about 5 minutes in {setup.group_label} Cloud
          Console.
        </p>
      )}
      {!setup.public_url_set && (
        <p>
          IRIS_PUBLIC_URL is not set on the server, so there is no address {setup.group_label} could
          send you back to. Set it with <code className="font-mono">set_server_env.sh --public-url</code>.
        </p>
      )}
    </div>
  );
}

/* Setup step 3: give the server the Web client JSON. Stored in the server's keyring;
 * the answer says only whether it is configured, never the secret. */
function ClientUpload({ route, setup }: { route: string; setup: ConnectionSetup }) {
  const qc = useQueryClient();
  const [open, setOpen] = useState(!setup.configured);
  const [busy, setBusy] = useState(false);
  // The toast fades after a few seconds and the form folds away, which left the owner
  // with no sign the upload took (2026-09-26). The outcome also stays on the panel.
  const [saved, setSaved] = useState<"ok" | "unlisted" | null>(null);
  const input = useRef<HTMLInputElement>(null);

  const onFile = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    try {
      const answer = await uploadConnectionClient(route, await file.text());
      qc.setQueryData(["connection-setup", route], answer);
      if (answer.redirect_uri_listed === false) {
        toast.warning(`Saved, but that client does not list ${answer.redirect_uri}`);
        setSaved("unlisted");
      } else {
        toast.success("Web client saved on the server");
        setSaved("ok");
      }
      setOpen(false);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "upload failed");
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  };

  if (!open) {
    return (
      <div className="space-y-1">
        {saved === "unlisted" ? (
          <p
            role="status"
            data-testid="connections-client-status"
            className="flex items-start gap-1.5 text-xs text-warning"
          >
            <TriangleAlert className="mt-0.5 size-3.5 shrink-0" aria-hidden="true" />
            <span>
              {setup.group_label} Web client saved, but it does not list{" "}
              <code className="break-all font-mono">{setup.redirect_uri}</code> as a redirect
              URI. Add it in the Cloud Console, or {setup.group_label} will refuse the reconnect.
            </span>
          </p>
        ) : (
          setup.configured && (
            <p
              role="status"
              data-testid="connections-client-status"
              className="flex items-center gap-1.5 text-xs text-success"
            >
              <Check className="size-3.5 shrink-0" aria-hidden="true" />
              <span>
                {setup.group_label} Web client saved on the server
                {saved === "ok" ? " just now" : ""}. Reconnect is ready.
              </span>
            </p>
          )
        )}
        <button
          type="button"
          className="min-h-[44px] text-xs text-fg-muted underline-offset-2 hover:underline"
          onClick={() => setOpen(true)}
        >
          Replace the {setup.group_label} Web client
        </button>
      </div>
    );
  }
  return (
    <div className="space-y-2 rounded-lg border border-border bg-surface px-3.5 py-3 text-xs">
      <p className="font-semibold text-fg">{setup.group_label} Web client</p>
      {setup.redirect_uri && (
        <p className="text-fg-muted">
          Authorised redirect URI:{" "}
          <code className="break-all font-mono text-fg" data-testid="connections-redirect-uri">
            {setup.redirect_uri}
          </code>
        </p>
      )}
      <p className="text-fg-muted">
        Download the client JSON from the Cloud Console and add it here. It is stored on the
        server and never shown again.
      </p>
      <label className="inline-flex min-h-[44px] cursor-pointer items-center">
        <span className="sr-only">Web client JSON</span>
        <input
          ref={input}
          type="file"
          accept="application/json,.json"
          disabled={busy}
          data-testid="connections-client-file"
          onChange={(e) => void onFile(e.target.files?.[0])}
          className="text-xs text-fg-muted file:mr-2 file:min-h-[36px] file:rounded file:border file:border-border file:bg-surface-raised file:px-2 file:text-fg"
        />
      </label>
    </div>
  );
}

/* The landing after the provider's redirect: reconnected, or why not. */
function ResultScreen({
  result,
  checks,
  onDone,
}: {
  result: ConnectResult;
  checks: HealthCheck[];
  onDone: () => void;
}) {
  const [error, setError] = useState<string | null>(null);
  const known = checks.find((c) => c.reconnect?.provider === result.provider)?.reconnect;
  const label = known?.label ?? result.provider ?? "Connection";
  const provider = known?.group_label ?? "the provider";
  const ok = result.result === "connected";

  const copy: Record<Exclude<ConnectResult["result"], "connected">, [string, string]> = {
    cancelled: [
      "Nothing changed",
      `You cancelled on ${provider}’s page. The connection still needs attention.`,
    ],
    wrong_account: [
      "Different account",
      `You approved as ${result.approved ?? "another account"}, but this reconnect is for ${
        result.account ?? "another account"
      }. The server saved nothing.`,
    ],
    expired: ["Link expired", "Each reconnect link works once, for 10 minutes. Start again."],
    failed: [
      "Reconnect failed",
      `${provider} did not finish the reconnect${
        result.reason ? ` (${result.reason.replace(/_/g, " ")})` : ""
      }. The server saved nothing.`,
    ],
  };
  const [title, body] = ok ? ["", ""] : copy[result.result as keyof typeof copy];

  const again = known && result.provider
    ? () =>
        beginReconnect({ ...known, account: result.account }).catch((e: unknown) =>
          setError(e instanceof Error ? e.message : "could not start"),
        )
    : null;

  return (
    <div
      className="flex min-h-[60vh] flex-col items-center justify-center gap-4 px-2 text-center"
      data-testid="connect-result"
    >
      <span
        className={`flex h-16 w-16 items-center justify-center rounded-full ${
          ok ? "bg-success/15 text-success" : "bg-warning/15 text-warning"
        }`}
      >
        {ok ? <Check size={32} /> : <TriangleAlert size={30} />}
      </span>
      {ok ? (
        <>
          <h2 className="text-xl font-semibold text-fg">{label} reconnected</h2>
          <span className="break-all text-sm text-fg-muted">{result.account}</span>
          <div className="rounded-lg border border-border bg-surface px-3.5 py-2.5 text-[13px] text-fg-muted">
            Live check with {provider}: connected. The health alert clears on the next check.
          </div>
          <Button type="button" className="min-h-[48px] px-6 text-[15px] font-semibold" onClick={onDone}>
            Back to connections
          </Button>
        </>
      ) : (
        <>
          <h2 className="text-xl font-semibold text-fg">{title}</h2>
          <span className="text-sm text-fg-muted">{body}</span>
          {error && <span className="text-xs text-danger">{error}</span>}
          <div className="flex w-full max-w-sm flex-col gap-2">
            {again && (
              <Button
                type="button"
                className="min-h-[48px] text-[15px] font-semibold"
                onClick={() => void again()}
              >
                Try again
              </Button>
            )}
            <Button type="button" variant="ghost" className="min-h-[44px]" onClick={onDone}>
              Back to connections
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
