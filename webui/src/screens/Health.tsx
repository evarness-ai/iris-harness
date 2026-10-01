/* System Health screen (ADR-0069) — read-only diagnostics: services, credentials
 * and hardware as green/yellow/red/grey rows over GET /health, plus the health
 * watch's incidents (ADR-0116): what broke, what IRIS tried, whether it asked the
 * owner. A thin renderer; all health logic lives in the harness. */
import { useState } from "react";
import { Link } from "react-router-dom";
import { Section } from "@/components/layout";
import { Tag } from "@/components/Tag";
import { Button } from "@/components/ui/button";
import { HealthStateTag, Notice, QueryState, fmtDateTime } from "@/components/control/parts";
import { CopyBlock } from "@/components/CopyBlock";
import { ReconnectSheet, type SheetTarget } from "@/components/connections/ReconnectSheet";
import {
  useCompactContext,
  useCost,
  useContextHealth,
  useHealth,
  useHealthIncidents,
  useRunHealthWatch,
  useWritesEnabled,
} from "@/lib/queries";
import type { HealthCheck, HealthIncident } from "@/lib/control";

/* Order is the answer to "is it alive", top down (plan decision 28): the box
 * it runs on, then what runs on it, then what those need. "Hardware" is called
 * VM because on the cloud harness that is what it is — and from a phone the
 * question is about the VM, not about a category of check. Track 2 PR 5 adds
 * disk, containers and uptime to this same group. */
const GROUPS: { label: string; kind: HealthCheck["kind"] }[] = [
  { label: "VM", kind: "hardware" },
  { label: "Services", kind: "service" },
  { label: "Credentials", kind: "credential" },
];

/* A red credential row its plugin made reconnectable gets a Reconnect button
 * (Reconnect Google prototype): the fix is one tap from where the owner sees it. */
function CheckRow({ c, onReconnect }: { c: HealthCheck; onReconnect?: () => void }) {
  const right = c.endpoint ?? c.action;
  return (
    <div
      className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-lg border border-border bg-surface px-3 py-2"
      data-testid="health-check"
    >
      <HealthStateTag state={c.state} />
      <span className="text-xs font-medium text-fg">{c.target}</span>
      <span className="text-xs text-fg-muted">{c.detail}</span>
      {onReconnect && (
        <Button
          type="button"
          size="sm"
          className="ml-auto min-h-[44px] px-3.5 font-semibold"
          onClick={onReconnect}
        >
          Reconnect
        </Button>
      )}
      {right && (
        // On a phone this row wraps, and `ml-auto` then pushed the endpoint to
        // the right of a line of its own, reading as an unrelated item. It
        // wraps inline and breaks instead: a truncated URL on a 390px screen
        // hides the port and path, which is the part worth seeing.
        <code className="w-full break-all font-mono text-[11px] text-fg-subtle sm:ml-auto sm:w-auto sm:truncate">
          {right}
        </code>
      )}
    </div>
  );
}

/* One incident: its state, what IRIS tried, whether it told the owner, the fix. */
function incidentTag(i: HealthIncident) {
  if (i.state === "needs_user") return <Tag kind="bad">needs you</Tag>;
  if (i.state === "repairing") return <Tag kind="warn">repairing</Tag>;
  return <Tag kind="ok">{i.resolution === "user_fixed" ? "fixed" : "fixed itself"}</Tag>;
}

function IncidentRow({ i }: { i: HealthIncident }) {
  const open = i.resolved_at === null;
  return (
    <div
      className="space-y-1 rounded-lg border border-border bg-surface px-3 py-2"
      data-testid="health-incident"
    >
      <div className="flex flex-wrap items-center gap-2">
        {incidentTag(i)}
        <span className="text-xs font-medium text-fg">
          {i.target}
          {i.subject ? ` (${i.subject})` : ""}
        </span>
        <span className="text-xs text-fg-muted">{i.detail}</span>
        <span className="font-mono text-[11px] text-fg-subtle sm:ml-auto">
          {fmtDateTime(i.opened_at)}
        </span>
      </div>
      {i.repairs.length > 0 && (
        <ul className="space-y-0.5 text-[11px] text-fg-muted">
          {i.repairs.map((r, n) => (
            <li key={n}>
              <span className={r.ok ? "text-success" : "text-danger"}>{r.ok ? "✓" : "✗"}</span>{" "}
              {r.tried}
              {!r.ok && r.detail ? ` — ${r.detail}` : ""}
            </li>
          ))}
        </ul>
      )}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-fg-subtle">
        {i.notify_count > 0 && (
          <span>
            told you {i.notify_count}× (last {fmtDateTime(i.notified_at)})
          </span>
        )}
        {!open && <span>resolved {fmtDateTime(i.resolved_at)}</span>}
        {open && i.action && (
          <span>
            fix: <code className="font-mono text-fg">{i.action}</code>
          </span>
        )}
      </div>
    </div>
  );
}

/* Health-watch incidents (ADR-0116). Open ones always show; resolved history is
 * one click away so a quiet week doesn't bury what is broken now. */
function IncidentsSection() {
  const { data, isLoading, isError } = useHealthIncidents();
  const canWrite = useWritesEnabled();
  const run = useRunHealthWatch();
  const [showHistory, setShowHistory] = useState(false);

  const open = data?.incidents.filter((i) => i.resolved_at === null) ?? [];
  const resolved = data?.incidents.filter((i) => i.resolved_at !== null) ?? [];

  const actions = (
    <>
      {resolved.length > 0 && (
        <Button
          type="button"
          size="sm"
          variant="ghost"
          onClick={() => setShowHistory((v) => !v)}
        >
          {showHistory ? "Hide history" : `History (${resolved.length})`}
        </Button>
      )}
      {canWrite && (
        <Button
          type="button"
          size="sm"
          variant="outline"
          disabled={run.isPending}
          onClick={() => run.mutate()}
          title="Refresh health, try repairs, and notify now (POST /health/watch)"
        >
          {run.isPending ? "Checking…" : "Check now"}
        </Button>
      )}
    </>
  );

  return (
    <Section title="Incidents" actions={actions}>
      <QueryState
        loading={isLoading}
        error={isError}
        empty={!data}
        emptyText="No incident data."
      >
        {data && !data.enabled && (
          <Notice>The health watch is off (IRIS_HEALTH_WATCH_ENABLED=0).</Notice>
        )}
        {data?.enabled && (
          <div className="space-y-1.5">
            {open.length === 0 && (
              <div className="rounded-lg border border-border bg-surface px-3 py-2 text-xs text-fg-muted">
                Nothing broken. IRIS fixes what it can and messages you on every channel
                when it can’t.
              </div>
            )}
            {open.map((i) => (
              <IncidentRow key={i.id} i={i} />
            ))}
            {showHistory && resolved.map((i) => <IncidentRow key={i.id} i={i} />)}
          </div>
        )}
        {run.isError && (
          <div className="mt-2 text-[11px] text-danger">{(run.error as Error).message}</div>
        )}
        {run.data && run.data.events.length > 0 && (
          <div className="mt-2 text-[11px] text-fg-subtle">
            last check: {run.data.events.join(" · ")}
          </div>
        )}
      </QueryState>
    </Section>
  );
}

/* What IRIS has spent (plan decision 30). LLM calls only: the Azure
 * resource-group bill needs the `azure_cost` plugin and a managed identity,
 * which is track 4, so this card says so rather than implying it is the whole
 * bill.
 *
 * The off state is the one that matters. `CostLimiter` is opt-in and is the
 * ledger's only writer, so on a box that never enabled it every figure is a
 * truthful 0.00 that reads as "IRIS is free". The card refuses to show numbers
 * it cannot stand behind and prints the line that turns recording on. */
function CostSection() {
  const { data, isLoading, isError } = useCost();
  const tiers = Object.entries(data?.by_tier_usd ?? {}).filter(([, usd]) => usd > 0);

  return (
    <Section title="Cost">
      <QueryState loading={isLoading} error={isError} empty={!data} emptyText="No cost data.">
        {data && (
          <div className="space-y-2 rounded-lg border border-border bg-surface px-3 py-2">
            {data.recording ? (
              <>
                <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1">
                  <span className="font-mono text-xl font-semibold text-primary">
                    ${data.month_usd.toFixed(2)}
                  </span>
                  <span className="text-xs text-fg-muted">this month</span>
                  <span className="font-mono text-xs text-fg sm:ml-auto">
                    ${data.today_usd.toFixed(2)} today
                  </span>
                </div>
                {tiers.length > 0 && (
                  <div className="border-t border-border pt-2">
                    {tiers.map(([tier, usd]) => (
                      <div key={tier} className="flex items-center justify-between py-0.5 text-xs">
                        <span className="text-fg-subtle">{tier}</span>
                        <span className="font-mono text-fg">${usd.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                )}
                <p className="text-[11px] text-fg-subtle">
                  {data.enforcing && data.daily_cap_usd !== null ? (
                    <>
                      Refusing calls past{" "}
                      <span className="font-mono text-fg">${data.daily_cap_usd.toFixed(2)}</span> a
                      day.{" "}
                    </>
                  ) : (
                    <>Recording only — no call is refused on cost. </>
                  )}
                  LLM calls only, from {data.entries.toLocaleString()} ledger rows. Local tiers
                  cost nothing, so a $0.00 tier ran on your own hardware. These are estimates from
                  a price table, not your Azure bill.
                </p>
              </>
            ) : (
              <>
                <Notice tone="muted">
                  Nothing is recording LLM spend on this harness, so $0.00 here would mean “nobody
                  is counting”, not “nothing was spent”.
                </Notice>
                {data.enable_hint && (
                  <CopyBlock
                    label="add to the harness env, then restart"
                    text={data.enable_hint}
                  />
                )}
                {data.entries > 0 && (
                  <p className="text-[11px] text-fg-subtle">
                    The ledger still holds {data.entries.toLocaleString()} rows from when it was
                    on — ${data.month_usd.toFixed(2)} of them this month.
                  </p>
                )}
              </>
            )}
          </div>
        )}
      </QueryState>
    </Section>
  );
}

/* Context-budget panel (ADR-0081): how the harness bounds its working context —
 * conversation-window fill, the in-loop transcript/memory split, and how many
 * proactive surfaces the user has suppressed. A thin renderer over /context-health. */
function ContextBudgetSection() {
  const { data } = useContextHealth();
  const canWrite = useWritesEnabled();
  const compact = useCompactContext();
  if (!data?.available || !data.window || !data.budgets || !data.suppression) return null;
  const w = data.window;
  const pct = Math.round(w.fill_pct * 100);
  const lc = w.last_compaction;
  return (
    <Section title="Context budget">
      <div className="space-y-2 rounded-lg border border-border bg-surface px-3 py-2">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs font-medium text-fg">window</span>
          <div className="h-2 min-w-[7rem] flex-1 overflow-hidden rounded bg-border">
            <div
              className={w.near_full ? "h-full bg-red-500" : "h-full bg-emerald-500"}
              // eslint-disable-next-line react/forbid-dom-props -- dynamic width requires inline style
              style={{ width: `${Math.min(100, pct)}%` }}
            />
          </div>
          <span className="font-mono text-[11px] text-fg-subtle">
            {pct}% · {w.current_tokens}/{w.budget_tokens} tok
          </span>
          {canWrite && (
            <Button
              type="button"
              size="sm"
              variant="outline"
              disabled={compact.isPending}
              onClick={() => compact.mutate("default")}
              title="Summarize older turns now to reclaim window space"
            >
              Compact now
            </Button>
          )}
        </div>
        {lc && (
          <div className="text-[11px] text-fg-muted">
            last compaction: {lc.trigger} trigger, archived {lc.archived_count} turns (
            {lc.tokens_before}→{lc.tokens_after} tok)
          </div>
        )}
        <div className="text-[11px] text-fg-muted">
          budgets: transcript {data.budgets.transcript_budget} tok · memory{" "}
          {data.budgets.memory_budget} tok
          {data.budgets.last_transcript_evicted
            ? ` · last evicted ${data.budgets.last_transcript_evicted} tok`
            : ""}
        </div>
        <div className="text-[11px] text-fg-muted">
          suppression: {data.suppression.active_suppressions} active ·{" "}
          {data.suppression.total_feedback} feedback total
        </div>
      </div>
    </Section>
  );
}

export function HealthScreen() {
  const { data, isLoading, isError } = useHealth();
  const canWrite = useWritesEnabled();
  const [sheet, setSheet] = useState<SheetTarget | null>(null);

  return (
    <div className="space-y-6">
      <Section title="Summary">
        <QueryState
          loading={isLoading}
          error={isError}
          empty={!data}
          emptyText="No health snapshot yet — the health_tick heartbeat may not have run."
        >
          {data && (
            <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2">
              <HealthStateTag state={data.state} />
              <span className="text-xs text-fg-muted">{data.summary}</span>
              <span className="font-mono text-[11px] text-fg-subtle sm:ml-auto">
                sampled {fmtDateTime(data.sampled_at)}
              </span>
            </div>
          )}
        </QueryState>
      </Section>

      <IncidentsSection />

      <CostSection />

      {data &&
        GROUPS.map(({ label, kind }) => {
          const rows = data.checks.filter((c) => c.kind === kind);
          if (rows.length === 0) return null;
          return (
            <Section key={kind} title={label}>
              <div className="space-y-1.5">
                {rows.map((c, i) => {
                  const r = c.reconnect;
                  const fix =
                    r && canWrite && c.state === "red"
                      ? () => setSheet({ reconnect: r, verb: r.account ? "Reconnect" : "Connect" })
                      : undefined;
                  return <CheckRow key={`${c.kind}:${c.target}:${i}`} c={c} onReconnect={fix} />;
                })}
                {rows.some((c) => c.reconnect) && (
                  <Link
                    to="/settings#connections"
                    className="flex min-h-[44px] items-center justify-center rounded-lg border border-border text-sm text-fg hover:bg-surface-raised"
                  >
                    Manage connections
                  </Link>
                )}
              </div>
            </Section>
          );
        })}

      <ContextBudgetSection />
      <ReconnectSheet target={sheet} onClose={() => setSheet(null)} />
    </div>
  );
}
