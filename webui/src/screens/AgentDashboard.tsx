/* Per-agent ops dashboard (ADR-0074) — pending actions, recent runs, and key
 * stores for one agent over GET /agents/{name}. Thin renderer; all composition
 * happens in the harness. */
import { Link, useParams } from "react-router-dom";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Tag } from "@/components/Tag";
import { CopyBlock } from "@/components/CopyBlock";
import { Section } from "@/components/layout";
import { QueryState, fmtDateTime } from "@/components/control/parts";
import {
  ConfigFiles,
  PluginStatusTag,
  ToolRow,
  kindLabel,
  STALE_API,
  sourceKind,
} from "@/components/plugins/parts";
import {
  useAgentDashboard,
  useAgentMetrics,
  usePlugin,
  useSetAgentToggle,
  useWritesEnabled,
} from "@/lib/queries";
import type { AgentRun, AgentSettings, AgentStores, PendingAction } from "@/lib/control";

function runTone(status: string): "ok" | "bad" | "warn" | "info" {
  const s = status.toLowerCase();
  if (s.includes("success") || s.includes("complete")) return "ok";
  if (s.includes("fail") || s.includes("error")) return "bad";
  if (s.includes("skip") || s.includes("pending") || s.includes("running")) return "warn";
  return "info";
}

function PendingActions({ actions }: { actions: PendingAction[] }) {
  return (
    <Section title={`Pending actions (${actions.length})`}>
      {actions.length === 0 ? (
        <p className="text-xs text-fg-subtle">Nothing blocked — resolve actions in the Action Center.</p>
      ) : (
        <div className="space-y-2">
          {actions.map((a) => (
            <div key={a.id} className="rounded-lg border border-border bg-surface p-3">
              <span className="text-sm font-medium text-fg">{a.title}</span>
              {a.action.kind === "copy_command" && a.action.command ? (
                <CopyBlock label="run locally" text={a.action.command} />
              ) : null}
            </div>
          ))}
          <Link to="/actions" className="text-xs text-primary hover:underline">
            Open in Action Center →
          </Link>
        </div>
      )}
    </Section>
  );
}

function RecentRuns({ runs }: { runs: AgentRun[] }) {
  return (
    <Section title={`Recent runs (${runs.length})`}>
      {runs.length === 0 ? (
        <p className="text-xs text-fg-subtle">No background runs recorded for this agent yet.</p>
      ) : (
        <div className="space-y-2">
          {runs.map((r, i) => (
            <div key={`${r.name}-${i}`} className="rounded-lg border border-border bg-surface p-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-[13px] text-fg">{r.name}</span>
                <Tag kind={runTone(r.status)}>{r.status}</Tag>
                {r.finished_at && (
                  <span className="ml-auto font-mono text-[11px] text-fg-subtle">
                    {fmtDateTime(r.finished_at)}
                  </span>
                )}
              </div>
              {r.error ? (
                <CopyBlock label="error" text={r.error} tone="text-danger" />
              ) : r.output ? (
                <CopyBlock label="output" text={r.output} />
              ) : null}
            </div>
          ))}
        </div>
      )}
    </Section>
  );
}

function Stores({ stores }: { stores: AgentStores }) {
  const { accounts, statements } = stores;
  return (
    <Section title="Stores">
      <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2">
        <Tag kind="info">{accounts} accounts</Tag>
        <Tag kind="info">{statements.total} statements</Tag>
        {statements.unextracted > 0 && <Tag kind="warn">{statements.unextracted} unextracted</Tag>}
      </div>
      {statements.groups.length > 0 && (
        <div className="mt-2 space-y-1.5">
          {statements.groups.map((g, i) => (
            <div
              key={`${g.institution}-${g.doc_label}-${i}`}
              className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2"
            >
              <span className="text-xs font-medium text-fg">{g.institution}</span>
              <span className="text-xs text-fg-muted">
                {g.count} {g.doc_label}
                {g.count === 1 ? "" : "s"}
              </span>
              {g.extracted < g.count && (
                <Tag kind="warn">
                  {g.extracted}/{g.count} extracted
                </Tag>
              )}
              {g.latest && (
                <span className="ml-auto font-mono text-[11px] text-fg-subtle">latest {g.latest}</span>
              )}
            </div>
          ))}
        </div>
      )}
    </Section>
  );
}

function Settings({ settings, name }: { settings: AgentSettings; name: string }) {
  const { llm, toggles, heartbeats } = settings;
  const canWrite = useWritesEnabled();
  const setToggle = useSetAgentToggle(name);

  const onToggle = (key: string, enabled: boolean) =>
    setToggle.mutate(
      { key, enabled },
      {
        onSuccess: (res) => {
          const r = res.applied[0];
          toast.success(
            `${key} ${enabled ? "on" : "off"}${r?.restart_required ? " (restart to apply)" : ""}`,
          );
        },
        onError: (e) => toast.error(e instanceof Error ? e.message : "update failed"),
      },
    );

  return (
    <Section
      title="Settings"
      actions={
        <span className="text-[11px] text-fg-subtle">{canWrite ? "editable" : "read-only"}</span>
      }
    >
      <div className="space-y-3">
        {llm.length > 0 && (
          <div>
            <div className="mb-1 text-[11px] font-semibold text-fg-muted">LLM tier</div>
            <div className="space-y-1.5">
              {llm.map((l) => (
                <div
                  key={l.intent}
                  className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2"
                >
                  <span className="font-mono text-[11px] text-fg-subtle">{l.intent}</span>
                  <Tag kind="info">{l.tier}</Tag>
                  <span className="ml-auto font-mono text-[11px] text-fg-muted">
                    {l.model} ({l.provider})
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {toggles.length > 0 && (
          <div>
            <div className="mb-1 text-[11px] font-semibold text-fg-muted">Toggles</div>
            <div className="space-y-1.5">
              {toggles.map((t) => (
                <div
                  key={t.key}
                  className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2"
                >
                  <Tag kind={t.enabled ? "ok" : "warn"}>{t.enabled ? "on" : "off"}</Tag>
                  <span className="text-xs font-medium text-fg">{t.label}</span>
                  <code className="font-mono text-[10.5px] text-fg-subtle">{t.key}</code>
                  <span className="ml-auto text-[10.5px] text-fg-subtle">
                    default {t.default ? "on" : "off"} · applies {t.applies}
                  </span>
                  {canWrite && t.guarded && (
                    <Link
                      to="/settings#agents"
                      className="text-[11px] font-medium text-primary hover:underline"
                    >
                      Guarded: change in Settings
                    </Link>
                  )}
                  {canWrite && !t.guarded && (
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      disabled={setToggle.isPending}
                      onClick={() => onToggle(t.key, !t.enabled)}
                    >
                      {t.enabled ? "Turn off" : "Turn on"}
                    </Button>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {heartbeats.length > 0 && (
          <div>
            <div className="mb-1 text-[11px] font-semibold text-fg-muted">Schedule</div>
            <div className="space-y-1.5">
              {heartbeats.map((h) => (
                <div
                  key={h.name}
                  className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2"
                >
                  <span className="font-mono text-[13px] text-fg">{h.name}</span>
                  <Tag kind={h.enabled ? "ok" : "warn"}>{h.enabled ? "enabled" : "disabled"}</Tag>
                  <span className="ml-auto font-mono text-[11px] text-fg-subtle">{h.schedule}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Section>
  );
}

function pct(x: number | null): string {
  return x === null ? "—" : `${Math.round(x * 100)}%`;
}

function Metrics({ name }: { name: string }) {
  const { data, isLoading } = useAgentMetrics(name);
  if (isLoading) return null;
  const m = data?.metrics;
  return (
    <Section title="Metrics" actions={<span className="text-[11px] text-fg-subtle">last 7d</span>}>
      {!data?.available || !m || m.volume === 0 ? (
        <p className="text-xs text-fg-subtle">
          No measured turns yet (metrics come from the learning store).
        </p>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2">
            <Tag kind="info">{m.volume} turns</Tag>
            <Tag kind="ok">{pct(m.success_rate)} success</Tag>
            {m.correction_rate !== null && m.correction_rate > 0 && (
              <Tag kind="warn">{pct(m.correction_rate)} corrected</Tag>
            )}
            {m.avg_tokens !== null && (
              <span className="ml-auto font-mono text-[11px] text-fg-subtle">
                {Math.round(m.avg_tokens)} tok/turn
              </span>
            )}
          </div>
          {m.by_tier.length > 0 && (
            <div className="mt-2 space-y-1.5">
              {m.by_tier.map((t) => (
                <div
                  key={t.tier}
                  className="flex items-center gap-2 rounded-lg border border-border bg-surface px-3 py-2"
                >
                  <span className="font-mono text-[11px] text-fg-subtle">{t.tier}</span>
                  <span className="text-xs text-fg-muted">{pct(t.success_rate)} success</span>
                  <span className="ml-auto font-mono text-[11px] text-fg-subtle">
                    n={t.volume}
                  </span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </Section>
  );
}

/** The plugin that registered this agent: identity, its tools, and its YAML. */
function AgentPlugin({ plugin }: { plugin: string | null | undefined }) {
  const { data, isLoading, isError } = usePlugin(plugin ?? "");
  if (plugin === undefined) {
    return (
      <Section title="Plugin">
        <p className="text-xs text-warning">{STALE_API}</p>
      </Section>
    );
  }
  if (plugin === null) {
    return (
      <Section title="Plugin">
        <p className="text-xs text-fg-subtle">
          Registered by the harness core itself — no plugin provides this agent.
        </p>
      </Section>
    );
  }
  return (
    <Section
      title="Plugin"
      actions={
        <Link to={`/agents/plugins/${plugin}`} className="text-xs text-primary hover:underline">
          Open plugin →
        </Link>
      }
    >
      <QueryState
        loading={isLoading}
        error={isError}
        empty={!data}
        emptyText="The plugin is not in the active profile."
      >
        {data && (
          <div className="space-y-3">
            <div className="rounded-lg border border-border bg-surface p-3">
              <div className="flex flex-wrap items-center gap-2">
                <Link
                  to={`/agents/plugins/${data.name}`}
                  className="font-mono text-[13px] font-semibold text-fg hover:text-primary"
                >
                  {data.name}
                </Link>
                {data.version && (
                  <span className="font-mono text-[11px] text-fg-subtle">v{data.version}</span>
                )}
                <PluginStatusTag status={data.status} />
                <span className="ml-auto font-mono text-[11px] text-fg-subtle">
                  {sourceKind(data.source)} · {data.trust}
                </span>
              </div>
              {data.description && (
                <p className="mt-1.5 line-clamp-3 text-xs text-fg-muted">{data.description}</p>
              )}
              <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-fg-subtle">
                {Object.entries(data.registration_counts).map(([kind, n]) => (
                  <span key={kind}>{kindLabel(kind, n)}</span>
                ))}
              </div>
            </div>
            <div>
              <div className="mb-1 text-[11px] font-semibold text-fg-muted">
                Tools declared by {data.name} ({data.tools.length})
              </div>
              {data.tools.length === 0 ? (
                <p className="text-xs text-fg-subtle">The manifest declares no tools.</p>
              ) : (
                <div className="space-y-1.5">
                  {data.tools.map((t) => (
                    <ToolRow key={t.name} tool={t} />
                  ))}
                </div>
              )}
            </div>
            <div>
              <div className="mb-1 text-[11px] font-semibold text-fg-muted">
                Configuration ({data.files.length} YAML, read-only)
              </div>
              <ConfigFiles files={data.files} emptyText="No YAML found for this plugin." />
            </div>
          </div>
        )}
      </QueryState>
    </Section>
  );
}

export function AgentDashboardScreen() {
  const { name = "" } = useParams();
  const { data, isLoading, isError } = useAgentDashboard(name);

  return (
    <div className="space-y-6">
      <Link to="/agents" className="text-xs text-primary hover:underline">
        ← All agents
      </Link>
      <QueryState
        loading={isLoading}
        error={isError}
        empty={!data}
        emptyText="No such agent in this runtime."
      >
        {data && (
          <>
            <div className="rounded-lg border border-border bg-surface p-3">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-fg">{data.title}</span>
                {data.source_kind && <Tag kind="info">{data.source_kind}</Tag>}
              </div>
              <p className="mt-1 text-xs text-fg-muted">{data.description}</p>
            </div>
            <AgentPlugin plugin={data.plugin} />
            <Metrics name={data.name} />
            <PendingActions actions={data.pending_actions.actions} />
            <RecentRuns runs={data.recent_runs} />
            {data.stores && <Stores stores={data.stores} />}
            {data.settings && <Settings settings={data.settings} name={data.name} />}
          </>
        )}
      </QueryState>
    </div>
  );
}
