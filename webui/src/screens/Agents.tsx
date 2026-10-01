/* Agents index (ADR-0074) — a card per registered agent over GET /agents, naming the
 * plugin that registered it, then the plugin inventory (every plugin in the profile,
 * with or without an agent). Thin renderer; the registry lives in the harness. */
import { Link } from "react-router-dom";
import { Tag } from "@/components/Tag";
import { Section } from "@/components/layout";
import { QueryState } from "@/components/control/parts";
import { PluginInventory } from "@/components/plugins/PluginInventory";
import { useAgents } from "@/lib/queries";

export function AgentsScreen() {
  const { data, isLoading, isError } = useAgents();
  const agents = data?.agents ?? [];

  return (
    <>
      <Section title={`Agents (${agents.length})`}>
        <QueryState
          loading={isLoading}
          error={isError}
          empty={agents.length === 0}
          emptyText="No agents registered in this runtime."
        >
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {agents.map((a) => (
              <Link
                key={a.name}
                to={`/agents/${a.name}`}
                className="rounded-lg border border-border bg-surface p-3 transition-colors hover:border-border-strong hover:bg-surface-raised"
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-semibold text-fg">{a.title}</span>
                  {a.source_kind && <Tag kind="info">{a.source_kind}</Tag>}
                </div>
                <p className="mt-1 text-xs text-fg-muted">{a.description || a.name}</p>
                <p className="mt-2 font-mono text-[11px] text-fg-subtle">
                  {a.plugin === undefined ? "" : a.plugin ? `plugin: ${a.plugin}` : "core"}
                </p>
              </Link>
            ))}
          </div>
        </QueryState>
      </Section>
      <PluginInventory />
    </>
  );
}
