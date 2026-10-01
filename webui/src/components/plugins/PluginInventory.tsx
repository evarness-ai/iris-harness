/* The plugin inventory over GET /plugins — a card per plugin in the active profile,
 * then the profile that chose them and its YAML. Rendered on the Agents page:
 * plugins are what provide the agents. Read-only. */
import { Tag } from "@/components/Tag";
import { Section } from "@/components/layout";
import { QueryState } from "@/components/control/parts";
import { usePlugins } from "@/lib/queries";
import type { PluginInventory as Inventory } from "@/lib/control";
import { ConfigFiles, PluginCard, STALE_API } from "./parts";

function ProfilePanel({ inventory }: { inventory: Inventory }) {
  const { profile, totals } = inventory;
  if (!profile) return null;
  const others = profile.available_profiles.filter((p) => p !== profile.name);
  return (
    <Section title="Profile">
      <div className="space-y-3 rounded-lg border border-border bg-surface p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-sm font-semibold text-fg">{profile.name}</span>
          {Object.entries(totals)
            .filter(([, n]) => n > 0)
            .map(([status, n]) => (
              <Tag
                key={status}
                kind={
                  status === "failed"
                    ? "bad"
                    : status === "degraded"
                      ? "warn"
                      : status === "loaded"
                        ? "ok"
                        : "info"
                }
              >
                {n} {status}
              </Tag>
            ))}
        </div>
        {profile.description && <p className="text-xs text-fg-muted">{profile.description}</p>}
        <div>
          <div className="mb-1 text-[11px] font-semibold text-fg-muted">Layers (apply order)</div>
          <ul className="space-y-0.5">
            {profile.layers.map((layer) => (
              <li key={layer} className="break-all font-mono text-[11px] text-fg-subtle">
                {layer}
              </li>
            ))}
          </ul>
        </div>
        {others.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] text-fg-subtle">other profiles</span>
            {others.map((p) => (
              <Tag key={p} kind="info">
                {p}
              </Tag>
            ))}
          </div>
        )}
        <details className="group">
          <summary className="cursor-pointer select-none text-[11px] font-semibold text-fg-muted hover:text-fg">
            Profile YAML ({profile.files.length})
          </summary>
          <div className="mt-2">
            <ConfigFiles
              files={profile.files}
              emptyText="No profile file applied — the built-in default plugin list is in use."
            />
          </div>
        </details>
      </div>
    </Section>
  );
}

export function PluginInventory() {
  const { data, isLoading, isError, error } = usePlugins();
  const plugins = data?.plugins ?? [];
  // A 404 here is an API older than the plugin inventory, not a missing plugin.
  const stale = isError && error instanceof Error && error.message.startsWith("HTTP 404");

  return (
    <>
      <Section
        title={`Plugins (${plugins.length})`}
        actions={<span className="text-[11px] text-fg-subtle">read-only</span>}
      >
        <QueryState
          loading={isLoading}
          error={isError && !stale}
          empty={plugins.length === 0}
          emptyText={stale ? STALE_API : "The profile names no plugins."}
        >
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {plugins.map((p) => (
              <PluginCard key={p.name} plugin={p} />
            ))}
          </div>
        </QueryState>
      </Section>
      {data && <ProfilePanel inventory={data} />}
    </>
  );
}
