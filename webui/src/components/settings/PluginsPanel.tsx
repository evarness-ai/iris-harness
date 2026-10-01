import { Switch } from "@/components/Switch";
import { Tag } from "@/components/Tag";
import { Section } from "@/components/layout";
import { QueryState } from "@/components/control/parts";
import { usePlugins } from "@/lib/queries";
import type { CatalogSetting, PluginSummary } from "@/lib/control";
import { effectiveValue, useSaveSetting } from "./SettingRow";

const DISABLE = "IRIS_PLUGINS_DISABLE";
const ENABLE = "IRIS_PLUGINS_ENABLE";

function names(value: string): string[] {
  return value
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

/**
 * Plugins on/off (ADR-0120). No new mechanism: a switch rewrites the
 * ``IRIS_PLUGINS_DISABLE`` / ``IRIS_PLUGINS_ENABLE`` settings (guarded, so each change
 * is confirmed), which the profile reads at the next start. A plugin whose manifest
 * is ``locked`` shows why and cannot be switched off here.
 */
export function PluginsPanel({
  catalog,
  canWrite,
}: {
  catalog: CatalogSetting[];
  canWrite: boolean;
}) {
  const { data, isLoading, isError } = usePlugins();
  const { save, dialog, busy } = useSaveSetting();
  const disable = catalog.find((s) => s.name === DISABLE);
  const enable = catalog.find((s) => s.name === ENABLE);
  const plugins = data?.plugins ?? [];

  const toggle = (p: PluginSummary, on: boolean) => {
    if (!disable || !enable) return;
    const off = names(effectiveValue(disable));
    if (!on) {
      void save(disable, [...new Set([...off, p.name])].join(","));
      return;
    }
    if (off.includes(p.name)) {
      void save(disable, off.filter((n) => n !== p.name).join(","));
    } else {
      // Off in the profile itself, not by the env list: switch it on explicitly.
      const onList = names(effectiveValue(enable));
      void save(enable, [...new Set([...onList, p.name])].join(","));
    }
  };

  return (
    <Section title={`Plugins (${plugins.length})`}>
      <p className="-mt-2 mb-3 text-xs text-fg-subtle">
        On/off applies after a restart. Each change asks you to confirm.
      </p>
      <QueryState
        loading={isLoading}
        error={isError}
        empty={plugins.length === 0}
        emptyText="No plugins."
      >
        <div className="space-y-2">
          {plugins.map((p) => (
            <div key={p.name} className="rounded-lg border border-border bg-surface p-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-sm text-fg">{p.name}</span>
                <Tag kind="warn">after restart</Tag>
                {p.status !== "loaded" && <Tag kind="info">{p.status}</Tag>}
                {p.set_by === "env" && <Tag kind="info">changed</Tag>}
                <span className="ml-auto" />
                <Switch
                  checked={p.enabled !== false}
                  label={`${p.name} on or off`}
                  disabled={!canWrite || busy || Boolean(p.locked) || !disable || !enable}
                  onChange={(next) => toggle(p, next)}
                />
              </div>
              {p.description && (
                <p className="mt-1 line-clamp-2 text-xs text-fg-muted">{p.description}</p>
              )}
              {p.locked && (
                <p className="mt-1 text-[11px] text-fg-subtle">Locked on: {p.locked}.</p>
              )}
              {p.agents.length > 0 && (
                <p className="mt-1 font-mono text-[10.5px] text-fg-subtle">
                  agents: {p.agents.join(", ")}
                </p>
              )}
            </div>
          ))}
        </div>
      </QueryState>
      {dialog}
    </Section>
  );
}
