/* One plugin, read-only, over GET /plugins/{name}: its manifest, what it registered,
 * the tools it declared, declared-vs-registered drift, and the YAML that configures
 * it. Thin renderer; the inventory is composed in the harness. */
import { Link, useParams } from "react-router-dom";
import { Tag } from "@/components/Tag";
import { CopyBlock } from "@/components/CopyBlock";
import { Section } from "@/components/layout";
import { Notice, QueryState } from "@/components/control/parts";
import {
  ConfigFiles,
  PluginStatusTag,
  STALE_API,
  ToolRow,
  kindLabel,
} from "@/components/plugins/parts";
import { usePlugin } from "@/lib/queries";
import type { PluginDetail } from "@/lib/control";

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] text-fg-subtle">{label}</dt>
      <dd className="break-all font-mono text-[12px] text-fg">{children}</dd>
    </div>
  );
}

function Overview({ plugin }: { plugin: PluginDetail }) {
  const m = plugin.manifest;
  return (
    <Section title="Overview">
      <div className="rounded-lg border border-border bg-surface p-3">
        <dl className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <Fact label="source">{plugin.source}</Fact>
          <Fact label="trust">{plugin.trust}</Fact>
          <Fact label="party">{plugin.party ?? "—"}</Fact>
          <Fact label="flavor">{plugin.flavor ?? "—"}</Fact>
          <Fact label="entrypoint">{m?.entrypoint ?? "—"}</Fact>
          <Fact label="cli">{m?.cli ?? "—"}</Fact>
          <Fact label="in profile">
            {plugin.enabled === null ? "—" : plugin.enabled ? "enabled" : "disabled"}
            {plugin.set_by ? ` (set by ${plugin.set_by})` : ""}
          </Fact>
          {m && <Fact label="requires python">{m.requires.python}</Fact>}
          {m && m.requires.packages.length > 0 && (
            <Fact label="requires packages">{m.requires.packages.join(", ")}</Fact>
          )}
          {m && m.requires.env_vars.length > 0 && (
            <Fact label="requires env">{m.requires.env_vars.join(", ")}</Fact>
          )}
        </dl>
        {plugin.provides.length > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] text-fg-subtle">provides</span>
            {plugin.provides.map((k) => (
              <Tag key={k} kind="info">
                {k}
              </Tag>
            ))}
          </div>
        )}
        {plugin.degraded_reason && (
          <CopyBlock label="degraded" text={plugin.degraded_reason} tone="text-warning" />
        )}
        {plugin.load_error && (
          <CopyBlock label="load error" text={plugin.load_error} tone="text-danger" />
        )}
        {plugin.last_error && (
          <CopyBlock
            label={`last runtime failure (${plugin.failure_count} total)`}
            text={plugin.last_error}
            tone="text-warning"
          />
        )}
      </div>
    </Section>
  );
}

function Registrations({ plugin }: { plugin: PluginDetail }) {
  const byKind = new Map<string, PluginDetail["registrations"]>();
  for (const r of plugin.registrations) {
    byKind.set(r.kind, [...(byKind.get(r.kind) ?? []), r]);
  }
  return (
    <Section title={`Registered at runtime (${plugin.registrations.length})`}>
      {byKind.size === 0 ? (
        <p className="text-xs text-fg-subtle">
          {plugin.status === "loaded" || plugin.status === "degraded"
            ? "Setup ran but registered nothing."
            : "Nothing registered — the plugin did not load."}
        </p>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          {[...byKind.entries()].map(([kind, regs]) => (
            <div key={kind} className="rounded-lg border border-border bg-surface p-3">
              <div className="mb-2 text-[11px] font-semibold text-fg-muted">
                {kindLabel(kind, regs.length)}
              </div>
              <div className="flex flex-wrap gap-1.5">
                {regs.map((r) =>
                  kind === "intent_handler" ? (
                    <Link
                      key={r.name}
                      to={`/agents/${r.name}`}
                      className="rounded-md border border-border px-1.5 py-0.5 font-mono text-[11px] text-primary hover:bg-surface-raised"
                    >
                      {r.name}
                    </Link>
                  ) : (
                    <span
                      key={r.name}
                      title={r.detail || undefined}
                      className="rounded-md border border-border px-1.5 py-0.5 font-mono text-[11px] text-fg-muted"
                    >
                      {r.name}
                    </span>
                  ),
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </Section>
  );
}

const DRIFT_LABEL: Record<string, string> = {
  tools_declared_not_registered: "Tools declared in the manifest but never registered",
  tools_registered_not_declared: "Tools registered but missing from the manifest",
  provides_not_registered: "Kinds under `provides` with nothing registered",
  registered_not_provided: "Kinds registered but not listed under `provides`",
  capabilities_declared_not_provided:
    "Capabilities under `capabilities: provides` never provided",
  search_providers_declared_not_registered:
    "Search providers under `search_providers` never registered",
};

function Drift({ drift }: { drift: Record<string, string[]> }) {
  const rows = Object.entries(drift).filter(([, v]) => v.length > 0);
  return (
    <Section title="Manifest vs runtime">
      {rows.length === 0 ? (
        <p className="text-xs text-fg-subtle">
          No drift — what the manifest declares is what registered.
        </p>
      ) : (
        <div className="space-y-1.5">
          {rows.map(([key, names]) => (
            <div
              key={key}
              className="flex flex-wrap items-center gap-2 rounded-lg border border-warning/40 bg-surface px-3 py-2"
            >
              <span className="text-xs text-fg">{DRIFT_LABEL[key] ?? key}</span>
              {names.map((n) => (
                <Tag key={n} kind="warn">
                  {n}
                </Tag>
              ))}
            </div>
          ))}
        </div>
      )}
    </Section>
  );
}

export function PluginDetailScreen() {
  const { name = "" } = useParams();
  const { data, isLoading, isError, error } = usePlugin(name);
  const notFound = isError && error instanceof Error && error.message.startsWith("HTTP 404");

  return (
    <>
      <Link to="/agents" className="text-xs text-primary hover:underline">
        ← Agents & plugins
      </Link>
      {notFound ? (
        <div className="mt-3">
          <Notice>
            No plugin named “{name}” in the active profile — or{" "}
            {STALE_API.charAt(0).toLowerCase() + STALE_API.slice(1)}
          </Notice>
        </div>
      ) : (
        <QueryState loading={isLoading} error={isError} empty={!data} emptyText="No data.">
          {data && (
            <div className="mt-3">
              <div className="mb-4 flex flex-wrap items-center gap-2">
                <h2 className="font-mono text-base font-semibold text-fg">{data.name}</h2>
                {data.version && (
                  <span className="font-mono text-xs text-fg-subtle">v{data.version}</span>
                )}
                <PluginStatusTag status={data.status} />
                <span className="ml-auto text-[11px] text-fg-subtle">read-only</span>
              </div>
              {data.description && <p className="mb-4 text-sm text-fg-muted">{data.description}</p>}
              <Overview plugin={data} />
              <Registrations plugin={data} />
              <Section title={`Declared tools (${data.tools.length})`}>
                {data.tools.length === 0 ? (
                  <p className="text-xs text-fg-subtle">The manifest declares no tools.</p>
                ) : (
                  <div className="space-y-2">
                    {data.tools.map((t) => (
                      <ToolRow key={t.name} tool={t} />
                    ))}
                  </div>
                )}
              </Section>
              <Drift drift={data.drift} />
              <Section title={`Configuration (${data.files.length} YAML)`}>
                <ConfigFiles
                  files={data.files}
                  emptyText="No YAML found — the plugin's directory is not available."
                />
                {data.directory && (
                  <p className="mt-2 break-all font-mono text-[11px] text-fg-subtle">
                    {data.directory}
                  </p>
                )}
              </Section>
            </div>
          )}
        </QueryState>
      )}
    </>
  );
}
