import { Card, Kpi } from "@/components/Card";
import { Tag } from "@/components/Tag";
import { Grid, Section } from "@/components/layout";
import { FlagGrid, QueryState, fmtBytes } from "@/components/control/parts";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ConfirmDialog, type ConfirmState } from "@/components/ConfirmDialog";
import { ModelsPanel } from "@/components/settings/ModelsPanel";
import { HealthWatchPanel } from "@/components/settings/HealthWatchPanel";
import { DigestPanel } from "@/components/settings/DigestPanel";
import { PluginsPanel } from "@/components/settings/PluginsPanel";
import { ConnectionsTab } from "@/components/connections/ConnectionsPanel";
import { SettingRow, useSaveSetting } from "@/components/settings/SettingRow";
import {
  useCapabilities,
  useInvalidateSettings,
  useRequestRestart,
  useResetSetting,
  useRestartStatus,
  useRuntimeInventory,
  useSettings,
  useSettingsCatalog,
  useSettingsHistory,
  useWritesEnabled,
} from "@/lib/queries";
import type { CatalogSetting, SettingChangeRow } from "@/lib/control";

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 py-1 text-xs">
      <span className="text-fg-subtle">{k}</span>
      <span className="break-all text-right font-mono text-fg">{v}</span>
    </div>
  );
}

/* What IRIS is currently running (fast-follow #5) — local inventory, zero egress.
 * No "newer available" check (that crosses the privacy boundary). */
function RuntimeSection() {
  const { data, isLoading, isError } = useRuntimeInventory();
  return (
    <Section title="Runtime">
      <QueryState loading={isLoading} error={isError} empty={!data} emptyText="No runtime info.">
        {data && (
          <Grid cols={3}>
            <Card title="Build">
              <Row k="IRIS" v={`v${data.iris_version}`} />
              <Row k="branch" v={data.git_branch ?? "—"} />
              <Row k="commit" v={data.git_rev ?? "—"} />
              <Row k="Python" v={data.python_version} />
            </Card>
            <Card title="Key packages">
              {Object.entries(data.packages).map(([name, version]) => (
                <Row key={name} k={name} v={version} />
              ))}
            </Card>
            <Card title="Ollama models">
              {data.ollama_models.length === 0 ? (
                <span className="text-xs text-fg-subtle">No models pulled.</span>
              ) : (
                data.ollama_models.map((m, i) => (
                  <Row
                    key={m.name ?? i}
                    k={m.name ?? "—"}
                    v={[m.parameter_size, m.size ? fmtBytes(m.size) : null]
                      .filter(Boolean)
                      .join(" · ")}
                  />
                ))
              )}
            </Card>
          </Grid>
        )}
      </QueryState>
    </Section>
  );
}

function SettingsBody() {
  const { data, isLoading, isError } = useSettings();
  return (
    <QueryState loading={isLoading} error={isError} empty={!data} emptyText="No settings.">
      {data && (
        <div className="space-y-6">
          <Section title="LLM tiers">
            <div className="space-y-2">
              {data.tiers.map((t) => (
                <div key={t.name} className="rounded-lg border border-border bg-surface p-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-sm font-medium text-primary">{t.name}</span>
                    <span className="font-mono text-xs text-fg">{t.model}</span>
                    <Tag kind="res">{t.provider}</Tag>
                    <span className="ml-auto font-mono text-[11px] text-fg-subtle">
                      temp {t.temperature} · {t.max_tokens} tok
                    </span>
                  </div>
                  {t.use_for.length > 0 && (
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {t.use_for.map((u) => (
                        <span
                          key={u}
                          className="rounded bg-bg px-1.5 py-0.5 font-mono text-[10px] text-fg-subtle"
                        >
                          {u}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Section>

          <Grid cols={2}>
            <Card title="Providers">
              {Object.entries(data.providers).map(([name, configured]) => (
                <div key={name} className="flex items-center justify-between py-1">
                  <span className="text-xs text-fg">{name}</span>
                  <Tag kind={configured ? "ok" : "info"}>
                    {configured ? "configured" : "not set"}
                  </Tag>
                </div>
              ))}
              <p className="mt-2 text-[11px] text-fg-subtle">
                Presence only — keys are never sent to the browser.
              </p>
            </Card>

            <Card title="Host">
              <Row k="RAM free" v={`${data.host.ram_free_gb.toFixed(1)} / ${data.host.ram_total_gb.toFixed(1)} GB`} />
              <Row k="CPU" v={`${Math.round(data.host.cpu_percent)}%`} />
              <Row
                k="thermal"
                v={data.host.thermal_throttled ? "throttled" : "nominal"}
              />
            </Card>
          </Grid>

          <Section title="Stores">
            <Grid cols={4}>
              <Kpi value={`${data.stores.skill_count}`} label="skills" />
              <Kpi value={`${data.stores.heartbeat_count}`} label="heartbeats" />
              <Kpi value={`${data.stores.filemanager_roots}`} label="FM roots" />
              <Kpi
                value={`${Object.values(data.stores.accounts).reduce((a, b) => a + b, 0)}`}
                label="accounts"
              />
            </Grid>
            <Card title="Databases">
              {Object.entries(data.stores.database_sizes).map(([name, bytes]) => (
                <Row key={name} k={name} v={fmtBytes(bytes)} />
              ))}
              {Object.keys(data.stores.database_sizes).length === 0 && (
                <span className="text-xs text-fg-subtle">No databases yet.</span>
              )}
            </Card>
          </Section>

          <Section title="Paths">
            <Card>
              {Object.entries(data.paths).map(([k, v]) => (
                <Row key={k} k={k} v={v} />
              ))}
            </Card>
          </Section>

          <Section title="Runtime flags">
            <FlagGrid flags={data.flags} />
            <p className="mt-2 text-[11px] text-fg-subtle">
              Read-only. Flags are set via environment variables at startup.
            </p>
          </Section>
        </div>
      )}
    </QueryState>
  );
}

// ── The editable settings (ADR-0120) ──────────────────────────────────────────

const TABS = [
  ["agents", "Agents & plugins"],
  ["features", "Features"],
  ["models", "Models"],
  ["health", "Health watch"],
  ["digest", "Digest"],
  ["connections", "Connections"],
  ["guards", "Guards"],
  ["advanced", "Advanced"],
  ["history", "History"],
  ["system", "System"],
] as const;
type TabKey = (typeof TABS)[number][0];

function initialTab(): TabKey {
  const hash = window.location.hash.replace("#", "");
  return (TABS.find(([k]) => k === hash)?.[0] ?? "agents") as TabKey;
}

export function SettingsScreen() {
  const [tab, setTab] = useState<TabKey>(initialTab);
  // A link or the back button changes only the hash; follow it.
  useEffect(() => {
    const onHash = () => setTab(initialTab());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const choose = (k: TabKey) => {
    setTab(k);
    window.history.replaceState(null, "", `#${k}`);
  };
  return (
    <div className="space-y-4">
      <nav className="flex gap-1.5 overflow-x-auto pb-1" aria-label="Settings sections">
        <Link
          to="/heartbeats"
          className="inline-flex min-h-[44px] shrink-0 items-center rounded-full border border-border px-3 py-1.5 text-[13px] text-fg-muted hover:text-fg sm:min-h-0"
        >
          Schedules
        </Link>
        {TABS.map(([k, label]) => (
          <button
            key={k}
            type="button"
            onClick={() => choose(k)}
            className={`min-h-[44px] shrink-0 rounded-full border px-3 py-1.5 text-[13px] sm:min-h-0 ${
              tab === k
                ? "border-primary bg-primary font-semibold text-primary-fg"
                : "border-border text-fg-muted hover:text-fg"
            }`}
          >
            {label}
          </button>
        ))}
      </nav>
      <RestartBanner />
      {tab === "system" ? (
        <div className="space-y-6">
          <RuntimeSection />
          <SettingsBody />
        </div>
      ) : tab === "history" ? (
        <HistoryPanel />
      ) : tab === "digest" ? (
        <DigestTab />
      ) : tab === "connections" ? (
        <ConnectionsTab />
      ) : (
        <CatalogTab tab={tab} />
      )}
    </div>
  );
}

function RestartBanner() {
  const { data } = useRestartStatus();
  const restart = useRequestRestart();
  const canWrite = useWritesEnabled();
  const [confirm, setConfirm] = useState<ConfirmState | null>(null);
  const invalidate = useInvalidateSettings();
  const startedAt = useRef<string | undefined>(undefined);
  // A new server process (after a restart) re-reads plugins and settings.
  useEffect(() => {
    if (data?.started_at && startedAt.current && data.started_at !== startedAt.current) {
      invalidate();
    }
    startedAt.current = data?.started_at;
  }, [data?.started_at, invalidate]);
  const waiting = data?.waiting_for_restart ?? [];
  if (waiting.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-xs">
      <div>
        <b>
          {waiting.length} change{waiting.length > 1 ? "s wait" : " waits"} for a restart:
        </b>{" "}
        <span className="font-mono">{waiting.join(", ")}</span>
      </div>
      {data?.supervised ? (
        canWrite && (
          <Button
            type="button"
            size="sm"
            className="ml-auto"
            onClick={() =>
              setConfirm({
                title: "Restart the harness?",
                description:
                  "Every server restarts to pick up the waiting changes. Chat, Telegram and " +
                  "heartbeats pause for about a minute; nothing is lost.",
                confirmLabel: "Restart",
                run: async () => {
                  try {
                    await restart.mutateAsync();
                    toast.success("Restarting… the app reconnects by itself");
                  } catch (e) {
                    toast.error(e instanceof Error ? e.message : "restart failed");
                    throw e;
                  }
                },
              })
            }
          >
            Restart harness
          </Button>
        )
      ) : (
        <span className="ml-auto text-fg-subtle">
          This harness has no supervisor: restart it yourself to apply them.
        </span>
      )}
      <ConfirmDialog state={confirm} onClose={() => setConfirm(null)} />
    </div>
  );
}

function ownerLabel(owner: string): string {
  if (owner === "core") return "Harness";
  return owner.replace(/^plugin:/, "");
}

/* Settings → Digest: the morning digest's own settings, not catalog rows. */
function DigestTab() {
  const canWrite = useWritesEnabled();
  const readOnlyDevice = useCapabilities().data !== undefined && !canWrite;
  return (
    <div className="space-y-6">
      {readOnlyDevice && (
        <p className="rounded-lg border border-info/40 bg-info/10 px-3 py-2 text-xs">
          This device is paired read-only: changing the digest needs a control device.
        </p>
      )}
      <DigestPanel canWrite={canWrite} />
    </div>
  );
}

function CatalogTab({
  tab,
}: {
  tab: Exclude<TabKey, "history" | "system" | "digest" | "connections">;
}) {
  const { data, isLoading, isError } = useSettingsCatalog();
  const canWrite = useWritesEnabled();
  const readOnlyDevice = useCapabilities().data !== undefined && !canWrite;
  const [query, setQuery] = useState("");
  const all = useMemo(() => data ?? [], [data]);
  const rows = useMemo(() => {
    if (tab === "advanced") {
      const q = query.trim().toLowerCase();
      // Advanced is also the search over every setting, read-only ones included.
      return q
        ? all.filter((s) =>
            [s.name, s.label, s.description, s.owner].some((f) => f.toLowerCase().includes(q)),
          )
        : all.filter((s) => s.tab === "advanced");
    }
    return all.filter((s) => s.tab === tab);
  }, [all, tab, query]);
  const groups = useMemo(() => {
    const by = new Map<string, CatalogSetting[]>();
    for (const s of rows) by.set(s.owner, [...(by.get(s.owner) ?? []), s]);
    return [...by.entries()].sort(([a], [b]) => (a === "core" ? -1 : b === "core" ? 1 : a < b ? -1 : 1));
  }, [rows]);

  return (
    <div className="space-y-6">
      {readOnlyDevice && (
        <p className="rounded-lg border border-info/40 bg-info/10 px-3 py-2 text-xs">
          This device is paired read-only: you can see every setting; changing one needs a
          control device.
        </p>
      )}
      {tab === "agents" && <PluginsPanel catalog={all} canWrite={canWrite} />}
      {tab === "models" && <ModelsPanel canWrite={canWrite} />}
      {tab === "health" && <HealthWatchPanel canWrite={canWrite} />}
      {tab === "advanced" && (
        <Input
          placeholder="Search all settings (name, words, plugin)"
          aria-label="Search all settings"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      )}
      <QueryState
        loading={isLoading}
        error={isError}
        empty={rows.length === 0}
        emptyText="No settings here."
      >
        <div className="space-y-6">
          {groups.map(([owner, list]) => (
            <Section key={owner} title={`${ownerLabel(owner)} (${list.length})`}>
              <div className="space-y-2">
                {list.map((s) => (
                  <SettingRow key={s.name} s={s} canWrite={canWrite} />
                ))}
              </div>
            </Section>
          ))}
        </div>
      </QueryState>
    </div>
  );
}

function describeValue(v: unknown): string {
  if (v === null || v === undefined) return "unset";
  if (typeof v === "object") {
    return Object.entries(v as Record<string, unknown>)
      .map(([k, x]) => `${k}=${String(x)}`)
      .join(", ");
  }
  return String(v);
}

function HistoryPanel() {
  const { data, isLoading, isError } = useSettingsHistory();
  const catalog = useSettingsCatalog().data ?? [];
  const canWrite = useWritesEnabled();
  const { save, dialog } = useSaveSetting();
  const reset = useResetSetting();
  const changes = data?.changes ?? [];

  const undo = (c: SettingChangeRow) => {
    const s = catalog.find((x) => x.name === c.key);
    if (!s) return;
    if (c.old === null || c.old === undefined) void reset.mutateAsync(c.key);
    else void save(s, String(c.old));
  };

  return (
    <Section title={`History (${changes.length})`}>
      <p className="-mt-2 mb-3 text-xs text-fg-subtle">
        Every change, with the device that made it. Saved on the harness's data volume, so
        it survives restarts and deploys.
      </p>
      <QueryState
        loading={isLoading}
        error={isError}
        empty={changes.length === 0}
        emptyText="No changes yet."
      >
        <div className="space-y-2">
          {changes.map((c) => (
            <div key={c.id} className="rounded-lg border border-border bg-surface p-3 text-xs">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-mono text-fg">{c.key}</span>
                <span className="text-fg-subtle">{c.section}</span>
                {c.action === "reset" && <span className="text-fg-subtle">reset</span>}
                <span className="ml-auto text-[11px] text-fg-subtle">
                  {new Date(c.at).toLocaleString()} · {c.actor_name ?? c.actor}
                </span>
              </div>
              <p className="mt-1 font-mono">
                <span className="text-danger line-through">{describeValue(c.old)}</span> →{" "}
                <span className="font-semibold text-primary">{describeValue(c.new)}</span>
              </p>
              {canWrite && c.section === "env" && catalog.some((x) => x.name === c.key) && (
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  className="mt-1"
                  onClick={() => undo(c)}
                >
                  Undo
                </Button>
              )}
            </div>
          ))}
        </div>
      </QueryState>
      {dialog}
    </Section>
  );
}
