/* Shared pieces for the plugin inventory screens (Agents -> Plugins, plugin detail).
 * Read-only renderers over GET /plugins and /plugins/{name}. */
import { useState } from "react";
import { Link } from "react-router-dom";
import { Tag } from "@/components/Tag";
import { CopyBlock } from "@/components/CopyBlock";
import type { ConfigFile, PluginStatus, PluginSummary, PluginTool } from "@/lib/control";

/** Shown when the API answers without plugin data: it predates the inventory. */
export const STALE_API =
  "The running IRIS API does not report plugins yet — restart it to load the plugin inventory.";

const STATUS_TONE: Record<PluginStatus, "ok" | "warn" | "bad" | "info"> = {
  loaded: "ok",
  degraded: "warn",
  failed: "bad",
  disabled: "info",
  unsupported: "info",
};

export function PluginStatusTag({ status }: { status: PluginStatus }) {
  return <Tag kind={STATUS_TONE[status] ?? "info"}>{status}</Tag>;
}

/** "builtin:iris_harness.plugins_builtin.system" -> "builtin". */
export function sourceKind(source: string): string {
  return source.split(":", 1)[0];
}

const KIND_LABEL: Record<string, [string, string]> = {
  intent_handler: ["agent", "agents"],
  tool: ["tool", "tools"],
  intercept: ["intercept", "intercepts"],
  heartbeat: ["heartbeat", "heartbeats"],
  channel: ["channel", "channels"],
  confirmation_executor: ["confirmation", "confirmations"],
};

export function kindLabel(kind: string, n: number): string {
  const pair = KIND_LABEL[kind];
  if (!pair) return `${n} ${kind}`;
  return `${n} ${n === 1 ? pair[0] : pair[1]}`;
}

export function PluginCard({ plugin }: { plugin: PluginSummary }) {
  const counts = Object.entries(plugin.registration_counts);
  return (
    <Link
      to={`/agents/plugins/${plugin.name}`}
      className="flex flex-col gap-2 rounded-lg border border-border bg-surface p-3 transition-colors hover:border-border-strong hover:bg-surface-raised"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-[13px] font-semibold text-fg">{plugin.name}</span>
        {plugin.version && (
          <span className="font-mono text-[11px] text-fg-subtle">v{plugin.version}</span>
        )}
        <span className="ml-auto">
          <PluginStatusTag status={plugin.status} />
        </span>
      </div>
      {plugin.description && (
        <p className="line-clamp-2 text-xs text-fg-muted">{plugin.description}</p>
      )}
      {plugin.load_error && (
        <p className="line-clamp-2 font-mono text-[11px] text-danger">{plugin.load_error}</p>
      )}
      {plugin.agents.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-[11px] text-fg-subtle">agents</span>
          {plugin.agents.map((a) => (
            <Tag key={a} kind="opp">
              {a}
            </Tag>
          ))}
        </div>
      )}
      <div className="mt-auto flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-border pt-2 text-[11px] text-fg-subtle">
        <span className="font-mono">{sourceKind(plugin.source)}</span>
        {counts.length === 0 ? (
          <span>no registrations</span>
        ) : (
          counts.map(([kind, n]) => <span key={kind}>{kindLabel(kind, n)}</span>)
        )}
      </div>
    </Link>
  );
}

function fmtSize(n: number): string {
  return n < 1024 ? `${n} B` : `${(n / 1024).toFixed(1)} KB`;
}

/**
 * A file's tab label. A plugin's files come relative to the plugin, and the folder is
 * part of the name — its memory vocabulary is `ontology/mappings.yaml`, not the core's
 * `mappings.yaml`. A profile's layers come as absolute paths: those show the file name.
 */
function fileLabel(path: string): string {
  return path.startsWith("/") ? (path.split("/").pop() ?? path) : path;
}

/** Read-only viewer for a set of YAML files: a file picker plus the selected file. */
export function ConfigFiles({ files, emptyText }: { files: ConfigFile[]; emptyText: string }) {
  const [selected, setSelected] = useState(0);
  if (files.length === 0) return <p className="text-xs text-fg-subtle">{emptyText}</p>;
  const file = files[Math.min(selected, files.length - 1)];
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Configuration files">
        {files.map((f, i) => (
          <button
            key={f.path}
            type="button"
            role="tab"
            aria-selected={i === selected}
            onClick={() => setSelected(i)}
            className={`max-w-full truncate rounded-md border px-2 py-1 font-mono text-[11px] transition-colors ${
              i === selected
                ? "border-primary/40 bg-primary/10 text-primary"
                : "border-border text-fg-muted hover:bg-surface-raised hover:text-fg"
            }`}
          >
            {fileLabel(f.path)}
          </button>
        ))}
      </div>
      {file.content === null ? (
        <p className="mt-3 text-xs text-fg-subtle">
          {file.truncated
            ? `Too large to display (${fmtSize(file.size)}).`
            : "Could not be read as text."}
        </p>
      ) : (
        <CopyBlock label={`${file.path} · ${fmtSize(file.size)}`} text={file.content} />
      )}
    </div>
  );
}

const EFFECT_TAG = { read: "info", write: "warn", destructive: "bad" } as const;

/** One declared tool: effect, confirm mode, flags and its prompt guidance. */
export function ToolRow({ tool }: { tool: PluginTool }) {
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        {/* A tool name is one unbroken token; without break-all a long one runs out
            of the card (the fixture in viewport.spec.ts is the proof). */}
        <span className="min-w-0 break-all font-mono text-[13px] text-fg">{tool.name}</span>
        <Tag kind={EFFECT_TAG[tool.effect]}>{tool.effect}</Tag>
        {tool.effect === "write" && <Tag kind="info">confirm {tool.confirm}</Tag>}
        {tool.effect === "destructive" && <Tag kind="info">approved per call</Tag>}
        {tool.undo && <Tag kind="res">undo {tool.undo}</Tag>}
        {tool.pinned && <Tag kind="res">pinned</Tag>}
        {tool.answers_directly && <Tag kind="res">answers directly</Tag>}
        {!tool.registered && <Tag kind="bad">not registered</Tag>}
      </div>
      {tool.guidance && <p className="mt-1.5 text-xs text-fg-muted">{tool.guidance}</p>}
    </div>
  );
}
