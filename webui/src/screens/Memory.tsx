import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Tag } from "@/components/Tag";
import { Notice, QueryState } from "@/components/control/parts";
import { MemoryMapTab } from "@/components/memory/MemoryMap";
import { RemovedTab } from "@/components/memory/removal";
import { TestSessionReview } from "@/components/memory/TestSessionReview";
import {
  AboutYouTab,
  ContextTab,
  ConversationsTab,
  HousekeepingTab,
  ReviewTab,
} from "@/components/memory/tabs";
import {
  useAckContradictions,
  useContradictions,
  useFactHistoryRetention,
  useLessons,
  useMemory,
  useLookalikes,
  useMemoryReview,
  usePruneFactHistory,
  useWritesEnabled,
} from "@/lib/queries";
import { getSessionId } from "@/lib/chat";

const RETENTION_DAYS = 180;

/** Detected same-key value conflicts. Read = always; acknowledge = write-gated.
 * Resolve a conflict by correcting the fact (CLI/chat); ack just clears the queue. */
function FactContradictions() {
  const { data, isLoading, isError } = useContradictions();
  const ack = useAckContradictions();
  const canWrite = useWritesEnabled();
  const rows = data?.contradictions ?? [];

  return (
    <div className="space-y-3 border-t border-border pt-4">
      <div>
        <h2 className="text-sm font-semibold text-fg">Contradictions</h2>
        <p className="text-[11px] text-fg-subtle">
          {data ? `${data.count} unreviewed` : "review queue"} — same key, conflicting values.
          Fix by correcting the fact; acknowledge to clear.
        </p>
      </div>

      <QueryState
        loading={isLoading}
        error={isError}
        empty={rows.length === 0}
        emptyText="No contradictions — IRIS hasn't seen conflicting values for any fact."
      >
        <div className="space-y-1.5">
          {rows.map((c) => (
            <div
              key={c.id}
              className="flex items-start justify-between gap-2 rounded-lg border border-border bg-surface p-2 text-xs"
            >
              <span className="flex-1">
                <span className="font-mono text-fg">{c.key}</span>{" "}
                <span className="text-fg-subtle">{c.stored_value}</span>
                <span className="text-fg-subtle"> vs </span>
                <span className="text-fg">{c.incoming_value}</span>{" "}
                <Tag kind={c.resolution === "blocked" ? "warn" : "info"}>{c.resolution}</Tag>
                {c.seen_count > 1 && (
                  <span className="ml-1 text-[10px] text-fg-subtle">{c.seen_count}×</span>
                )}
                <span className="ml-1 text-[10px] text-fg-subtle">
                  {c.source} · {c.detected_at.slice(0, 10)}
                </span>
              </span>
              {canWrite && (
                <Button
                  type="button"
                  variant="outline"
                  disabled={ack.isPending}
                  onClick={() => ack.mutate([c.id])}
                >
                  Ack
                </Button>
              )}
            </div>
          ))}
        </div>
      </QueryState>

      {!canWrite && (
        <Notice>
          Review only — set IRIS_WEBUI_ALLOW_WRITES=1 to acknowledge here (or use `iris facts ack`).
        </Notice>
      )}
      {ack.isError && (
        <Notice tone="danger">
          {ack.error instanceof Error ? ack.error.message : "acknowledge failed"}
        </Notice>
      )}
    </div>
  );
}

/** Human-reviewed retention of the fact-history audit trail. Lists entries older than
 * the window and lets the user prune chosen ones — nothing is ever auto-deleted. The
 * prune action is write-gated (IRIS_WEBUI_ALLOW_WRITES); read/review is always available. */
function FactRetentionReview() {
  const { data, isLoading, isError } = useFactHistoryRetention(RETENTION_DAYS);
  const prune = usePruneFactHistory();
  const canWrite = useWritesEnabled();
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const entries = data?.entries ?? [];
  const toggle = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  const allSelected = entries.length > 0 && selected.size === entries.length;
  const toggleAll = () =>
    setSelected(allSelected ? new Set() : new Set(entries.map((e) => e.id)));

  const onPrune = () => {
    if (selected.size === 0) return;
    const plural = selected.size === 1 ? "y" : "ies";
    if (
      !window.confirm(
        `Permanently prune ${selected.size} history entr${plural}? This cannot be undone.`,
      )
    )
      return;
    prune.mutate(Array.from(selected), { onSuccess: () => setSelected(new Set()) });
  };

  return (
    <div className="space-y-3 border-t border-border pt-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-sm font-semibold text-fg">Fact history retention</h2>
          <p className="text-[11px] text-fg-subtle">
            {data
              ? `${data.total} entries · ${data.count} older than ${data.older_than_days}d`
              : "review queue"}{" "}
            — review and prune; nothing is auto-deleted.
          </p>
        </div>
        {canWrite && entries.length > 0 && (
          <div className="flex items-center gap-2">
            <Button type="button" variant="outline" onClick={toggleAll}>
              {allSelected ? "Clear" : "Select all"}
            </Button>
            <Button
              type="button"
              variant="destructive"
              disabled={selected.size === 0 || prune.isPending}
              onClick={onPrune}
            >
              Prune{selected.size ? ` ${selected.size}` : ""}
            </Button>
          </div>
        )}
      </div>

      <QueryState
        loading={isLoading}
        error={isError}
        empty={entries.length === 0}
        emptyText={`Nothing older than ${RETENTION_DAYS} days — the history trail is within retention.`}
      >
        <div className="space-y-1.5">
          {entries.map((e) => (
            <label
              key={e.id}
              className="flex items-start gap-2 rounded-lg border border-border bg-surface p-2 text-xs"
            >
              {canWrite && (
                <input
                  type="checkbox"
                  className="mt-0.5"
                  checked={selected.has(e.id)}
                  onChange={() => toggle(e.id)}
                  aria-label={`Select history entry ${e.id}`}
                />
              )}
              <span className="flex-1">
                <Tag kind="info">{e.reason}</Tag>{" "}
                <span className="font-mono text-fg">{e.key}</span>
                {e.old_value != null && <span className="text-fg-subtle"> · was {e.old_value}</span>}
                {e.new_value != null && <span className="text-fg-subtle"> → {e.new_value}</span>}
                <span className="ml-1 text-[10px] text-fg-subtle">
                  {e.source} · {e.changed_at.slice(0, 10)}
                </span>
              </span>
            </label>
          ))}
        </div>
      </QueryState>

      {!canWrite && (
        <Notice>
          Review only — set IRIS_WEBUI_ALLOW_WRITES=1 to prune here (or use `iris facts prune`).
        </Notice>
      )}
      {prune.isError && (
        <Notice tone="danger">
          {prune.error instanceof Error ? prune.error.message : "prune failed"}
        </Notice>
      )}
    </div>
  );
}

/** Live in-runtime conversation buffer for a session (GET /memory/{id}). This is
 * the working memory the runtime holds this process — distinct from the
 * persisted session logs shown in Sessions / Call Trace. */
/** The running runtime's in-process window for one session (was the whole screen). */
function LiveBuffer() {
  const [sessionId, setSessionId] = useState<string>(getSessionId);
  const [input, setInput] = useState<string>(sessionId);
  const { data, isLoading, isError } = useMemory(sessionId);
  const turns = data?.turns ?? [];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex-1">
          <span className="mb-1 block text-[11px] uppercase tracking-wide text-fg-subtle">
            session id
          </span>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") setSessionId(input.trim());
            }}
            placeholder="session id"
            aria-label="Session id"
            className="w-full rounded-lg border border-border bg-surface px-3 py-2 font-mono text-sm text-fg placeholder:text-fg-subtle focus:border-primary/50 focus:outline-none"
          />
        </label>
        <Button type="button" onClick={() => setSessionId(input.trim())} disabled={!input.trim()}>
          Load
        </Button>
        <Button
          type="button"
          variant="outline"
          onClick={() => {
            const cur = getSessionId();
            setInput(cur);
            setSessionId(cur);
          }}
        >
          My chat session
        </Button>
      </div>

      <div className="flex items-center gap-2">
        <Tag kind="res">{data?.turn_count ?? 0} turns</Tag>
        <span className="font-mono text-[11px] text-fg-subtle">{sessionId || "(none)"}</span>
      </div>

      <QueryState
        loading={isLoading}
        error={isError}
        empty={turns.length === 0}
        emptyText="No live turns for this session. The buffer only holds sessions active in the current runtime process."
      >
        <div className="space-y-2">
          {turns.map((t, i) => (
            <div
              key={`${t.role}-${i}`}
              className="rounded-lg border border-border bg-surface p-3"
            >
              <Tag kind={t.role === "user" ? "opp" : "ok"}>{t.role}</Tag>
              <p className="mt-1.5 whitespace-pre-wrap break-words text-sm text-fg">{t.content}</p>
            </div>
          ))}
        </div>
      </QueryState>

      <Notice>
        Working memory held by the running runtime — distinct from persisted logs (see Sessions /
        Call Trace).
      </Notice>

      <FactRetentionReview />

      <FactContradictions />
    </div>
  );
}


type TabKey =
  | "about"
  | "review"
  | "conversations"
  | "context"
  | "housekeeping"
  | "map"
  | "removed";

const TABS: { key: TabKey; label: string }[] = [
  { key: "about", label: "About you" },
  { key: "review", label: "Review" },
  { key: "conversations", label: "Conversations" },
  { key: "context", label: "Context" },
  { key: "housekeeping", label: "Housekeeping" },
  { key: "map", label: "Map" },
  { key: "removed", label: "Removed" },
];

/** Memory: what IRIS knows about you, what is waiting for your yes, and what it keeps.
 *
 * Facts used to have no screen at all — only contradictions and history pruning lived
 * here, and review was split between Digital Twin and Self-Learning. */
export function MemoryScreen() {
  const [tab, setTab] = useState<TabKey>("about");
  const [inspectSession, setInspectSession] = useState(getSessionId());
  const review = useMemoryReview();
  const lessons = useLessons();
  const lookalikes = useLookalikes();
  const waiting =
    (review.data?.pending_count ?? 0) +
    (lessons.data?.lessons.length ?? 0) +
    (lookalikes.data?.count ?? 0);

  return (
    <div className="space-y-5">
      <nav className="flex flex-wrap gap-1 border-b border-border pb-2">
        {TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setTab(t.key)}
            className={`min-h-11 rounded-lg px-3 py-1 text-xs sm:min-h-0 ${
              tab === t.key ? "bg-surface text-fg" : "text-fg-subtle hover:text-fg"
            }`}
          >
            {t.label}
            {t.key === "review" && waiting > 0 && (
              <span className="ml-1 rounded bg-warn/20 px-1 text-[10px] text-warn">{waiting}</span>
            )}
          </button>
        ))}
      </nav>

      {tab === "about" && <AboutYouTab />}
      {tab === "review" && (
        <div className="space-y-6">
          <ReviewTab />
          <TestSessionReview />
          <div className="border-t border-border pt-4">
            <FactContradictions />
          </div>
        </div>
      )}
      {tab === "conversations" && (
        <ConversationsTab
          onInspect={(sid) => {
            setInspectSession(sid);
            setTab("context");
          }}
        />
      )}
      {tab === "context" && (
        <div className="space-y-6">
          <ContextTab sessionId={inspectSession} />
          <div className="border-t border-border pt-4">
            <LiveBuffer />
          </div>
        </div>
      )}
      {tab === "map" && <MemoryMapTab />}
      {tab === "removed" && <RemovedTab />}
      {tab === "housekeeping" && (
        <div className="space-y-6">
          <HousekeepingTab />
          <div className="border-t border-border pt-4">
            <FactRetentionReview />
          </div>
        </div>
      )}
    </div>
  );
}
