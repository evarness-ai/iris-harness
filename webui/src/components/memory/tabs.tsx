/** The Memory screen's tabs: About you, Review, Conversations, Context, Housekeeping.
 * (Map lives in MemoryMap.tsx, Removed in removal.tsx.)
 *
 * Every action here has a CLI twin (`iris facts …`, `iris memory …`) and an API route
 * behind it — the screen is a client of the same surface, never the only way in. */
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Tag } from "@/components/Tag";
import { Notice, QueryState, fmtBytes, fmtDateTime } from "@/components/control/parts";
import {
  useApproveLesson,
  useApproveProposal,
  useContextInspection,
  useCorrectFact,
  useCuratedProfile,
  useForgetFact,
  useForgetMatching,
  useHousekeepingRuns,
  useLessons,
  useLookalikes,
  useLearnedTerms,
  useRejectTerm,
  useActivateTerm,
  useMarkEntitiesDistinct,
  useMarkEntitiesSame,
  useMemoryReview,
  useRejectLesson,
  useRejectProposal,
  useRunHousekeeping,
  useStoredFacts,
  useStoredSessions,
  useWriteCuratedProfile,
  useCompactSession,
  useWritesEnabled,
} from "@/lib/queries";

export function WriteGate({ canWrite }: { canWrite: boolean }) {
  if (canWrite) return null;
  return (
    <p className="text-[11px] text-fg-subtle">
      Read-only — set <code>IRIS_WEBUI_ALLOW_WRITES=1</code> to act from here, or use the
      CLI.
    </p>
  );
}

/* ─────────────────────────── About you ─────────────────────────── */

export function AboutYouTab() {
  const canWrite = useWritesEnabled();
  const profile = useCuratedProfile();
  const write = useWriteCuratedProfile();
  const facts = useStoredFacts(true);
  const forget = useForgetFact();
  const correct = useCorrectFact();
  const [draft, setDraft] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [value, setValue] = useState("");

  const text = draft ?? profile.data?.profile ?? "";
  const rows = facts.data?.facts ?? [];

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <div>
          <h2 className="text-sm font-semibold text-fg">Your profile</h2>
          <p className="text-[11px] text-fg-subtle">
            The part of USER.md you write. It goes into every prompt in full. The
            auto-detected block below it is the fact store's projection and is not edited
            here.
          </p>
        </div>
        <QueryState
          loading={profile.isLoading}
          error={profile.isError}
          empty={false}
          emptyText=""
        >
          <textarea
            className="h-64 w-full rounded-lg border border-border bg-surface p-3 font-mono text-xs text-fg"
            value={text}
            readOnly={!canWrite}
            onChange={(e) => setDraft(e.target.value)}
          />
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-fg-subtle">{text.length} characters</span>
            {canWrite && draft !== null && (
              <>
                <Button size="sm" onClick={() => write.mutate(text, { onSuccess: () => setDraft(null) })}>
                  Save
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setDraft(null)}>
                  Cancel
                </Button>
              </>
            )}
          </div>
          <WriteGate canWrite={canWrite} />
        </QueryState>
      </section>

      <section className="space-y-2 border-t border-border pt-4">
        <div>
          <h2 className="text-sm font-semibold text-fg">Confirmed facts</h2>
          <p className="text-[11px] text-fg-subtle">
            {rows.length} fact(s) you approved. Only these reach a prompt — anything
            unconfirmed waits in Review.
          </p>
        </div>
        <QueryState
          loading={facts.isLoading}
          error={facts.isError}
          empty={rows.length === 0}
          emptyText="Nothing confirmed yet — approve a proposal in Review and it lands here."
        >
          <div className="space-y-1.5">
            {rows.map((f) => (
              <div
                key={f.id ?? f.key}
                className="flex items-start justify-between gap-2 rounded-lg border border-border bg-surface p-2 text-xs"
              >
                <span className="flex-1">
                  <span className="font-mono text-fg">{f.key}</span>{" "}
                  {editing === (f.id ?? f.key) ? (
                    <input
                      className="rounded border border-border bg-bg px-1 text-fg"
                      value={value}
                      onChange={(e) => setValue(e.target.value)}
                    />
                  ) : (
                    <span className="text-fg-subtle">{f.value}</span>
                  )}
                  <span className="ml-1 text-[10px] text-fg-subtle">
                    {f.source} · confirmed {f.times_confirmed}×
                  </span>
                </span>
                {canWrite && (
                  <span className="flex gap-1">
                    {editing === (f.id ?? f.key) ? (
                      <>
                        <Button
                          size="sm"
                          onClick={() =>
                            correct.mutate(
                              { key: f.key, value, id: f.id },
                              { onSuccess: () => setEditing(null) },
                            )
                          }
                        >
                          Save
                        </Button>
                        <Button size="sm" variant="ghost" onClick={() => setEditing(null)}>
                          Cancel
                        </Button>
                      </>
                    ) : (
                      <>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => {
                            setEditing(f.id ?? f.key);
                            setValue(f.value);
                          }}
                        >
                          Edit
                        </Button>
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => forget.mutate({ key: f.key, id: f.id })}
                        >
                          Forget
                        </Button>
                      </>
                    )}
                  </span>
                )}
              </div>
            ))}
          </div>
        </QueryState>
      </section>
    </div>
  );
}

/* ─────────────────────────── Review ─────────────────────────── */

export function ReviewTab() {
  const canWrite = useWritesEnabled();
  const review = useMemoryReview();
  const lessons = useLessons();
  const approve = useApproveProposal();
  const reject = useRejectProposal();
  const approveLesson = useApproveLesson();
  const rejectLesson = useRejectLesson();
  const lookalikes = useLookalikes();
  const same = useMarkEntitiesSame();
  const distinct = useMarkEntitiesDistinct();
  const learned = useLearnedTerms();
  const rejectTerm = useRejectTerm();
  const activateTerm = useActivateTerm();

  const proposals = review.data?.proposals ?? [];
  const words = (learned.data?.terms ?? []).filter(
    (t) => t.status === "candidate" || t.status === "active",
  );
  const lessonRows = lessons.data?.lessons ?? [];
  const pairs = lookalikes.data?.decisions ?? [];
  const total = proposals.length + lessonRows.length + pairs.length;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-sm font-semibold text-fg">{total} item(s) waiting</h2>
        <p className="text-[11px] text-fg-subtle">
          Nothing here reaches a prompt until you approve it. Confidence never approves
          anything — the store once held <code>name = ollama</code> at 1.00.
        </p>
        <WriteGate canWrite={canWrite} />
      </div>

      <QueryState
        loading={review.isLoading || lookalikes.isLoading}
        error={review.isError}
        empty={total === 0}
        emptyText="Nothing waiting — every stored fact and lesson is confirmed."
      >
        <div className="space-y-4">
          {proposals.length > 0 && (
            <section className="space-y-1.5">
              <h3 className="text-xs font-semibold text-fg">Proposed facts</h3>
              {proposals.map((p) => (
                <div key={p.id} className="rounded-lg border border-border bg-surface p-2 text-xs">
                  <div className="flex items-start justify-between gap-2">
                    <span className="flex-1">
                      {p.subject && <span className="text-fg-subtle">{p.subject} · </span>}
                      <span className="font-mono text-fg">{p.key}</span>{" "}
                      <span className="text-fg">{p.value}</span>{" "}
                      {p.kind === "changed" && (
                        <Tag kind="warn">was: {p.current_value ?? "—"}</Tag>
                      )}
                      <span className="ml-1 text-[10px] text-fg-subtle">
                        {p.source} · {p.created_at.slice(0, 10)}
                      </span>
                    </span>
                    {canWrite && (
                      <span className="flex gap-1">
                        <Button size="sm" onClick={() => approve.mutate(p.id)}>
                          Approve
                        </Button>
                        <Button size="sm" variant="ghost" onClick={() => reject.mutate(p.id)}>
                          Reject
                        </Button>
                      </span>
                    )}
                  </div>
                  {p.evidence && (
                    <p className="mt-1 text-[10px] text-fg-subtle">you said: {p.evidence}</p>
                  )}
                </div>
              ))}
            </section>
          )}

          {lessonRows.length > 0 && (
            <section className="space-y-1.5">
              <h3 className="text-xs font-semibold text-fg">Lessons</h3>
              {lessonRows.map((l) => (
                <div key={l.id} className="rounded-lg border border-border bg-surface p-2 text-xs">
                  <div className="flex items-start justify-between gap-2">
                    <span className="flex-1">
                      <span className="text-fg-subtle">when</span>{" "}
                      <span className="text-fg">{l.trigger}</span>{" "}
                      <span className="text-fg-subtle">→ do</span>{" "}
                      <span className="text-fg">{l.lesson}</span>
                      <span className="ml-1 text-[10px] text-fg-subtle">{l.source}</span>
                    </span>
                    {canWrite && (
                      <span className="flex gap-1">
                        <Button size="sm" onClick={() => approveLesson.mutate(l.id)}>
                          Approve
                        </Button>
                        <Button size="sm" variant="ghost" onClick={() => rejectLesson.mutate(l.id)}>
                          Reject
                        </Button>
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </section>
          )}

          {pairs.length > 0 && (
            <section className="space-y-1.5">
              <h3 className="text-xs font-semibold text-fg">Look-alike names</h3>
              <p className="text-[10px] text-fg-subtle">
                Kept apart until you decide. Unanswered, a pair merges once it turns up in
                enough conversations — any merge can be undone (<code>iris memory unmerge</code>).
              </p>
              {pairs.map((d) => (
                <div key={d.id} className="rounded-lg border border-border bg-surface p-2 text-xs">
                  <div className="flex items-start justify-between gap-2">
                    <span className="flex-1">
                      <span className="text-fg">{d.b.label}</span>{" "}
                      <span className="text-fg-subtle">— same as —</span>{" "}
                      <span className="text-fg">{d.a.label}</span>
                      <span className="ml-1 text-[10px] text-fg-subtle">
                        seen in {d.evidence_count} conversation(s)
                        {d.asked_at ? " · asked in chat" : ""}
                      </span>
                    </span>
                    {canWrite && (
                      <span className="flex gap-1">
                        <Button size="sm" onClick={() => same.mutate(d.id)}>
                          Same
                        </Button>
                        <Button size="sm" variant="ghost" onClick={() => distinct.mutate(d.id)}>
                          Different
                        </Button>
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </section>
          )}
        </div>
      </QueryState>

      {words.length > 0 && (
        <section className="space-y-1.5">
          <h3 className="text-xs font-semibold text-fg">Words memory is learning</h3>
          <p className="text-[10px] text-fg-subtle">
            Kinds of fact memory had no word for. A new one is only counted; it is learned
            once it keeps coming up, and its facts then wait for review like any other.
            Nothing here needs you — say never to one that is noise.
          </p>
          {words.map((t) => (
            <div key={t.name} className="rounded-lg border border-border bg-surface p-2 text-xs">
              <div className="flex items-start justify-between gap-2">
                <span className="flex-1">
                  <span className="font-mono text-fg">{t.label}</span>{" "}
                  {t.status === "active" ? (
                    <Tag kind="info">learned</Tag>
                  ) : (
                    <span className="text-[10px] text-fg-subtle">counting</span>
                  )}
                  <span className="ml-1 text-[10px] text-fg-subtle">
                    seen {t.observations}× in {t.conversations} conversation(s)
                    {t.examples.length > 0 ? ` · e.g. “${t.examples.join("”, “")}”` : ""}
                  </span>
                </span>
                {canWrite && (
                  <span className="flex gap-1">
                    {t.status === "candidate" && (
                      <Button size="sm" onClick={() => activateTerm.mutate(t.name)}>
                        Learn now
                      </Button>
                    )}
                    <Button size="sm" variant="ghost" onClick={() => rejectTerm.mutate(t.name)}>
                      Never
                    </Button>
                  </span>
                )}
              </div>
            </div>
          ))}
        </section>
      )}
    </div>
  );
}

/* ─────────────────────────── Conversations ─────────────────────────── */

export function ConversationsTab({ onInspect }: { onInspect: (sessionId: string) => void }) {
  const canWrite = useWritesEnabled();
  const sessions = useStoredSessions();
  const forget = useForgetMatching();
  const [needle, setNeedle] = useState("");
  const rows = sessions.data?.sessions ?? [];
  const preview = forget.data?.preview;

  return (
    <div className="space-y-6">
      <section className="space-y-2">
        <div>
          <h2 className="text-sm font-semibold text-fg">Forget about something</h2>
          <p className="text-[11px] text-fg-subtle">
            Searches turns, summaries and facts. Shows you what would go before anything
            is deleted.
          </p>
        </div>
        <div className="flex gap-2">
          <input
            className="flex-1 rounded-lg border border-border bg-surface px-2 py-1 text-xs text-fg"
            placeholder="e.g. Northwind"
            value={needle}
            onChange={(e) => setNeedle(e.target.value)}
          />
          <Button
            size="sm"
            variant="ghost"
            disabled={!needle.trim()}
            onClick={() => forget.mutate({ needle, confirm: false })}
          >
            Preview
          </Button>
          {canWrite && preview && preview.total > 0 && (
            <Button size="sm" onClick={() => forget.mutate({ needle, confirm: true })}>
              Delete {preview.total}
            </Button>
          )}
        </div>
        {forget.data?.deleted && (
          <Notice>
            Forgotten — {forget.data.turns ?? 0} turn(s), {forget.data.summaries ?? 0}{" "}
            summary(ies), {forget.data.facts ?? 0} fact(s).
          </Notice>
        )}
        {preview && (
          <div className="space-y-1 rounded-lg border border-border bg-surface p-2 text-xs">
            <p className="text-fg">
              {preview.total} item(s) match “{preview.needle}”
            </p>
            {preview.facts.map((f) => (
              <p key={f.key} className="text-fg-subtle">
                fact — {f.key}: {f.value}
              </p>
            ))}
            {preview.summaries.map((s) => (
              <p key={s.session_id} className="text-fg-subtle">
                summary — {s.session_id}
              </p>
            ))}
            {preview.turns.slice(0, 8).map((t) => (
              <p key={t.id} className="text-fg-subtle">
                turn — [{t.session_id}] {t.role}: {t.content}
              </p>
            ))}
          </div>
        )}
        <WriteGate canWrite={canWrite} />
      </section>

      <section className="space-y-2 border-t border-border pt-4">
        <div>
          <h2 className="text-sm font-semibold text-fg">Stored conversations</h2>
          <p className="text-[11px] text-fg-subtle">
            Full text is kept for {sessions.data?.hot_days ?? 90} days; after that the
            summary stands in — and a session without a summary keeps its text.
          </p>
        </div>
        <QueryState
          loading={sessions.isLoading}
          error={sessions.isError}
          empty={rows.length === 0}
          emptyText="No stored conversations."
        >
          <div className="space-y-1.5">
            {rows.map((s) => (
              <div key={s.session_id} className="rounded-lg border border-border bg-surface p-2 text-xs">
                <div className="flex items-start justify-between gap-2">
                  <span className="flex-1">
                    <span className="font-mono text-fg">{s.session_id}</span>{" "}
                    <Tag kind={s.state === "hot" ? "info" : "warn"}>{s.state}</Tag>
                    {s.is_run && <Tag kind="warn">run</Tag>}
                    <span className="ml-1 text-[10px] text-fg-subtle">
                      {s.turns} turns · {fmtDateTime(s.last_activity)}
                    </span>
                  </span>
                  <Button size="sm" variant="ghost" onClick={() => onInspect(s.session_id)}>
                    Context
                  </Button>
                </div>
                {s.summary && (
                  <p className="mt-1 text-[10px] text-fg-subtle">{s.summary.slice(0, 220)}</p>
                )}
              </div>
            ))}
          </div>
        </QueryState>
      </section>
    </div>
  );
}

/* ─────────────────────────── Context ─────────────────────────── */

export function ContextTab({ sessionId }: { sessionId: string }) {
  const [sid, setSid] = useState(sessionId);
  const inspection = useContextInspection(sid);
  const compact = useCompactSession();
  const canWrite = useWritesEnabled();
  const blocks = inspection.data?.blocks ?? [];

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-sm font-semibold text-fg">What the next turn would carry</h2>
        <p className="text-[11px] text-fg-subtle">
          Every block with its token cost. A block marked absent is one the model will not
          see — which is how six features shipped wired to nothing.
        </p>
      </div>
      <div className="flex gap-2">
        <input
          className="flex-1 rounded-lg border border-border bg-surface px-2 py-1 font-mono text-xs text-fg"
          value={sid}
          onChange={(e) => setSid(e.target.value)}
          placeholder="session id"
        />
        {canWrite && (
          <Button size="sm" variant="ghost" onClick={() => compact.mutate(sid)}>
            Compact now
          </Button>
        )}
      </div>
      <QueryState
        loading={inspection.isLoading}
        error={inspection.isError}
        empty={blocks.length === 0}
        emptyText="Nothing to show for that session."
      >
        <div className="space-y-1">
          {blocks.map((b) => (
            <div
              key={b.block}
              className="flex items-start justify-between gap-2 rounded-lg border border-border bg-surface p-2 text-xs"
            >
              <span className="flex-1">
                <span className="text-fg">{b.block}</span>{" "}
                {!b.present && <Tag kind="warn">absent</Tag>}
                {b.preview && (
                  <span className="ml-1 text-[10px] text-fg-subtle">{b.preview}</span>
                )}
              </span>
              <span className="font-mono text-[10px] text-fg-subtle">{b.tokens} tok</span>
            </div>
          ))}
          {inspection.data?.message && (
            <p className="pt-1 text-[11px] text-fg-subtle">
              Memory graph block computed for: “{inspection.data.message.slice(0, 120)}”
            </p>
          )}
          <p className="pt-1 text-[11px] text-fg-subtle">
            {inspection.data?.total_tokens ?? 0} tokens total
            {inspection.data?.compaction_in_flight ? " · summary roll in flight" : ""}
          </p>
          {(inspection.data?.pointers ?? []).map((p) => (
            <p key={p} className="text-[11px] text-fg-subtle">
              Note: {p}
            </p>
          ))}
        </div>
      </QueryState>
    </div>
  );
}

/* ─────────────────────────── Housekeeping ─────────────────────────── */

export function HousekeepingTab() {
  const canWrite = useWritesEnabled();
  const runs = useHousekeepingRuns();
  const run = useRunHousekeeping();
  const rows = runs.data?.runs ?? [];

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-sm font-semibold text-fg">Retention</h2>
        <p className="text-[11px] text-fg-subtle">
          Runs daily. It cools old conversations to their summary, sweeps vectors whose row
          is gone and rotates logs. It never deletes a summary, a confirmed fact, or
          anything you wrote.
        </p>
      </div>
      <div className="flex gap-2">
        <Button size="sm" variant="ghost" onClick={() => run.mutate(true)}>
          Dry run
        </Button>
        {canWrite && (
          <Button size="sm" onClick={() => run.mutate(false)}>
            Run now
          </Button>
        )}
      </div>
      <WriteGate canWrite={canWrite} />
      <QueryState
        loading={runs.isLoading}
        error={runs.isError}
        empty={rows.length === 0}
        emptyText="No runs recorded yet."
      >
        <div className="space-y-1.5">
          {rows.map((r) => (
            <div key={r.started_at} className="rounded-lg border border-border bg-surface p-2 text-xs">
              <span className="text-fg">{fmtDateTime(r.started_at)}</span>{" "}
              {r.dry_run && <Tag kind="info">dry run</Tag>}
              {r.errors.length > 0 && <Tag kind="warn">{r.errors.length} error(s)</Tag>}
              <p className="mt-1 text-[10px] text-fg-subtle">
                cooled {r.sessions_cooled} · kept unsummarized {r.sessions_kept_unsummarized} ·{" "}
                {r.turns_deleted} turns and {r.vectors_deleted} vectors removed ·{" "}
                {r.orphan_vectors_swept} orphan vectors swept · logs {r.logs_compressed}{" "}
                compressed / {r.logs_deleted} deleted ({fmtBytes(r.log_bytes_reclaimed)})
              </p>
            </div>
          ))}
        </div>
      </QueryState>
    </div>
  );
}
