# Memory subsystem

**Status:** Current as of 2026-06-25 — written from code (`src/iris_harness/memory/`).
Engineering doc; PRD intent lives in `project-iris-prd/05-memory-and-learning.md`.
The harness-level summary is in `iris-harness.md` §5; the **operator/user how-to for
curation** (capture gates, recall filter, correct/forget/restore, retention,
contradictions) is `docs/usage-guides/memory-curation.md`; this is the architecture
detail.

IRIS memory is a **dual store**: durable rows in SQLite (`MemoryStore`) plus a
semantic index over ChromaDB (`SemanticIndex`). The retriever blends both into the
`MemoryContext` assembled for every turn.

## The five ChromaDB collections

All embedded with ChromaDB's default local ONNX MiniLM-L6 function (no Ollama
dependency; ~80 MB one-time download).

| Collection | Holds | Doc id |
|---|---|---|
| `iris_user_facts` | one doc per `UserFact` | fact key |
| `iris_learning_signals` | one doc per learning signal | signal id |
| `iris_conversation_turns` | one doc per stored turn | SQLite row id |
| `iris_wiki_pages` | one doc per `WikiPage` | slug |
| `iris_episodic_patterns` | one pattern per line from `~/.iris/memory/episodic.md` | pattern |

(Note: RAG documents are a **separate** index — `iris_documents` in `data/chroma_docs`,
owned by `src/iris_harness/services/rag/`, not this subsystem.)

The RAG index follows the same rule (canonical store plus rebuildable projection), by a
different route. The canonical chunk text and each chunk's classification live in
`data/rag.db` (`DocumentStore`); the documents stay where the user keeps them;
`iris_documents` only mirrors the stored chunks. If `data/chroma_docs` is lost or damaged,
`iris docs reindex` (`reindex_all`) rebuilds it from `rag.db` without reading any source
file. `iris docs sync` is not a rebuild: it skips files whose mtime or hash is unchanged, so
it never refills an empty index. It does prune: a registered file that is gone from disk
is removed from the store and the index and reported to the file domain (`removed`), but
only when its parent folder is readable; otherwise (an unmounted volume) it is left alone and
counted as unavailable. If `rag.db` is lost instead, re-run `iris docs add <path>`
for each source; chunks ingested that way carry no classification stamp unless they go
back through the ingest gate (`execute_rag_ingest`), which is what sets it.

Two properties of `reindex_all` to know. It refuses (`EmptyStoreRefused`; `--force` on the
CLI overrides) when `rag.db` holds zero chunks but the index does not, since the rebuild
would delete every vector; the usual cause is a wrong `IRIS_DATA_DIR`. And it is not atomic
with a running server: it snapshots the stored chunks, then upserts and prunes. A chunk the
server ingests during the rebuild is missing from the snapshot, so the prune can drop it
from the index until the next `iris docs sync` (or re-ingest) puts it back; it stays safe
in `rag.db`. Run the rebuild while ingestion is quiet.

One failure `reindex_all` cannot repair by itself: if the embedding model changed, Chroma
refuses to open the persisted collection under the new embedder (an "embedding function
already exists" conflict), so the index is unavailable and retrieval falls back to keyword
search; labels and sources in `rag.db` are untouched. `iris docs reindex` then says so and
names the fix: `iris docs reindex --reset-collection` (`reset_and_reindex`), explicit and
off by default. It deletes and re-creates only the `iris_documents` collection (no other
collection, never `rag.db`, never a source file) and refills it from `store.iter_chunks()`,
so every source and chunk keeps its label; it applies the same empty-store refusal (and
`--force`) before it deletes anything, and logs one INFO line with the chunk count. If the rebuild fails after the delete (disk full, embedder error) it raises `IndexRebuildIncomplete`: the index is partial, `rag.db` is untouched, and rerunning the command is idempotent. Unlike a
plain rebuild, a reset invalidates other handles on the collection: the API server keeps
one on `app.state` for its lifetime. The CLI cannot detect a running server, so stop the
server before the reset and restart it after. A server left running recovers on its next
call: `DocumentIndex` catches Chroma's `chromadb.errors.NotFoundError` ("Collection [...]
does not exist"), logs one WARNING, reopens the collection under the current embedder and
retries once; if the reopen fails it marks the index unavailable (WARNING) and retrieval
searches by keyword. A failed `index_chunks` now logs a WARNING with the chunk count (no
text) instead of a debug line. There is no HTTP route for either `reindex` or the reset, by design.

## Components

| Component | File | Role |
|---|---|---|
| `MemoryStore` | `memory/store.py` | SQLite system of record (facts, turns, signals). |
| `SemanticIndex` | `memory/semantic_index.py` | ChromaDB wrapper; `query_*` + `index_*` + `sync_from_store`. |
| `MemoryRetriever` | `memory/retriever.py` | `build_context()` → `MemoryContext`; blends semantic + keyword ranking; attaches identity layers. |
| `ConversationCompactor` | `memory/compactor.py` | Summarizes older turns when the history hits ~80% of the model window (or a turn-count floor) — **through the governance kernel** (see below). |
| fact extraction/validation | `memory/fact_extractor.py`, `fact_validation.py` | LLM fact extraction + **three** gates before a fact is stored: plausibility, durability, grounding (see Curation & integrity). |
| `UserProfile` | `memory/profile.py` | Structured view of stored user facts. |
| triage | `memory/triage.py` | Decides where a learned item belongs. |

## Integration points (the part that was undocumented)

### Incremental turn indexing (watermark)

`SemanticIndex.sync_from_store(store)` is called at startup and indexes only rows
added since the last run, tracked by a watermark file at `data/chroma_watermark`:

```
watermark = self._load_watermark()
new_turns = store.load_turns_since(min_id=watermark)   # only the delta
... index ...
self._save_watermark(max_id)
```

During a live session the individual `index_*` methods are called after each SQLite
write, so the two stores stay in sync without a full rescan.

### Cross-session recall

`query_turns(query, *, exclude_session=...)` retrieves semantically similar prior
turns but **filters out the current session's own turns** (`where = {"session_id":
{"$ne": exclude_session}}`). The retriever passes the live `session_id`, so recall
surfaces relevant history from *other* conversations without echoing the one in
progress.

### Governance-hooked compaction

`ConversationCompactor` does not call the model directly — `_invoke_with_governance`
fires `PreClassify` + `PreLLMCall` through `kernel_from_env()` first, and **raises if
the kernel blocks** the call. Summarizing your conversation history is itself a
governed LLM call (subject to egress/redaction/cost), not a side channel.

### Window-aware auto-compaction (ADR-0079)

Compaction fires on **two triggers, whichever comes first**:

- **window (load-bearing)** — the running history's estimated tokens reach
  `compaction_ratio` (default **0.8**) of `token_budget`, the chat tier's
  window-derived prompt budget (`budget_for(num_ctx)` for the
  `communication`/Tier-2 model). This is the Claude-Code-style "compact at ~80% of
  the context window" behaviour: a few large turns fill the window long before any
  turn count is hit.
- **count (floor)** — more than `compaction_threshold` (default 20) turns, for
  tiny-turn chats where no window is wired.

On compaction the oldest turns are summarized and the most-recent turns are kept
verbatim, **bounded by a token sub-budget** (≈ half the trigger) rather than a fixed
count, so even a single huge turn compacts while ≥ 1 recent turn is always kept.
Empty / whitespace-only turns are dropped from the summary input (keep only the last
*good* turns). The archived span is also exposed (`CompactedHistory.archived_turns`) so the
behavior miner can mine it before it leaves the working window — "learn before you forget"
(ADR-0082; see `digital-twin.md` Layer 1). The compactor is built window-derived in
`build_runtime` from the live
`tier_router`; env overrides win (`IRIS_COMPACTION_{TOKEN_BUDGET,RATIO,THRESHOLD,
KEEP_RECENT}`). Each event emits telemetry (`trigger`, `archived`, `tokens
before->after`, `kept_turns`) via `_record_turn`. This is the cross-turn twin of the
ADR-0077 P3 in-loop `ContextBudgetController` (`src/iris_harness/agent/context_budget.py`),
which bounds a single ReAct transcript with the same window-derived discipline.

### Context-health surface (ADR-0081)

The compactor (above), the P3 in-loop controller, and the surface-feedback
suppression spine (ADR-0078) each bound the working context but logged in isolation.
`IrisRuntime.context_health(session_id)` composes them into one read-only snapshot —
window fill % + last compaction, the transcript/memory budget split + latest in-loop
eviction (via a `budget_observer` the `AgenticCore` fires per iteration into a shared
sink), and the suppression roll-up (`SurfaceFeedbackStore.summary()`). Pure
composition in `src/iris_harness/agent/context_health.py`; surfaced at `GET /context-health`,
`iris context-health`, and the web Health screen's Context-budget panel.

**Acting on it (ADR-0084).** Auto-compaction is already eager (every turn boundary), so
the surface's job isn't to trigger it — it's to let you *act* on the pressure you see.
`IrisRuntime.compact_now(session_id)` force-compacts on demand (`compact(..., force=True)`
summarizes the older span even below the auto trigger; a short conversation is a no-op),
applied through the shared `_apply_compaction` helper that the reactive `_record_turn` path
also uses. Surfaced at write-gated `POST /context-health/compact` and a **Compact now**
button on the Context-budget panel.

**The agent acting on itself (ADR-0086).** The same controls are exposed to the **agent**
as ReAct tools — `context_health`, `compact_context`, `learning_status` — so IRIS can
introspect and manage its own state mid-turn (*"I'm near full, let me compact"*), not just
the user. Built per-turn in `_core_for` (capturing the live `session_id`), late-bound to the
runtime via a holder, relevance-shortlisted so they surface only on self-referential turns.
Opt-in (`IRIS_AGENT_SELF_MANAGEMENT`, off by default — one tool mutates the conversation),
and **hot-toggleable** from the web Experiment console (`self_management` flag) with no
restart — `_core_for` reads `runtime.learning.self_management_enabled()` (override > env) per turn.

### Identity layers

`MemoryRetriever._attach_identity_context` loads the always-on identity files
(SOUL/USER and the ACTIVE/EPISODIC/BEHAVIORS layers) into the context each turn, each
capped (`max_identity_chars`) so identity markdown can't dominate a small-tier context.

### Fail-safe degradation

Every `SemanticIndex` method is safe when ChromaDB failed to initialize
(`is_ready == False`): it returns empty results and the retriever falls back to keyword
ranking. Memory degrades, the agent keeps working.

Degrading is never silent, though: a store read or a semantic query that fails *after*
startup returns the same empty/partial result but logs a WARNING naming the operation
and the exception type (never the query or memory content), so a broken store is not
mistaken for an empty one. `tests/unit/test_logging_coverage.py` keeps it that way: a
broad `except` under `memris/`, `memory/`, `services/rag/` or `runtime/session_memory.py`
must log, re-raise, or carry a `silent-ok: <reason>` marker. One read fails *closed*:
if the removal ledger (ADR-0119) cannot be read, no cross-session turn is recalled,
since the retriever can no longer tell which sessions the owner removed.

## How a turn uses it

```
MemoryRetriever.build_context(query, session_id):
  facts      = SemanticIndex.query_facts(query)        ∪ keyword-ranked store facts
  turns      = SemanticIndex.query_turns(query, exclude_session=session_id)
  episodic   = SemanticIndex.query_episodic(query)
  + identity layers (SOUL/USER/ACTIVE/EPISODIC/BEHAVIORS, capped)
  → MemoryContext  (signals are fetched but not currently injected — see harness §5)
```

## Curation & integrity

Memory of the user is kept **clean, reversible, and auditable**. Two independent lines
of defense plus full provenance. (User-facing how-to: `usage-guides/memory-curation.md`.)

### Capture gates (keep junk OUT of the store)

At the persistence choke point (`TurnCapture.extract_and_store_facts` in
`runtime/turn_capture.py`, reached as `runtime.capture`), every candidate clears three deterministic gates in
`fact_validation.py`, each logged on rejection:

1. **`is_plausible_fact`** — no sentence fragments, first-person clauses, or over-long
   values (the gate that pinned the `name="working on a project called Aur"` corruption).
2. **`is_durable_fact`** — rejects ephemeral/conversational tokens: ephemeral *keys*
   (`greeting`, `day`, `task`, `reminder_time`, …) and ephemeral *values* (greetings,
   relative dates, `am/pm` clock times). The deterministic floor under the extractor's
   durable-only prompt (`FACT_EXTRACTION_SYSTEM_PROMPT`).
3. **`is_fact_grounded`** — the value must be supported by the user's message (stops the
   LLM bleeding its own few-shot examples).

Questions/commands and "for this conversation" statements are skipped before extraction.
The upsert is **confidence-gated** (a weak value can't clobber a strong one).

### Recall quality filter (keep junk OUT of the prompt)

`MemoryRetriever` filters stored facts by confidence before injection
(`min_fact_confidence=0.35` drop, `uncertain_below=0.60` mark `(unconfirmed)`;
env-tunable). Applies only to extracted store facts — the curated `USER.md` block is
never filtered.

### Measuring both defences (ADR-0085)

`iris_harness.memory.gate_audit` is the meter for the two defences above. Given **labelled**
candidates it scores the capture gates and the recall filter — **false-admit** (junk
that slipped the gates), **false-recall** (junk that reached the prompt), **false-reject
/ false-drop** (real facts over-blocked) — with per-gate attribution, and sweeps the drop
threshold for a balanced-accuracy suggestion. `run_full_audit` models the real pipeline
(gates on all; recall on the *admitted* subset). Pure + deterministic; a built-in
synthetic `DEFAULT_CORPUS` makes it run out of the box. Surface: `iris facts gate-eval
[--corpus x.json]`. It **measures and suggests, never auto-tunes** — changing
`IRIS_MEMORY_MIN_FACT_CONFIDENCE` stays a human, real-data call. Distinct from `iris facts
audit`, which buckets your *real* facts by the live thresholds (no labels).

### Provenance, reversibility, retention, contradictions (store tables)

`MemoryStore` carries three sidecar tables beyond `user_facts`:

| Table | Purpose | Store methods |
|---|---|---|
| `user_fact_history` | append-only audit trail of every value change (`capture`/`supersede`/`correct`/`forget`/`restore`) | `correct_user_fact`, `delete_user_fact` (reversible forget), `restore_user_fact`, `fetch_fact_history` |
| `user_fact_history` (retention) | human-reviewed retention — entries past the window (180d) surfaced, never auto-deleted | `fetch_history_retention_candidates`, `prune_history_entries` |
| `user_fact_contradictions` | detected same-key value conflicts (`superseded`/`blocked`, deduped via `seen_count`) | `fetch_contradictions`, `acknowledge_contradictions` |

Surfaces (CLI `iris facts`, chat tools `memory_correct`/`memory_forget`/`memory_restore`,
web UI Memory screen, and `/memory/*` API endpoints) are all thin layers over these
methods.

## Related

- `iris-harness.md` §5 — harness-level memory summary.
- `usage-guides/memory-curation.md` — operator/user how-to for curation.
- `learning-subsystem.md` — the `iris_learning_signals` producer.
- `src/iris_harness/services/rag/` — the separate `iris_documents` RAG index.
- `DOCS-DRIFT-AUDIT.md` — why this doc exists.
