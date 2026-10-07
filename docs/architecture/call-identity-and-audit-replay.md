# Design note: one unique, parent-linked id per record, and a gap-free replay (issue #134)

Status: Design for issue #134; stages 1, 2 and 3 implemented.
Evidence labels: [RUN] = observed by running current code; [READ] = read from source only.

## 0. Headline findings

1. `tool_call_id` appears in NO audit_log row [RUN: 0 of 82 payloads]. It lives only in the side-effect key
   (`run:step:tool_call_id`, high-risk calls only) and in session-log `tool.invoke.*` events.
2. The held attempt of an approved call loses its call id entirely: the held PRE rows and the approval row carry
   run/step/approval_id, and the id minted for that attempt is written to no store [RUN].
3. A call made on behalf of another call (plugin `api.tools.call`, capability call) gets a fresh run_id and
   step=None; there is no parent pointer anywhere [RUN].
4. Audit writes fail open: a failed insert is a logger warning, the call proceeds, no marker, no id hole
   (AUTOINCREMENT does not burn an id on a failed insert), `audit_gaps()` stays empty [RUN].
5. `audit_gaps()` is count-based, so deleting one of two `pre_llm_call` rows is not detected [RUN].
6. `SideEffectLedger.record()` still uses INSERT OR IGNORE and returns the key as if recorded; a second, different
   effect under the same key is dropped silently [RUN] (#102 is still open on main).
7. Compaction deletes hot rows and the Parquet archive schema has no `id` column (cold view uses `NULL AS id`):
   retention destroys the only sequence there is [READ].
8. The turn id (`turn_id`, session log + learning.db) is not in audit_log; the trace builder joins audit to session
   log by session_id plus a timestamp window [READ].

## 1. Inventory of stores (verified)

Legend for "silent drop": can a write be lost with only a log line.

| Store (file) | Written by | Identifier columns today | Unique / not unique | Silent drop? | Retention | Read by |
|---|---|---|---|---|---|---|
| governance `audit_log` (`<gov dir>/audit.db`, `~/.local/share/iris/audit.db`) | `GovernanceKernel._audit` (one row per hook firing, ~10-14 rows per tool call), approvals queue, response curator, evaluator judge, `react.py` synthesis fallback, governance_judge, plugins via stable `sdk.audit.AuditLog.record` [READ] | `id` INTEGER AUTOINCREMENT (global per DB), `run_id` TEXT, `step_id` INT null; `session_id`, `trace_id`, `caller`, `tool_name`, `approval_id`, `side_effect_id` only inside `payload_json` | `id` unique, global, gap-free while nothing is deleted [RUN: ids 1..82 contiguous]. run_id not unique per anything (see sec 2). No call id. | YES: `kernel._audit` catches everything, `logger.warning`, proceeds [RUN]. Curator/judge/router sites also swallow [READ]. Kernel built with `audit_log=None` skips persistence silently (hand-built kernels) [READ]. `IRIS_GOVERNANCE_ENABLED` off = no kernel, tool runner runs ungoverned [READ]. | 30 days hot (`iris audit compact`, manual CLI) then Parquet [READ] | audit view, `GET /governance/audit`, CLI, trace builder, proof bundle, harness `audit_rows`/`audit_gaps` |
| side-effect ledger (`side_effects.db`) | `PreToolUseLedgerHook` (write-ahead, high-risk only), `PostToolUseLedgerHook` (settle, or insert for other non-read when `..._ALL`) | `side_effect_id` PK = `run:step:tool_call_id` (call id `-` when absent), `run_id`, `step_id` | Unique per key. NOT unique when call id omitted/reused. Row is mutated (pending -> completed/error), so history of transitions is not kept | YES: `INSERT OR IGNORE`; post-hook write failure is warn-and-allow; pre-hook failure fails closed (good) [READ+RUN] | none (never pruned) | `iris run resume`, post hook |
| router audit (`<data dir>/audit.db` table `router_decisions`) | `RouterAuditLogger.record` per turn (`route` stage) | `id` AUTOINCREMENT, `session_id`; no turn id, no run id | id unique | YES: sqlite error swallowed, warning [READ]. Append-only triggers | none | operators (sql) |
| governor guard audit (same filename, other dir/table `governor_guard_audit`) | `GovernorAuditLogger.record_decision` from MCP bridge `_guard_mcp_action` and the governor service | `id` AUTOINCREMENT, `route`, `action`, `metadata_json`; NO run/session/call id | id unique | Raises `GovernorAuditError`, not caught by the bridge, so the MCP call fails (fail closed) [READ] | none | governor API |
| session logs (`<IRIS_HOME>/logs/session-<id>.jsonl`) | `session_log._append_event`, `log_timeline_event`, `llm_call_scope` | `session_id`, `turn_id` (uuid4 hex per `turn_scope`), `agent_type`, `iteration` (None for tool events [RUN]); tool events carry `tool_call_id` | No event id, no sequence. File append only | YES: `logger.exception ... continuing`; no-op outside `session_scope` (approved code-caller call and held attempts leave nothing [RUN, p4]) | none | trace builder, memory, proof bundle observations |
| Parquet archive (`audit/year=*/month=*/audit-<uuid>.parquet`) | `AuditCompactor` (write, fsync, then DELETE hot rows) | all audit columns EXCEPT `id` | rows have no id | n/a; delete happens after write, but nothing records which id range went where | forever | DuckDB `iris audit query` (UNION hot + cold, `id` NULL for cold) |
| traces / telemetry (OTel, `trace_builder.py`) | spans around turn and stages; `trace_id` stamped into audit payload only when a span records [READ; RUN: absent in harness] | `trace_id`, span ids | n/a | best-effort by design | backend dependent | trace builder joins audit by session + time window |
| learning.db `signals` | learning loop | `id`, `session_id`, `turn_id`, `trace_id`, `span_id` (the only store with turn + trace) | id unique | swallowed [READ] | long | learning screens |
| memory.db `conversations` | session memory | int `id`, `session_id`; no turn id | id unique | n/a | long | memory |
| approvals queue (`approvals.db` `approval_queue`) | `ApprovalQueue.enqueue/respond/claim_execution` | `approval_id` (uuid4) PK, `run_id`, `checkpoint_id` (= run id in practice [RUN]), `session_id`, `caller`; NO step, NO tool_call_id | unique id | enqueue is not silent; audit row for each transition is swallow-on-failure only at the audit layer | none (rows mutate: pending -> approved -> executed_at) | action center, resume |
| cost ledger (`cost-ledger.db`) | `CostLimiter._record` (only when a cap is configured; off by default, no file in harness) | `id`, `run_id`, no step/session | id unique | YES: `ledger write failed; allowing call` [READ] | none | cost summary |
| checkpoints (`checkpoints.db`) | agentic loop on halt | PK `(run_id, step_id)`, `session_id` | upsert: `ON CONFLICT DO UPDATE` overwrites silently by design | by design | 7 days TTL | resume |
| continuations (`memory/state/continuations.py`) | confirmations | `continuation_id`, `run_id`/`step_id` nullable, session | unique | n/a | TTL | intercept dispatch |
| vault (`vault_secrets`) | vault store | handle only; has no audit table. Handle resolution shows up only as `credential_broker` PRE rows (handles, never values) | n/a | follows audit_log | n/a | broker |
| egress / ingress logs | `log_egress`, `log_ingress` (python loggers `iris.egress`/`iris.ingress`) | none (free text; ingress has `session=` only) | no records, just log lines on stdout/uvicorn log files | YES: `log_egress` swallows all exceptions; log level/handler dependent | whatever the process log policy is | humans, grep |
| proof bundle (`audit/proof_bundle.py`) | export | row ids, ts, hook, decision; "observations" from session logs | closed schema (v1: unexpected keys fail verify) | n/a | n/a | auditors |

Every place the kernel warns and proceeds when an audit/ledger write fails [READ unless noted]:
`kernel.py:_audit` [RUN]; `PostToolUseLedgerHook` record and settle ("could not record/settle", `severity=warn`, allow);
`CostLimiter._record`; `RouterAuditLogger.record`; `response_curator` audit-signal write; `evaluator/judge.py`
`_audit_verdict`/`_audit_warn`; `session_log._append_event`; `log_egress`; capability `_capability_failed`
("could not settle its failed call", `logger.exception`); `_settle_unrun_tool`; `AuditLog._safe_json` falls back to `{}`;
`email_workflows` and `connections/google.py` audit writes. Fail-CLOSED today: `PreToolUseLedgerHook` (high-risk call
denied if its pre row cannot be written), audit-key unavailable (`AuditKeyUnavailable` refuses governed calls),
governor guard audit.

## 2. What the ids mean today

| Id | Minted where | Scope / lifetime | Notes |
|---|---|---|---|
| `session_id` | the caller of `chat`/`chat_stream` (web, CLI REPL via API, Telegram via gateway proxy to the API); MCP serve mints `mcp-<12hex>` per connection; harness `harness-N` | a conversation; many turns, many processes | held in a ContextVar (`session_scope`); kernel stamps it into audit payload |
| `turn_id` | `session_log.turn_scope()` in `facade.chat/chat_stream/resume_halted_run/open_turn` (uuid4 hex) | one user message; a resumed run gets a NEW turn id [RUN: 599f.. then 36d8..] | in session log and learning.db only; NOT in audit_log, ledger, approvals |
| `run_id` | no single meaning. (a) ReAct loop run: `uuid4` in `AgenticCore.run`; survives a halt, resume re-enters it via checkpoint (spans two turns [RUN]). (b) one per governed model call outside the loop: `uuid4` in `llm/client._governance_pre_llm` (router, planner, compactor...). (c) PRE_TURN: `uuid4().hex`. (d) a code-caller or capability call: fresh `uuid4` per call. (e) general-lane tool loop: one per turn's tool loop. (f) curator / response_safety rows: `run_id = session_id`. (g) MCP bridge outbound: constant `mcp-<server>`. (h) `react.py` synthesis fallback: fresh uuid | so a run is NOT a session, NOT a turn, NOT a step: it is "whatever the minting site chose" | seen vocabularies in one real session [RUN]: loop run, 2 model-call runs, 1 `S1`, 1 hex PRE_TURN run |
| `step_id` | ReAct iteration index (`range(start_iteration, max)`); PRE_LLM_CALL and the tool call of that iteration share it; a resumed approval executes at `start_iteration-1` | within a run | None for non-loop calls (stored as 0 in the ledger key via `ctx.step_id or 0`) |
| `tool_call_id` | `GovernedToolRunner.execute`: `uuid4().hex[:12]` (48 bits) minted before PRE_TOOL_USE; capability calls: `tool_call_id = run_id` (full uuid); MCP bridge: `None` | one call attempt | NOT in audit_log. In ledger key and session log. A second, unrelated "tool_call_id" exists in the general lane: the model/provider id (`LLMToolCall.id`, `react_{turn}` which repeats every turn, `xml-0`) [READ] |
| `approval_id` | `ApprovalQueue.enqueue` uuid4 | one approval; held -> approved -> executed | in audit payload of `approval_queue` rows and of the `destructive_approval` PRE row (before and after approval): the one existing held<->approved join [RUN] |
| `side_effect_id` | key string `run:step:call` | one high-risk call | in audit_metadata of the two ledger-hook rows only |
| `trace_id` | OTel | one trace | stamped only if a span records |

Relations: session 1-n turn; turn 1-n (loop run | model-call run | tool run); a loop run n-1 turn when halted and resumed;
step 1-n calls in a run (a model call and a tool call share the step number). No store records these edges except by
coincidence of ids.

Table: which stores carry which (Y = column, P = only inside json payload, - = absent)

| Store | session | turn | run | step | tool_call | approval | parent |
|---|---|---|---|---|---|---|---|
| audit_log | P | - | Y | Y | - | P | - |
| side-effect ledger | - | - | Y | Y | in key | - | - |
| router_decisions | Y | - | - | - | - | - | - |
| governor_guard_audit | - | - | - | - | - | - | - |
| session log | Y | Y | - | `iteration` (None for tools) | tool events | - | - |
| approval_queue | Y | - | Y | - | - | Y | - |
| checkpoints | Y | - | Y | Y | - | - | - |
| cost ledger | - | - | Y | - | - | - | - |
| learning.db signals | Y | Y | - | - | - | - | - (has trace/span) |
| Parquet archive | P | - | Y | Y | - | P | - (no id) |
| egress/ingress log lines | ingress only | - | - | - | - | - | - |

## 3. Every path that mints (or fails to mint) a call id

| Path | run_id | step_id | tool_call_id | Parent edge recorded? |
|---|---|---|---|---|
| ReAct loop -> `_execute_tool` -> `runner.execute` | loop run | iteration | runner `hex[:12]` | no (session, run, step only) |
| general-lane plugin tool `_run_plugin_tool` | per-turn tool-loop uuid | None -> 0 | runner-minted | no |
| general-lane builtin and skill tools (memory_search, wiki_search, local skills...) | per-turn tool-loop uuid | None -> 0 | runner-minted (since #155; they used to run with no PRE/POST_TOOL_USE and no audit row, under the model/provider id) | no. The lane serves only with `IRIS_AGENTIC_CORE_ENABLED` off or `shadow`, never by default |
| `ToolService._run` (plugin `api.tools.call`, `core:<workflow>`, `call_for_client`) | fresh uuid per call | None | runner-minted | no: `caller=plugin:x` only [RUN: child run 3b349fd7 vs parent fb0273a3] |
| approved code-caller call `_run_approved` | `row.run_id` (the original child run) | None -> 0 | NEW runner id; held attempt's id lost | approval_id in audit rows; no tool_call link [RUN p4] |
| approved loop call `_settle_pending_approval` | same loop run | `start_iteration-1` (same step as the held attempt) | NEW id; held attempt's id lost | approval_id only [RUN] |
| capability call (`execute_call`, `aexecute_call`, `aexecute_stream`) | fresh uuid per call | None | `= run_id` | no parent (caller string only) [READ] |
| MCP bridge outbound `invoke_external_tool` (only caller: HTTP route `mcp_server.py`, no run, persona, session scope) | `mcp-<server>` constant, shared by all calls | None | `None` in POST metadata, absent in PRE | no; ledger never sees it (effect None) [READ] |
| `mcp_serve` inbound (`call_for_client`) | fresh uuid per call | None | runner-minted | session `mcp-<12hex>` only; no turn scope |
| Model calls (`CodingLLMClient`, router, planner, compactor, extractor) | fresh uuid per call, PRE_CLASSIFY + PRE_LLM_CALL share it | None (loop: iteration) | no id, and no settle record (there is no POST_LLM_CALL hook) | no |
| PRE_TURN (`screen.py`) | `uuid4().hex` | None | - | session only |
| `iris run resume` CLI | re-enters run via `resume_halted_run`; its own process/kernel | - | - | `side_effect_ledger.pending(run_id)` is read, "re_exec" only printed, retry is never recorded |
| Telegram/web/CLI REPL | all converge on `chat_stream` in the API process (`turn_scope`), so one turn id site; channel_gateway ingress log lines carry no turn id | | | |
| heartbeats/routines/ticks | outside `session_scope`: session None [READ, not run] | | | |

Edges that exist but are not recorded: tool -> child call (plugin or capability), held attempt -> approved re-execution
(only approval_id), crash -> resumed run (run id reused, new turn id, no "resumed_from"), retry (`iris run resume`
re_exec, loop repeated-action guard), model call -> tool call (same step number only), turn -> turn (resume).

## 4. Proposed scheme

### 4.1 Record identity
- `record_id`: opaque, kernel-minted, globally unique, time-sortable (ULID, 26 chars; uuid7 acceptable). Minted INSIDE the
  store write method (`AuditLog.record`, `SideEffectLedger.record`, `ApprovalQueue.enqueue`, session-log append), never taken
  from a caller kwarg. Because `sdk.audit.AuditLog.record` is stable-tier and plugins call it, minting there covers plugins
  without any new caller obligation. Existing integer `id` stays (local ordering, compaction key).
- `call_id`: a second minted id, one per call ATTEMPT, minted in exactly one place: the runner, before PRE_TOOL_USE
  (replaces `hex[:12]`; same ULID format). Every row about the call carries it: all ~14 PRE/POST hook rows, the ledger row,
  the session-log events, the approval row (so a held attempt keeps its id). Capability calls and MCP calls get the
  same minting (`call_id` != `run_id`; fixes `tool_call_id=None` and the `run_id`-as-call-id overload). Model calls get a
  `call_id` too (kind=`model`), with a settle record (new `post_llm_call` row), so a model call has PRE and settle like a tool call.
  A caller cannot supply it: `ToolCall` has no such field; hooks read it from kernel-stamped metadata like `tool_call_id` today.
- Relationship columns (real columns, not payload): `session_id`, `turn_id`, `run_id` (documented as "a loop/agent run"),
  `step_id`, `call_id`, `parent_call_id`, `attempt` (1..n), `replay_of` (a call_id or record_id), `resumed_from_run` (run
  level, on the first record of a resumed run), `kind` (`hook|tool|capability|mcp|model|approval|resume|gap|compaction`).
- Propagation (the parent edge): a ContextVar `current_call` (call_id, run_id, turn_id) set by the runner around
  `tool.call(...)`. `ToolService._run`, capability `invoke`, MCP bridge read it to fill `parent_call_id`, `run_id`(inherit)
  and `turn_id`. This fixes the three fresh-run mints. Held -> approved: the approval row stores `call_id` of the held attempt;
  the approved execution is a NEW call with `attempt=2`, `parent_call_id=<held call_id>`, `replay_of=<held call_id>` only if it
  is a re-execution of the same effect (see 4.3). Resume: first record of the resumed run carries `resumed_from_run`.
  Turn id must move to the kernel context (read `current_turn_id()` in `_audit`, as it already does for session_id).
- URN rendering, derived, never stored as truth:
  `urn:iris:session/<S>/turn/<T>/run/<R>/step/<N>/call/<C>` (+ `/attempt/<A>`). Parse is lossless from the columns.

### 4.2 Completeness (sequence, gap detection)
Two facts decide the design: (a) [RUN] a failed insert leaves no id hole, so an AUTOINCREMENT or an in-transaction counter
cannot reveal a lost write; (b) [RUN bench] a per-session counter allocated with `BEGIN IMMEDIATE` costs about the same as
today single-process (630 vs 814 us/row) but collapses under 4 writer processes (3,268 vs 991 us/row per process; 1,188 vs
3,651 rows/s total) on this laptop; and sessions are touched by several processes (API, CLI `iris approvals`/`iris run resume`).
Therefore: per-WRITER sequence, allocated in memory BEFORE the write attempt.
- `writer_id` = ULID minted at process start (an "incarnation"); `writer_seq` = in-process monotonic counter (a lock + int),
  incremented for every record the process tries to write to ANY audit store (or one counter per store, recorded as `store`).
  Columns on every record: `writer_id`, `writer_seq`.
- A lost write (fail-open) now leaves a hole in `(writer_id, writer_seq)` that the next successful record exposes. Plus each
  writer emits a `writer.start` (seq 0) and `writer.close` (last seq) record; a missing close with no later activity is
  reported as "writer ended without close: tail unknown", never as complete.
- Per-session order: the session timeline is ordered by `(ts, writer_id, writer_seq)`; the issue's "per-session monotonic
  sequence" is derived at replay time per writer, and completeness is proven per writer then merged. If the owner insists on one
  per-session number, the option is a `session_seq` allocated by a session-owner writer (the process holding the session lock);
  not recommended given multi-process writers (decision D2).
- Batch: write the ~12 hook rows of one `fire()` in one transaction (`record_many`); the bench shows connection open per
  row dominates (~0.8 ms), so this is a net performance win that pays for the extra columns.

### 4.3 Idempotency versus identity
- Identity: every attempt/record has its own `record_id`/`call_id`. A key collision is an ERROR (raise `LedgerKeyConflict`),
  never `INSERT OR IGNORE`. `record()` returns `inserted: bool` (#102 asks for this).
- Idempotency is explicit and separate: an optional `idempotency_key` (sha256 of tool + canonical args digest + approval_id or
  caller-supplied) stored on the call. Same key seen again:
  1. a stream item of the same call (same `call_id`): no new row (today's "capability stream items recorded once");
  2. a deliberate re-execution (approved resume of a held call, `iris run resume` retry, loop retry): NEW call_id,
     `attempt=n+1`, `replay_of=<first call_id>`, recorded as a replay; policy decides whether the effect runs (probe first);
  3. a duplicate delivery (same idempotency key AND same call_id arriving twice): second write is a `duplicate_of` record that
     points at the first and is counted, not dropped.
  The HMAC digest machinery (`audit/digest.py`) already produces keyed `args_digest`; reuse it so no argument text is stored.
- Ledger key becomes `side_effect_id = call_id` (unique by construction); the old `run:step:call` string is kept as a derived
  display value for existing rows.

### 4.4 Fail-open audit failure becomes a recorded/noticed gap
Tiered by effect (owner decision D3):
- Effectful boundaries (write/destructive tools, cloud-tier model call, outbound egress): fail CLOSED when neither the DB nor
  the spool accepts the PRE record (extends what `PreToolUseLedgerHook` already does for high-risk calls).
- Everything else: on DB failure append the record (already carrying its writer_id/seq and record_id) to a local spool file
  `<IRIS_HOME>/governance/audit-spool.jsonl` (append + fsync, 0600, no argument text: same closed field set), keep going, and
  surface a counter via `iris system status` and a `gap` record at the next success. A drainer replays spool lines into the DB
  idempotently by `record_id`. If the spool write also fails, the in-memory sequence still exposes the hole on the next write
  (reported as "unrecorded range, cause unknown").
- Surfacing: `audit_log` gets a `kind=gap` row (`writer_id`, first/last missing seq, cause class only); the kernel's warning
  also increments a process-level counter exposed on `/governance/audit` and `iris governance audit --gaps`.

### 4.5 Stable tier additive
- `TurnAuditRow` is a frozen dataclass; `tests/unit/test_stable_tier.py::test_the_turn_audit_row_fields_are_the_declared_ones`
  pins the exact field tuple. Add fields at the END with defaults (`record_id`, `call_id`, `parent_call_id`, `turn_id`,
  `attempt`, `writer_seq` all `None`-defaulted), update that test and `docs/reference/stable-api.md` + `names.txt` in the
  same PR (additions are not breaking, the pin test just needs the update). `AuditLog.record(...)` keeps its signature; new
  kwargs are optional and ignored when a plugin does not pass them (the store fills them). `AuditRow` (internal) can change.
- Proof bundle format v1 is a closed schema (`verify` rejects unexpected keys): do not add fields to v1; either add them in
  v2 or leave the bundle on its current columns (id, ts, hook, decision, tier).
- `public_payload` (`PUBLIC_PAYLOAD_FIELDS`, #131) is a closed allowlist: add `turn_id`, `call_id`, `parent_call_id`, `attempt`
  to it (identifiers only) so the audit view/CLI/trace can show them. Same masking test pattern as #131.

### 4.6 Migrations (existing user databases under `~/.local/share/iris`)
All additive, nullable columns, idempotent `ALTER TABLE ADD COLUMN` guarded by `PRAGMA table_info` (the checkpoints store
already does exactly this for `session_id`: precedent). Old rows keep NULL ids; replay labels them "pre-identity era" and does
not claim completeness for them (honest boundary: a `migration` marker row records the first id/ts that carries identity).
| Store | Change |
|---|---|
| audit_log | add `record_id, session_id, turn_id, call_id, parent_call_id, attempt, replay_of, resumed_from_run` (stage 3; `writer_id, writer_seq, kind` are stage 4); `session_id` is a column but is NOT backfilled from the payload (D6): readers use `COALESCE(session_id, json_extract(payload_json, '$.session_id'))`; keep payload copies for compatibility |
| side_effect_ledger | add `call_id, parent_call_id, attempt, replay_of, record_id`; new `side_effect_events` append-only table (pending/settled transitions) so the mutated row stops being the only history |
| approval_queue | add `call_id` (held attempt), `step_id`, `turn_id`; transitions already audited |
| router_decisions / session log | add `turn_id`, `record_id`, `writer_seq` (JSONL: new keys, readers ignore unknown) |
| governor_guard_audit | add `run_id, call_id, session_id` (bridge passes them) |
| cost ledger, checkpoints | add `call_id`/`turn_id` (cost); checkpoints unchanged except `resumed_from` |
| Parquet archive | add `id`, `record_id`, `writer_id`, `writer_seq` columns; compaction writes a `compaction` marker (per writer: seq ranges, row count, chunk name, sha256) before deleting hot rows |
| egress/ingress | emit `call_id`/`turn_id`/`record_id` in the line; and (with #103) a structured durable egress record |

### 4.7 Effects on neighbouring issues
- #103 egress rows: each governed outbound call is a record `kind=egress` with `parent_call_id` = the tool call that made it
  (ContextVar), `call_id` its own; the governed HTTP client is the minting site; "tool/plugin/host/method/decision" are columns
  of that row. Without #134 a row cannot be tied to the tool call that caused it.
- #114 MCP: bridge mints `call_id`, reads `current_call` for run/turn/parent (fixes constant `mcp-<server>` run and
  `tool_call_id=None`), stamps caller; server identity (name, signer, spec hash) stays on the PRE row. Acceptance bullet
  "the turn's run id" is answerable once `run_id` is inherited.
- #131 audit view: `audit_entry` adds `turn_id`, `call_id`, `parent_call_id`, `attempt`, `record_id`; model/provider stay.
  The `by` CLI column is unaffected.
- #102: shared ledger handle + collision handling become straightforward: one `SideEffectLedger` per process, `record()` returns
  `inserted`, collision raises; write-ahead for destructive MCP tools needs the bridge to mint `call_id` first (this design).
  Multiple processes still open the same file; per-writer sequence avoids a shared counter row.

## 5. Replay design

### 5.1 Command / API
`iris audit replay --session S [--turn T | --run R] [--json]` and `GET /governance/replay?session=S` (read-only, owner auth,
identifiers and decisions only: same closed field set as `audit_view`, never argument/result text). Implementation lives in
`kernel/governance/audit/replay.py` reading stores through their own readers (hot SQLite + Parquet cold + ledger + approvals
+ checkpoints + session log + router rows).
Output: one ordered timeline of records `(ts, writer_id, writer_seq, record_id, kind, call_id, parent_call_id, attempt, ...)`
grouped into a tree turn -> run -> step -> call (children by `parent_call_id`), plus a `gaps[]` list.

### 5.2 Gap classes it reports
| Gap | Rule | Proves |
|---|---|---|
| sequence hole | per `writer_id`, `writer_seq` not contiguous between first and last seen (minus declared compaction/spool ranges) | a record was attempted and is not stored |
| writer tail | no `writer.close` and no later record from that writer | unknown tail; reported "open writer", never "complete" |
| open call | `call_id` with PRE record and no settle record (`post_tool_use`, `post_llm_call`, ledger `completed|error`), and not the latest call of a live writer | crash/kill mid-call; ledger row `pending` agrees |
| missing parent | `parent_call_id` / `replay_of` / `resumed_from_run` / `approval.call_id` with no matching record | lost parent or cross-store loss |
| witness mismatch | a record in an independent store with no counterpart: session-log `tool.invoke.start` without PRE row (and vice versa); approval row without audit transition; `log_egress` line without egress record; checkpoint without resume | a write that never happened but was witnessed elsewhere |
| orphan settle / duplicate | settle without PRE; same `record_id` twice with different content; `idempotency_key` collisions without `replay_of` | aliasing |
| identity era | rows with NULL ids (pre-migration) | replay is best-effort for them, flagged |

### 5.3 What it can and cannot prove
- CAN prove: completeness of what each writer tried to write (sequence), that every started call settled or is flagged, that
  every parent link resolves, that independent witnesses agree.
- CANNOT prove: a record the code path never attempted to write (no sequence slot is consumed). "Never written" becomes
  detectable only by construction: (1) the write-ahead rule (the PRE record is written before the effect, fail closed for
  effectful calls, so an effect without a PRE record means the effect did not go through the runner: detectable via the
  independent witnesses: ledger/session-log/egress/approval/provider-side logs); (2) every boundary crossing has a second store
  (the proof bundle's "observations" are the existing precedent); (3) lint/conformance tests that every code path that executes a
  tool or model call goes through the runner (`tests/security/test_no_bypass.py` style). Known unwitnessed paths today: general-lane
  builtin tools (no hooks at all), MCP bridge (no ledger), `kernel=None` (governance off). It also cannot prove integrity against
  an attacker who can delete rows AND rewrite sequence numbers; that needs a hash chain (`prev_digest` HMAC under the audit key per
  writer): optional stage 6, owner decision D5. Compaction is covered only if `compaction` markers exist (today they do not).

### 5.4 Prototype on a real temp session (RUN against current code)
Setup [RUN]: `testing.harness` with a scripted model and one in-process plugin: `lookup_doc` (read; internally calls
`api.tools.call("word_count")`) and `shred_doc` (`effect: destructive`). Turn 1 "Shred the memo" -> loop step 0 `lookup_doc`
(+ nested `word_count`), step 1 `shred_doc` held for approval; then `respond_to_approval(approve=True)` resumes the run and
executes it. Dumps are in `proto134/dump-output.txt`, `proto134/join_matrix-output.txt` (identifiers only). Second scenario p4:
a plugin tool calls `api.tools.call("shred_doc")` (code-caller approval path), approved later (`proto134/dump4-output.txt`).

Rows (identifiers only, ids shortened):
| Store | Row |
|---|---|
| audit_log id 8-17, 32-35 | PRE/POST of lookup_doc: run fb02, step 0, caller `model:system`, session S1 |
| audit_log id 18-31 | PRE/POST of word_count: run 3b34 (NEW), step None, caller `plugin:shredder`, session S1 |
| audit_log id 39-47 | PRE of shred_doc attempt 1: run fb02 step 1; id 46 approval_queue `require_approval` appr 1e90; id 47 `destructive_approval` require_approval appr 1e90 |
| audit_log id 55 | approval_queue `approved` run fb02 appr 1e90 |
| audit_log id 59-72 | PRE/POST of shred_doc attempt 2: SAME run fb02, SAME step 1, appr 1e90 in id 66; ids 68/71 carry `side_effect_id` fb02:1:d081 |
| side_effect_ledger | one row `fb02..:1:d081ba6a8d9b` completed (key = attempt 2's call id) |
| approval_queue | 1e90, run fb02, checkpoint_id fb02, session S1, status approved; no step, no call id |
| checkpoints | (fb02, step 1, require_approval, S1) |
| session log | turn 599f: tool.invoke 2d4f lookup_doc, a8a3 word_count (nested); turn 36d8: tool.invoke d081 shred_doc |
| router_decisions | id 1, id 2: session S1 only |

Where the joins hold and break today (from `join_matrix.py`):
| Join | Holds? |
|---|---|
| audit_log <-> session via payload `session_id` | holds: 68/82 rows stamped; the other 14 are curator/response rows whose run_id IS the session id |
| audit PRE <-> approval row | holds, via `approval_id` in audit payloads (the one explicit link) |
| audit PRE <-> session-log tool event | BREAKS: no `tool_call_id` in audit; lookup_doc and word_count match only by (tool, session) |
| the 2 attempts of shred_doc | BREAKS: same (run, step, tool); distinguishable only by audit `id` order and `approval_id`; session-log call id d081 matches 2 candidate PRE attempts; ledger joins attempt 2 only via the rows that carry `side_effect_id` |
| held attempt's call id | BREAKS: recorded nowhere |
| word_count -> lookup_doc (child -> parent) | BREAKS: no parent column; different run, step None; inferable only from time containment |
| turn 1 vs turn 2 (resume) | BREAKS in audit: `turn_id` is not in audit_log; the resumed run spans 2 turns; session log has the turn ids |
| router_decisions <-> turn | BREAKS: session only; 2 rows, no turn id |
| approved code-caller call (p4) | audit rows join by run e30e + approval a372; session log has NO events for the child or the approved call (no session_scope) |
| completeness | BREAKS: `audit_gaps()` count-based; delete one of two `pre_llm_call` rows and nothing is reported [RUN p3]; fail-open write leaves no trace [RUN p3] |

### 5.5 What the replay would output for the same session (design, not implemented)
turn T1 -> run R (loop) -> step 0 -> call C1 `lookup_doc` -> child call C2 `word_count` (parent C1); step 1 -> call C3 attempt 1
`shred_doc` HELD (approval A) ; turn T2 (resumed_from_run R) -> step 1 -> call C4 attempt 2, parent C3, `replay_of` C3 (approved
re-execution) -> ledger settle completed. Gaps: none; if audit id 47 were deleted: "missing record: writer W seq n, between
seq n-1 and n+1"; if the process died after id 68: "open call C4: PRE + ledger pending, no settle".

## 6. Staged implementation plan (smallest safe first)

| Stage | Content | Owner decisions | Tests |
|---|---|---|---|
| 0 (docs + test only) | This design in `docs/architecture`; characterization tests that pin today's join breaks as strict xfails (call id absent in audit, parent missing, held id lost, fail-open silent) | D1-D6 below | the xfails themselves |
| 1 | Runner mints `call_id` once, stamps it on every hook row (metadata -> `_AUDITED_PAYLOAD_KEYS`, payload only, no schema change) and on session-log events and the approval row (new nullable col). Kill `hex[:12]`; capability/MCP get a minted id. `SideEffectLedger.record` returns `inserted` and raises on conflict (closes #102 item 2). Add `call_id` to `PUBLIC_PAYLOAD_FIELDS` | key format (D1) | collision raises; no caller can supply it (`ToolCall` has no field); both `chat` and `chat_stream`; approved resume; ToolService; mcp_serve; capability; masking test |
| 2 | Relationship columns + ContextVar `current_call`: `turn_id` in kernel stamp; `parent_call_id`, `attempt`, `replay_of`, `resumed_from_run`; `ToolService`, capability, MCP inherit run/turn/parent; held->approved link; `TurnAuditRow` additive fields | stable-tier addition (D4) | nested call has parent; approved attempt 2 links to attempt 1; resume links runs; pin test updated |
| 3 | Schema migration of audit_log/ledger/approvals/router/governor columns + `record_id` minted in stores; `session_id` promoted to a column; `record_many` per fire | migration policy for old rows (D6) | upgrade a DB created by the current release in a temp home; idempotent migration run twice; old rows read fine |
| 4 | Completeness: `writer_id`/`writer_seq`, `writer.start/close`, `gap` records, spool, fail-closed tiers, compaction markers + archive `id` | D2, D3 | injected write failure produces a gap record/hole and spool line; kill mid-call yields open call; compaction leaves reconcilable markers |
| 5 | Replay command/API + `audit_gaps()` rebuilt on it; the acceptance test from the issue (approved call, capability call from a tool, bridged MCP call, resumed run, deleted record reported) | none | as listed in the issue; run on `chat` and `chat_stream` |
| 6 (optional) | per-writer HMAC hash chain; proof bundle v2; independent witnesses for egress (#103) | D5 | tamper test |

Owner decisions: D1 id format (ULID vs uuid7, display length); D2 per-writer sequence (recommended) vs one per-session
counter; D3 fail-closed set vs spool for the rest (which effects count as effectful); D4 additive `TurnAuditRow`/PUBLIC fields
accepted under the stable-tier pin; D5 tamper evidence in scope or not; D6 old rows: label "pre-identity era" (recommended) vs
backfill; D7 should general-lane builtin tools and `kernel=None` be brought under the runner or declared out of scope.

Risks: `kernel/governance/` is CODEOWNER-reviewed on every stage (1, 3, 4 touch it); migrations run on users' live
`~/.local/share/iris/*.db` shared by several processes (ALTER under WAL, concurrent first-open: guard with
`PRAGMA table_info` + tolerate "duplicate column" race, precedent `checkpoints`); a writer counter is process-local and must
survive forks (`writer_id` re-minted post-fork) and thread safety (a lock); stable-tier names pinned by tests; audit payload
whitelist must stay free of argument text; extra columns per row (about 14 rows per call) grow the DB, offset by `record_many`;
Parquet schema change needs a reader that handles old chunks (the DuckDB view must union old and new schemas); the proof bundle
v1 closed schema.
What to test: each stage's row above, plus a two-process writer test (API + CLI writing one session), a crash test (hard exit
between PRE and POST in a subprocess), and the mutation checks used for #131.

### 6.1 Stage 2 as built (parent, attempt, turn, resumed run)

Payload-only: no schema change and no migration (the audit payload is schema-free; stage 3 owns the columns).
- `kernel/governance/call_context.py`: `call_scope(call_id)` makes a call current while its code runs (the runner sets it
  around `tool.call` and around a capability provider, and around each step of a streamed capability so a governed call
  started inside a stream's body still sees its parent); `register_call` records `(parent_call_id, attempt, replay_of)`
  when the id is minted; `mark_run_resumed(run_id)` is called from `AgenticCore._settle_pending_approval`, the one
  re-entry point of both loops. The kernel reads them by the row's own `call_id`, never from a payload or an argument.
- Fields on the audit payload (and the public view): `turn_id` (from the turn scope, absent outside a turn),
  `parent_call_id` (containment only; an egress request keeps its own), `attempt` (1, or 2 for the approved re-execution of
  a held call), `replay_of` (the held attempt), `resumed_from_run` (the run id, as a marker on the rows the re-entered run
  writes: the run id survives a halt, so it is not a new id). `TurnAuditRow` gains seven optional fields at its end:
  `call_id`, `held_call_id`, `turn_id`, `parent_call_id`, `attempt`, `replay_of`, `resumed_from_run`.
- A nested call keeps its own `run_id` (D4: no existing field changes); the parent link is the new field.
- Context limits (stated, tested): a `ContextVar` follows `await`, `asyncio.run` and `asyncio.to_thread`, not a bare
  `threading.Thread` or `loop.run_in_executor`; a governed call started there has no parent (the egress scope has the same
  limit). The lineage registry is bounded (8192 calls): an evicted lineage leaves the fields off a row, nothing is invented.
- Not stamped: retries (`iris run resume` re-exec, the loop's repeated-action guard) until stage 4; calls that never reach
  `GovernedToolRunner` get no call id, no parent and no turn link: since #155 that is only a run with governance switched
  off (`kernel=None`, the operator's explicit opt-out, with one startup warning when the legacy general lane then serves).
  The general lane's own tools now go through the runner like the loop's (D7, first half, closed).

### 6.2 Stage 3 as built (identity columns, migration)

- Shared helper `foundation/persistence/sqlite.add_columns_if_missing`: its OWN connection, `BEGIN IMMEDIATE`, then
  `PRAGMA table_info` and `ALTER TABLE ... ADD COLUMN` for what is missing (nullable, no default, no row rewritten),
  index creation, an `on_added` callback in the same transaction, `COMMIT`. Why not inside a store's transaction: a
  read followed by an `ALTER` in a deferred transaction fails with `database is locked` when another process commits in
  between, and the 5 s busy timeout does not help (`SQLITE_BUSY_SNAPSHOT`). Measured with 8 real interpreters opening
  one release-shaped file at the same instant: the deferred variant failed 37 of 40 rounds, `BEGIN IMMEDIATE` 0 of 40
  (`tests/unit/iris_harness/kernel/test_governance/test_identity_migration.py` runs the real race per store and shows
  the naive variant failing).
- `audit_log` gains `record_id` (minted by `AuditLog.record`, ULID, unique partial index), `session_id`, `turn_id`,
  `call_id`, `parent_call_id`, `attempt`, `replay_of`, `resumed_from_run`, filled from the same payload keys the kernel
  stamps (one source; payload copies stay). `AuditLog.record_many` writes several rows on one connection with per-row
  failure semantics (a failed row is `None` in the result and the others are kept); the kernel does not batch a firing
  yet (atomic-per-firing waits for D3, stage 4, because `PRE_EGRESS` reads each write's result).
- Pre-identity era (D6): a row with no `record_id` was written before identity (an older process still writing to a
  migrated file produces such rows too). The process that adds `record_id` also writes one `audit_meta` row
  (`key='identity'`, `{schema, first_identity_row_id, migrated_at}`) in the same transaction; a fresh database writes none.
  Nothing is backfilled or rewritten.
- `side_effect_ledger` gains `call_id, parent_call_id, attempt, replay_of, record_id` (the key stays
  `<run>:<step>:<call_id>`, so `list_by_run`/`pending`/`iris run resume` are unchanged) and a new append-only
  `side_effect_events` table (one row per pending/settled transition). `approval_queue` gains `step_id, turn_id`;
  `router_decisions` gains `record_id, turn_id`; `governor_guard_audit` gains `run_id, call_id, session_id, record_id`
  (the MCP bridge passes the run and the call id it minted). Router and governor migrations that fail are logged at error
  level and the old schema keeps being written.
- Not in stage 3: cost ledger and checkpoints (#193), the Parquet archive (stage 4: a compacted row keeps its identity
  in `payload_json`, but loses `record_id` and the columns until the archive schema grows them), the session log.

## 7. Verified vs not verified
Verified by running (current main fbfb7e8, scripted model, temp IRIS_HOME, throwaway vault key): the two scenarios and their
dumps; ids contiguous; `tool_call_id` absent from audit payloads; held-attempt id absent; nested call fresh run; ledger collision
silently dropped; audit write failure silently tolerated with no hole; `audit_gaps()` misses a deleted row; no session-log events
for code-caller approved calls; the sequence-counter microbenchmark (single laptop, sqlite per-row connection, 1 and 4 processes;
the "seq" variant writes fewer columns, so compare trends, not absolutes).
Read only, NOT run: MCP bridge paths (`mcp-<server>` constant run, `tool_call_id=None`, only caller is the HTTP route);
`mcp_serve`; capability call minting (`execute_call` family); general-lane builtin tools bypassing hooks and whether that lane is
reachable by default; heartbeats/routines session scope; Telegram/gateway end to end; `iris run resume` CLI; `iris audit compact`
and the archive schema loss of `id` (not executed); cost limiter failure path; governor guard failure behaviour; multi-process
migration races; real macOS Keychain never touched. Parquet/DuckDB reader union across schemas untested. The proposed design
(sequence, spool, replay) is not implemented beyond the benchmark; the replay output in 5.5 is a design sketch.
