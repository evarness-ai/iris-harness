# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- The sandbox egress proxy (`egress_proxy.py`) now listens on `127.0.0.1` by default, not `0.0.0.0`. The sidecar the harness starts is unaffected: it is started with `IRIS_EGRESS_BIND=0.0.0.0` because the sandbox reaches it from a sibling container. Anyone who runs `egress_proxy.py` by hand or from their own compose file and relied on it listening on every interface must set `IRIS_EGRESS_BIND=0.0.0.0` (or a specific address). A value that is not an IP address is refused, and a non-loopback bind logs one warning.

### Added

- Audit compaction now keeps what the ledger knows and accounts for what it moved (issue #134,
  stage 4b). The Parquet archive keeps the identity and sequence columns (`id`, `record_id`,
  `session_id`, `turn_id`, `call_id`, `parent_call_id`, `attempt`, `replay_of`,
  `resumed_from_run`, `writer_id`, `writer_seq`, `kind`) and archives the ledger's own
  `writer` and `gap` rows. Each run writes and fsyncs its chunks, then ONE transaction inserts a
  `compaction` marker (chunk files, SHA-256, row counts, id range, each writer's lowest and
  highest sequence number and count) and deletes exactly the archived rows by id. A crash between
  the two leaves chunks no marker names: the next run adopts them (their rows are still in the
  ledger) or moves them to `<archive>/.orphans/` (never deleted); a chunk written by an older
  release, or any file it does not recognise, is never touched. `audit_archive` de-duplicates a
  row found in both tiers or in two chunks by `record_id`, hides the ledger's own rows by default
  (`iris audit query|export --include-store-rows` shows them), and reads old and new chunks
  together. `iris audit verify` checks every marker against its chunks. Compactions are
  serialised by a lock in the archive directory. Known limit: a spool line that survives a crash
  between being applied and being removed can re-insert a row archived in the meantime; the view
  collapses the copy.
- A session can be replayed from the stored records, and what is missing is named (issue #134,
  stage 5). `iris audit replay --session S` and `GET /governance/replay?session=S` rebuild one
  session's ordered timeline from the audit ledger (hot and archive), the approval queue, the
  side-effect ledger and the session log, with its calls hung on their parents, and list the gaps
  found: a sequence number no store holds, an archive chunk that no longer matches its marker, a
  call that started and never settled in a process that is gone, a parent or held call that does not
  exist, a witness with no counterpart, a settle with no start, a record stored twice with
  different content. Rows from before records had ids, a writer that never closed and a call still
  running are notes, not gaps. The command exits 1 on a gap (`--no-fail` for scripts) and the
  endpoint is capped (rows and seconds) and says when it cut the read short. `Harness.audit_gaps()`
  still returns a list of strings and is empty for a healthy turn; it now also lists what the
  replay proves lost. `writer.start` rows now carry the process's start time so a recycled pid is
  not mistaken for a running writer (older rows fall back to the pid alone). The egress witness is
  not part of this stage.

### Security

- An unexpected failure in `/chat/stream` or a web slash command now tells the client the exception class and to see the server log, not the exception text. The full traceback is logged. Curated errors (validation, removal, allow-list) are unchanged.
- Request-derived values in log lines are escaped (new stable `iris_harness.sdk.logging.log_safe`); the shared ingress line escapes method, path, source and session id.
- Session ids that contain a path separator or leave the session log directory are refused.
- Copilot token fragments are no longer logged.

## [0.1.0] - 2026-10-07

First public release of the IRIS harness.

### Added

- The audit ledger can show what is missing (issue #134, stage 4a). Every process that writes to
  `audit.db` is a writer with its own id and numbers its rows before writing them
  (`writer_id`, `writer_seq`), so a write that fails leaves a hole in the sequence. A row the
  database refuses is kept in a local spool (`audit-spool.jsonl`, beside the database, `0600`) and
  put into the ledger on the next successful write or restart; the next row that lands also records a
  `gap` row (the missing numbers and the failure class, never a message). A call that guards an
  effect (a write or destructive tool, a cloud-tier model call, an outbound request) is refused
  when neither the database nor the spool could keep its row. `writer.start` / `writer.close` rows
  bound each writer's life; a writer with no close ended without saying so. A spool line can add
  a missing row but cannot rewrite one that exists, and malformed lines are set aside and counted.
  The lost and waiting counts show in System Health (red when a row was kept nowhere),
  `iris system status`, `GET /governance/state` and `GET /governance/audit` (`write_health`), and
  the Governance screen. A database an earlier release created gets the columns when it is opened
  (several processes opening it at once are safe); old rows are never backfilled. The archive
  compaction leaves the ledger's own rows in place until the marker work lands (stage 4b).

- A stored conversation summary records whether it absorbed third-party text (issue #145,
  summaries). `conversation_summaries.has_external` is 1 once any external-origin turn was folded
  into the summary (and stays 1: a paraphrase cannot be un-marked), 0 when every folded turn is known
  not to be external, and NULL (unknown) for every summary written before the column. A flagged
  summary comes back into a prompt inside the untrusted-content envelope, as well as scanned, from the
  session summary, `memory_search` over sessions and `recall_conversation`'s summary fallback;
  unflagged and unknown summaries are scanned as before. A database an earlier release created gets
  the column when it is opened (several processes opening it at once are safe).

- A stored assistant turn records where it came from (issue #145, steps 2-3). The loop notes
  whether the run read third-party text (a `content: external` tool's result) before answering,
  and `conversations.turn_origin` stores `external` or `internal` on the assistant row (NULL
  is unknown: every row written before this change, never backfilled, and any answer the loop
  did not produce). A turn marked `external` comes back into a prompt inside the
  untrusted-content envelope, as well as scanned, from the conversation window, the history
  reload after a restart, the related turns from other sessions, `recall_conversation`,
  `memory_search` over sessions and the intent router's recent-turns block. Internal and
  unknown turns are scanned as before; the owner's own turns come back verbatim. A database an
  earlier release created gets the column when it is opened (several processes opening it at once
  are safe).

- `iris_harness.sdk.persistence.ensure_columns(db_path, table, {column: declaration}, *, indexes=(),
  on_added=None)` adds the columns an older table lacks, safely when several processes open it at
  once: a plugin that reads `PRAGMA table_info` and then runs `ALTER TABLE` itself loses that race
  with `duplicate column name` (issue #201). It looks first, then takes the write lock on a
  connection of its own (`BEGIN IMMEDIATE`) and decides again inside it; `on_added` runs in that
  transaction, only in the process that added a column, for a one-time backfill. It is the harness's
  own function, re-exported as a new stable name.

- The audit stores carry an identity in real columns (issue #134, stage 3). `audit_log` gains
  `record_id` (minted by the store, unique), `session_id`, `turn_id`, `call_id`,
  `parent_call_id`, `attempt`, `replay_of` and `resumed_from_run`; the side-effect ledger gains the
  call's identity and an append-only `side_effect_events` table; the approval queue gains
  `step_id` and `turn_id`; the router and governor audit tables gain their ids. A database an
  earlier release created is migrated when it is opened: columns are added, nothing is rewritten
  or backfilled, a row without a `record_id` is one written before identity (an `audit_meta` row
  records the boundary), and several processes opening the file at once are safe.

- The plugin raw-network lint also names `paramiko`, `boto3`, `openai`, `redis`, `pymongo` and
  `aiosmtplib` (issue #175, item 4). `NETWORK_MODULES` grows by six entries; no name is added or
  removed, so the stable-tier snapshot (`tests/fixtures/stable_tier/names.txt`) is unchanged. A
  plugin that imports one of them now fails `check_network_imports` the way one that imports
  `httpx` does; route the call through `api.http`, or declare why it cannot be. The scan finds
  one first-party file, `research/cache.py` (the opt-in Redis cache, a plain TCP connection to
  the operator's own server), pinned as listed debt with its reason in the research manifest and
  `plugin-egress.md`.

- An audit row says where its call sits (issue #134, stage 2): `parent_call_id` (the governed
  call it ran inside: a tool a tool called, a capability a tool called, a bridged MCP call),
  `attempt` (2 on the approved re-execution of a held call, with `replay_of` naming the held
  attempt), `turn_id` (the chat turn the row was written in) and `resumed_from_run` (the run
  id, on the rows a halted run writes after it is re-entered). The harness writes them, never a
  caller, an argument or a hook. `TurnAuditRow` gains `call_id`, `held_call_id`, `turn_id`,
  `parent_call_id`, `attempt`, `replay_of` and `resumed_from_run`, all optional and `None` on a
  row they do not describe or on one written before calls carried an identity. No existing
  field, `run_id` included, changes, and no database migration is involved.

- An operator can declare what an MCP tool does: `governance.tools: {<name>: {effect:
  read|write|destructive}}` under a server in `mcp-servers.yaml` (issue #102). A destructive
  tool is refused before the server is reached unless `invoke_external_tool` is given
  `approved_by`, an approved queue row that pinned exactly that call, and it leaves a pending
  write-ahead ledger row before it runs. A tool not listed behaves as before.

- The governed HTTP client's response cap is configurable per plugin (issue #175). The manifest
  key `egress.max_response_bytes` sets the most decoded body bytes a response may have: default
  10 MiB, lowering is always allowed, and the harness ceiling is 64 MiB (a larger value fails
  manifest validation, so the plugin does not mount). A cap other than the default is recorded as
  `cap` on every `pre_egress` row and shown by `iris plugins`. Adding a manifest key is not a
  breaking change to the stable surface; the key sits under `egress`, which is already in it.

- A capability method can declare `sends_to` (`MethodSpec.sends_to`, as a tool does), and
  `CapabilityCall` carries it: every governed call of the method stamps where its arguments go
  at `PRE_TOOL_USE`, so the owner-PII guards read them. `weather.forecast` declares
  `external_service` for its `location` (issue #100).

- Health shows the model guard when it is not doing what the owner may believe (issue #136):
  a yellow `governance` row when `IRIS_GOVERNANCE_PROMPT_GUARD` is on but its classifier cannot
  run (packages missing, weights not in the local cache, or a load already failed), and one when
  it is off while tools that return third-party text are mounted, naming how many and which
  plugins. Silent when the posture is fine. The probe never loads the model, and the always-on
  floor is named as still running. `GET /governance/state` gains a `model_guard` object from the
  same probe. New `CheckKind.GOVERNANCE`; the web Health screen gains Plugins and Governance
  groups (plugin rows had no group there).

- `ToolResult` (what `api.tools.call` returns) gains `external`, `source` and `tool`, set by the
  tool service from the tool's declared `content`, and `for_model()`: the text inside the
  untrusted-content envelope when the tool is external, unchanged when it is not (issue #147).
  `result.text` stays the owner-facing form. The conformance suite adds a `content` check: for
  every external tool with an example call, the result must say it is external and
  `for_model()` must wrap it. `wrap_external_content` and `for_model()` share one
  implementation (`wrap_scanned`).

- The owner is told when text they are reading was redacted by the external-content floor, and
  can allow known false positives (issue #139). A shared turn stage appends one plain sentence
  once to an answer that shows the redaction marker (generated and deterministic, `chat` and
  `chat_stream`, before the turn is recorded); a brief and a directly answered skill add it
  where they render. `iris governance redactions` and `GET /governance/redactions` list what
  the floor cut by pattern id, count, tool and source, never the text. The allow-list
  (`config/governance/external-content.yaml`, `iris governance allow add|remove|list`) names one
  floor pattern id and a namespaced source (`plugin:`, `skill:`, `mcp:` or `core:`), optionally
  narrowed to a tool: no wildcard, never a hidden-character pattern,
  optional `until`, ignored when expired, and a file with a refused entry allows nothing. Each
  edit is a ledger row, each use is recorded on the floor's row, and `GET /governance/state`
  gains `external_content_allow`. `ToolResult.for_model()` honours the allow-list for the scope the
  tool service stamped, so an allowed source's text is not re-redacted there. A plugin cannot edit it, and the plugin name `system` (the stamp
  of the core's own tools) is reserved for the builtin: any other plugin of that name fails to
  load.

- A model-called memory write after the run read outside text needs the owner's approval
  (issue #149). A run is tainted once a step ran a tool declared `content: external`; in a
  tainted run, `memory_correct`, `memory_forget` and `memory_restore` (the list is
  `config/governance/taint-policy.yaml`, `approval_when_tainted`) are held on the existing
  approval card, which says why. The loop derives the taint from the run's recorded steps, so a
  run resumed after an approval keeps it; an ordinary correction in a run that read nothing
  from outside is unchanged, and the next message starts clean. New defaulted
  `ToolCall.tainted`. A missing or malformed file falls back to the three memory writes. Taint
  comes only from a tool step declared external in the run: external text that reaches the prompt
  with no tool step (recalled memory, injected retrieval context, stored text from earlier
  turns) does not taint it.

- `search_docs`: the core can search its own shipped documentation (architecture, concepts,
  guides, reference, usage-guides) by keyword at section level, as an internal read tool of
  the `system` plugin. The corpus is an allow-list in `config/docs_search.yaml`; it cannot
  reach the identity files, the vault, `.env` files, the owner's data or any store, and a
  document that classifies `secret` is dropped. An install without the docs says so.

- Plugin manifests can declare `egress:` (issue #103): the hosts a plugin's code may
  contact through the SDK's governed HTTP client (#103b; see below). The host
  grammar refuses IP literals in any spelling, `localhost`, newlines, repeated dots,
  suffix-only wildcards (`*.com`) and over-long hosts; `iris plugins show` and
  `--dump-config` print `none declared (raw network calls by this plugin are not governed)` for a plugin
  without one.

- Governed outbound HTTP for plugins (issue #103, part 2): `api.http` (and
  `iris_harness.sdk.http.current_http()` for a declarative plugin) sends a request only to a
  host the plugin's manifest `egress:` declares, fires the new `PRE_EGRESS` / `POST_EGRESS`
  hook points and writes a ledger row per request (host, port, method, plugin, tool, run,
  status, bytes, duration; never a path, query, header or body). A host that is not declared,
  an IP literal, a missing kernel or a missing `plugin_egress` hook is refused with
  `EgressDenied`. This governs calls made through that client only: it does not stop a plugin
  that opens its own socket, and no shipped plugin uses the client yet. Also new:
  `testing.fake_http`, `testing.check_network_imports` and a conformance check `egress`.
  Each request has one total time budget (default 10 s, at most 60) and a 10 MiB decoded-body
  cap, ignores proxy and netrc environment variables, refuses `Host` and `Proxy-*` headers and
  names such as `localhost`, `*.local` and `*.internal` (even under `open_web`), connects only
  to a checked public address (the name is resolved once), records a malformed URL as a
  denial, and is not sent when its `pre_egress` ledger row cannot be written. A name an
  operator or attacker points at an internal address is refused; see
  `docs/architecture/plugin-egress.md` for the limits. A request is made only inside a
  governed tool or capability call and acts only for the plugin whose tool is running (a
  tool of plugin `evil` using `GovernedHttp("weather")` is denied and recorded against
  `evil`); one made outside any call, such as from a thread the tool started, is denied.
  `EgressDenied` is a `RuntimeError`, not an `OSError` (a stable name whose base changed
  before release). `check_network_imports` also reports attribute use of an imported
  package (`urllib.request.urlopen`), `asyncio.open_connection` / `start_server`, `httpcore`,
  `h11` and `http.server`, and an unparsable file as a finding.
  The response body is decoded by the client with a bounded decompressor, never by httpx:
  it asks for `identity`, accepts at most one `gzip` or `deflate` layer, and refuses any other
  encoding (layered, `zstd`, `br`) before reading it, so a compression bomb cannot exceed the
  10 MiB cap in memory. A `Host`, `Proxy-*`, `Connection`, `Upgrade`, `TE`, `Transfer-Encoding`
  or `Content-Length` request header is refused in any spelling (dict, pairs, bytes keys).

### Changed

- **Default flipped (issue #180): an MCP tool nobody declared is now treated as
  `destructive`.** Before, a tool with no `governance.tools` entry ran with no effect
  stamp, no write-ahead row and no approval. Now every such call is refused before the
  server is reached unless `approved_by` names an approved queue row that pinned exactly
  that call, and it leaves a write-ahead row. **What breaks:** every current MCP server
  whose tools are not listed under `governance.tools` will need an approval on each call
  (over HTTP it is refused with a 403, since that surface grants no approval). **The
  one-line fix:** for a server that only reads, add `governance: {undeclared_tools: read}`
  to its entry in `mcp-servers.yaml`; or declare each tool under `governance.tools`. After
  `tools/list` the bridge logs a warning naming the undeclared tools. A server's own
  `readOnlyHint` never relaxes the default.
- **Default flipped (issue #104): text a tool, capability or MCP server declares
  `content: external` is now marked and scanned on every install.** The new
  external-content floor needs no model, no weights and no network. An external result
  reaches the model inside an `<external_content source=... trust="untrusted">` envelope
  (plugin or core code calling through `api.tools` gets the redaction only, no envelope),
  and a short list of deterministic patterns (instruction overrides, chat-template and
  tool-call syntax, exfiltration instructions, hidden characters) is redacted with a ledger
  row naming the pattern ids, the tool and the source, never the text. It is on by default
  (`IRIS_GOVERNANCE_EXTERNAL_CONTENT_FLOOR`, a plain boolean: unset, blank and any
  unrecognised value leave it on (an unrecognised one logs a warning), only
  `0`/`false`/`no`/`off` turns it off, with a warning; with the floor off, text a third party wrote reaches the model unmarked and
  unscanned) and acts at `POST_TOOL_USE`, so the agent loop, `api.tools`, `iris mcp serve`,
  the MCP bridge and capability results all get it. A text that quotes an attack phrase is
  redacted too: see "The external-content floor" in `docs/concepts/governance.md` for the
  pattern list and its limits. The model guard (`IRIS_GOVERNANCE_PROMPT_GUARD`) is
  unchanged: opt-in, shadow-first, fail-open.
- **`code_exec` output is treated as external content (issue #140).** The tool stays
  `effect: read`, behind its Docker mount and sandbox limits, and its egress allowlist is
  unchanged; what changed is that what a script printed is no longer assumed clean (a
  script can print a page it fetched). The tool is declared `content: external`, so the
  governed loop marks and scans its result; and the paths that do not run through the
  loop's runner apply the same tripwire (`iris_harness.sdk.content.redact_external_content`,
  new): the intent route's answer and trace, the nested planner's view of `run_shell`
  stdout and stderr (inside the envelope), the session log's `tool_run` record, and a
  code_exec lesson before it is stored and again before it is re-injected into a planner
  prompt. The streamed answer is scanned with a sliding overlap (the last two lines are held
  back and rescanned with each new line, a long line is force-scanned at 64 KB), so a phrase
  split across up to two line breaks is caught; the planner's progress text, the command in
  the activity hint, trace and log, and artifact file names (including the "Artifacts:"
  block on every path) are scanned too. These calls honour
  `IRIS_GOVERNANCE_EXTERNAL_CONTENT_FLOOR` (off: verbatim) and a run writes at most five
  counts-only ledger rows however many lines match (later matches are redacted and logged
  once). A redaction cut by the 64 KB forced flush also drops the rest of its sentence when
  that arrives, instead of emitting it raw. Known limit: the session log's `llm_call` record of
  the planner's raw prompt and reply is a log, not a prompt, and is not scanned.
- The retrieved-content guard's `guard unavailable` ledger row now records the tool, how
  many segments went unscanned, and the classifier's backend and detail. Docs and manifest
  comments that said external content is always scanned now say what is on by default.
- **Behavior change:** `config/governance/threat-detection.yaml` no longer has a `fail_mode` key. It was
  parsed and read by nothing, so `closed` promised what never happened; a guard that
  cannot run still lets the text through and writes a `guard unavailable` row. A config
  override that still sets `fail_mode` now stops startup: when a guard that reads the file is
  requested (`IRIS_GOVERNANCE_PROMPT_GUARD`, `IRIS_GOVERNANCE_INPUT_SAFETY` or
  `IRIS_CURATOR_OUTPUT_SAFETY`) the build raises an error that names the file and the key and
  says to delete the line, instead of running on with the guards silently off (there is no
  deprecation shim). Other config errors still turn the guards off with a warning, as before.
- New stable name: `iris_harness.sdk.content.wrap_external_content(text, *, source, tool=None)`
  applies the floor's tripwire and envelope (the kernel's own implementation) to external text
  that plugin code puts into a prompt of its own. Idempotent and offline.
  `iris_harness.sdk.content.redact_external_content(text)` is the same tripwire without the
  envelope, for text a plugin shows the owner or logs rather than hands to a model. It goes
  through the kernel's `redact_text`, so it honours `IRIS_GOVERNANCE_EXTERNAL_CONTENT_FLOOR`
  (off: the text comes back unchanged) and a matching call writes a counts-only ledger row,
  for the first five matches per scope (a run for a plugin that opens one, else a session);
  later matches are still redacted and only counted in one warning.
- All nine email skill tools (`email-triage`: `email_inbox_summary`, `email_focus`,
  `email_needs_reply`, `email_judged_yesterday`, `classify_email_by_id`, `run_email_triage`;
  `gmail-inbox`: `fetch_new_emails`; `email-followup`: `detect_followups`,
  `list_open_followups`) declare `content: external`. Everything they return is derived from
  the mailbox pipeline, so they are external as a class, not because each returns email text:
  six of them carry only counts, ids or status today, so the envelope is conservative there,
  and a later change that adds free text to one cannot silently skip the floor. Code callers
  get no envelope. `rag/docs-search` is undecided and unchanged.
- A skill package's tool can declare `content: internal | external` in its manifest
  (`web-fetch`'s `fetch_web_content` is external), served and in the loop alike.

- Destructive tools and pinned writes now get a durable side-effect ledger row before
  they run (issue #73). This needs no flag: with `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER`
  unset the ledger covers that high-risk class only, and plain writes and reads are
  unchanged (no row, no file). Behavior change for deployments that set
  `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=0`: a destructive tool or pinned write is now denied
  rather than run with no durable record. Remove the setting, or set it to `1`, to run
  them again.

- Behavior change: `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=1` (or `true`/`yes`/`on`) is now
  the same as leaving it unset: the high-risk class only. It used to also record every
  non-read call, so an explicit `true` silently widened audit scope compared with the
  default. That scope is now its own boolean, `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL`
  (default off). If you set `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=1` expecting every
  non-read call to be recorded, also set `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_ALL=1`.
  `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=0` is unchanged (no ledger; destructive tools and
  pinned writes are denied), and `..._ALL` set while the ledger is off logs a warning that
  it has no effect. An explicit `IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER=1` (or `true`/`yes`/`on`)
  without `..._ALL` logs one warning at startup saying it covers high-risk calls only;
  leaving the setting unset logs nothing.

### Added

- **Stored assistant text and summaries are scanned when they re-enter a prompt (issue
  #145, step one).** The external-content floor screens a tool result once; the model's
  restatement of it, stored as a turn, came back unscreened through the conversation window,
  the session summary, `recall_conversation`, `memory_search` over sessions, the related
  earlier turns and the intent router's recent-turns context. `kernel/governance/reentry.py` now runs the floor's tripwire over assistant
  turns and summaries at those readers (the owner's own turns are returned verbatim), cuts
  anything over 16 KB per text or 128 KB per read with a visible `[not scanned: ...]` marker,
  and writes one counts-only `audit_log` row per read when something matched or a cap was hit.
  It follows `IRIS_GOVERNANCE_EXTERNAL_CONTENT_FLOOR`; there is no new setting. Phrase-level
  only; the recalled text is redacted (with the marker `[redacted: instruction-like text in stored content]`, since it is the model's own earlier text and not external content) but not yet marked as untrusted. See "Stored text
  coming back into a prompt" in `docs/concepts/governance.md`.

- The governed agent harness (`iris_harness`): intent routing, planning, a ReAct
  tool loop, response curation, and tiered LLM routing across local (Ollama, LM
  Studio) and opt-in cloud models.
- The governance kernel every model and tool call passes through: data
  classification, an egress gate, a Fernet vault, hooks, an evaluator, human
  approvals and an audit ledger.
- A plugin SDK (`iris_harness.sdk`) with profiles and three discovery paths
  (built-in, `iris_harness.plugins` entry points, `$IRIS_HOME/plugins`), plus the
  reference plugins `system`, `research`, `code_exec`, `telegram_channel`,
  `web_channel`, `web_push_channel` and `graphiti_import`.
- memris (`memris`), the memory graph of claims the harness remembers with.
- The IRIS API and Governor services, the `iris` CLI, a React web console, and a
  Docker Compose stack with a bundled Ollama.
- The plugin loader logs a warning when it mounts a plugin whose manifest omits
  `party` (the plugin is treated as `untrusted`), including an entry-point plugin
  with no manifest; `iris plugins` and `iris plugins show` print `party`; the
  examples and `iris plugin new` scaffolds declare `party: untrusted` explicitly.

### Fixed

- The compaction summariser reads stored text through the re-entry scan (issue #161). The
  summary the model writes is stored, so a reworded instruction would defeat a scan at read
  time; the assistant turns and the previous summary now go through the scan on the way INTO
  the summariser (the owner's own turns stay verbatim, nothing is enveloped, the summary's
  provenance is still tracked by its flag). One fix point, `_summary_prompt`, covers `compact`,
  `summarize_all` and the legacy dict API. Audited as reader `compactor`, counts only.

- `httpcore` is a declared dependency (issue #174): the governed client's pinned transport imports it directly and replaces httpx's private `_pool`. A new test pins that attribute, so an httpx upgrade that moves it fails in CI rather than at runtime.

- The behavior miner reads stored turns through the re-entry scan (issue #162). The turns it
  mines are stored text, so an assistant turn that held third-party text is redacted before it is
  put in the miner's prompt and the owner's own turns stay verbatim. The scan sits in
  `mine_behavior_patterns`, the one entry the console preview, the periodic run and the
  compaction-archived span all pass through. Audited as reader `behavior_miner`, counts only.
  The miner's output was already propose-only and human-approved.

- The general lane's direct skill answer is a governed call (issue #155, second part). When a
  request matches a skill, the lane answers it before any model by running the skill's first
  tool; that call used to run in-process with no `PRE_TOOL_USE` / `POST_TOOL_USE`, no approval
  check and no audit row. It now runs through `ToolService` under the `core:general_lane` caller
  (a `core:` caller needs no caller-policy entry, and a test pins that): audit rows with one call
  id, the external-content floor (one scan, where there were two), and the tool-policy and
  approval checks. The owner's answer is unchanged byte for byte for a structured result
  (the tool's result travels as JSON inside the governed call). **Visible change for flag-off
  and shadow installs:** a skill tool that governance will not run without the owner's approval
  is now refused with the governance message instead of running unasked (nothing is queued: a
  skill tool is not a registered tool, so an approval for it could never execute). Governed text
  that is not the tool's JSON (a result governance replaced or withheld) is shown as the text it
  is and records no pending actions. The brief slots (`direct_brief_response`) are a separate
  follow-up, #199.

- The legacy general lane (`IRIS_AGENTIC_CORE_ENABLED` off, or `shadow` where the lane's answer is
  the one read) runs every tool through the governed runner (issue #155, #134 D7). Its
  `memory_search`, `memory_graph`, `wiki_search`, `propose_skill_from_sandbox` and every skill
  tool, `fetch_web_content` included, used to run through a private dispatch table with no
  `PRE_TOOL_USE` / `POST_TOOL_USE`, no external-content floor and no audit row, so an injected
  instruction in a fetched page reached the next prompt, the session log, the answer and the memory
  stores raw. The table is gone: the lane's tool set is built from the loop's own builders and each
  tool carries the loop's declaration (`effect`, `confirm`, `content`). What changes for a
  flag-off or shadow install: a page or wiki result reaches the model inside the untrusted-content
  envelope with the instruction redacted and leaves audit rows; `memory_search` is the loop's
  scoped search (`facts`, `patterns`, `behaviors`, `sessions`); and `propose_skill_from_sandbox`,
  a write that needs the owner's approval, is refused in this lane (it has no checkpoint to pause
  on) instead of writing a draft unasked. With governance switched off
  (`IRIS_GOVERNANCE_ENABLED`) the lane's tools still run ungoverned, as the operator's explicit
  opt-out; one startup warning says so.

- `httpcore` is a declared dependency (issue #174): the governed client's pinned transport imports it directly and replaces httpx's private `_pool`. A new test pins that attribute, so an httpx upgrade that moves it fails in CI rather than at runtime.

- The governed HTTP client's name lookup is inside the request's deadline (issue #173).
  `getaddrinfo` has no timeout, so a resolver that hangs held the request before the deadline
  even started. The lookup now runs on a daemon thread and the request waits for it at most
  what is left of the deadline, then fails as the usual deadline abort. A thread stuck in the C
  resolver cannot be cancelled: it ends when the resolver returns, never keeps the process
  alive, and at most 8 can be stuck at once (a further lookup fails closed at once, "name
  lookups are backed up"). The decompression half of the issue was already done in #171
  (`_decoded` decodes at most 64 KiB at a time).

- The email store and the email onboarding store add their columns the same safe way (issue #201,
  second part): opening an older `email.db` from several processes at once no longer fails with
  `duplicate column name`, and the one-time stamp of old classifications as IRIS's (`classified_source`)
  runs once, in the process that added the column. The `memris` versioned upgrades were measured
  (twelve interpreters, fifteen rounds) and already tolerate the race, so they are unchanged.

- The email judge's user message is no longer wrapped in the untrusted-content envelope
  (undoing that part of the issue #148 change). Measured on a real mailbox, 100 emails judged
  by the local `email_judge` model before and after the wrapping: about 5 stably changed bucket
  at the same confidence, and by reading them the wrapping looked slightly worse (two
  promotional bank loan offers became `bill`). The judge's reply is a schema-constrained
  verdict, so the envelope only guarded against a manipulated label. The digest narration and the
  triage picker keep their envelope.

- Opening an older database from several processes at once no longer fails in the stores that
  migrate their tables (issue #201). The checkpoint, continuation, task, activity, learning-signal,
  document (RAG), reminder and approval stores each read `PRAGMA table_info` and then ran
  `ALTER TABLE ... ADD COLUMN`; two processes (the API, the CLI, a heartbeat) opening the same older
  file both saw the column missing, and the loser's `ALTER` raised `duplicate column name`, which
  stopped that process from starting its store. Measured with twelve real interpreters opening an
  older file at once: every one of these stores failed. They now add their columns through
  `ensure_columns` (a cheap look, then `BEGIN IMMEDIATE` on a connection of its own), keeping
  their original column declarations; the three one-time backfills (document classification,
  reminder lifecycle) run inside the transaction that adds the column, so only the process that added
  it does them, once. The email and onboarding stores and the `memris` versioned upgrades are the
  follow-up.

- The kernel's content classifier runs in linear time on hostile text (issue #156). The `email`
  pattern backtracked quadratically (a 200 KB `"a."*100000 + "@b."` took 19.5 s, and
  `"+1-"*130000` and `"123-45-"*57000` over 20 s each) and so did the voice-transcript marker;
  anything that classifies untrusted text (tool and MCP results, files, RAG ingest, the docs
  search) could be stalled with a small input. Both are now small linear matchers that mean what
  the old patterns meant (checked against them on 40,000 random strings); the same inputs take
  under 0.1 s.

- The external-content scan is idempotent (issue #166). A redaction could expose a phrase the
  first pass had no word boundary for (`...?q=1disregard the above prompt` after a swallowed
  URL), so scanning the scan's own output could still redact. `scan` now re-scans until
  nothing matches (at most 8 passes; spans and ids accumulate; `ScanResult.passes` and
  `.exhausted` report it; reaching the cap logs a warning and still returns the redacted text).
  Text that needed one pass is returned as before. Found by fuzzing 200,000 inputs, 7 of which
  needed a second pass; they are pinned as regression cases beside a seeded CI fuzz.

- An MCP call a `PreToolUse` hook answers with `require_approval` no longer reaches the
  server: the bridge refused only `deny` before the call, though the post step already refused
  both (issue #181).
- The timing test for the linear-time classifier no longer flakes on a loaded runner (issue #156
  follow-up). Its absolute one-second bound failed once at 1.2 s on a hosted runner for a case that
  takes about 0.05 s on a laptop. The bound is now a generous eight seconds (the old quadratic
  patterns took 20-50 s), and linearity is proven by a runner-independent check: doubling the input
  must not much more than double the time, best of three on each side, which a quadratic pattern
  fails whatever the machine; a second test shows the check catches the original email regex.

- `iris docs sync` (`sync_all`) removes a registered file that is gone from disk: its chunks
  and index entries are dropped and the file domain is told (`removed`), the same as
  `iris docs remove`. Before, a deleted file stayed searchable and the file domain kept
  listing it as indexed (issue #130). It prunes at once, but only when the file's parent
  directory exists and can be read: if the parent is missing or unreadable (an unmounted
  volume, a dropped share, a denied folder) nothing is pruned or reported, the entry is
  counted in `sources_unavailable` (the summary says "N skipped: location unavailable") and
  one warning says how many. Only RAG's own entries are removed, never a file. `IngestResult`
  gains defaulted `sources_removed` and `sources_unavailable` counts.

- The re-entry scan's audit rows are deduplicated (issue #164). A poisoned turn left in the
  history window used to write a `reentry_scan` row on every later read (about two a turn: 100
  reads, 100 rows). Rows are now keyed on session, reader, origin and the hashes of the matched
  or cut texts, in this process: the first sighting writes a row and a later one is written at
  the 2nd, 4th, 8th, 16th... sighting, each carrying `sightings` (the running count) in its
  payload. The audit log is append-only, so the count rides on new rows, not on an edit of the
  first. A different poisoned text is a different key and always gets its own row. The counts
  are in-process: a restart resets them. The existing payload fields are unchanged. This is a
  deliberate change to the kernel's audit behaviour.

- `iris run resume` reads the process's shared side-effect ledger for the database instead of
  building a second handle of its own, and opens it before it lists pending rows, so an
  unusable ledger database stops the command there (issue #102).

- Raw mail no longer reaches a model unmarked (issue #148). The email judge's user message
  (sender, subject, body), the inbox digest narration prompt and the triage picker's email
  envelope each go in one untrusted-content envelope with instruction-like spans redacted,
  through `wrap_external_content`; the owner's own instructions stay outside it, and the plain
  digest list the owner reads is unchanged. The offline demo's scripted model rule that matched
  the end of the prompt now allows for the closing tag. Nothing is scanned at ingest: stored
  text is not rewritten, and retrieval is marked where it is read.

- Learned memory is scanned where it is read back into a prompt (issue #163). `active.md`,
  `episodic.md` and a lesson's body are written from conversations, some of which held third-party
  text, and used to reach the model unscanned. The retriever (`active`, the episodic digest, the
  matched lesson and its pointers, the semantic episodic patterns), `memory_search` (`patterns`,
  `behaviors`), the intention-rollup context and the mission proposer now scan them after the size
  cut. Text the scan redacts also comes back inside the untrusted-content envelope
  (`source="learned_memory"`); clean text is unchanged and not enveloped, because a lesson is a
  recipe the model follows. Audit rows carry origin `learned_memory`.

- A document index made under a different embedding model can be repaired (issue #144).
  Chroma refuses to reopen the persisted collection under a new embedder, so the index was
  unavailable, retrieval fell back to keyword search and `iris docs reindex` could not
  rebuild it. `iris docs reindex` now says so in plain words and names the repair, the new
  explicit `iris docs reindex --reset-collection`: it deletes only the vector collection and
  rebuilds it from `rag.db`, keeping every source and every label. Plain `iris docs reindex`
  is unchanged. A rebuild that fails part-way (disk full, embedder error) leaves the index
  partial; the command says so, `rag.db` is untouched, and rerunning it is idempotent. The
  command cannot detect a running IRIS server, so stop the server before
  the reset and restart it after. A server that was left running no longer fails silently:
  on its next request it logs one warning, reopens the collection and retries once (and
  falls back to keyword search with a warning if the reopen fails).
- A governed request whose outcome row cannot be written is no longer invisible (issue #175).
  A failed `post_egress` ledger write used to be a log line only. The request has already
  happened by then, so the response is still returned (failing it would report a POST that took
  effect as failed and invite a retry); the loss is now counted (`unrecorded_outcomes()`),
  logged at error with the call id, and shown as a red `Egress ledger` row in System Health, so
  `GET /health`, the web Health screen, the system_health tool and the CLI all see it. The same
  ledger being down fails the next `pre_egress` write, so later requests are refused. The count
  is per process and clears on restart.

- The external-content floor bounds what a hostile text can grow to without erasing what
  follows it. The first 64 redacted spans in a text keep the full-size marker; each further
  span is redacted with the 3-character `[~]` and the legitimate text between spans is kept
  (an earlier cap collapsed the whole rest of the text into one marker, so one hostile item
  could erase the content after it). Output is asymptotically at most 2x the input plus a fixed 7 KB (the first 64 spans of
  each pass carry the full marker), every span is still redacted, and the span and pattern counts in the ledger row stay exact. A `source` or
  `tool` label with a newline or other control or invisible character can no longer inject a log line or
  ledger value: `redact_text` and the tool-result floor hook clean and cap both (200 characters).
- The research plugin's keyed providers and page fetch go through the governed HTTP client
  (issue #172). Brave, Exa and Tavily, and the page fetch behind `fetch_content`, used to open
  their own connections; each request is now checked against the plugin's `egress` declaration,
  made to the address the client checked (a changing DNS answer no longer matters) and recorded
  as a PRE and a POST ledger row. Redirects are followed one governed request per hop, capped at
  5, with each hop's scheme checked first; a redirect to a loopback, private, link-local or
  metadata address is refused and never connected to, and so is a redirect from https to http
  (http to https and http to http are followed). The fetch carries the tool call's context
  onto its worker threads, and a fetch with no tool call running is refused, not sent. Behaviour
  changes: a bare IP address in a URL is denied (even `open_web` contacts host names only), the
  timeout is the request's total budget rather than per operation, and a page body is read up
  to 10 MiB. The extracted text of a golden set of pages is byte-identical to before. SearXNG
  (the operator's own server), the DuckDuckGo library and the Crawl4AI browser are unchanged.
  The raw-network debt list shrinks by four pairs.

- The side-effect ledger is one shared handle per database, not one per kernel, and a ledger
  key that already holds a row is no longer ignored silently (issue #102). A destructive or
  pinned call whose key is taken is denied before it runs; a call's post-run record that
  finds its key taken is reported as a warning instead of confirmed.
  If the database file is removed while the process runs, the ledger creates it again,
  empty, and logs one warning (pending write-ahead rows it held are gone). Not covered by
  this change: `iris run resume` (`main.py`) still builds its own `SideEffectLedger`
  instead of the shared handle, and MCP-bridge calls get no write-ahead record (they declare
  no effect today, and calls with no call id need a key scheme first; see #134).
- A wildcard egress declaration over a public suffix is refused (issue #175). `*.co.uk`,
  `*.com.au` and the like passed because no public-suffix list shipped, so a plugin could
  declare every registrant under them. The harness now vendors Mozilla's Public Suffix List,
  ICANN section only (`kernel/governance/public_suffix_icann.dat`, MPL-2.0, the notice, version
  and source commit kept in its header; refreshed by `scripts/refresh_public_suffixes.py`), and
  a wildcard whose base is a public suffix fails manifest validation, so the plugin does not
  mount; `*.example.co.uk` and `*.github.io`-style declarations (the PRIVATE section is not
  vendored) are accepted. A manifest that declared such a wildcard needs the domain it means.

- A ReAct step's `pre_llm_call` audit row names the model and provider the step
  actually called: the step resolves its model once and the row and the call share
  it, where the row used to carry a snapshot from when the loop was built and the call
  asked the tier router again (a governor downshift or a tier edit could make them
  disagree, and a turn paid one extra router ask, a governor acquire, for the snapshot).
  A router answer of the wrong type still fails loudly rather than yielding a row with
  no model. `model` and `provider` now
  appear in `GET /governance/audit`, `iris governance audit` and the trace, and the
  CLI table has a `by` column (tool owner, capability provider or model).
- A core-only start (no `email` extra, no domain plugins) stays quiet without going
  blind. A skill whose tools module genuinely fails to load is logged once with its
  traceback (as before #118's one-line warning), while a skill blocked by a declared
  missing package gets one INFO line naming the extra to install; `gmail-inbox` now
  declares all three Google packages its import chain needs and `requires.extra: email`.
  A `heartbeats.yaml` entry names its owning plugin with `plugin:`: an unmounted owner is
  skipped quietly (one INFO summary, "plugin not mounted"), but a mounted plugin that
  registered no handler, a mistyped handler with no `plugin:`, or no plugin lookup bound
  still warns. Refs #110.
- A typo in a heartbeat's `plugin:` no longer reads as "plugin not installed":
  `config/heartbeats.yaml` declares every owner it may name (`plugins_in_tree`,
  `plugins_external`), a test checks each `plugin:` against them (and the in-tree names
  against the shipped manifests), and an undeclared owner with a missing handler warns.
  A skill blocked by a missing env var, config file or credential, not only a package,
  now logs the same one INFO line and shows `blocked: env:NAME` in `iris skills list`;
  `requires.env_vars` is now read from the process environment (it was checked against
  an empty mapping, so a declared variable always counted as missing). Refs #110.
