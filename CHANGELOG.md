# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First public release of the IRIS harness.

### Added

- `search_docs`: the core can search its own shipped documentation (architecture, concepts,
  guides, reference, usage-guides) by keyword at section level, as an internal read tool of
  the `system` plugin. The corpus is an allow-list in `config/docs_search.yaml`; it cannot
  reach the identity files, the vault, `.env` files, the owner's data or any store, and a
  document that classifies `secret` is dropped. An install without the docs says so.

### Changed

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
  only; the recalled text is redacted but not yet marked as untrusted. See "Stored text
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
- The side-effect ledger is one shared handle per database, not one per kernel, and a ledger
  key that already holds a row is no longer ignored silently (issue #102). A destructive or
  pinned call whose key is taken is denied before it runs; a call's post-run record that
  finds its key taken is reported as a warning instead of confirmed.
  If the database file is removed while the process runs, the ledger creates it again,
  empty, and logs one warning (pending write-ahead rows it held are gone). Not covered by
  this change: `iris run resume` (`main.py`) still builds its own `SideEffectLedger`
  instead of the shared handle, and MCP-bridge calls get no write-ahead record (they declare
  no effect today, and calls with no call id need a key scheme first; see #134).
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
