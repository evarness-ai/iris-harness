# The IRIS plugin contract

*Status: v1 (OSS plan M1, 2026-09-10). Everything outside the governed core is a plugin;
the governed loop and the governance kernel are the one thing you cannot unplug.*

## What a plugin is

A directory (or installed package) with two files:

```
my-plugin/
├── manifest.yaml
└── plugin.py        # exposes setup(api)
```

```yaml
# manifest.yaml
name: my-plugin            # [a-z][a-z0-9_-]*
version: 0.1.0
description: One line.
summary: What it lets IRIS do, for the owner (optional; the first-chat welcome lists it)
entrypoint: plugin:setup   # module:function (default)
trust: in-process          # in-process (default) | mcp (out-of-process, M2+)
party: first-party         # first-party (default) | trusted-third-party | untrusted
provides: [intercept, tool]   # advisory; feeds --dump-config and the drift panel
requires:                  # checked before setup(); unmet → plugin not loaded
  packages: []
  env_vars: []
```

```python
# plugin.py
from iris_harness.sdk import PluginAPI

def setup(api: PluginAPI) -> None:
    api.register_tool("my_tool", "What the model reads to decide to call it.", my_tool)
```

A plugin imports only `iris_harness.sdk` (and whatever public library code it needs),
never another core package: a name outside the SDK carries no stability promise. An
import-linter contract (`plugins-use-the-public-api`, release gate 2) holds the
reference plugins to this with no exceptions. The reference plugin
`src/iris_harness/plugins_builtin/system/` is the worked example.

## The six registration kinds (v1)

| Kind | Call | Handler shape | On failure |
|---|---|---|---|
| intercept | `api.register_intercept(name, fn, trace_text=..., trace_fields=(), passes_channel=False, activity_hint=None, guard_output=False)` | `fn(message, *, session_id, span[, channel]) -> ChatResult \| None` | returns `None`: the chain falls through |
| tool | `api.register_tool(name, description, call)` — the tool must be declared under `tools:` in the manifest (ADR-0110) | `call(args: dict) -> str` | the model gets an error observation; an undeclared tool is refused at mount |
| intent handler | `api.register_intent_handler(agent_type, fn, stream_handler=...)` | `fn(task: AgentTask) -> str \| (str, dict)` | an apology answer; streams re-raise |
| heartbeat | `api.register_heartbeat(name, fn, schedule="interval:60", description=...)` | `fn(definition) -> HeartbeatRun` | a `FAILED` run, reported not lost |
| channel | `api.register_channel(connector)` | `IChannelConnector` (`name`, `send`, `healthy`) | recorded, then re-raised |
| confirmation executor | `api.register_confirmation_executor(kind, fn)` | resolves a pending approval of `kind` | recorded, then re-raised |

Every registration is recorded in the `PluginRegistry` and wrapped by one fault
boundary. Failures count against the plugin and show as a yellow (degraded) or red
(failed to load) `plugin:<name>` row in System Health (`system_health` tool, `GET
/health`, the web Health screen). Judges and LLM providers are not pluggable in v1.

**Governance by construction.** A plugin tool joins the same ReAct tool pool as a
built-in and executes through the same `PRE_TOOL_USE` path; intent handlers land on
the same `AgentExecutor`; heartbeats on the same scheduler; channels on the same
gateway. There is no side door.

## Deterministic handlers are governed like generated answers

An intercept is a **deterministic handler**: it answers a turn with no model. It gets the
same governance as a generated answer. The user's message passed the `PRE_TURN` screen
before the handler ran, and the handler's answer passes the kernel's model-free response
check (`PRE_RESPONSE`: credentials, identity secrets, architecture disclosure) before it
ships. A blocked answer is replaced with the refusal text, and every check is audited with
`deterministic: true` and the handler's name.

Declare `guard_output=True` when the answer **repeats text someone else wrote** (an email
subject or sender, statement text, a news headline). When the operator enables the
model-based output guard (`IRIS_CURATOR_OUTPUT_SAFETY`), it then also runs on your
handler's answers, with the same halt and warning banner a generated answer gets. A handler
that answers only in its own words leaves it off. `config/intercepts.yaml` can set it for a
handler too; either side saying yes is enough. Design:
[`deterministic-path-parity.md`](./deterministic-path-parity.md).

## Calling another plugin's tool from code

Never import another plugin's tool function: that call skips `PRE_TOOL_USE`, the approval
rules and the audit row. Use `api.tools` — a `BoundTools` (`iris_harness.sdk.tools`) the
harness binds to your plugin as the caller:

```python
result = api.tools.call("search_inbox", {"query": "statement"})
if result.ok:
    use(result.text)
elif result.held:
    ...  # governance stopped it: result.text says why
```

`api.tools.describe()` lists every registered tool with what it declares (effect, confirm).

**You need permission.** Your plugin may always call its own tools. To call another
plugin's tool, list it in your manifest — the kernel's caller policy denies anything else,
and the call comes back `held`:

```yaml
uses:
  tools: [search_inbox, read_email]
```

An operator can take a tool away from a plugin in `config/governance/tool-access.yaml`
(never grant one: a grant lives in your manifest, where the dependency is visible). A
`uses:` entry naming a tool no plugin registers shows as drift (`plugin_uses`).
Your code cannot answer an approval, so a destructive tool, a pinned write or a
`confirm: once` write comes back `held` — never written silently. Design:
[`plugin-capabilities.md`](./plugin-capabilities.md).

## Sharing a service interface: capabilities

When another plugin (or the core) should use your data or service in code, provide a
**capability** instead of letting it import your package: a `domain.verb` name whose
`Protocol` is published in `iris_harness.sdk.capabilities`. Declare it, then register the
implementation in `setup`:

```yaml
capabilities:
  provides: [mail.read]    # you implement it
  uses: []                 # you work without these, degraded
  requires: []             # you are not loaded without these
```

```python
api.provide("mail.read", GmailReader())         # provider
reader = api.capability("mail.read")            # consumer: the implementation, or None
```

Undeclared use is refused both ways and shows as your plugin's failure. Providers mount
before consumers; a `requires` nobody provides keeps you unloaded (red in System Health), a
`uses` nobody provides leaves `capability()` returning None. A failure inside a provider is
attributed to the provider. With several providers you get one implementation that fans
out to all of them. A declared capability never provided shows as drift
(`plugin_capabilities`). The catalogue is closed: a new capability needs an SDK release,
not a plugin. Design: [`plugin-capabilities.md`](./plugin-capabilities.md) §2.

**Every call is governed like a tool call** (`capability:<name>.<method>`): the kernel checks
your manifest allows it, applies the method's declared effect (a `confirm: once` write is
refused with `require_approval` until the approval executor is extended to capability
calls), audits it, and masks the owner's identity out of the result's text
fields. You receive a masked *copy*: plain data (dataclasses and the like), never the
provider's live object. An operator can take a capability, or one method, away from you in
`config/governance/tool-access.yaml`.

**The consumer contract.** Catch `CapabilityUnavailable` to take your degraded path: it is
raised when the provider is no longer mounted, and its subclass `CapabilityDenied` when
governance stopped the call or withheld its result. Any other exception is the provider's,
and is already recorded against the provider in System Health. Call a sync method from sync
code; from async code use the capability's async methods (a sync call inside a running
event loop cannot be governed, so it is refused).

```python
try:
    messages = reader.search("statement") if reader is not None else []
except CapabilityUnavailable:
    messages = []  # degraded: no mail source right now
```

**Who you are is the harness's stamp.** `api.tools`, `api.capability` and
`api.register_owner_identity_source` are bound to your plugin, and they are the only bound
entries: the registry is private and `services.tools` is
a catalogue (`describe` only). Your plugin runs in the harness's process (`trust:
in-process`), so this is a contract, not a sandbox -- reaching into private attributes or
importing runtime internals is unsupported. `trust: mcp` is the real boundary.

## Supplying the owner's identity

A plugin that knows an address of the owner's (the account it signs in to) hands it to the
guards, so they protect it like the rest of the owner's identity. Declare the kinds, then
register a provider that returns `{kind: literals}`:

```yaml
identity:
  provides: [email]        # name, email, phone, address, handle
```

```python
api.register_owner_identity_source(lambda: {"email": [account.address]})
```

The source is `plugin:<your name>`, never a name you choose. A kind you return but did not
declare is dropped and shows as your plugin's failure; registering with nothing declared is
refused. `secret` is never providable. Pass `fingerprint=` (a cheap probe that changes
when your literals do) if they can change while the harness runs. Design:
ADR-0125.

## Tools declare their effect (ADR-0110)

Every tool a plugin registers is declared in its manifest, and `register_tool` checks
the declaration: a registered tool missing from `tools:` is refused and shows as the
plugin's failure; a declared tool never registered shows in the drift report.

```yaml
provides:
  - tool
tools:
  calendar_lookup:
    effect: read
  create_reminder:
    effect: write          # changes something on the user's behalf
    confirm: once          # default for a write: the loop asks the user once per run first
  daily_plan:
    effect: read
    guidance: >-           # shown in the prompt only while this tool is on the loop
      For a whole-day question call daily_plan; use calendar_lookup only for a
      specific calendar question.
```

Two words buy the loop mechanics: a `write` tool is marked in the prompt and, with
`confirm: once`, is turned back until the run has asked the user (`ask_user`) — a write
the user's own sentence already authorises says `confirm: never`. `guidance` is the
plugin's routing prose in the plugin's words; the core renders it and authors none.
`pinned: true` keeps a tool on the prompt's menu whatever the query sounds like (the
menu is capped and ranked by relevance); use it for a tool other tools' guidance names.
A skill can never shadow a registered tool's name; nothing needs reserving.

A tool that removes or overwrites the user's data is `effect: destructive` (ADR-0118). It
takes no `confirm`: every call waits for the owner's itemised approval. It names its
`undo` tool when the service has a reversible form, and must use that form (trash, not
delete):

```yaml
tools:
  trash_email:
    effect: destructive
    undo: restore_email    # declared here too, as a write with confirm: never
    undo_window_days: 30   # how long the undo stays possible; shown on the card
  restore_email:
    effect: write
    confirm: never         # "undo that" is its own authorisation
```

Each call waits for the owner: the run halts on an approval row pinning the tool and its
exact arguments, and only approving runs exactly that call. **Take a list** (`ids: [...]`)
so one request is one approval; a one-item-at-a-time tool costs the owner one approval per
item. With governance off or approvals disabled, a destructive call is refused.

A write that is not data loss but cannot be taken back (sending an email) declares
`approval: pinned` instead of `confirm` (ADR-0118 amendment). It stays `effect: write`,
and every call takes the same path as a destructive one: `validate`, a card from
`describe`, a row pinning the exact call, approve runs exactly it, reject runs nothing.
The card says it acts on the owner's behalf, not that it deletes anything.

```yaml
tools:
  send_email:
    effect: write
    approval: pinned       # no confirm: the card is the confirmation
```

Give the owner words, not ids: register the tool with `describe=`, a function from the
call's arguments to a `ToolDescription` (from `iris_harness.sdk.types`) holding a title
("Trash 3 emails") and one line per item ("Your weekly deals — Store X · 20 Sep"), looked
up in your own data. It runs once, when the approval is created, and is frozen into it;
if it raises, the card falls back to the raw call. A tool with no `undo` is shown as
"This cannot be undone."

```python
def _describe(args):
    mails = [mailbox.get(i) for i in args["ids"]]
    return ToolDescription(
        title=f"Trash {len(mails)} emails",
        lines=tuple(f"{m.subject} — {m.sender} · {m.date:%d %b}" for m in mails),
    )

api.register_tool("trash_email", "Move emails to the trash.", _trash, describe=_describe)
```

## Services a plugin may use

`api.services` (a `HarnessServices`) carries: `config_dir`, `data_dir`, `tier_router`,
`agent_executor`, `heartbeats`, `channels`, `deterministic_reply(message=, session_id=,
response=, metadata=, span=)` (the governed way to answer a turn from a template; a
domain may label its turn with `intent=` / `agent_type=` / `sources=` and report a
failed action with `has_errors=` / `error_summary=`), `events` (the runtime's event
bus — use `api.subscribe` / `api.publish`, below; note the `scope` argument, since the
domain chains run on the process-global bus), `submit_activity(...)` (below),
`react_handler` / `react_stream_handler` (below), `continuations` (open an owned
question, see ADR-0106), `skill_registry` (the loaded skill packages — read, never
owned), `conversation_in_flight(session_id)` (true while a harness-owned multi-turn
conversation is mid-flight in the session; an on-demand intercept steps aside),
`default_channel()` (the validated default delivery channel name, read at call time),
`classify_intent(message)` (the harness's one intent router,
for an intercept that wants a semantic gate instead of its own classifier — read
`.intent` and `.source`; `None` when no runtime is wired), and
`heartbeat_diagnostics()`. If you need something else, that is a contract change: open
an issue rather than reaching into the runtime.

**What each handle promises** (`iris_harness.sdk.services`, core/SDK boundary plan PR 1).
The handle fields are typed with Protocols that list only the methods plugins call, so
that is all you may rely on (and all a test fake needs):

| Field | Protocol | Methods |
|---|---|---|
| `tier_router` | `TierRouterService` | `get_llm_config(intent)`, `get_tier(intent)`, `get_tier_by_name(name)` |
| `agent_executor` | `AgentExecutorService` | `register`, `register_stream` (use `api.register_intent_handler`) |
| `heartbeats` | `HeartbeatService` | `register_handler`, `register`, `trigger_by_name` |
| `channels` | `ChannelService` | `register`, `channels()`, `broadcast(message, channels=)` |
| `events` | `EventBusService \| None` | `on`, `emit_sync` (use `api.subscribe` / `api.publish`) |
| `lessons` | `LessonService \| None` | `find_similar`, `render_prior_lessons`, `handle` |
| `continuations` | `ContinuationService \| None` | `ask`, `pending`, `answered` |
| `skill_registry` | `SkillRegistryService \| None` | `list_packages(agent_name=, only_loadable=)` |

`iris_harness.sdk` itself re-exports only the author surface: `PluginAPI`,
`HarnessServices`, `PluginCLI`, `register_plugin_commands`, `PluginManifest` and
`RegistrationKind`. The loader, registry and profile resolution are host machinery in
`iris_harness.runtime.plugin_host`.

**Provider plugins plug into keyed core registries, not a registration kind.** A plugin
that *is how something external gets reached* — a mailbox, a credential, a pending-action
source — registers an implementation of a core interface in its `setup()`:
`email.provider_api.register_mail_provider(provider)` (the `MailProvider` protocol: fetch,
cursor reset, body, attachments, attachment search; looked up by the account's provider
name; `provider_api` is the whole stable surface a mail provider is written against,
and its `connect_account(provider, address)` records the owner's account so the sweep
syncs it),
`health.service.register_check_provider(key, fn)` (a credential's rows go through
`api.register_credential_check`, below), and
`iris_harness.sdk.pending_actions.register_provider(...)`. Registration is keyed, so a
second runtime in one process replaces rather than stacks; an entry outlives an unmount
within that process, so a composed package is a fresh process -- except under
`iris_harness.testing.harness`, which puts every declared registry back on exit
(`foundation/process_state.py`). Such a plugin may declare `provides: []`. The `gmail`
plugin is the worked example.

**Surfaces the core serves, filled by your plugin** go through `PluginAPI` (core/SDK
boundary plan, PR 3c-2). Each forwards to a keyed core registry, is recorded against your
plugin (`seams` in `GET /plugins/{name}` and `iris plugins show NAME`), and runs inside
the fault boundary: a failure is charged to your
plugin in System Health and re-raised, and the surface skips the broken entry.

| Method | What the core does with it |
|---|---|
| `api.register_api_router(key, factory)` | The API service mounts `factory()` (a `fastapi.APIRouter`) after its own routes: how a capability that left the core keeps its API (the owner's rule: every capability needs an API, never a UI-only one). |
| `api.register_public_callback(path)` | `GET path` (exact, under `/api/v1/`) skips the token and cookie check, for a redirect back from an OAuth consent page. Your route must authenticate the request itself with a one-time secret it issued (an OAuth `state`). |
| `api.register_agent_panel(agent, build)` | `GET /agents/{agent}` shows `build()` as that agent's "key stores" panel. |
| `api.register_learned_source(name, source)` | The digest's "learned yesterday" line includes `source(start, end)`'s phrases for the owner's previous local day. |
| `api.register_footer_line(name, line)` | The digest footer adds `line(start, end)` (one short line, or None) after "learned yesterday", for the owner's previous local day. |
| `api.register_credential_check(name, check)` | System Health's credential rows add `check(net_probe)`'s `HealthCheck`s after the core's own (cloud keys, audit key), for a credential your plugin owns. `net_probe` is the owner's opt-in to a live check. A check that raises is a yellow row under `name`, not a missing one. The Gmail, Calendar and Drive rows arrive this way (`iris_personal.connections.google.google_credential_checks` builds them); the core names no provider. |
| `api.register_search_provider(name, provider, priority=None)` | The `research` tool's provider chain tries `provider` (an `iris_harness.sdk.research.SearchProvider`: `is_available()` and `search(query, *, max_results, ...) -> list[SearchHit]`) at its place in `config/search_providers.yaml` (else `priority`, else `default_priority`; lower first). `SearchHit` is frozen and keyword-only: `url`, `title`, `snippet`, `published` (a `datetime` or None), `source`, `extra` (short string labels); scoring, trust and page content are the engine's. A call that returns anything but a list of `SearchHit` is refused and charged to your plugin (degraded), and the chain moves on. The name must be declared under `search_providers:` in the manifest -- an undeclared one is refused and charged, as an undeclared tool is. The chain's guards, cache, rerank, injected-instruction scan and audit apply to it; no tool of its own is declared. The research plugin's five built-ins register this way too. A name another mounted plugin holds is refused; the provider leaves the chain when your plugin is no longer mounted. |
| `api.register_loop_intent(intent, fallback=fn)` | With the governed loop on, the loop answers `intent` (over every registered tool) and `fn(task)` is its degrade path when a turn errors, answers nothing or answers without reading; an apology if `fn` itself fails (not re-raised). With the loop off, nothing: `register_intent_handler` decides the lane. |

These are not registration kinds (`provides:` does not list them): like `api.subscribe`,
they fill a surface the core owns rather than add a capability of a new kind.

The same shape answers a core surface that reports on data a plugin owns. `iris system
status` counts file roots and connected accounts through
`system.status.register_file_root_counter(fn)` and `register_account_counter(fn)` (a
provider -> count mapping); with nothing registered it reports zero roots and no accounts.
The digest settings check a `section_config` knob whose values are a plugin's vocabulary
through `iris_harness.sdk.digest.register_section_knob_validator(knob, allowed)`; a knob
nobody registered is kept as saved. Status runs with no runtime built, so a counter is
registered from the plugin's CLI `register()` as well as its `setup()`.

**Expiry kinds are manifest data** (boundary plan PR 5, decision D1). A plugin whose dated
items age out by local days declares each kind under `expiry:` in `manifest.yaml`
(`{default, min, max, description}`; not a core key) and reads the owner's value with
`iris_harness.sdk.digest.expiry_days(key)`. The owner tunes it in `digest.yaml`'s
`expiry:` by the same key. Declarations are read from every installed plugin, mounted or
not, so the file stays valid while a plugin is off; a key nobody declares is refused in
`digest.yaml` (a warning, built-in policy) and raises `KeyError` from `expiry_days`.

**A stateful plugin's SDK modules** (core/SDK boundary plan, PR 3). A plugin that keeps
data or works on the owner's clock imports these, never the core module behind them;
each name is the core object itself, so switching an import changes no behaviour:

| Module | Names | Use it for |
|---|---|---|
| `iris_harness.sdk.persistence` | `data_path`, `data_dir`, `connect`, `sqlite_conn`, `with_locked_retry`, `collection_kwargs` | Your files and tables in the data dir (`$IRIS_DATA_DIR`, else `$IRIS_HOME/data`, else the checkout's `data/`, else `~/.iris/data`; never a bare `Path("data/x.db")`, which lands in whatever directory `iris` runs from), SQLite in WAL mode with a busy timeout, and the one shared ChromaDB embedder. |
| `iris_harness.sdk.time` | `iris_timezone` | The owner's zone (`IRIS_TZ`, else UTC) for every "today" and "at 9". |
| `iris_harness.sdk.events` | `get_default_bus`, `EventBus`, `EventHandler` | The process bus, for code that runs with no runtime built (below). |
| `iris_harness.sdk.digest` | `load_digest_settings`, `news_group_topics`, `DigestSettings`, `register_section_knob_validator` | Reading what the owner set for the digest; validating your section knob. |
| `iris_harness.sdk.tasks` | `Task`, `TaskStore`, `TaskAction`, `ActionCard` / `ActionChoice` / `ActionFact` / `ActionEvidence` / `ActionOptions` / `ActionOptionValue`, `SourceKind`, `WaitFor`, `TASK_COMPLETED`, `short_title` | Raising work for the owner into the one task store, with a one-tap action and the card that explains it. |
| `iris_harness.sdk.reminders` | `Reminder`, `ReminderStore`, `TERMINAL_STATUSES`, `NOT_YET`, `parse_snooze`, `SNOOZE_CHOICES`, `REMINDER_COMPLETED`, `REMINDER_SNOOZED` | Scheduling a reminder the harness delivers, and reacting when the owner finishes or snoozes it. |
| `iris_harness.sdk.pending_actions` | `PendingActionProvider`, `ChoiceActionProvider`, `DesiredAction`, `register_provider`, `reconcile`, `provider_for`, `PendingActionsSummary` | Surfacing blockers in the Action Center: your provider says what should be open, `reconcile` raises and resolves the tasks. |
| `iris_harness.sdk.vault` | `save_token`, `load_token`, `delete_token`, `CredentialRevokedError`, `SecretStore`, `get_secret_store` | An OAuth blob in the OS keychain (keyed by provider and account); any other secret through the configured secret backend. |
| `iris_harness.sdk.rag` | `DocumentStore`, `DocumentIndex`, `search_documents`, `propose_rag_ingest`, `execute_rag_ingest`, `IngestDeniedError`, `IngestProposal`, `IngestResult`, `IngestSource`, `KnownFile`, `IndexedDocument`, `register_ingest_source`, `DocumentCatalog`, `RagDocument`, `register_document_catalog` | Searching the owner's documents, asking to ingest one (governed), and owning where documents come from. |

Three older SDK modules gained names in the same PR: `sdk.llm` (`make_narrative_llm_call`,
`embed_corpus`, `EMBED_MODEL_DEFAULT`, `TierConfig`, `governance_tier_for_intent`,
`provider_root_url`), `sdk.health` (`HealthCheck`, `HealthState`, `CheckKind`,
`Reconnect`, `net_probe_enabled`) and `sdk.cli` (`console`, `print_error`: the console
the core CLI prints to, so a plugin command reads like a core one).

**Published for the public email slice** (OSS plan R2), same shape — each name is the
core object:

| Module | Names | Use it for |
|---|---|---|
| `iris_harness.sdk.activity` (new) | `chat_in_progress` | Background work (an embedding pass, a batch of model calls) yielding while the owner chats. |
| `iris_harness.sdk.audit` (new) | `AuditLog`, `audit_db_path` | Recording a governed decision the kernel does not see (an OAuth connect) in the one ledger `iris audit` reads. Never a token, a code or content. |
| `iris_harness.sdk.config` | + `PUBLIC_URL_ENV`, `public_base_url` | The address the owner reaches IRIS at from outside (an OAuth redirect, a link). |
| `iris_harness.sdk.config` | + `iris_home`, `workspace_dir` | Where this IRIS lives (`$IRIS_HOME`, else `~/.iris`) and the owner's workspace in it, for what a plugin keeps beside the identity files. Never `Path.home() / ".iris"`, which ignores a relocated home. |
| `iris_harness.sdk.health` | + `register_account_counter`, `register_file_root_counter` | The counts `iris system status` shows. |
| `iris_harness.sdk.memory` | + `WIKI_INGEST_REQUESTED`, `WikiIngestEvent`, `WikiEngine`, `subscribe_wiki_ingest_consumer` | Feeding the knowledge wiki; a backfill command wiring its own engine. |
| `iris_harness.sdk.types` | + `Principal` | Who is calling your API route (owner or paired device). |
| `iris_harness.sdk.types` | + `HeartbeatRun`, `HeartbeatStatus`, `HeartbeatHandler` | A heartbeat handler's return value and Protocol, beside `HeartbeatDefinition` (step 2). |
| `iris_harness.sdk.llm` | + `JsonReply`, `LLMUnreachable`, `LLMBadReply` | The governed structured call, `CodingLLMClient.invoke_json(system_prompt=, user_prompt=, schema=)` (a classifier, a judge; Ollama tiers): stop a batch on `LLMUnreachable`, skip the item on `LLMBadReply`. |
| `iris_harness.sdk.heartbeat` (new) | `heartbeat_runs`, `HeartbeatRunHistory`, `StoredRun`, `tally`, `Tally`, `clock` | Reading whether your scheduled jobs ran (a health check, a footer line): `heartbeat_runs(data_dir)` is a read-only view, since the scheduler is the one writer. `clock` is display text whose format may change. |
| `iris_harness.sdk.heartbeat` | + `load_heartbeats`, `describe_schedule`, `HeartbeatConfigError` | Telling the owner when your job runs ("daily at 06:15"), read from `heartbeats.yaml`; read-only. |
| `iris_harness.sdk.approvals` (new) | `ApprovalQueue`, `ApprovalStore`, `ApprovalCard`, `ApprovalRow`, `ApprovalNotFoundError`, `ApprovalAlreadyAnsweredError`, `respond_to_approval` | A flow the owner drives step by step that must leave a governed approval before it acts (email setup's "may IRIS change your mailbox?"): enqueue a row with a card (it shows in the Action Center and `iris approvals`), answer it only on the owner's explicit word through `respond_to_approval`, the function every surface answers with. |
| `iris_harness.sdk.vault` | + `master_key_status`, `MasterKeyStatus`, `fix_master_key`, `KeyFix` | A setup flow checking the vault master key (#741) and, when the owner agrees, creating one with `iris doctor --fix`'s own function (it never replaces a key). |

A plugin's **vocabulary is data, not code.** Trigger phrases, verbs and nouns an
intercept matches belong in a YAML the plugin ships (the calendar plugin's `nlu.yaml`,
overridable from `<config_dir>/calendar/nlu.yaml`), compiled at load; the core carries
no plugin's keywords. A **governed write** needs no service of its own: stash the action
on a continuation with an `executor_kind`, register a confirmation executor for that
kind, and the harness re-runs it on "approve" (the calendar plugin is the worked example).

## Personas: answering on the harness's own loop

An intent handler normally means "I supply the handler". Sometimes you want the opposite:
the agent type should answer on the **harness's governed ReAct loop**, with the intent
biasing the tier and the prompt and the tools coming from your skill packs (ADR-0077).
Pass the loop straight through:

```python
handler = api.services.react_handler
if handler is not None:                       # None when the loop is off
    api.register_intent_handler(
        "filemanager", handler, stream_handler=api.services.react_stream_handler
    )
```

Register it only when you mean it: with the agent type unregistered, routing falls back
exactly as it did before, because the router gates on the registered agents. If your
persona has no deterministic fallback lane, gate the *registration* on your flag rather
than gating the behaviour inside the handler.

## Long jobs: submit, don't block

An intercept must not hold a chat turn open while it walks ten thousand files. Hand
the work to the Activity spine and answer immediately:

```python
def setup(api):
    def organize(message, *, session_id, span=None, channel=None):
        folder = ...                         # your parse; return None if not yours
        def work(progress):
            from iris_harness.services.activities import ActivityOutcome
            progress(0.1, "indexing folder")
            ...
            return ActivityOutcome(result_summary="Organized 412 files.", metadata={})
        api.services.submit_activity(
            kind="filemanager.organize", title=f"Organize {folder}",
            work=work, origin=f"chat:{session_id}", metadata={"channel": channel},
        )
        return api.services.deterministic_reply(
            message=message, session_id=session_id,
            response="I've started that — watch it under Activity.", metadata={}, span=span,
        )
    api.register_intercept(
        "organize_request", organize, passes_channel=True,
        activity_hint=lambda m: "scanning the folder…" if _looks_like_organize(m) else None,
    )
```

- `work(progress) -> ActivityOutcome`; `progress(frac, message)` streams onto the row.
- **The harness sends the completion notice** — the in-chat notice on the originating
  session plus a channel message. Do not write your own. Set `metadata["channel"]` to
  steer which connector it uses.
- Park anything a follow-up turn needs (a plan id, a proposal) **inside `work`**, before
  it returns, not in a completion subscriber.
- `activity_hint(message) -> str | None` is the line streamed *before* the chain runs,
  so a slow intercept does not leave the user watching silence. It is guarded like the
  handler and degrades to `None`; it is part of the intercept, not a registration of
  its own.

## Events: work that finishes outside a turn

Some work does not end when the turn does. A plugin submits a long job to the
Activity spine, the turn replies *"I've started that"*, and minutes later the job
finishes on a background thread. There is no seventh registration kind for that
callback — the producer already publishes a typed payload on a named topic, so a
plugin subscribes to the same contract the core does:

```python
from iris_harness.services.activities import ACTIVITY_COMPLETED, ActivityCompletedPayload

def setup(api):
    def on_done(payload: ActivityCompletedPayload) -> None:
        if payload.kind != "organize":
            return          # not ours — every subscriber sees every event
        ...
    api.subscribe(ACTIVITY_COMPLETED, on_done)
```

- `api.subscribe(topic, handler, *, scope="runtime")` — `handler(payload)`, sync or async.
- `api.publish(topic, payload, *, scope="runtime")` — emit from synchronous code.

Subscribing is **consuming a service, not providing a capability**: it does not appear
in `provides:` or in the plugin's `kinds()`. It is still inside the fault boundary — a
raising subscriber is recorded against your plugin (a yellow row in System Health) and
degrades to `None`, so the other subscribers on that topic still run. Every subscriber
on a topic sees every event, so filter on the payload yourself.

### Which bus: `scope="runtime"` vs `scope="process"`

IRIS runs **two** event buses, and picking the wrong one fails *silently* — your handler
simply never fires. Which bus a topic lives on is a property of its **producer**, so
check the producer before you subscribe.

| `scope` | Bus | Use it for |
|---|---|---|
| `"runtime"` (default) | `HarnessServices.events`, private to this runtime | Activities, and anything else published by the runtime you are mounted in. Private so completion subscribers don't cross-talk between runtimes in tests. |
| `"process"` | `iris_harness.sdk.events.get_default_bus()`, the process-global singleton | The domain chains — `email.new_arrived` → `email.classified` → wiki/followup. These have to be process-global because CLI commands (`iris email recategorize`, `iris email reingest-wiki`) emit on them with **no runtime built at all**. |

Both scopes go through the same fault boundary and both are recorded against your
plugin: `GET /plugins/{name}` lists them as `subscriptions` (`topic` + `scope`), and
`iris plugins show NAME` prints them as `topic @scope`. An unrecognised scope raises
rather than subscribing to nothing.

Topic names and payloads live with their producer, never in a shared enum:
`iris_harness.services.activities.events`, `iris_harness.services.tasks.events`,
`iris_harness.services.notifications.events`. Follow `<domain>.<verb>` and ship a frozen
dataclass payload.

## CLI: adding `iris` subcommands

A command group is a **static** surface: `iris files --help` must not pay for
building a runtime. So CLI contributions come through a second, much smaller entry
point than `setup(api)` — declare it in the manifest:

```yaml
cli: cli:register          # module:function, beside plugin.py
```

```python
# cli.py
def register(cli):                       # cli is a PluginCLI
    files = cli.group("files")           # the core's existing `iris files` app
    files.add_typer(organize_app, name="organize")
    files.command("cleanup")(cmd_cleanup)
```

`cli.group(name)` returns the harness's app for that group, so you add commands
**into** a group the core already publishes; ask for one it does not define and you
get a new group attached to `iris`. A dotted name reaches a nested group the core has
published for extension (`cli.group("files.photos")` is `iris files photos`); an
unpublished dotted name raises, rather than quietly making a top-level group. At `iris` start-up the manifest is read and only
this one module is imported — **no `setup`, no `HarnessServices`, no runtime**. A
command body that needs the runtime builds it itself, the way the core's own do.

Failures are contained: if your CLI module is missing, fails to import, or throws
while registering, it is logged and skipped and the rest of `iris` still works.
Because the decorator on a module-level function would bind before `register` runs,
attach top-level commands inside `register`, not with `@app.command` at import time.

## Web console screens

The console's navigation is the core's screens (`config/webui/nav.yaml`) plus those of
every **mounted** plugin (OSS plan R17). A plugin declares the screens it owns:

```yaml
webui:
  screens:
    - id: inbox              # lowercase, unique
      label: Inbox           # sidebar / More entry
      route: /inbox          # one segment; /inbox/<id> belongs to it too
      title: Inbox           # page header (default: label)
      subtitle: What IRIS sorted each email into
      icon: mail             # a lucide name the console bundles; else a generic icon
      group: operations      # a core group id (nav.yaml); unknown -> default_group
      order: 30              # place inside the group; core entries step by 10
      nav: true              # false: owned and gated, but no menu entry
```

`GET /api/v1/webui/nav` (`runtime/plugin_host/nav.py`) returns the ordered groups, the
off-nav screens, the phone bar's pins (core only) and, for each installed plugin that is
not mounted (failed, disabled, not in the profile), its screens under `unavailable` with
the reason. The console draws only that: a mounted plugin's screen appears, an unmounted
one's is absent, and a direct link to it shows which plugin is missing rather than a
screen with no API behind it. A route the core or an earlier plugin already owns is
refused and reported under `problems`. The screen's component still ships in the console
bundle; the manifest says whose it is.

## Profiles: what mounts, in what order

`config/profiles/<name>.yaml` lists plugins in mount order. `IRIS_PROFILE` picks the
profile; shipped: `minimal`, `default`, `email`, `personal-assistant`. Unset, it is
`default`, or the first profile in `default.yaml`'s `prefer_when_installed` whose every
plugin is installed and loadable (found, `requires` met): with the email extra
installed that is `email`, so `iris-harness[email]` needs no setting. A profile named
by `IRIS_PROFILE` or `--profile` is always taken as named.

Layers, later wins: shipped profile → `$IRIS_HOME/profile.yaml` (merge by plugin name;
may add plugins; `intercept_order` replaces) → `IRIS_PLUGINS_DISABLE=a,b` /
`IRIS_PLUGINS_ENABLE=c,d`. A missing shipped profile falls back to the built-in
default (`system`), like every other `config/` file.

Intercept order: `config/intercepts.yaml` rows run in file order. A row whose
`handler` is `plugin:<name>` is served by the plugin's registration of the same
intercept name. Plugin intercepts not declared there run after the declared ones, in
registration order. A profile `intercept_order: [a, b]` moves those names to the front.

```
iris --dump-config [--profile NAME] [--json-config]   # layers, sources, kinds; no boot
iris plugins [--json]                                  # the running IRIS: status + counts
iris plugins show NAME [--json]                        # registrations, subscriptions, seams, drift
```

## Discovery order

1. builtin — `iris_harness.plugins_builtin.<name>`
2. entry point — group `iris_harness.plugins`, name `<name>` (pip-installed plugins)
3. home — `$IRIS_HOME/plugins/<name>/manifest.yaml` (personal plugins, no packaging)

Not discovered at all: an **in-process** plugin, handed to `build_runtime` as its
`setup` and manifest (`in_process_plugins=`), mounts after the profile's own rows with
the same checks. That is how a plugin's tests and the examples run one without
packaging it: `iris_harness.testing.harness(plugins=[plugin(setup, manifest=...)])`.

## What you may import

Only the stable tier: `iris_harness.sdk`, `iris_harness.testing`, and the mail-provider
names. The list, the deprecation rule and how it is enforced:
[`docs/reference/stable-api.md`](../reference/stable-api.md).

## Declarative plugins (`flavor: declarative`)

A skill is the declarative flavor of plugin (OSS plan decision 5): the manifest is the
whole plugin, and the loader mounts it with no `setup()` of its own
(`runtime/plugin_host/declarative.py`).

```yaml
name: units
flavor: declarative
provides: [tool]
tools:
  units_convert:
    effect: read                       # effect/content/confirm/... as for any tool
    description: Convert a length.     # what the model reads (the args are appended)
    impl: my_units.functions:convert   # package.module:function, importable
    args:
      value: {type: number}
      unit: {type: enum, options: [m, km]}
      digits: {type: integer, required: false, default: 3}
```

- Argument types: `string`, `number`, `integer`, `boolean`, `enum` (with `options`).
  Required unless `required: false`; an optional one may give a `default`.
- Every call's arguments are checked before the function runs (and before any approval
  is queued): an unknown or missing argument or a wrong type is an `error: ...`
  observation, never an exception. The function returns a string or anything JSON
  encodes.
- The tools register through `register_tool`, so the declaration's effect and gate, the
  governed runner (`PRE_TOOL_USE`, approvals, `POST_TOOL_USE`, audit) and the fault
  boundary apply as for a python plugin's.
- Refused at load (the plugin is `FAILED` with the reason): a tool without `description`
  or `impl`; `entrypoint`, `cli`, a non-`tool` `provides` or `trust: mcp`; an `impl` that
  does not import or cannot take the declared args. A python plugin may not carry
  `description`/`impl`/`args` (it says them in code).
- Installed, the entry point names the package (`my-units = "my_units"`); in a test,
  `iris_harness.testing.plugin(manifest=PATH)` with no setup. A home-directory plugin
  (`$IRIS_HOME/plugins/<name>/manifest.yaml`) works when its `impl` module is importable.
- Not the `config/skills/` skill system (code-first skills, the semantic router): that
  is a separate, older mechanism and is unchanged.

## Starting a plugin: `iris plugin new`

```bash
iris plugin new weather-now --kind tool [--dir PATH] [--force]
cd weather-now && pip install -e ".[test]" && pytest
```

Writes `<dir>/<name>/`: `pyproject.toml` (the `iris_harness.plugins` entry point, the
manifest as package data, pytest on `src/`), the package `iris_plugin_<name>` with `plugin.py`
(`setup(api)`) and `manifest.yaml` (a `skill` is manifest-only), a README, and tests that pass out of the box, offline:
the plugin mounted in-process in `iris_harness.testing.harness` on the scripted model,
plus a `check_stable_imports` test over the plugin itself. `--force` overwrites the
scaffold's own files in an existing directory and leaves the rest.

| `--kind` | What it generates |
|---|---|
| `tool` | one read tool (`effect: read`, `content: internal`), called by the scripted model through the governed loop |
| `channel` | a connector implementing `IChannelConnector` (`name`, `send`, `healthy`) on the gateway, delivering to an outbox file in the data dir |
| `mail-provider` | a `MailProvider` over a demo mailbox: `connect_account`, incremental sync through `MailSyncStore`, the write gate, and the owner's address to the identity guards (`identity: provides: [email]`); needs `iris-harness[email]` |
| `research-provider` | a `SearchProvider` returning `SearchHit`s, registered into the `research` tool's chain (`api.register_search_provider`) and declared under `search_providers:`, no tool of its own; its test drives a `research` call through the harness (`default` profile) and proves the answer came from it |
| `skill` | a `flavor: declarative` plugin: tools declared in the manifest (description, `impl`, typed args), bound to plain functions, no `plugin.py`; the entry point names the package |

The kinds and their one-line summaries are `src/iris_harness/cli/templates/plugin/kinds.yaml`;
each kind is the template tree of that name, laid over `_common/`. The templates are
Python files the stable-tier test holds to the tier as they are (`enforced_roots`), and
`tests/unit/iris_harness/test_cli/test_plugin_scaffold.py` generates every kind, lints it
with its own config and runs its tests in a subprocess (OSS plan L2 exit).

## Writing your first plugin (five minutes)

```bash
mkdir -p ~/.iris/plugins/hello
cat > ~/.iris/plugins/hello/manifest.yaml <<'EOF'
name: hello
version: 0.1.0
description: Answers "say hello" without a model.
provides: [intercept]
EOF
cat > ~/.iris/plugins/hello/plugin.py <<'EOF'
def setup(api):
    def hello(message, *, session_id, span=None):
        if message.strip().lower() != "say hello":
            return None
        return api.services.deterministic_reply(
            message=message, session_id=session_id,
            response="Hello from a plugin.", metadata={"hello_plugin": True}, span=span,
        )
    api.register_intercept("hello", hello, trace_text="hello plugin answered")
EOF
cat > ~/.iris/profile.yaml <<'EOF'
plugins:
  - name: hello
EOF
iris --dump-config          # shows hello under home:...
iris -p "say hello"          # → Hello from a plugin.
```

## Related

- Decision record: `OSS-PLUGIN-HARNESS-PLAN.md` (decisions 1, 3, 4, 5, 8).
- Code: `src/iris_harness/sdk/` (the author-facing facade + `cli`) and `src/iris_harness/runtime/plugin_host/` (api, manifest, registry, profile, loader, dump),
  `src/iris_harness/plugins_builtin/system/`.
- Tests: `tests/unit/iris_harness/test_sdk` + `tests/unit/iris_harness/runtime/test_plugin_host/`, `tests/unit/iris_harness/runtime/test_bootstrap/test_plugin_wiring.py`.
