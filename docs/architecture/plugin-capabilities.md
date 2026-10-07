# Plugins using each other: governed tool calls and declared capabilities — design note

Status: **approved 2026-09-29** (the four decisions as recommended, plus the owner's
permission rule, §4). Measured against `main` at `87f77472`.

## The requirement (owner, 2026-09-29)

If a plugin's tools or services can be useful to another plugin, a workflow, or the core,
the **contract** must make that possible: the harness exposes hooks or interfaces through
which any of them can use, bridge to, or hook into another plugin. It must not depend on
private imports between plugins, and it must not change how the finance workflows behave
today.

Added on approval (owner, 2026-09-29): **config-driven, as little code as possible.** The
harness exposes one interface contract, and every tool the harness uses follows it, so the
model, core code and plugins all use tools under governance. **A plugin cannot call a tool
or an agent directly:** it needs a permission — an allow-list defined by the governance
contract — and the harness enforces it.

## What exists today

| Mechanism | What it gives | Governed? |
|---|---|---|
| **Shared tool pool** — `api.register_tool` | Every plugin's tools join the one governed ReAct loop (`runtime/handlers/react.py:671`), ranked per turn; manifests declare `effect` / `confirm` / `approval` / `guidance` (ADR-0110, ADR-0118). The *model* can use email's `search_inbox` while answering a finance question. | Yes: `PRE_TOOL_USE` / `POST_TOOL_USE`, per-call approval, audit |
| **Event bus** — `api.subscribe` / `api.publish` | Async hand-offs between plugins: `email.classified` → wiki, finance. | n/a (no action taken on the owner's behalf) |
| **Keyed core seams** — `register_provider`, `register_ingest_source`, `register_api_router`, … (`PluginRegistry.add_seam`) | A plugin fills a slot the core defined; the registry records it and attributes failures. | Per seam |
| **Direct Python imports** | Finance imports the email library (`store` ×13, `contracts` ×4, `providers` ×3, `categories`, `agent_tools`). | **No** — a code dependency, not a contract |

## The gaps

1. **Code cannot call a tool through governance.** The governed execution path lives inside
   the loop class (`agent/agentic_core.py`: `_execute_tool` → `_governance_pre_tool` /
   `_governance_post_tool`, the per-call approval path, confirm-once). A heartbeat, a
   deterministic handler, another plugin or a core workflow that wants "email's
   `search_inbox` with these arguments" can only import the function — skipping
   `PRE_TOOL_USE`, approvals and the audit row.
2. **Plugin-to-plugin dependencies are invisible.** `requires:` covers Python, packages and
   env vars only (`plugin_host/manifest.py:37`). Nothing tells the harness that finance uses
   email, so there is no mount-time check, no uniform "email is not mounted, so this
   degrades", and `--dump-config` cannot show the graph.
3. **No typed interface registry.** A plugin cannot publish "my mail-reading service" as an
   interface others (or the core) look up by name. Bridges become ad-hoc hooks or private
   imports — the pattern PR #724 just removed.

## Design

Three mechanisms, one rule for which to use:

- **Tool** — anything that *acts* or reads on the owner's behalf in response to a request.
  Always governed. Usable by the model (today) and by code (new, §1).
- **Capability** — a typed *service interface* one plugin provides and others consume in
  code: data access, lookups, stores (§2). In-process; no model involved.
- **Event** — something *happened*; whoever cares reacts (unchanged).

### 1. Governed tool calls for code

- **One governed tool runner.** Extract the governance half of `_execute_tool` (pre/post
  hooks, per-call approval, argument checks, audit, the fault boundary) into a runner the
  loop and code callers both use. One implementation: a code call cannot be governed
  differently from a model call, for the same reason deterministic answers now pass the
  same response check as generated ones.
- **`services.tools`** on `HarnessServices`, published in the SDK:
  - `call(name, args, *, caller) -> ToolResult` — runs the tool through the runner.
    `ToolResult` carries `ok`, the output text, and when halted, the reason or the
    approval id. `caller` (the plugin, or `core:<workflow>`) lands on the audit row.
  - `describe(name=None)` — the registered tools and their manifest declarations, so a
    caller can check what exists before it calls.
- **The manifest `tools:` block is the contract for code callers too**: `effect`,
  `confirm`, `approval` mean the same thing whoever calls.
- **Effect rules for a code caller** (no chat turn to ask in): `read` and `write` with
  `confirm: never` run; `destructive` and `approval: pinned` go to the approval queue as
  today; `write` with `confirm: once` — see decision 1.

### 2. Declared capabilities

- **The interface lives in the SDK.** A capability is a name (`mail.read`) plus a
  `Protocol` published in `iris_harness.sdk.capabilities`, so provider and consumer agree
  on a stable, versioned shape (part of the SDK stable tier, OSS plan R16).
- **The manifest declares it:**
  ```yaml
  capabilities:
    provides: [mail.read]          # email_workflows
    uses: [mail.read]              # finance_workflows: works without it, degraded
    requires: []                   # not loaded at all without these
  ```
- **The API resolves it:** `api.provide("mail.read", impl)` in the provider's `setup`;
  `api.capability("mail.read")` in a consumer, returning the implementation or `None`.
  The registry checks declarations against registrations (a declared-but-unprovided
  capability shows in the drift report, as tools do today).
- **Mount rules:** a missing `requires` stops the plugin loading, with the reason in
  System Health (as a missing package does); a missing `uses` leaves the plugin running,
  and `api.capability()` returns `None` so the consumer takes its degraded path. System Health shows that plugin
  yellow, naming the capability (`optional capability X unavailable (degraded)`), until a
  provider mounts (derived live, so a late mount or an unmount flips it).
  Providers mount before consumers.
- **Visible:** `--dump-config` and `iris plugins show` print who provides and who uses
  what; provider calls go through the registry guard, so a failure is attributed to the
  provider plugin.
- **The core can consume capabilities too.** Decision D of the email-slice work becomes
  the first core consumer: `credentials.accounts`, provided by the Google connector (and
  later IMAP), consumed by the core's credential health checks — so the core stops naming
  Google.

### 3. Events

Unchanged.

### 4. The permission contract (governance-enforced, config-driven)

- **Every call names its caller.** The governed runner stamps `caller` on the
  `PRE_TOOL_USE` context: `model:<agent>` for the loop, `plugin:<name>` for a plugin's
  `services.tools.call`, `core:<workflow>` for the core.
- **The allow-list is config, not code.** A plugin's manifest declares what it may use:
  ```yaml
  uses:
    tools: [search_inbox, read_email]   # tools it may call from code
    agents: []                          # agents it may hand work to
    capabilities: [mail.read]
  ```
  The harness compiles every mounted manifest into one caller policy; an operator can
  narrow it in `config/governance/tool-access.yaml` (the same override shape as
  `IRIS_GOVERNANCE_ALLOWED_TOOLS` / `BLOCKED_TOOLS`). The model's loop keeps today's
  surface: its tools are the ones on the governed pool, as now.
- **The kernel enforces it.** A `CallerPolicyHook` at `PRE_TOOL_USE` — the pattern
  `PersonaSurface` already uses for coding-agent personas against `persona-policy.yaml` —
  denies a call whose caller is not allowed that tool, with an audit row naming both. It
  sits in the kernel, so no path can skip it; `test_no_bypass.py` gains the code-caller
  path.
- **Capabilities too** (as built in step 3b). A capability method call is governed as the
  pseudo-tool `capability:<name>.<method>`: the same caller stamp, the same
  `CallerPolicyHook` (the grant is the manifest's `capabilities: uses/requires`), the same
  `tool-access.yaml` narrowing (a whole capability or one method), the method's declared
  effect read by the tool policy and the approval hooks, an audit row of metadata and
  digests, and its result redacted in place at `POST_TOOL_USE` before the consumer sees it.
- **Digests are keyed; no key, no call** (owner, 2026-09-30). Every governed call -- the
  runner's tool calls, capability calls, the MCP bridge -- carries `args_digest` (of the
  arguments as written, never the credential broker's rewrite), `result_digest` and
  `digest_alg` (`hmac-sha256/v1/<key-id>`): HMAC-SHA256 under a key HKDF-derived from the
  vault master key (`kernel/governance/audit/digest.py:101,128`). It resolves lazily on the
  first governed call, never at kernel build. With a kernel bound and no key, the call is
  refused before it runs: `GovernedToolRunner.execute` returns `refused`
  (`agent/tool_runner.py:259`), a capability call raises `CapabilityDenied`
  (`tool_runner.py:770`), the MCP bridge raises `PermissionError` (`tools/mcp_bridge.py:483`).
  Chat without tools still answers.
- **Code calls carry the turn's label.** The turn pipeline publishes the turn's label
  (`kernel/governance/turn_label.py`; scoped in `runtime/turn/pipeline.py:137`, seeded by
  `screen`); the harness stamps it on every `api.tools` call (`runtime/tool_service.py:132`)
  and `api.capability` call (`runtime/plugin_host/registry.py:221`). A plugin cannot pass or
  override it. It is a floor: a governed call's `POST_TOOL_USE` lifts it, never lowers it,
  and every model call in the turn is floored on it (`apply_turn_floor`: the loop, the
  shared LLM client, planner, router, compactor, entity extractor; only the cloud
  search-synthesis client reads personal as internal, never `secret`). Outside a turn
  (heartbeats, CLI, the approved-call executor) nothing is stamped.
- **Agents too.** Handing work to another agent (`services.agent_executor`) goes through
  the same check against `uses.agents`.
- **No back door.** An import-linter contract forbids one plugin importing another
  plugin's package, so the governed paths are the only paths. Existing edges go on a
  burn-down ignore list rather than being cut in the same PR. They are debt, not permitted
  dependencies: each is retired before release 1 (rollout step 5).
- **Little code.** The allow-list, the operator override and the tool declarations are
  YAML; the code is one hook, one runner and the manifest schema fields.

### 5. The closure rule (owner's final policy, 2026-09-29)

Every tool attached to the harness follows the rules and contracts **and closes the loop**:
it sits in one connected graph anchored in the harness, and its whole lifecycle completes
with no orphan and no loose end. Closure is not asserted; it is drawn and checked.

| Stage | Closed when | Loose end |
|---|---|---|
| Declared | the manifest declares it (`effect`, `confirm`, `approval`, `guidance`) | registered but undeclared (refused today) |
| Registered | a mounted plugin registers it | declared but never registered |
| Reachable | at least one caller is allowed it (the model's pool, or a `uses:` allow-list) | no caller can reach it |
| Permitted | every allow-list entry names a real, mounted tool / agent / capability | an entry naming nothing |
| Governed | every call passes the one runner (`PRE_TOOL_USE` → run → `POST_TOOL_USE`) | a path that skips the kernel |
| Audited | every call leaves an audit row naming caller and tool | a call with no record |
| Reversible / approved | a destructive tool declares its undo, or is approved per call | neither |
| Observed | outcomes reach health and learning signals | failures nobody sees |
| Removable | unmounting removes the tool and every edge to it | a dangling allow-list, subscription or capability |

The same rule covers everything else that connects: a capability required but not provided
(or provided with no declared consumer), an event published with no subscriber (or
subscribed with none published), an intercept in `intercepts.yaml` with no registered
handler, a seam key filled for a plugin that is not mounted.

**Enforcement: the closure check.** It builds the graph from config (manifests,
`intercepts.yaml`, governance YAML) plus the live registry, fails on any orphan or loose
end, and draws the graph. It runs in CI and as `iris plugins graph`, extending today's drift
report and `--dump-config` rather than adding a mechanism. **Every rollout step's exit test
is a closed graph** for what that step touches.

## No change to finance's workflows (what finance does), but finance never imports email

The model (owner, 2026-09-29): **finance never imports email.** It declares what it
needs — `uses: tools:` for an email tool, or `uses: [mail.read]` for the capability — and
the harness resolves the provider, checks the permission contract and hands over the
governed tool or implementation. No plugin-to-plugin or cross-domain import is an
acceptable dependency.

Finance's existing email-library imports are **debt**, listed on step 3c's burn-down list
so the gate stops new ones. Retiring them is **required before release 1** (rollout step
5): each finance call site moves onto `api.tools` / `api.capability`, one call site per
PR, each pinned by finance's existing tests so what finance does for the owner does not
change. Declaring `uses: [mail.read]` in finance's manifest makes the dependency visible
immediately.

## Rollout (one PR each, suite-gated)

1. The governed tool runner extracted from the loop; the loop uses it. **No behaviour
   change** — pinned by the existing loop and governance tests, plus mutation checks.
2. `services.tools.call` / `describe` in the SDK, on the runner; tests for each effect rule.
3. Capabilities and the permission contract: the manifest `capabilities:` and `uses:`
   blocks, `api.provide` / `api.capability`, the compiled caller policy and
   `CallerPolicyHook`, the operator override file, the plugin-to-plugin import contract
   (with the burn-down list), mount rules, `--dump-config`, drift report.
4. First capabilities: `credentials.accounts` (decision D) and `mail.read` (email provides;
   finance declares `uses`).
4b. The closure check (§5) and `iris plugins graph`, then run against every step above.
   Step 1's exit test, before the checker exists: every tool call path — model loop and
   the runner — reaches the kernel, pinned by `test_no_bypass.py` and mutation checks.
5. **Required before release 1:** retire the burn-down list. Every finance call site (and
   every other listed edge) moves onto `api.tools` / `api.capability`, one call site per
   PR, each pinned by finance's existing tests; the lists end empty.

## Step 1 as built (2026-09-29)

`agent/tool_runner.py`: `GovernedToolRunner` holds the governance half of a tool call —
the per-call approval rule (`approved_per_call`), the plugin's argument check, the approval
card, `PRE_TOOL_USE` (`pre`), fail-closed when a per-call tool is allowed with no approval,
the timeline events, the call, and `POST_TOOL_USE` (`post`). It holds no state between
calls. `AgenticCore._execute_tool` keeps only what a loop has — resolving the model's tool
name, and turning an approval into a halt — and calls `self._tool_runner().execute`; its
`_governance_post_tool` delegates to the runner. No behaviour change: the full suite,
including the approval and refusal tests, is unchanged.

**Payload contract (2026-09-30).** Every tool-hook payload is built by
`kernel/governance/hooks/tool_payload.py` (`pre_tool_payload`: `tool_name` + `args`;
`post_tool_payload`: `tool_name` + `result`), and the runner stamps the tool's declaration
on the `POST_TOOL_USE` metadata (`tool_effect`, `tool_content`, `tool_verify`,
`call_id`, with `tool_call_id` as the same value under its old name). `execute` now runs `POST_TOOL_USE` itself and its `ToolOutcome` carries the
`PostOutcome` (`agent/tool_runner.py`, `post`): a deny withholds the result behind
`governance_block_message`, a transform is what the caller hands on, on the sync and stream
loops, the approved-resume path and both `ToolService` sites; `_governance_post_tool` is
gone. `tests/unit/iris_harness/kernel/test_governance/test_tool_payload_contract.py` pins
every producer to the builders and runs the tool hooks over built payloads.

`tests/security/test_no_bypass.py` follows the governed call into the runner and is
stricter than before: the runner's single `tool.call` is dominated by `self.pre`, `pre`
fires the kernel at `HookPoint.PRE_TOOL_USE` (checked by argument), and the loop's
`_execute_tool` makes no `tool.call` and reaches tools only through the runner. Four
mutations (pre on one branch only, the wrong hook point, a direct call in the loop, a
runner built without the kernel) each fail it.

**Loose end the closure rule exposes:** `PlannerHandlers.handle_brief_config_turn` (the
planner plugin) calls `configure_brief` with a direct `tool.call` — a reviewed exception
(ADR-0103) on the no-bypass baseline, and a path that skips `PRE_TOOL_USE`. Step 2's
`services.tools.call` closes it.

## Step 2 as built (2026-09-29)

`runtime/tool_service.py`: `ToolService` runs a registered tool through the governed
runner — `PRE_TOOL_USE`, the approval rules, the call, `POST_TOOL_USE` — and returns a
`ToolResult` (`ok`, `text`, `held`). `describe()` returns each tool's declaration
(`ToolInfo`: effect, confirm). `HarnessServices.tools` holds it; `sdk.tools` publishes
`BoundTools`, `ToolInfo`, `ToolResult`.

**Refinement of §1: callers are bound, not named.** The note had
`services.tools.call(name, args, *, caller)`. A caller the plugin passes itself could claim
to be anyone, and the permission contract (§4) allows or denies by caller — so a plugin gets
`api.tools`, a `BoundTools` the harness binds to `plugin:<name>`, with no way to change it;
the core asks `services.tools.for_caller("core:<workflow>")`. The runner stamps the caller
on the `PRE_TOOL_USE` context (`metadata["caller"]`, `model:<agent>` for the loop).

**Effect rules, as the kernel already enforces them:** `read` and `write` with
`confirm: never` run; a `confirm: once` write is turned back by the tool policy; a
destructive tool or a pinned write is refused by the approval hook, which denies an
approval no run can resume from. All come back `held`; nothing writes silently.
**Decision 1** (queue a code caller's approval and run it once approved) is built below:
"Decision 1 as built".

**Loose end closed:** the planner's brief-config intercept now calls `configure_brief`
through `api.tools` (as `plugin:planner`) instead of a direct `tool.call`, so it passes
`PRE_TOOL_USE` and the operator's tool policy applies to it; the no-bypass baseline has one
entry left, the runner. A related loose end, not tool-shaped, remains: the same handler
imports `iris_personal.finance.dues_filters` (planner → finance), for step 3's import
contract to catch.

## Step 3a as built (2026-09-29): the permission contract for tools

- **Manifest:** `uses: tools: [...]` (`PluginUses`). Only `tools` for now — `agents` and
  `capabilities` join when they are enforced, so the schema declares nothing it does not
  enforce (the closure rule).
- **The grant:** a plugin may call its own tools and the ones its manifest lists
  (`PluginRegistry.caller_denial`).
- **The operator override:** `config/governance/tool-access.yaml` (`deny: {plugin: [tools]}`)
  can only narrow. Unreadable, it denies every plugin call rather than widening.
- **Enforcement:** `CallerPolicyHook` at `PRE_TOOL_USE` (priority 12, before the tool
  policy), registered in every default kernel. `model:` and `core:` callers keep their
  access; a `plugin:` call is checked against the policy `build_runtime` compiles
  (`runtime/tool_access.py`) and registers once plugins have mounted
  (`kernel/governance/caller_policy.py`, the same registered-from-above seam as identity
  redaction). No policy loaded: plugin calls fail closed, with that reason.
- **Closure:** a grant naming no registered tool is drift (`plugin_uses` surface, one-sided:
  an unused tool is normal, a permission for nothing is a loose end).
- **Behaviour today:** no mounted plugin calls another plugin's tool through `api.tools` yet
  (the planner calls its own `configure_brief`), so nothing a user does changes.

## Step 3b as built (2026-09-29): declared capabilities

- **Manifest:** a `capabilities:` block with `provides`, `uses`, `requires`
  (`PluginCapabilities`), separate from the top-level `provides:`, which keeps meaning
  registration kinds. Names are `domain.verb` (decision 2), checked in the manifest; a name
  may hold one role per plugin (a plugin does not consume what it provides, and a need is
  optional or required, never both). §4's `uses: capabilities` is this block's `uses`: one
  place to declare a capability dependency, not two.
- **The catalogue:** static and closed. The Protocols, their `CapabilitySpec`s (the
  Protocol and, for a capability several plugins may provide, its `fan_out`) and the
  `CAPABILITIES` map are defined in `foundation/capabilities.py`; `iris_harness.sdk.capabilities`
  re-exports the same objects and is the stable import path. Defined low because the host
  (runtime) and the core's own consumers (services) must import it, and neither may import
  the SDK above them; nothing is registered at import time, so the host works in a process
  that never imported the SDK (pinned by a test), and the map is a read-only
  `MappingProxyType`. A new capability ships in an SDK release. It publishes `weather.forecast` so far; the
  rest arrive with step 4.
- **Plain data, declared.** Protocol methods return plain data — dataclasses, pydantic
  models, `TypedDict`s, sequences, scalars — never live objects or handles. Each method has a
  `MethodSpec`: its `effect` (`read`, or `write` with `confirm: once|never`, as a tool's
  manifest entry) and the `fields` of its return type that carry text, as path patterns
  (`[].subject`, `[].sender.name`, `""` for a bare `str`), which must be exactly the `str`
  leaves of the type. A method may also declare `sends_to` (`external_service` or
  `search_engine`), as a tool does: where its arguments go when they leave the machine. Every
  call of it then carries that declaration at `PRE_TOOL_USE`, so the owner-PII guards read
  its arguments (`weather.forecast` declares `external_service`, since `location` may be the
  owner's home address). Refused when the spec is defined: a member that is not a method, a
  method with no return type or with `*args` / `**kwargs` / positional-only parameters, and a
  return type the harness cannot see into (`bytes`, `dict`, `Any`, a handle, a fixed-length
  tuple, a recursive type, a dataclass field with `init=False`).
- **The API:** `api.provide(name, impl)` in a provider's `setup`; `api.capability(name)` in
  a consumer, the implementation or `None`. Refused, recorded as the plugin's failure and
  without effect: a `provide` the manifest does not list under `capabilities: provides`, of
  a capability the SDK does not publish, of an implementation missing part of the
  Protocol, or a second provider of a capability with no `fan_out`; a `capability()` not
  listed under `uses` or `requires` (it returns `None`). Declared equals registered.
- **Several providers (decision 3):** `capability()` returns one implementation — the
  provider's, or the spec's `fan_out` over every mounted provider in mount order.
- **The fault boundary:** a consumer holds a facade exposing the Protocol's methods and
  nothing else (any other attribute is an `AttributeError`). Each call goes through the
  registry guard, including what it returns lazily — an async method's coroutine, an async
  generator, a
  generator — so a failure is recorded against the *provider* (`capability:<name>.<method>`,
  a yellow row) and re-raised to the consumer, which owns its degraded path. A provider
  whose `setup` later failed provides nothing, and a facade resolved before it failed
  raises `CapabilityUnavailable` instead of calling it.
- **Governed like a tool (§4).** Every method call runs through the governed runner
  (`GovernedToolRunner.execute_call`, `aexecute_call`, `aexecute_stream`) and the kernel as
  the pseudo-tool `capability:<name>.<method>`, with the canonical tool keys (`tool_name`,
  `args`, `result`) the kernel's hooks read:
  - **Caller.** Bound by the harness when the facade is built: `plugin:<consumer>` from
    `api.capability`, `core:<module>` from `PluginRegistry.capability_for_core`; never named
    by the consumer. A plugin has no public way to reach either: `PluginAPI`'s registry
    is private, and the `services.tools` it sees is a `ToolCatalogue` (`describe` only),
    not the `ToolService` that can bind any caller -- `api.tools` and `api.capability`
    are its only bound entries. **The in-process trust limit:** a `trust: in-process`
    plugin shares the harness's interpreter, so this is a contract, not a sandbox;
    reaching into private attributes or importing runtime internals is unsupported (and,
    for this repository's code, forbidden by the import-linter contracts). `trust: mcp` is
    the real boundary.
  - **Permitted.** `CallerPolicyHook` grants a `plugin:` caller the methods of a capability
    its manifest lists under `capabilities: uses/requires`; `tool-access.yaml` can take a
    capability away whole (`capability:mail.read`) or one method at a time
    (`capability:mail.read.search`), deny-only, and unreadable it denies every plugin call.
    The operator's global allowed-tools list does not apply to `capability:` names.
  - **Effect.** The tool policy and the approval hooks read the method's declared effect as
    they read a tool's: a read, or a `confirm: never` write, runs; a `confirm: once` write is
    refused with `require_approval` (Decision 1). Nothing is queued: the approval executor
    for code callers (`ToolService.execute_approved_call`, "Decision 1 as built") runs
    approved *tool* calls only, and a capability call stays refused until it is extended to
    them. A capability call carries no `deferred_executor` stamp, so the tool policy's
    confirm-once rule, not the queue, answers it.
  - **Redacted in place.** `POST_TOOL_USE` gets the result's text fields as
    `payload["fields"]` (`{concrete path: text}`, e.g. `[0].body`) beside their joined text
    in `payload["result"]`. `CapabilityRedactionHook` (priority 5, `capability:` calls only)
    masks the owner's secret-shaped identity literals — the ones the response check refuses
    in an answer, read through `identity_redaction.identity_texts()` — and the runner writes
    the final context's field map back into a *copy* of the typed result. The copy rule:
    containers are rebuilt through the type's own copy path — `dataclasses.replace` (frozen
    dataclasses are copied, never mutated), pydantic `model_copy(update=...)`, a new dict /
    list / tuple — so the consumer never holds the provider's object. As for a plain tool
    (below), the final context is used: the transform reaches the consumer, and a
    `POST_TOOL_USE` deny withholds the result.
  - **Strict values.** A result is walked along its *declared* type, never its own shape:
    a subclass of a declared dataclass or model, pydantic extras, `TypedDict` keys the
    type does not name, stray instance attributes, a `str` where the type says `int`, a
    list where it says tuple, text at an undeclared path, or an object of any other kind
    is refused (`CapabilityDenied`) rather than passed through unredacted. Leaves are exact
    too: a subclass of `int` or `datetime` (which can carry attributes) is refused, and
    numbers are exact -- an `int` in a `float` field, or a `bool` in an `int` field, is
    refused (declare `int | float` where either is meant). The copy is rebuilt from the
    declared fields alone -- a dataclass as `object.__new__` plus `object.__setattr__` per
    field, so no `__init__` (hand-written or not) runs in the consumer's path; pydantic
    `model_construct` -- leaves fresh, and a
    pydantic model declaring private attributes is refused at spec time. The copy is built
    *without* re-validation (masked text may not satisfy a validator), so records must be
    plain data with no construction hooks: a dataclass with a `__post_init__` or any
    `InitVar`, or a model with a `model_post_init`, is refused at spec time -- those would
    run again on masked values, twice per call. Any other failure building the copy reaches
    the consumer as `CapabilityDenied`. No code may run on access either: refused at spec
    time are a field name that is a data descriptor (a slot member excepted), an overridden
    `__getattribute__` / `__getattr__` / `__setattr__` (a frozen dataclass's own guard
    excepted), a pydantic computed field, and a property that may return text. A provider
    field that cannot be read (a slots value built bare, a `default_factory` never run) is
    refused rather than raised raw.
  - **Result-rewriting hooks rewrite the field map.** The map is what reaches the
    consumer, so a `POST_TOOL_USE` hook that redacts text must rewrite
    `payload["fields"]` (and `result` to their join). `PromptGuardRetrievedHook` does so
    when a field map is present. A final `result` that disagrees with the join of the
    final map is refused, naming the hooks that transformed the payload (the kernel now
    records them in `metadata["transformed_by"]`).
  - **Streams.** Governed at the call (an async stream: when iteration starts), then
    `POST_TOOL_USE` per yielded item (each item redacted), then once more at the end --
    also when the consumer stops early or the provider raises (`stream_partial`). There
    is no sync method returning an awaitable: its `PRE_TOOL_USE` would have to run
    synchronously inside the event loop it is awaited in; such a method is `async def`.
  - **Audited.** Each hook firing's row names the caller, the provider, the capability, the
    method and the decision, with a digest of the arguments and of the result — never their
    text (the kernel's audit whitelist gains only these metadata keys).
  - **Fail closed.** No kernel bound (`PluginRegistry.bind_kernel`, done by `build_runtime`)
    means no call; a sync method called inside a running event loop cannot be governed
    (the kernel is async) and is refused — the async method is the way from async code. A
    denial raises `CapabilityDenied`, a `CapabilityUnavailable`, so a consumer's degraded
    path catches both.
  - **No bypass.** `tests/security/test_no_bypass.py` pins facade → runner → kernel: the
    guard hands the provider method to the runner and never calls it, each runner entry
    point calls the provider only after its `PRE_TOOL_USE` step, and the POST steps return
    the kernel's result.
- **Mount rules:** plugins mount in profile order except that a capability's providers
  mount before its consumers (`_mount_order`; in a cycle, the first remaining plugin in
  profile order goes first, logged). A `requires` no mounted plugin provides keeps the
  plugin unloaded — `FAILED` with "required capability not provided: …", the red System
  Health row a missing package gets. A missing `uses` loads the plugin, and
  `capability()` returns `None`.
- **Closure:** two drift surfaces. `plugin_capabilities` compares each mounted plugin's
  declared `provides` with what it provided (`plugin:capability`, two-sided);
  `capability_uses` lists a capability mounted plugins use that nothing provides
  (one-sided, like `plugin_uses`). A provided capability with no declared consumer is not
  drift yet: the core consumes capabilities too (step 4) without a manifest, so the
  "no consumer" edge waits for step 4b's closure check.
- **Visible:** `--dump-config` prints, per capability, who provides, uses and requires it
  (from manifests, before boot); `GET /plugins/{name}` and `iris plugins show` print each
  plugin's side — what it provides and who uses it, what it uses and who provides it — and
  a declared-but-unprovided capability under drift.
- **Closure (§5) for capabilities:** declared (manifest `capabilities:` and each
  `MethodSpec`), registered (`api.provide`, drift otherwise), permitted (the caller policy),
  governed (the runner and the kernel, pinned by `test_no_bypass.py`), audited (one row per
  hook firing, metadata and digests).
- **Behaviour today:** no manifest declares a capability and the SDK publishes none, so the
  mount order, every plugin and every audit row are unchanged.

## Step 3c as built (2026-09-29): no plugin-to-plugin imports

§4's "No back door", as import-linter contracts in `pyproject.toml` (run by
`poetry run lint-imports`, so by `scripts/ci_local.sh` and `.github/workflows/ci-linux.yml`
with the other contracts). It only pins: no import was removed. The pinned edges are
debt, not permitted dependencies; step 5 retires every one before release 1.

- **What a plugin is.** A package under `iris_harness/plugins_builtin/` or
  `iris_personal/plugins/`. `iris_personal/{email,finance,filemanager,calendar}` are
  domain *libraries*, not plugins; a plugin may import its own. Domains: email = library +
  `connections` + `email_workflows` + `gmail`; finance = library + `finance_workflows`;
  files = library + `file_organizer`; calendar = library + plugin; `planner` has no
  library. `connections` (the Google reconnect routes) is part of the public email slice
  (R2): it reads email's account store, so calendar and file_organizer reach that store
  through it — debt on the list, retired via `credentials.accounts`. `market` is a shared
  library no plugin owns, so importing it reaches no other domain.
- **`plugins-are-independent`** (independence, both roots by wildcard): no plugin imports
  another plugin. The public tree keeps the `plugins_builtin.*` half.
- **`<domain>-imports-no-other-domain`** (forbidden, direct imports): a domain's library
  and plugins import no other domain's library. Finance's email imports are on the list
  as debt to retire before release 1. **Email's list is empty and checks indirect chains too**: email never
  imports finance, or any other domain, by any route.
- **`libraries-import-no-plugin`** (empty): a library never imports a plugin.
  `plugins-use-the-public-api` now also forbids `iris_personal`.
- **Coverage.** `tests/unit/test_import_contract_coverage.py` fails when a new plugin or
  library is in no domain, or a domain contract does not forbid a library.

**The burn-down list: 53 exact pairs of debt, all retired before release 1 (step 5).
Each group names what retires it:**

| Edge | Count | Debt to retire before release 1 via |
|---|---|---|
| `email_workflows.cli`, `finance_workflows.cli` → `plugins.gmail.provider` | 2 | `mail.read` (step 4): the CLI resolves the capability instead of building Gmail's provider |
| `finance_workflows` → `email.{store,contracts,providers,categories}` | 18 | `mail.read` (step 4), call sites moved in step 5 |
| `finance_workflows.handlers` → `email.agent_tools` | 1 | `api.tools` under `uses: tools:` (steps 2 + 3a, built) |
| `finance.registry` → `email.store` (`domain_of_address`) | 1 | move the pure helper beside finance's registry |
| `finance.custody_migration`, `finance_workflows.{extract,ingest}` → `filemanager.custodian` | 3 | `files.custody` — **deferred to release 2** (owner, 2026-09-30) |
| `finance_workflows.plugin` → `calendar.store` | 1 | `calendar.read` (**not yet in the plan**) |
| `filemanager.gdrive_oauth`, `file_organizer.{plugin,seams}`, `calendar.{cli,gcalendar_oauth,plugin}` → `email.accounts` | 6 | `credentials.accounts` (decision D, step 4) |
| `file_organizer.plugin`, `calendar.plugin` → `connections.google` | 2 | `credentials.accounts` (decision D, step 4) |
| `planner` → `email.{store,digest}` | 8 | `mail.read` (step 4) |
| `planner` → `calendar.store` | 4 | `calendar.read` (**not yet in the plan**) |
| `planner` → `finance.{store,bills,dues_filters}` | 7 | `finance.dues` (**not yet in the plan**) |

**Open follow-ups (not committed scope).** Three pointers above name capabilities no
rollout step delivers: `files.custody`, `calendar.read` and `finance.dues`. They are
names in decision-2 form for the edges they would retire, not yet designed; each needs its
own design entry before a PR. The edges they cover are still debt to retire before
release 1.

**Amended (owner, 2026-09-30): `files.custody` and the skills rule move to release 2.**
Both edges `files.custody` would retire run between finance and filemanager, and R1 keeps
both domains private until release 2, so nothing in the release-1 export depends on them.
The three pairs stay on the burn-down list and the gate still stops new ones. The skills
rule (cross-domain calls from `config/skills/*/tools.py` go only through the harness as
caller `skill:<name>`, under a `uses:` allow-list, with an AST guard and burn-down) is
deferred for the same reason. The rest of step 5, `calendar.read` and `finance.dues`
included, is still required before release 1.

**L3 export item.** The email contract, `libraries-import-no-plugin` and the
`iris_personal` line in `plugins-use-the-public-api` all sit inside private export
fences, so the public tree (which ships the email slice) will need a public email
contract of its own. The coverage test must then skip per missing domain, not, as today,
per missing `iris_personal` root. Recorded here; no code in this step.

Deliberate breakage, then reverted: a new finance → email-workflows import, a new
finance → email-library import, email library → finance, `connections` → finance,
builtin `research` → builtin `system`, `market` → `planner` and builtin `research` →
the email library each failed the gate.

## Decision 1 as built (2026-09-29): a code caller's approved call runs once

A code caller's `confirm: once` write, pinned write or destructive call is **queued**, not
refused, and the harness runs it once the owner approves. Owner's answers: the caller and
the claim are two nullable columns on the queue; `ToolResult` carries the approval id; no
inline terminal prompt for these rows; and the caller hears the outcome on an event.

- **Queued.** `ToolService` builds every code call as `ToolCall(deferred=True)`; the runner
  stamps `deferred_executor` and `per_call_approval` on the `PRE_TOOL_USE` context (the
  harness's stamp — no caller can set either). For a deferred call, `confirm: once` is
  approved per call like `approval: pinned` (`approved_per_call_for`), so the tool
  policy's ask-first rule steps aside and `DestructiveApprovalHook` enqueues the row with
  its `caller` — instead of refusing, as it still does for a run that cannot resume. The
  plugin gets `ToolResult(held=True, approval_id=...)`, whose text names the approval in
  no one surface's words ("Queued for approval <id>; ..."). The loop is unchanged: its
  `confirm: once` write is still turned back with "ask first".
- **Stored.** `approval_queue` gains `caller` and `executed_at` (nullable, migrated like
  `session_id`); `ApprovalRow.is_deferred_call` is a row with a caller and pinned items.
  `respond()` is now a conditional `UPDATE ... WHERE status='pending'` checked by
  rowcount, so two answers racing across processes cannot both land.
- **Run once, governed.** `respond_to_approval(..., executor=)` hands an approved deferred
  row to the executor, `ToolService.execute_approved_call`, which runs the pinned call
  through the same `GovernedToolRunner` as the caller (`approved_by` = the approval): the
  caller policy is re-checked at execution (an operator narrowing after the ask wins),
  the tool policy, then the approval hook verifies the row (approved, this exact call,
  this caller) and **claims** it — `claim_execution`, one conditional write — so a
  double approve, a replay or a concurrent executor is denied. `POST_TOOL_USE` follows.
  Approving with no executor present does not record the answer: it stays pending, so
  no approved row is left that nothing will run.
- **Never run.** Rejected: nothing runs. Expired: `sweep_expired(executor=)` (the timeout
  heartbeat passes the runtime's) settles it. Every lapse notice — the in-chat one when no
  executor is here, and each channel's timeout message — says the call was not run
  (`ApprovalRow.lapse_consequence`), never "the run stays halted", which describes a run a
  code call does not have.
- **Told.** Every outcome (`ran`, `failed`, `denied`, `rejected`, `expired`) is audited
  against the approval (`call_<status>`, with caller and tool; the claim is audited
  too), returned in the surface's `detail`, posted as a notice into the conversation the
  call came from, and emitted as `approval.call_completed`
  (`ApprovalCallCompletedPayload`: approval id, caller, tool, `status` — a `Literal` of
  the five — and a display-masked, truncated summary). A plugin hears only its own calls
  with `api.on_approved_call(h)`, a subscription recorded on the registry like any other.
  `api.subscribe` and `api.publish` refuse the topic outright, and so does
  `services.events`, which plugins get as a `GuardedEventBus` view refusing it on every
  verb (`on`, `off`, `emit`, `emit_sync`) — so no plugin reads another's summaries or
  forges an outcome. The refusal is one table, `runtime/plugin_host/harness_topics.py`
  (`HARNESS_TOPICS`), that both use; the harness's own emitter keeps the raw bus. The SDK (`sdk.tools`) publishes the topic and payload.
- **Every surface.** API `POST /governance/approvals/{id}/respond` (the web UI and Action
  Center, and the gateway's Telegram through `ApiApprovalBackend`), `iris approvals
  approve|reject`, and the runtime's in-process
  Telegram poller (`LocalApprovalBackend`) all pass `runtime.tool_service` as the
  executor. `CLIChannel` does not prompt inline for a deferred row — a yes there would
  run inside the call's own `PRE_TOOL_USE` — and prints `iris approvals approve <id>`.
  One test per surface: `tests/unit/iris_harness/runtime/test_runtime/test_approved_call_executor.py`.
- **The CLI answers through the server.** For a deferred row, `iris approvals` posts to
  the API endpoint (as `ApiApprovalBackend` does), so the call runs in the server and its
  `approval.call_completed` reaches the plugin that asked, there. Only when the API is
  unreachable does it build a local runtime, and it says the event then stays in that
  process. `--no-resume` on a deferred row answers nothing: approving would mean running,
  so the row stays pending and the CLI says so.
- **Loose end (closure rule):** no mounted plugin makes a gated code call or subscribes to
  `approval.call_completed` yet, so in a production profile the event has a producer and
  no subscriber. The closure check (step 4b) should list it until the first consumer
  lands; it is not suppressed. The operator's `IRIS_GOVERNANCE_CONFIRM_ONCE_TOOLS`
  override by name is not escalated for code callers: such a call is still held with no
  approval id. And a call denied at execution *before* the claim (the caller policy or
  the tool policy refused it) leaves its row `approved` with `executed_at` NULL: audited
  as `call_denied` and announced, but in the store it looks like an approval that never
  ran. The closure check (4b) should flag such rows.

## Decisions (owner, 2026-09-29: all as recommended, plus §4)

1. **A code caller and `confirm: once` writes.** There is no chat turn in which to ask.
   Recommendation: route them to the approval queue like `approval: pinned` — the owner
   approves once per call, nothing writes silently.
2. **Naming.** Recommendation: `domain.verb` (`mail.read`, `credentials.accounts`), and a
   capability's `Protocol` versioned with the SDK.
3. **One provider or many.** Recommendation: a capability may have several providers
   (Gmail and IMAP both provide `mail.read`); `api.capability()` returns one
   implementation that fans out to all of them, so consumers never iterate providers.
4. **Scope of the first slice.** Recommendation: rollout steps 1–4, which also deliver
   decision D. **Amended (owner, 2026-09-29):** step 5 is not optional — it is required
   before release 1 (see "No change to finance's workflows"). **Amended again (owner,
   2026-09-30):** except `files.custody`, deferred to release 2 (see the burn-down list).
