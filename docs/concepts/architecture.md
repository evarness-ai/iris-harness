# Architecture

IRIS is a harness: a small set of primitives around a model loop. The loop and the
governance kernel are the core; everything else is a plugin on the SDK.

```text
  CLI · web console · Telegram · MCP clients
                    │
              IRIS API (:8003)
                    │
  ┌──────────────── turn pipeline ────────────────┐
  │ input screen → deterministic handlers ───┐    │
  │      │ (no handler claimed the turn)     │    │
  │      ▼                                   ▼    │
  │ IntentRouter → TaskPlanner         model-free │
  │ → ReAct loop → AgentExecutor         guards   │
  │ → curator (model-free checks + judges)   │    │
  │      ▼                                   ▼    │
  │      record: session log + audit → answer     │
  └───────────────────────────────────────────────┘
        │ every model call, every tool call
        ▼
  governance kernel: classify · egress gate · vault · tool policy
                     approvals · evaluator · audit ledger
        │
  tiers: router model · local tiers (Ollama, LM Studio) · opt-in cloud tier
```

## The turn

Every message is one turn through one pipeline (`src/iris_harness/runtime/turn/`), the
same for the REPL, the API, streaming and every channel:

1. **Input screen.** The message is classified and screened before anything acts on it.
2. **Deterministic handlers.** Plugins' intercepts see the message in order; the first
   that claims it answers with no model ([deterministic handlers](deterministic-handlers.md)).
3. **The agentic loop**, when no handler claimed the turn: the `IntentRouter` classifies
   the intent and picks a model tier; the `TaskPlanner` splits a compound request into
   tasks only when they need different agents; the ReAct loop alternates thought,
   tool call and observation until it has an answer or hits its step or cost cap; the
   `AgentExecutor` runs the agent the intent names.
4. **Curate or guard.** A generated answer goes through the curator: the model-free
   checks (credentials, identity secrets, internal details) and the model-based judges
   you turn on (faithfulness, grounding, output safety). A deterministic handler's
   answer goes through the same model-free checks. Each check writes an audit row.
5. **Record.** The answer goes into the session log through one recording path,
   however it was produced.

## Model tiers

`config/llm_tiers.yaml` names the models: a small router model, local tiers for most
work, and an optional cloud tier. Routing follows the intent and the data: personal or
secret data never leaves the local tiers, whatever the intent asks for. The cloud tier
is off until you configure it, and even then it sees only data the egress gate lets
out ([governance](governance.md)).

## Services

| Service | Port | Module |
|---|---|---|
| IRIS API: chat, streaming chat, the MCP server, the web console and its data | 8003 | `iris_harness.server.iris_api.main:app` |
| Governor: the HTTP front of the governance kernel | 8080 | `iris_harness.server.governor.main:app` |
| Evaluator (out-of-process, optional) | 8090 | `iris_harness.server.evaluator.main:app` |
| Channel gateway (Telegram) | 8006 | `iris_harness.server.channel_gateway.main:app` |

The CLI commands in the [quickstart](../getting-started/install.md) run in-process and
need none of them; the scheduled jobs and the web console run in the IRIS API.

## Background work

Heartbeats are scheduled jobs (the email sweep and judge are two). They run on one
scheduler inside the IRIS API, under the same governance as a turn, and report each run
to System Health. An event bus carries work that finishes outside a turn to whoever
subscribed, and channels deliver the results.

## Memory

IRIS remembers in **memris**, a graph of claims with provenance (`src/memris/`): what
was said, by whom, from which source, with what confidence. Capture gates keep
implausible or ungrounded facts out of the store, a recall filter keeps low-confidence
ones out of the prompt, and every change is reversible.

## Where the code is

| Path | What lives there |
|---|---|
| `src/iris_harness/runtime/` | The composition root, the turn pipeline, the plugin host. |
| `src/iris_harness/agent/` | The agentic core: router, planner, ReAct loop, executor, curator. |
| `src/iris_harness/kernel/governance/` | The governance kernel. |
| `src/iris_harness/sdk/` | The plugin SDK: the stable surface plugins import. |
| `src/iris_harness/plugins_builtin/` | The reference plugins: system, channels, research, code execution. |
| `src/iris_personal/` | The email assistant: the mail record and the email plugins. |
| `src/iris_harness/server/` | The four HTTP services. |
| `webui/` | The web console (React). The released package carries a production build. |
