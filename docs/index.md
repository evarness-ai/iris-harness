# IRIS

IRIS is a governed agent harness for small local models. It runs an agent loop
(classify, plan, act with tools, curate the answer) on models that fit on a laptop,
sends every model call and every tool call through a governance kernel you cannot
unplug, and lets you answer a turn without a model at all when the answer can be
computed. Everything outside the loop and the kernel is a plugin.

The email assistant that ships with it is the proof: a private inbox assistant that
reads, sorts and labels your mail on a local model, writes nothing until you approve
it, and leaves an audit row for every call.

[Get started](getting-started/install.md){ .md-button .md-button--primary }
[Write a plugin](guides/write-a-plugin.md){ .md-button }

## Three minutes, no credentials

```text
uv tool install "iris-harness[email]"   # or: pipx install "iris-harness[email]"
iris doctor                             # can IRIS run here, and what to fix
iris email demo                         # a synthetic mailbox, a scripted model, no network
iris email setup                        # then your own mailbox, step by step
```

## What you can build

| Use case | What IRIS gives you |
|---|---|
| **A private email assistant** | Triage, a daily digest and labels on a local model. Personal data never reaches a cloud model; nothing changes in your mailbox until you approve it once per account. |
| **Your own domain agent** | A plugin registers tools, deterministic handlers and an agent on the same governed loop the email assistant uses ([example](guides/examples.md)). |
| **Proactive routines** | Heartbeats run your jobs on a schedule and deliver to a channel (Telegram, web push, the web console). |
| **Governed research** | One `research` tool over a provider chain (SearXNG, Tavily, Brave, DuckDuckGo), cached and reranked, behind the egress gate. |
| **Safe code execution** | A bounded model-to-sandbox loop that runs shell commands in a container. |

## Why a harness, not a bigger model

- **Determinism is a primitive.** When an action is grounded and parseable, parse it
  and call the function. On one measured action, a 7B model called the right tool 25%
  of the time and a 35B model 0%; a deterministic handler did it 100% of the time. IRIS
  lets a plugin answer such turns with no model, and governs those answers like
  generated ones ([deterministic handlers](concepts/deterministic-handlers.md)).
- **Governance is a passage, not a filter.** Data is classified before it moves;
  personal and secret data stays on local models; secrets reach tools as vault handles,
  never as prompt text; destructive actions wait for your approval
  ([governance](concepts/governance.md)).
- **Every call is audited.** Each model call, tool call and answer writes a ledger row,
  and the [proof bundle](reference/proof-bundle.md) lets a CI job verify the email
  assistant's invariants offline.
- **The model-free guards hold.** In an adversarial battery of 14 probes against the
  full guard stack, 11 were fully defended; the misses were in model-based judges, and
  the deterministic guards held. That result is why every answer, generated or not,
  now passes the model-free guards.

## Where to go next

- [Install](getting-started/install.md), then [try the demo](getting-started/demo.md)
  and [connect your mailbox](getting-started/connect-your-mailbox.md).
- [Write a plugin](guides/write-a-plugin.md) and [test it](guides/test-your-plugin.md)
  against the [stable API](reference/stable-api.md).
- [Architecture](concepts/architecture.md) for how the pieces fit.

IRIS is licensed under the Apache License 2.0, and it stays that way. It runs on
macOS (Apple Silicon) and Linux; on Windows, use WSL2. It collects no telemetry.
