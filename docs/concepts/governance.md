# Governance

The governance kernel (`src/iris_harness/kernel/governance/`) is a mandatory passage:
every model call and every tool call goes through it, whether the core, an agent or a
plugin makes it. It cannot be unplugged, and a plugin has no way around it.

## Hook points

The kernel runs its checks at fixed points in a turn, each check a small plugin of the
kernel with a priority:

| Hook point | When | Typical checks |
|---|---|---|
| `pre_turn` | A user message arrives, before anything acts on it | Classification, the input safety screen |
| `pre_classify` | Text is about to enter a model | Data classification |
| `pre_llm_call` | Before each model call | The egress gate, the cost limiter |
| `pre_tool_use` | Before each tool call | Tool policy, approvals, the filesystem jail, network and command allowlists, vault handles |
| `post_tool_use` | After each tool call | The side-effect ledger, scanning external content |
| `post_step` | After each step of the loop | The evaluator: loops, goal drift, the step cap |
| `pre_response` | Before an answer ships | Credentials, identity secrets, internal details |

Each check's decision (allow, deny, or ask the owner) is written to the audit ledger.

## Data classification and the egress gate

Everything that moves is classified:

| Class | Meaning | Where it may go |
|---|---|---|
| `public` | Freely shareable | Any tier |
| `internal` | Not personal, not for publishing | Local tiers; the cloud tier when you opt in for that kind of work |
| `personal` | Your data: profile, mail, notes | Local tiers only, never the cloud |
| `secret` | Credentials, keys, vault values | Never in a prompt |

The egress gate enforces the table on every model call. Email is `personal`, so the
email assistant runs on local models by construction.

## Secrets as handles

Credentials live in a Fernet-encrypted vault under one master key (the one
`iris doctor` checks). A tool that needs a secret declares it, and receives it at
`pre_tool_use` through a `vault://` handle: the value never appears in a prompt, a
transcript or a log line. Outbound text is also scanned and redacted.

## Effects, approvals and the side-effect ledger

A plugin's manifest declares each tool's effect, and the kernel enforces it:

- `read` tools run freely.
- `write` tools ask the owner once before the first write of a run.
- `destructive` tools wait for the owner's approval on every call.

Approvals are durable rows in an approval queue. The owner answers in the CLI
(`iris approvals`), in chat, or in the web console's Actions screen, and the halted run
resumes from a checkpoint with the approved call, governed again. Mailbox writes are a
standing approval per account (`iris email writes approve`). Every side effect a tool
reports lands in the side-effect ledger, which is on by default. A destructive call or a
pinned write gets its ledger row before it runs, so a crash mid-call still leaves a
record for `iris run resume` to check; if that row cannot be written, the call is denied
and nothing runs.

## Run limits

An evaluator watches each run for loops, goal drift and a runaway step count, and a
per-run cost ceiling stops spend on paid tiers. The evaluator can run out of process,
so the agent it watches cannot influence it.

## Threat detection

Retrieved content a third party wrote (a web page, an email) is marked `external` and
scanned for injected instructions before the model reads it. Optional model-based
guards (Prompt Guard on input, Llama Guard on output) and response judges
(faithfulness, grounding) add a second layer; each ships off or in shadow mode first.
The model-free checks are the floor every answer passes, whatever is turned on.

## The audit ledger

Every check writes a row: the hook point, the check, the decision and why, the run,
step and session, the data class and the tier. Recent rows stay in SQLite; older ones
move to a compressed Parquet archive you can query. `iris audit` and the web console's
Governance and Call trace screens read it, and the
[proof bundle](../reference/proof-bundle.md) turns it into evidence a CI job verifies
offline.
