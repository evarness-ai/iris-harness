# Deterministic-path parity — design note

Status: **approved 2026-09-29** (A, B, C as recommended, below). Rollout step (a) in progress. Tracks OSS plan R15 (launch blocker), public issue
evarness-ai/iris-harness#1, private detail #713. Measured against `main` at `90645a49`.

## The claim we want to make

"Determinism is a first-class primitive, and it is governed." A deterministic handler
(`api.register_intercept`) answers a turn with no model: parse, call typed tools, return a
grounded answer. Every answer IRIS gives, deterministic or generated, must pass the same
model-free guards and leave the same audit trail.

## What the code does today

Four facts, each checked in the source:

1. **An intercept hit ends the turn before `curate`.** `runtime/turn/pipeline.py:120`:
   `if state.intercepted: break`. `curate` is `ResponseCurator.curate` →
   `_run_pre_response_judges` (`agent/response_curator.py:496`), which is where every
   response check lives: the deterministic `_judge_safety` (credential/SSN/dangerous
   patterns, identity-secret egress, architecture disclosure, dump arbitration), schema,
   consistency, faithfulness, grounding, the Llama Guard `output_safety` guard and the
   escalation shadow judge. Each writes an audit row via `_audit_signal`. An intercept
   answer gets none of it; only display masking (`mask_text`, pipeline.py) applies.
2. **Input governance is tied to LLM calls, not to turns.** `HookPoint.PRE_CLASSIFY`
   fires inside the governed client (`llm/client.py:778`), the intent router
   (`agent/intent_router.py:329`), the compactor and the entity extractor — i.e. only when
   text is about to enter a model. The data classifier (`kernel/governance/plugins/classifier.py`,
   regex packs) and the inbound Prompt Guard hook are registered there. A deterministic
   handler that calls no model is never screened. This is the 2026-07-06 red-team case: a
   self-harm message was captured by a keyword intercept and answered with no guard at all.
3. **`HookPoint.PRE_RESPONSE` is declared and never fired.** It exists in
   `kernel/governance/hooks/types.py:27`; no call site fires it. The curator runs its judges
   directly instead of through the kernel, so the kernel has no response-side hook point
   that any path uses.
4. **Two recording paths.** `record` (stage 8) is skipped on an intercept hit too; the
   intercept dispatcher logs the answer itself (`runtime/intercept_dispatch.py:190`,
   `log_agent_response`). Same outcome today, but two writers that can drift.

**Root cause (one sentence):** governance is attached to *model calls* and to the
*generated-answer stage*, while the harness's own unit of work is the *turn*, and a turn
can be answered without either.

## Design

Move governance to the turn boundary, where both paths meet. Three changes, one principle:
a stage says which turns it serves; nothing `break`s out of the pipeline.

### 1. Turn-level input screen (new stage `screen`, before `intercept`)

- Fire the kernel once per turn on the raw user message at a new hook point
  **`PRE_TURN`** (see open question A). The data classifier, the inbound Prompt Guard (when
  enabled) and any input threat hook register for it.
- Outcome handling mirrors the kernel's existing verbs: `deny` ends the turn with the
  governance refusal text and an audit row; `require_approval` goes to the approval queue
  (already wired, M5.C6); `transform` rewrites the message the rest of the turn sees;
  `allow` continues.
- The classification result (`public/internal/personal/secret`) is stored on `TurnState`
  so deterministic handlers and the generated path see the same label. Today a handler
  cannot know the turn is `secret`-classified.
- The existing per-LLM-call `PRE_CLASSIFY` stays as is: it screens every text that enters
  a model (retrieved context, tool output), which is a different job.

### 2. Split `curate` into `guard` (always) and `judge` (generated only)

- **`guard`** — model-free, runs on every answer:
  the deterministic half of `_judge_safety` (patterns, identity-secret egress, architecture
  disclosure; dump arbitration keeps its fail-closed behaviour: with no leak judge it halts),
  the ReAct-scaffolding and tool-instruction strippers (harmless on a deterministic answer),
  and one audit row per signal. Fired through the kernel at the existing
  **`PRE_RESPONSE`** hook point, so the point finally has a caller and plugins' response
  hooks (if any are ever allowed) run on both paths.
- **`judge`** — model-based, generated answers only by default: schema, consistency,
  faithfulness, grounding, `output_safety` (Llama Guard), escalation. Faithfulness and
  grounding have nothing to judge on a templated answer built from tool data; that is the
  point of determinism. `output_safety` is the one debatable member (open question B).
- A `guard` halt replaces the answer with the same blocked text `curate` uses today, on
  both paths.

### 3. One pipeline, stages declare their audience

```
screen → intercept → classify → route → plan → execute → judge → guard → record
  all       all       gen       gen     gen     gen      gen     all     all
```

- `TurnState.intercepted` becomes a routing fact, not an exit: generated-only stages
  return immediately when it is set. `pipeline.py`'s `break` goes.
- `record` runs for every turn; the dispatcher's own `log_agent_response` is removed, so
  there is one recording path.
- The turn span and every audit row of an intercept turn carry `deterministic: true` and
  the handler name, so the Governance / Call-trace screens can show "answered without a
  model, guarded, audited".
- `RESUME_STAGES` drops `screen` as well as `intercept`: a resume carries no new user
  message, and the original one was screened when it arrived.
- `guard` runs after `judge` on purpose: `judge` may rewrite the text (retries), and the
  guard must check the text that actually ships.

### What does not change

- The `register_intercept` API and handler contract (R15: keep the name).
- The ADR-0106 continuation shield in the dispatcher.
- Per-LLM-call governance inside the client, and `tests/security/test_no_bypass.py`'s
  hook-before-LLM guarantee.

## Tests (each seen failing first, per the M5.C7 mutation rule)

1. **Invariant, parametrised over every registered intercept in the default and
   personal-assistant profiles:** a handler answer containing an identity-secret literal
   is blocked; every intercept turn writes `pre_turn` and `pre_response` audit rows with
   `deterministic: true`. On `chat` and `chat_stream`.
2. **The red-team case as a regression:** the 2026-07-06 self-harm message, with a
   keyword intercept that would claim it, is screened at `PRE_TURN` before dispatch.
3. **Classification is recorded for the turn:** a `secret`-classified message sets
   `TurnState.classification`. Handlers do not receive `TurnState`, so a handler-facing
   accessor lands with the SDK stable-tier work (public issue #15), not here.
4. **One recorder:** an intercept turn produces exactly one session-log response entry.
5. **Mutation checks:** reintroduce the `break`; skip `screen`; drop the `guard` audit
   write — each must fail a test above.
6. Playground diff green; latency budget: `screen` + `guard` add < 20 ms to a deterministic
   turn with the model-based hooks off (they are regex and literal checks).

## Rollout

One PR per step, each suite-gated: (a) `PRE_TURN` + `screen` stage; (b) `guard`/`judge`
split with `PRE_RESPONSE` fired; (b2) the opt-in input safety screen; (c) remove the `break`,
stage audiences, single recorder;
(d) the invariant + regression + mutation tests land with (a)-(c) as they become true.
No flag: this closes a governance gap, and a flag that turns governance off on one path is
the gap itself.

**Step (a) as built (2026-09-29).** `HookPoint.PRE_TURN`; `GovernanceKernel.register(hook,
at=...)` so one hook instance serves two points; the data classifier and the opt-in inbound
Prompt Guard registered at `PRE_TURN` as well as `PRE_CLASSIFY`; the `screen` stage (logs the
user message, fires `PRE_TURN`, records the classification, refuses on `deny` /
`require_approval`); a refused turn skips to `record` only and never reaches session memory;
`GOVERNANCE_BLOCKED_TEXT` is the one refusal text; `IrisRuntime.governance_kernel` (from
`kernel_from_env()`) is the kernel the pipeline fires on; resumes skip `screen`.

**What step (a) does not do.** The 2026-07-06 self-harm case is not caught by default: the
screen gives every turn the *same* input checks, and the default input checks are the regex
data classifier (credentials, PII) and the opt-in Prompt Guard (injection). Neither detects
self-harm. Catching it at `PRE_TURN` needs an input hazard hook (for example the Llama Guard
classifier the output guard already uses, applied to the user turn).

**Step (b) as built (2026-09-29).** The model-free response check moved out of the
curator into the kernel: `kernel/governance/plugins/response_safety.py` holds
`check_response` (credential/SSN/dangerous patterns, identity-secret literals read through
the kernel's identity-text seam with the issue-0022 URL/email exclusion, architecture
disclosure) and `ResponseSafetyHook` at `PRE_RESPONSE`, registered in `build_default_kernel`.
The disclosure detector moved with it (`kernel/governance/disclosure.py`). The curator
fires `PRE_RESPONSE` (the point's first caller), lets the kernel write the audit row, and
runs `check_response` directly when governance is disabled or the fire fails, so response
safety never weakens with the kernel. Dump phrasing is only flagged by the check; the
curator's leak judge still decides it, failing closed. The guard re-runs on the final text
whenever a judge rewrote it. `build_runtime` builds one kernel and hands it to both the
curator and the pipeline. Deterministic answers reach `PRE_RESPONSE` in step (c).

**Step (c) as built (2026-09-29).** `pipeline.STAGE_AUDIENCE` + `serves()` replace the
`break`: every stage declares the turns it serves (`all` / `generated` / `handled`), and a
refused turn runs `record` alone. A handled turn runs `screen → intercept → guard →
record`. The new `guard` stage calls `ResponseCurator.guard()`, the same kernel-fired
`PRE_RESPONSE` check generated answers pass inside `curate`, with `deterministic: true` and
the handler's name in the payload (both now audited kernel keys) and on the turn span; a
halt swaps in the one refusal text. Every answer therefore passes `PRE_RESPONSE` exactly
once — inside `curate` for a generated answer, in `guard` for a handled one — rather than
the design's single `guard` stage for all turns: moving the generated path's check out of
`curate` would have split it from the dump arbitration and the re-check after a judge
rewrite, which live there. `record` is the one recorder: the dispatcher's own
`log_agent_response` is gone. `TurnHost` gained `response_curator` (21 members).

**Step (b2) as built (2026-09-29).** `kernel/governance/plugins/input_safety.py`:
`InputSafetyHook` at `PRE_TURN` (priority 7) runs the output guard's Llama Guard classifier
on the user's message. `threat-detection.yaml` gained an `input_safety` section; its
`enforce` / `log_only` default to the output guard's lists (`self_harm` enforced).
`_input_safety_from_env` installs it only when `IRIS_GOVERNANCE_INPUT_SAFETY` is truthy
**and** the section is enabled, and logs at WARNING when the flag is on but the screen is
not (disabled section or a failed build), so its absence never looks like it is working.
Shadow mode audits and allows; enforce refuses an enforced or uncategorized hazard and
audits a `log_only` one; a model error lets the turn through. The flag is surfaced where
its siblings are: `.env.example`, `environment-variables.md`, the settings catalog (web
Settings, guards tab), `/governance` flags, `scripts/governance_preflight.py`, the
enforce-flips runbook. Test 2 (the red-team regression) runs with a stub classifier on
both entry points: enforced, a message a deterministic handler would answer is refused
first; in shadow it is answered.

**Decision B as built (2026-09-29).** `InterceptSpec.guard_output`, declared by the plugin
(`api.register_intercept(..., guard_output=True)`) or in `config/intercepts.yaml` —
either side saying yes is enough, the `resolves_confirmation` merge rule. (The decision
said "the manifest"; intercepts are declared at registration and in the chain config, not
in `manifest.yaml`, so the flag lives where the declaration does.) When set, the `guard`
stage also runs `ResponseCurator.guard_output()`: the same Llama Guard judge, verdicts and
fail-open-with-banner as `curate`, audited with the deterministic marker; `skipped` when
`IRIS_CURATOR_OUTPUT_SAFETY` is off. `governance_warning_banner` is the one banner wording.

Opted in, after a read of each handler's reply (the rule: the answer repeats text someone
else wrote): `email_rebucket` (email subject and sender), `brief_request` (event, followup,
inbox and news text), `dues_request` and `bill_amount_request` (issuer parsed out of the
email), `account_statement` (statement email subject), `statement_email_details` (inbox
search output), `bill_paid` (issuer, when no configured institution matches). Declared in
both places. **Left off, owner's call:** `folder_files` and `cleanup_request` print local
filenames, which a download may have named. Not traced: agent-created task titles reaching
`reminder_action`.

**Decided (owner, 2026-09-29): an opt-in input safety screen, after step (b).** Same shape as
the existing model-based checks: an `IRIS_GOVERNANCE_INPUT_SAFETY` flag, **off by default**;
an `input_safety` section in `config/governance/threat-detection.yaml` reusing the output
guard's model (`llama-guard3:1b`) and category lists (so `self_harm` is covered); registered at
`PRE_TURN` only when the flag is on; shadow-first (logged and audited, not blocking, until
the YAML's mode is flipped to enforce); a slow or missing model lets the turn through, as the
inbound Prompt Guard does. Test 2 (the red-team regression) runs with the flag on and a stub
classifier. A default install keeps step (a)'s behaviour: the regex classifier on every turn.

## Decisions (owner, 2026-09-29: all as recommended)

- **A. New hook point or reuse?** Decided: add `PRE_TURN`. Reusing `PRE_CLASSIFY` at
  turn level would double-fire it (once per turn, again per model call) and blur two jobs:
  screening what the user said vs screening what enters a model.
- **B. Llama Guard on deterministic answers?** A deterministic email digest reproduces
  third-party text (subjects, senders), which can carry unsafe content the harness did not
  write. Decided: `output_safety` stays in `judge` (off for deterministic answers by
  default), with a per-handler manifest opt-in (`guard_output: true`) for handlers that echo
  third-party text; the email digest handlers opt in.
- **C. Blocked deterministic answers:** same blocked text as generated answers, or a
  handler-specific fallback? Decided: same text, one refusal voice.
