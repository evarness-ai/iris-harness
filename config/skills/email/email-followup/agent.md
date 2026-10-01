# email-followup skill

Tracks emails the user owes a personal reply to. Two halves:

- **Detection** (Tier 3 local, user-invoked via `iris email detect-followups`):
  Scans recent classified emails for one account, asks the LLM "does
  the user need to send a personal reply?", and upserts a followup
  Task on positive verdict.
- **Auto-resolution** (no LLM, auto-fires on `email.new_arrived`):
  When a new message arrives in a thread that has a tracked followup,
  marks the followup's `wait_for` resolved. The user decides whether
  to mark the task done.

## Prerequisites

1. Phase 1 email subsystem operational (`iris email triage` has run —
   followups attach to classified mail).
2. Tier-3-local llama-server on `localhost:8090` for detection.
   `bash scripts/serve_tier3_local.sh`.

## Behavior

- **Dedup contract**: `followup_key(provider, thread_id)`. Exactly one
  followup Task per email thread. Re-detection over the same thread is
  a no-op.
- **Auto-resolution does not auto-complete** (ADR-0014 §10). The
  task's `wait_for_resolved_at` is set; the user marks done via
  `iris task complete` after seeing the resolved followup in the
  brief.
- **Threadless emails are skipped**. Without a thread_id there is no
  reliable anchor for auto-resolution.

## Why detection is CLI-only

Per ADR-0022 (resource-aware triage revision), Tier 3 local runs only
when the user explicitly invokes it. Detection joins `triage-batch` as
a user-invoked Tier 3 caller; auto-resolution is the cheap half that
runs continuously.

## Related

- ADR-0005 — Task vocabulary (followup is a Task with `wait_for`).
- ADR-0014 — TaskStore implementation shape (dedup, resolve_wait
  semantics).
- ADR-0022 — Tier 3 only on demand.
- Canonical doc §3.1 — Email skill split.
