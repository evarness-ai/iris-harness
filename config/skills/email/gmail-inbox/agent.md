# gmail-inbox skill

Read-only Gmail access. Fetches new mail since the last sync cursor and
persists envelope + snippet to `data/email.db`. Used by:

- The `email-sweep` heartbeat (Track 1D+) — periodic background sync.
- The harness on demand when the user asks "fetch my new mail."

## Prerequisites

Run `iris auth gmail login --user <address>` once per Gmail account
before this skill can run. The skill reads tokens from the macOS
Keychain (per ADR-0003) and the account row from `data/iris.db`
(per ADR-0016).

## Behavior

- **Cold start** (no cursor): fetches the last 30 days via
  `messages.list(q="newer_than:30d")`. Tunable per-call via
  `max_messages`.
- **Warm sync** (cursor present): uses `history.list` for delta sync
  from the last known `historyId`. Trivial quota cost; suitable for
  a 2–5 minute heartbeat cadence.
- **Stale cursor** (Gmail 404/410): auto-falls-back to cold-start.

## What's NOT here

- Body fetching. Canonical §3.1: bodies are pulled on demand from
  Gmail, not cached. A future `fetch_message_body` tool covers that.
- Attachment metadata. Requires `format="full"` from Gmail; deferred
  until a downstream skill (likely finance-statements) needs it.
- Send / archive / label operations. Those land in a future
  `gmail-compose` skill.

## Related

- ADR-0003 — Keychain credentials.
- ADR-0007 — multi-currency model (uses email_accounts.currency_default
  hint to tag fetched mail downstream).
- ADR-0016 — email_accounts registry shape.
- Canonical doc §3.1 — Email skill split + sweep model.
