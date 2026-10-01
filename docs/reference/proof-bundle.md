# The proof bundle (format version 1)

R14 asks IRIS to ship a proof for email onboarding that a CI job can verify offline. The
proof bundle is that proof: the audit ledger's evidence for three invariants, as one
versioned JSON document, and a verifier that needs nothing but the document. It has no
Evarness dependency; it is the format an Evarness integration (or any CI) reads.

## The invariants

| id | statement |
|---|---|
| `no-private-to-cloud` | No model call whose data was classified `personal` or `secret` was allowed to a cloud tier (`tier_3`). |
| `mailbox-write-approved` | No mailbox write without an approved approval row: every account written to has a `mailbox_writes` approval in the ledger, recorded before the write when the write has a time. |
| `every-call-and-answer-audited` | Every model call and every answer has an audit row: at least one `pre_llm_call` `(run_id, step_id)` per model call observed in a session, and a `pre_response` row in each session that answered. |

The statements are carried in every bundle verbatim; `verify` fails a bundle that altered
one. The cloud tier set (`tier_3`) is fixed by the format version, never read from the
bundle.

### How invariant 2 is judged: the approval in force at the write's time

An account's `mailbox_writes` rows are its approval history: decision `allow` is an
approval, decision `deny` a revoke. A write with a time (`at`) is authorised only when
the **latest** of those rows at or before that time is an `allow`:

- approve, then write: holds;
- approve, revoke, then write: a violation ("came after its approval was revoked");
- approve, revoke, approve again, then write: holds;
- a write before any approval: a violation ("came before its approval"); a write made
  before a later revoke still holds, since a revoke takes back only what follows it.

Equal timestamps: a row stamped at exactly the write's time counts as before it, and
rows that share one time are ordered by their ledger `id`, the last recorded deciding.
So an approval at the write's own instant authorises it and a revoke at that instant
fails it (fail closed). An approval row whose `ts` is not ISO 8601 fails every timed
write to its account, since it could be the revoke that decides.

**A write with no time is refused** (owner's decision, 2026-09-30). It cannot be ordered
against a revoke, so it fails invariant 2 even when its account was approved and never
revoked ("have no time, so they cannot be ordered against a revoke"). Every write the
email library records carries its row's time; only a caller's `--observations` can
lack one, so a write supplied there must carry `at`.

Until 2026-09-30 the verifier asked only for *some* `allow` row earlier than the write,
so a write after a revoke passed, and a write with no time passed on any approval. That
was a verifier bug, not the format: the fix
reads rows every version-1 bundle already carries (`hook_point`, `decision`, `ts`,
`id`, and every `mailbox_writes` row up to `--until`, revokes included), and the
statement above already says "an approved approval row". The format stays version 1;
a version-1 bundle with a write after a revoke, or an untimed write, that an older
verifier passed now fails, which is the point.

## Commands

```bash
iris governance proof-bundle export --out bundle.json \
    [--since ISO] [--until ISO] [--run RUN_ID ...] [--subject TEXT] \
    [--db AUDIT_DB] [--session-logs DIR] [--observations observed.json]
iris governance proof-bundle verify bundle.json    # exit 0 verified, 1 a violation, 2 unreadable
iris governance proof-bundle check [--days 7] [--json]   # export + verify in memory, per invariant
```

`check` writes nothing: it exports the last `--days` of the ledger with the session logs'
observations and verifies it, reporting per invariant whether it holds, why not, and the
evidence it was judged on (an invariant over zero observations holds trivially, and says
so). `GET /governance/proof-bundle/check?days=7` returns the same, and the web Governance
screen renders it (`check_window` in the module below). Mailbox writes are observed from
the ledger's `mailbox_write_performed` rows (below), so `check` and the web see them too.

From Python (stable tier): `iris_harness.testing.proof_bundle` -- `export_bundle`,
`verify_bundle`, `observations_from_session_logs`, `Observations`, `MailboxWrite`,
`read_bundle` / `write_bundle`, `content_sha256`. The logic is
`src/iris_harness/kernel/governance/audit/proof_bundle.py`; the CLI is a thin wrapper.

## What a bundle holds

Two kinds of evidence, kept apart:

- **`ledger`** -- the audit rows in scope (`--since` / `--until` / `--run`; default: the
  whole ledger), plus every `mailbox_writes` approval row up to `--until` whatever the
  scope, since an approval granted earlier still authorises a write in the window. Each
  row is cut down to a closed set of fields: `id`, `ts`, `run_id`, `step_id`,
  `agent_type`, `hook_point`, `plugin`, `decision`, `classification`, `tier`,
  `severity`, `session_id`, `tool_name`, `account_ref`, `args_digest`, `result_digest`,
  `digest_alg`. Never the row's reason or its payload text: no prompt, argument, result
  or message content reaches a bundle, and `verify` rejects a row with any other key.
- **`observations`** -- what happened, recorded where it happened rather than at a
  governance decision:
  - `model_calls` and `answers`, read from the session logs (`--session-logs`, default
    this home's `logs/`): one per `llm_call` event (written by the transport side after a
    call returns) and per `agent_response` event. Only `kind`, `session_id`, `ts` and
    `tier` are read, never the text. A call with no session counts against every
    `pre_llm_call` row;
  - `mailbox_writes`, one per `mailbox_write_performed` ledger row in scope (`at` = the
    row's `ts`, `count` and `account` from its payload). The email library writes that
    row (`write_approvals.mailbox_write`, plugin `email_write_approvals`) once a provider's
    write -- labels, Trash, restore, a created label -- has reached the mailbox, with how
    many messages or labels it changed and the kind (`op`); a write that fails part way
    records what landed. The Gmail, IMAP and demo providers all write through it. Such a
    row is an observation, never an approval: invariant 2 still needs the account's
    latest `mailbox_writes` row at or before the write to be decision `allow` (a
    revoke is decision `deny`; see above). A third-party provider writes through the
    same `mailbox_write`, exported from the stable `iris_personal.email.provider_api`;
    one that only calls `require_mailbox_writes` records no observation.
  - Anything the caller adds with `--observations`, e.g.
    `{"mailbox_writes": [{"account": "...", "count": 3, "at": "ISO"}]}`, or `model_calls` /
    `answers` entries.

Invariants 2 and 3 are claims about observations against the ledger: a bundle with no
observations proves only invariant 1. That is deliberate. The ledger cannot vouch for
what it never saw; the observations are the independent witness. The write rows share
the ledger file but not its writer: the provider's write path records them after the
write, the approval path records approvals.

No ledger row the email slice writes names an address in its free-text `reason`
("mailbox writes approved for a gmail account", "email sweep: a gmail account waits for
email setup ...", "google connected: gmail"); the account id, and for a Google connect
the actor, are in the payload only.

### Accounts are pseudonymised

A mailbox account names a person, so it never appears in a bundle. Each export makes a
random key, uses it to turn every account (in approval rows and in observations alike)
into `acct-<24 hex>`, and throws the key away. Equal accounts get equal refs inside one
bundle, so the checks can match an approval to a write; refs link to nothing outside it,
and nobody, the exporter included, can reverse them.

### Integrity

`content_sha256` is the SHA-256 of the canonical JSON (sorted keys, no whitespace,
UTF-8) of the document without that field. An edit that does not recompute it fails
`verify` with `integrity`. It is integrity, not authenticity: whoever edits a bundle can
recompute it, and then only the invariant checks stand between the edit and a pass.
**A signed format version 2 is planned** (owner's decision, 2026-09-30); version 1 stays
integrity-only, with `tier_3` as its fixed cloud set.

## Document shape

```json
{
  "format": "iris-proof-bundle",
  "schema_version": 1,
  "subject": "email-onboarding",
  "created_at": "2026-09-30T10:00:00+00:00",
  "producer": {"name": "iris-harness", "version": "0.x"},
  "scope": {"since": null, "until": null, "run_ids": []},
  "invariants": [{"id": "no-private-to-cloud", "statement": "..."}, "..."],
  "ledger": [{"id": 1, "ts": "...", "hook_point": "pre_llm_call", "classification": "personal", "tier": "tier_1", "...": "..."}],
  "observations": {"model_calls": [{"session_id": "s-1", "tier": "tier1"}], "answers": [{"session_id": "s-1"}], "mailbox_writes": [{"account_ref": "acct-...", "count": 3, "at": "2026-09-30T09:20:00+00:00"}]},
  "content_sha256": "..."
}
```

The JSON schema is `src/iris_harness/kernel/governance/audit/proof_bundle.schema.json`
(also `SCHEMA_FILE`). `verify` checks the same closed structure in code, without a schema
library at runtime.

## Versioning

`schema_version` is 1. A verifier refuses any other version rather than guess. A change
to a field, an invariant's statement or the cloud tier set is a new version.

## Example

`examples/07-verify-with-evarness` exports and verifies the bundle of `iris email demo`'s
onboarding run with the CLI, and of an email chat turn from Python, then tampers with a
bundle (a private call moved to `tier_3`, an unaudited model call, a write with no
approval, an edit without a new digest) and shows each fail.
