# 07 · Verify email onboarding (the R14 proof bundle)

IRIS promises three things about email onboarding, and this example proves them from
the audit ledger rather than taking the code's word for it -- as a **proof bundle**, a
versioned JSON document a CI job verifies offline
([docs/reference/proof-bundle.md](../../docs/reference/proof-bundle.md)):

1. **No personal content to a cloud tier.** No model call whose data was classified
   `personal` or `secret` was allowed to a cloud tier (`tier_3`).
2. **No mailbox write without an approved approval row.** Labels reach an account only
   when the ledger holds the owner's approval of mailbox writes for that account.
3. **Every model call and every answer has an audit row.**

| File | What it is |
|---|---|
| `test_email_onboarding_proof.py` | The proof: the demo's onboarding run and an email chat turn, each exported as a bundle and verified; then tampered bundles, each failing. |

## Run it

```bash
pytest examples/07-verify-with-evarness -q
```

Expected output: `3 passed` in under a minute:

- **the onboarding run** -- `iris email demo` in a temporary home (fetch 200 synthetic
  emails, judge them, preview the labels, approve mailbox writes for the demo's own
  account, write the labels, build the first digest), then

  ```bash
  iris governance proof-bundle export --db <home>/governance/audit.db \
      --session-logs <home>/logs --subject email-onboarding --out bundle.json
  iris governance proof-bundle verify bundle.json
  ```

  The labels the run wrote are observed from the ledger itself: the demo provider
  writes through `mailbox_write`, which leaves a timed `mailbox_write_performed` row
  per write, and invariant 2 checks each against the approval in force at its time
  (an `--observations` write must carry `at` for the same reason). The bundle's model calls carry `personal` data, every one on `tier_1`, and
  the approval row is in it -- under a pseudonym, never the address;
- **an email chat turn** -- "Which emails need a reply?" in a governed harness with the
  `email` profile, exported from Python (`iris_harness.testing.proof_bundle`) with the
  model calls and the answer its session log recorded, and verified;
- **the checks bite** -- a private call moved to `tier_3`, a model call the ledger never
  saw, a mailbox write with no approval row (each with its digest recomputed, so only
  the invariant can catch it), and an edit without a new digest: each fails.

## Evarness

R14 makes IRIS Evarness's reference harness, with an Evarness proof bundle for email
onboarding verified offline in CI. **Evarness is not a dependency of IRIS** (R14:
optional, dev/CI only, never runtime). The bundle format is IRIS's own, documented and
versioned, so wiring Evarness in means reading `bundle.json` (or calling
`verify_bundle`) from its side; nothing here imports it.
