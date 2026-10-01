"""Kernel — the governance layer: what may run, on which tier, and with whose approval.

`governance/` is the unified kernel (classification, policy, judges, the audit ledger,
the approval queue, checkpoints, the vault); `governor/` is its legacy HTTP frontend,
served by `src/iris_harness/server/governor/`. Everything above depends on the kernel; the kernel
depends only on foundation.

The vault holds every secret the harness keeps -- including the credential and
secret stores that were `identity/`'s until M6.2 (OSS plan M6, decision 4). Where a
secret LIVES is a kernel question, not an identity one.
"""
