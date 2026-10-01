"""The proof bundle (OSS plan R14), for a test or a CI job: export one, verify one.

Stable tier. A bundle is the audit ledger's evidence for the three email-onboarding
invariants -- no personal content to a cloud tier; no mailbox write without an approved
approval row; every model call and every answer audited -- as one versioned JSON
document that :func:`verify_bundle` checks offline, with nothing but the document. The
``iris governance proof-bundle export|verify`` commands are the same two functions.
Format: docs/reference/proof-bundle.md (schema: ``SCHEMA_FILE``).

``export_bundle(ledger, observations=...)`` takes the ledger (a harness's ``audit_db``,
or a home's) and what was observed outside it: the model calls and answers in the
session logs (:func:`observations_from_session_logs`, a harness's ``home / "logs"``),
plus mailbox writes the caller reports (:class:`MailboxWrite`).
"""

from __future__ import annotations

from iris_harness.kernel.governance.audit.proof_bundle import (
    FORMAT,
    INVARIANTS,
    SCHEMA_FILE,
    SCHEMA_VERSION,
    MailboxWrite,
    Observations,
    ProofBundleError,
    Violation,
    content_sha256,
    export_bundle,
    observations_from_session_logs,
    read_bundle,
    verify_bundle,
    write_bundle,
)

__all__ = [
    "content_sha256",
    "FORMAT",
    "INVARIANTS",
    "SCHEMA_FILE",
    "SCHEMA_VERSION",
    "MailboxWrite",
    "Observations",
    "ProofBundleError",
    "Violation",
    "export_bundle",
    "observations_from_session_logs",
    "read_bundle",
    "verify_bundle",
    "write_bundle",
]
