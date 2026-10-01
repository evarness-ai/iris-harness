"""The ledger as a reader sees it: one shape for ``GET /governance/audit`` and the CLI.

A row's columns, the documented payload fields (``foundation.observability.audit_view``:
who called, whether a deterministic handler answered, what was called, the digest key),
where its tier runs, and its reason with every email address masked
(``display_mask``). A reason is text a hook wrote, and older ones name an account
(``mailbox writes approved for gmail:<address>``, before the email library's reasons
named only the provider); the append-only ledger keeps those bytes, the screen does not
show them. Nothing else of the payload is ever returned.
"""

from __future__ import annotations

from typing import Any

from iris_harness.foundation.observability.audit_view import public_payload, tier_locality
from iris_harness.kernel.governance.audit.log import AuditLog, AuditRow
from iris_harness.kernel.governance.display_mask import mask_text

#: The most rows one read returns.
MAX_LIMIT = 500


def audit_entry(row: AuditRow) -> dict[str, Any]:
    """One row for a screen: columns, documented payload fields, locality, masked reason."""
    return {
        "id": row.id,
        "ts": row.ts,
        "run_id": row.run_id,
        "step_id": row.step_id,
        "agent_type": row.agent_type,
        "hook_point": row.hook_point,
        "plugin": row.plugin,
        "decision": row.decision,
        "classification": row.classification,
        "tier": row.tier,
        "locality": tier_locality(row.tier),
        "severity": row.severity,
        "reason": mask_text(row.reason),
        **public_payload(row.payload_json),
    }


def audit_view(
    log: AuditLog,
    *,
    decision: str | None = None,
    caller: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    """The newest ``limit`` rows matching the filter, newest first, and the callers seen.

    ``caller`` is exact, or a namespace when it ends in ``:`` (``AuditLog.query``).
    ``callers`` lists every caller the ledger names, so a reader can offer the filter.
    """
    capped = max(1, min(limit, MAX_LIMIT))
    rows = log.query(decision=decision, caller=caller or None)
    recent = list(rows)[-capped:][::-1]
    return {
        "count": len(recent),
        "total": log.count(),
        "audit_db": str(log.db_path),
        "callers": list(log.callers()),
        "entries": [audit_entry(r) for r in recent],
    }


__all__ = ["MAX_LIMIT", "audit_entry", "audit_view"]
