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


#: The plugin name the floor's rows carry (``ExternalContentFloorHook.name``).
FLOOR_PLUGIN = "external_content_floor"


def redaction_view(log: AuditLog, *, limit: int = 100) -> dict[str, Any]:
    """What the external-content floor redacted or the owner's allow-list kept, newest first.

    One entry per ledger row of the floor that cut a span (``patterns``) or kept an allowed
    match (``allowed``): when, the tool and source, the pattern ids and the span count, who
    called, and which turn (``session_id``). Ids and counts only: the ledger never holds the
    text, and neither does this view (issue #139).
    """
    import json

    capped = max(1, min(limit, MAX_LIMIT))
    entries: list[dict[str, Any]] = []
    for row in reversed(log.query(plugin=FLOOR_PLUGIN)):
        try:
            payload = json.loads(row.payload_json)
        except ValueError:
            continue
        if not isinstance(payload, dict) or not (payload.get("patterns") or payload.get("allowed")):
            continue
        entries.append(
            {
                "ts": row.ts,
                "tool": payload.get("tool"),
                "source": payload.get("source"),
                "patterns": list(payload.get("patterns") or []),
                "spans": payload.get("spans", 0),
                "allowed": list(payload.get("allowed") or []),
                "caller": payload.get("caller"),
                "session_id": payload.get("session_id"),
            }
        )
        if len(entries) >= capped:
            break
    return {"count": len(entries), "audit_db": str(log.db_path), "entries": entries}


__all__ = ["FLOOR_PLUGIN", "MAX_LIMIT", "audit_entry", "audit_view", "redaction_view"]
