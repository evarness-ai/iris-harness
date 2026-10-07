"""Health rows for the model guard (issue #136): off while external tools are mounted, or on
but unable to run.

The guard fails open, so without these rows an owner can believe external content is being
scanned by the model when it is not. Silent when the posture is fine. The always-on
deterministic floor (marker and instruction tripwire) is independent of this and runs
either way; the detail says so.
"""

from __future__ import annotations

from collections.abc import Callable

from iris_harness.kernel.governance.plugin_egress import unrecorded_outcomes
from iris_harness.kernel.governance.threat.availability import (
    PROMPT_GUARD_FLAG,
    model_guard_state,
)
from iris_harness.services.health.models import CheckKind, HealthCheck, HealthState

TARGET = "Model guard"

#: ``(plugin or skill, tool)`` for every mounted tool or capability that returns third-party text.
ExternalTools = Callable[[], list[tuple[str, str]]]


def model_guard_provider(external_tools: ExternalTools) -> Callable[[], list[HealthCheck]]:
    """A ``register_check_provider`` callable. ``external_tools`` is re-read each pass."""

    def provider() -> list[HealthCheck]:
        state = model_guard_state()
        if state.on:
            if state.classifier == "available":
                return []
            return [
                HealthCheck(
                    target=TARGET,
                    kind=CheckKind.GOVERNANCE,
                    state=HealthState.YELLOW,
                    detail=(
                        f"on, but its classifier cannot run: {state.reason}. External text is "
                        "still marked and tripwire-scanned by the always-on floor, not by the "
                        "model."
                    ),
                    action=state.fix,
                )
            ]
        mounted = external_tools()
        if not mounted:
            return []
        owners = sorted({owner for owner, _tool in mounted})
        shown = ", ".join(owners[:5]) + (f" and {len(owners) - 5} more" if len(owners) > 5 else "")
        return [
            HealthCheck(
                target=TARGET,
                kind=CheckKind.GOVERNANCE,
                state=HealthState.YELLOW,
                detail=(
                    f"off while {len(mounted)} external-content tool(s) are mounted ({shown}). "
                    "The always-on floor still marks and tripwire-scans their text; the model "
                    "guard would scan it too."
                ),
                action=f"set {PROMPT_GUARD_FLAG}=1 (and install the ml extra)",
            )
        ]

    return provider


UNRECORDED_TARGET = "Egress ledger"


def unrecorded_egress_provider() -> list[HealthCheck]:
    """A ``register_check_provider`` callable: governed requests that completed with no
    ``post_egress`` row on the ledger (issue #175). Silent while the count is zero.

    The request had already happened when its outcome row failed to write, so the response was
    returned; the ledger is the thing that is broken, and the next request is refused until it
    is back (a request is sent only with its ``pre_egress`` row written). The count is this
    process's: it clears on restart, the log line names each call.
    """
    lost = unrecorded_outcomes()
    if not lost:
        return []
    return [
        HealthCheck(
            target=UNRECORDED_TARGET,
            kind=CheckKind.GOVERNANCE,
            state=HealthState.RED,
            detail=(
                f"{lost} governed request(s) completed but their outcome could not be written "
                "to the audit ledger. Plugin requests are refused while the ledger cannot be "
                "written. The log names each call."
            ),
            action="check the audit database (disk space, permissions, IRIS_GOVERNANCE_AUDIT_DB_PATH)",
        )
    ]


AUDIT_WRITES_TARGET = "Audit ledger"


def audit_writes_provider() -> list[HealthCheck]:
    """A ``register_check_provider`` callable: audit rows the ledger would not take (#134).

    RED when a row was kept nowhere (the database and the local spool both refused it, so the
    ledger has a hole this process cannot repair; calls that guard an effect are refused while
    that holds). YELLOW while rows wait in the spool for the database, or lines were set
    aside as malformed. Silent when neither. The lost count is this process's (it clears on
    restart; the hole itself stays on the ledger as a gap in the writer's sequence); the
    spool is on disk.
    """
    from iris_harness.foundation.paths import audit_db_path
    from iris_harness.kernel.governance.audit.write_health import write_health

    state = write_health(audit_db_path())
    if state["writes_lost"]:
        return [
            HealthCheck(
                target=AUDIT_WRITES_TARGET,
                kind=CheckKind.GOVERNANCE,
                state=HealthState.RED,
                detail=(
                    f"{state['writes_lost']} audit row(s) could not be written to the ledger or the "
                    f"local spool (last cause: {state['last_error_class'] or 'unknown'}). Write, "
                    "destructive, cloud-model and outbound calls are refused while this holds."
                ),
                action="check the audit database (disk space, permissions, IRIS_GOVERNANCE_AUDIT_DB_PATH)",
            )
        ]
    if state["spool_pending"] or state["spool_rejected"]:
        return [
            HealthCheck(
                target=AUDIT_WRITES_TARGET,
                kind=CheckKind.GOVERNANCE,
                state=HealthState.YELLOW,
                detail=(
                    f"{state['spool_pending']} audit row(s) wait in the local spool for the ledger; "
                    f"{state['spool_rejected']} malformed spool line(s) were set aside."
                ),
                action="they are written back on the next successful audit write or restart",
            )
        ]
    return []


__all__ = [
    "AUDIT_WRITES_TARGET",
    "TARGET",
    "UNRECORDED_TARGET",
    "audit_writes_provider",
    "ExternalTools",
    "model_guard_provider",
    "unrecorded_egress_provider",
]
