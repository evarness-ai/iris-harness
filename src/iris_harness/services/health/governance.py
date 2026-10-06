"""Health rows for the model guard (issue #136): off while external tools are mounted, or on
but unable to run.

The guard fails open, so without these rows an owner can believe external content is being
scanned by the model when it is not. Silent when the posture is fine. The always-on
deterministic floor (marker and instruction tripwire) is independent of this and runs
either way; the detail says so.
"""

from __future__ import annotations

from collections.abc import Callable

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


__all__ = ["TARGET", "ExternalTools", "model_guard_provider"]
