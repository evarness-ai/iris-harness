"""Self-repair for red health checks (ADR-0116).

A *repairer* looks at one red ``HealthCheck`` and either says "not mine" (``None``)
or tries a fix and reports a ``RepairOutcome``. The watch never trusts the outcome
alone: the next snapshot decides whether the check is green again.

The core ships the two repairs it can do without knowing any external system:

* re-run a heartbeat whose last run failed (``heartbeat:<name>`` rows), and
* restart a local service, with the command taken from ``config/health_watch.yaml``.

Everything that talks to an external system — refreshing a Google token, say — is a
plugin's, registered with :func:`register_repairer` next to the plugin's own
``register_check_provider``. Keyed, so a second runtime in one process replaces a
repairer instead of doubling it (the same rule as the check providers).
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from iris_harness.foundation.process_state import track_globals
from iris_harness.services.health.models import CheckKind, HealthCheck

logger = logging.getLogger(__name__)

HEARTBEAT_PREFIX = "heartbeat:"


@dataclass(frozen=True)
class RepairOutcome:
    """What one repair attempt did.

    ``ok`` — the fix itself ran cleanly (the check is still re-read before anyone
    calls it healed). ``final`` — trying again cannot help (a revoked token); the
    watch goes straight to asking the owner.
    """

    tried: str
    ok: bool
    detail: str = ""
    final: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"tried": self.tried, "ok": self.ok, "detail": self.detail, "final": self.final}


Repairer = Callable[[HealthCheck], RepairOutcome | None]

_lock = Lock()
_repairers: dict[str, Repairer] = {}


def register_repairer(key: str, repairer: Repairer) -> None:
    """Install (or replace) the repairer under ``key``."""
    with _lock:
        _repairers[key] = repairer


def clear_repairers() -> None:
    """Drop all registered repairers (tests)."""
    with _lock:
        _repairers.clear()


def registered_repairers() -> list[Repairer]:
    with _lock:
        return list(_repairers.values())


def heartbeat_retry_repairer(heartbeats: Any, *, skip: Sequence[str] = ()) -> Repairer:
    """Re-run a failed heartbeat once through the scheduler (``trigger_by_name``).

    ``skip`` names heartbeats never re-run from here — the one running the watch
    (``health_tick``) above all, which would recurse.
    """
    skipped = set(skip)

    def repair(check: HealthCheck) -> RepairOutcome | None:
        if not check.target.startswith(HEARTBEAT_PREFIX):
            return None
        name = check.target[len(HEARTBEAT_PREFIX) :]
        if name in skipped or heartbeats is None:
            return None
        run = heartbeats.trigger_by_name(name)
        if run is None:
            return RepairOutcome(f"re-run {name}", ok=False, detail="no such heartbeat")
        status = getattr(getattr(run, "status", None), "value", str(getattr(run, "status", "")))
        ok = status == "success"
        detail = (getattr(run, "error", "") or getattr(run, "output", "") or status).strip()
        return RepairOutcome(f"re-run {name}", ok=ok, detail=detail[:300])

    return repair


def service_restart_repairer(
    commands: Mapping[str, Sequence[str]],
    *,
    cwd: Path,
    never: Sequence[str] = (),
    launcher: Callable[[Sequence[str], Path], None] | None = None,
) -> Repairer:
    """Restart a local service with its configured command (fire and forget).

    The launch returns at once; the service's own health probe on the next tick is
    the verdict. ``never`` names services this process must not restart — itself.
    """
    blocked = set(never)
    launch = launcher or _spawn

    def repair(check: HealthCheck) -> RepairOutcome | None:
        if check.kind is not CheckKind.SERVICE or check.target.startswith(HEARTBEAT_PREFIX):
            return None
        argv = commands.get(check.target)
        if not argv or check.target in blocked:
            return None
        try:
            launch(list(argv), cwd)
        except Exception as exc:  # noqa: BLE001 — a failed launch is an outcome, not a crash
            return RepairOutcome(f"restart {check.target}", ok=False, detail=str(exc)[:300])
        return RepairOutcome(f"restart {check.target}", ok=True, detail=" ".join(argv))

    return repair


def credential_refresh_repairer(target: str, refresh: Callable[[str], bool | None]) -> Repairer:
    """Force-refresh one account's token for a plugin's credential rows.

    ``refresh(account)`` is the plugin's own: True = refreshed, False = the provider
    refused (revoked — only a re-login helps, so the watch asks the owner now),
    None = nothing stored that could refresh (same). The plugin names its target;
    the core knows no provider.
    """

    def repair(check: HealthCheck) -> RepairOutcome | None:
        if check.target != target or check.kind is not CheckKind.CREDENTIAL:
            return None
        if not check.subject:
            return None
        result = refresh(check.subject)
        if result is True:
            return RepairOutcome("token refresh", ok=True)
        detail = (
            "the provider refused the refresh (revoked or expired)"
            if result is False
            else "no stored token that can refresh"
        )
        return RepairOutcome("token refresh", ok=False, detail=detail, final=True)

    return repair


def _spawn(argv: Sequence[str], cwd: Path) -> None:
    # argv comes from the owner's config (health_watch.yaml), never from a turn.
    subprocess.Popen(  # noqa: S603
        list(argv),
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def run_repairs(check: HealthCheck, repairers: Sequence[Repairer]) -> RepairOutcome | None:
    """The first repairer that claims ``check`` runs; ``None`` when none does."""
    for repairer in repairers:
        try:
            outcome = repairer(check)
        except Exception as exc:  # a broken repairer must not break the watch
            logger.warning("health repair failed for %s", check.key, exc_info=True)
            return RepairOutcome("repair", ok=False, detail=f"{type(exc).__name__}: {exc}"[:300])
        if outcome is not None:
            return outcome
    return None


__all__ = [
    "HEARTBEAT_PREFIX",
    "RepairOutcome",
    "Repairer",
    "clear_repairers",
    "credential_refresh_repairer",
    "heartbeat_retry_repairer",
    "register_repairer",
    "registered_repairers",
    "run_repairs",
    "service_restart_repairer",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_repairers")
