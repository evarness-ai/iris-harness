"""System Health rows for accounts the scheduled sweep waits on (owner decision 2026-09-30).

An account whose email setup has not reached "Keep it current" is held out of the
sweep (``email.sweep_gate``), so no new mail arrives for it. One yellow row per such
active mailbox account, target ``Mail sweep``, subject the address, action the command
that goes on with setup. Yellow, not red: it is setup in progress, not a failure.

Local only: reads the account table and the gate, no network.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from iris_harness.sdk.health import CheckKind, HealthCheck, HealthState

from .writes_health import _active_accounts, _provider_for

TARGET = "Mail sweep"


def _held() -> dict[str, str]:
    from iris_personal.email.sweep_gate import SweepGate

    return {account_id: entry.reason for account_id, entry in SweepGate().held().items()}


def sweep_wait_checks(
    *,
    accounts: Callable[[], list[tuple[str, str]]] = _active_accounts,
    provider_for: Callable[[str], Any] = _provider_for,
    held: Callable[[], dict[str, str]] = _held,
) -> list[HealthCheck]:
    """The rows; ``accounts`` returns ``(account id, address)`` pairs (tests pass one)."""
    waiting = held()
    rows: list[HealthCheck] = []
    for account_id, address in accounts():
        if account_id not in waiting or provider_for(account_id) is None:
            continue
        rows.append(
            HealthCheck(
                TARGET,
                CheckKind.SERVICE,
                HealthState.YELLOW,
                f"{address}: not swept on the schedule yet -- email setup has not turned "
                f"the sweep on ({waiting[account_id]})",
                action=f"iris email setup --account {account_id}",
                subject=address,
            )
        )
    return rows


__all__ = ["TARGET", "sweep_wait_checks"]
