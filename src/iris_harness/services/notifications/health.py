"""The ``push_delivery`` health check (loop-proof D13/D14, graph §8).

Red while a reminder no channel accepted (``failed``) has not yet been shown to the
owner in a digest's "missed" line, green otherwise. Red pages the owner through the
health watch on every channel — the one that recovered first delivers the notice.
Once the digest has shown the reminder (or it expired), the owner knows, and the
check is green again.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import tzinfo
from pathlib import Path

from iris_harness.services.health.models import CheckKind, HealthCheck, HealthState

from .channels import describe_failed
from .store import ReminderStore

logger = logging.getLogger(__name__)

TARGET = "push_delivery"


def push_delivery_checks(store: ReminderStore, tz: tzinfo) -> list[HealthCheck]:
    if not store.db_path.exists():
        return [_check(HealthState.GREEN, "no reminders yet")]
    store.ensure_schema()
    unseen = [r for r in store.list_missed() if r.missed_digests == 0]
    if not unseen:
        return [_check(HealthState.GREEN, "every reminder was delivered")]
    return [_check(HealthState.RED, describe_failed(unseen, store.db_path, tz))]


def _check(state: HealthState, detail: str) -> HealthCheck:
    return HealthCheck(target=TARGET, kind=CheckKind.SERVICE, state=state, detail=detail)


def push_delivery_provider(
    data_dir: Path, tz_fn: Callable[[], tzinfo]
) -> Callable[[], list[HealthCheck]]:
    """A ``register_check_provider`` callable over ``<data_dir>/tasks.db``."""

    def provider() -> list[HealthCheck]:
        return push_delivery_checks(ReminderStore(db_path=Path(data_dir) / "tasks.db"), tz_fn())

    return provider


__all__ = ["TARGET", "push_delivery_checks", "push_delivery_provider"]
