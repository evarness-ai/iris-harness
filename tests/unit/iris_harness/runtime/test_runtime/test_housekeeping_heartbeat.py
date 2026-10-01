"""The memory_housekeeping heartbeat: checked hourly, runs when due (2026-09-19).

``interval:86400`` restarted its clock with every process, so on a stack restarted
more often than daily the retention pass never ran. The handler now reads the last
recorded pass and decides.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from iris_harness.memory import retention as retention_module
from iris_harness.runtime.facade import IrisRuntime
from iris_harness.services.heartbeat.models import HeartbeatDefinition, HeartbeatStatus


class _Retention:
    def __init__(self, last: dict[str, Any] | None) -> None:
        self._last = last
        self.runs = 0

    def last_run(self) -> dict[str, Any] | None:
        return self._last

    def run(self) -> Any:
        self.runs += 1
        return retention_module.HousekeepingReport(started_at="now", sessions_close_backlog=4)


def _defn(**params: Any) -> HeartbeatDefinition:
    return HeartbeatDefinition(
        name="memory_housekeeping",
        handler="memory_housekeeping",
        schedule="interval:3600",
        params=params,
    )


@pytest.fixture(autouse=True)
def _enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retention_module, "_CONFIG_CACHE", {"housekeeping": {"enabled": True}})


def _tick(last: dict[str, Any] | None, **params: Any) -> tuple[Any, _Retention]:
    retention = _Retention(last)
    run = IrisRuntime._housekeeping_heartbeat(SimpleNamespace(retention=retention), _defn(**params))  # type: ignore[arg-type]
    return run, retention


def _ago(hours: float) -> dict[str, Any]:
    started = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
    return {"started_at": started, "dry_run": False, "sessions_close_backlog": 0}


def test_the_first_check_after_any_start_runs_a_pass_that_never_ran() -> None:
    run, retention = _tick(None, min_hours_between=23)
    assert retention.runs == 1 and run.status is HeartbeatStatus.SUCCESS
    assert "4 waiting" in run.output


def test_an_hourly_check_skips_until_a_day_has_passed() -> None:
    run, retention = _tick(_ago(3), min_hours_between=23)
    assert retention.runs == 0 and run.status is HeartbeatStatus.SKIPPED
    assert "next after 23h" in run.output


def test_min_hours_comes_from_the_heartbeat_params() -> None:
    _, retention = _tick(_ago(3), min_hours_between=2)
    assert retention.runs == 1
