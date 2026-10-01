"""Health-watch repairers (ADR-0116): which red check each one claims, and what it reports."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

from iris_harness.services.health.models import CheckKind, HealthCheck, HealthState
from iris_harness.services.health.repair import (
    RepairOutcome,
    credential_refresh_repairer,
    heartbeat_retry_repairer,
    run_repairs,
    service_restart_repairer,
)


def _red(target: str, kind: CheckKind = CheckKind.SERVICE, subject: str | None = None):
    return HealthCheck(target, kind, HealthState.RED, "down", subject=subject)


class _Scheduler:
    def __init__(self, status: str = "success", error: str = "") -> None:
        self.triggered: list[str] = []
        self._status, self._error = status, error

    def trigger_by_name(self, name: str):
        self.triggered.append(name)
        return SimpleNamespace(
            status=SimpleNamespace(value=self._status), error=self._error, output="done"
        )


# ── heartbeat retry ──────────────────────────────────────────────────────


def test_heartbeat_retry_re_runs_the_failed_heartbeat() -> None:
    scheduler = _Scheduler()
    outcome = heartbeat_retry_repairer(scheduler)(_red("heartbeat:email_sweep"))
    assert scheduler.triggered == ["email_sweep"]
    assert outcome == RepairOutcome("re-run email_sweep", ok=True, detail="done")


def test_heartbeat_retry_reports_a_second_failure() -> None:
    outcome = heartbeat_retry_repairer(_Scheduler("failed", "invalid_grant"))(
        _red("heartbeat:email_sweep")
    )
    assert outcome is not None and outcome.ok is False
    assert outcome.detail == "invalid_grant"


def test_heartbeat_retry_never_re_runs_the_watchs_own_tick() -> None:
    scheduler = _Scheduler()
    repair = heartbeat_retry_repairer(scheduler, skip=["health_tick"])
    assert repair(_red("heartbeat:health_tick")) is None
    assert repair(_red("governor")) is None
    assert scheduler.triggered == []


# ── service restart ──────────────────────────────────────────────────────


def test_restart_launches_the_configured_command_from_the_repo(tmp_path: Path) -> None:
    launched: list[tuple[list[str], Path]] = []
    repair = service_restart_repairer(
        {"governor": ["scripts/start_iris.sh", "--restart=governor"]},
        cwd=tmp_path,
        launcher=lambda argv, cwd: launched.append((list(argv), cwd)),
    )
    outcome = repair(_red("governor"))
    assert launched == [(["scripts/start_iris.sh", "--restart=governor"], tmp_path)]
    assert outcome is not None and outcome.ok and outcome.tried == "restart governor"


def test_restart_never_touches_the_process_running_the_watch(tmp_path: Path) -> None:
    launched: list[Sequence[str]] = []
    repair = service_restart_repairer(
        {"iris_api": ["x"], "governor": ["y"]},
        cwd=tmp_path,
        never=["iris_api"],
        launcher=lambda argv, cwd: launched.append(argv),
    )
    assert repair(_red("iris_api")) is None
    assert repair(_red("channel_gateway")) is None  # no command configured
    assert repair(_red("governor", CheckKind.CREDENTIAL)) is None  # not a service row
    assert repair(_red("heartbeat:governor")) is None
    assert launched == []


def test_a_failed_launch_is_an_outcome(tmp_path: Path) -> None:
    def boom(argv: Sequence[str], cwd: Path) -> None:
        raise FileNotFoundError("open: not found")

    outcome = service_restart_repairer({"ollama": ["open"]}, cwd=tmp_path, launcher=boom)(
        _red("ollama")
    )
    assert outcome is not None and outcome.ok is False and "not found" in outcome.detail


# ── credential refresh ───────────────────────────────────────────────────


def test_credential_refresh_maps_the_plugins_verdicts() -> None:
    answers = {"ok@x": True, "revoked@x": False, "none@x": None}
    repair = credential_refresh_repairer("Gmail", answers.__getitem__)

    ok = repair(_red("Gmail", CheckKind.CREDENTIAL, "ok@x"))
    revoked = repair(_red("Gmail", CheckKind.CREDENTIAL, "revoked@x"))
    missing = repair(_red("Gmail", CheckKind.CREDENTIAL, "none@x"))

    assert ok == RepairOutcome("token refresh", ok=True)
    assert revoked is not None and revoked.final and "refused" in revoked.detail
    assert missing is not None and missing.final and "no stored token" in missing.detail


def test_credential_refresh_claims_only_its_own_rows() -> None:
    calls: list[str] = []

    def refresh(account: str) -> bool:
        calls.append(account)
        return True

    repair = credential_refresh_repairer("Gmail", refresh)
    assert repair(_red("Calendar", CheckKind.CREDENTIAL, "a@x")) is None
    assert repair(_red("Gmail", CheckKind.SERVICE, "a@x")) is None
    assert repair(_red("Gmail", CheckKind.CREDENTIAL, None)) is None
    assert calls == []


# ── dispatch ─────────────────────────────────────────────────────────────


def test_first_claiming_repairer_wins_and_errors_become_outcomes() -> None:
    def not_mine(check: HealthCheck) -> RepairOutcome | None:
        return None

    def broken(check: HealthCheck) -> RepairOutcome | None:
        raise OSError("network down")

    def never(check: HealthCheck) -> RepairOutcome | None:
        raise AssertionError("must not run after a claim")

    outcome = run_repairs(_red("governor"), [not_mine, broken, never])
    assert outcome is not None and outcome.ok is False
    assert outcome.detail == "OSError: network down"
    assert outcome.final is False  # a network blip is worth another try
    assert run_repairs(_red("governor"), [not_mine]) is None
