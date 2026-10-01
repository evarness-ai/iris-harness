"""Unit tests for the embedded IRIS governor service."""

from __future__ import annotations

from pathlib import Path

from iris_harness.kernel.governor.audit import GovernorAuditLogger
from iris_harness.kernel.governor.policy import GovernorPolicyEngine, load_governor_policy
from iris_harness.kernel.governor.rate_limiter import TokenBucketRateLimiter
from iris_harness.kernel.governor.service import IRISGovernorService


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


def write_policy(repo_root: Path) -> None:
    policy_dir = repo_root / "config" / "governor"
    policy_dir.mkdir(parents=True)
    (policy_dir / "policy.yaml").write_text(
        "version: '1'\n"
        "routes:\n"
        "  - route: coding/github\n"
        "    allowed_actions:\n"
        "      - create_pull_request\n"
        "    rate_limit:\n"
        "      requests: 1\n"
        "      window_seconds: 60\n",
        encoding="utf-8",
    )


def test_governor_service_applies_rate_limits_and_records_audit(tmp_path: Path) -> None:
    write_policy(tmp_path)
    fake_clock = FakeClock()
    service = IRISGovernorService(
        policy_engine=GovernorPolicyEngine(load_governor_policy(tmp_path)),
        audit_logger=GovernorAuditLogger(tmp_path / "data" / "audit.db"),
        rate_limiter=TokenBucketRateLimiter(clock=fake_clock),
    )

    first = service.guard(
        "coding/github",
        {
            "action": "create_pull_request",
            "metadata": {"task_id": "task-1"},
        },
    )
    second = service.guard(
        "coding/github",
        {
            "action": "create_pull_request",
            "metadata": {"task_id": "task-2"},
        },
    )
    fake_clock.advance(60)
    third = service.guard(
        "coding/github",
        {
            "action": "create_pull_request",
            "metadata": {"task_id": "task-3"},
        },
    )

    assert first.allowed is True
    assert second.allowed is False
    assert second.retry_after_seconds == 60
    assert "rate limit exceeded" in second.reason
    assert third.allowed is True

    entries = service.audit_logger.list_entries()
    assert len(entries) == 3
    assert entries[1].allowed is False
    assert service.health_report().route_count == 1
