"""Compatibility tests for the thin IRIS governor HTTP wrapper."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

# A plain import. This used to be a `spec_from_file_location` over a hand-built path to
# `services/governor/main.py`, because the service lived outside the package and was not
# importable. M6.2 layer 10 moved it in, so the twelve lines of loader ceremony -- and the
# path they spelled out, which the move broke -- are simply gone.
from iris_harness.foundation.auth import auth_headers
from iris_harness.kernel.governor.audit import GovernorAuditLogger
from iris_harness.kernel.governor.policy import GovernorPolicyEngine, load_governor_policy
from iris_harness.kernel.governor.rate_limiter import TokenBucketRateLimiter
from iris_harness.kernel.governor.service import IRISGovernorService
from iris_harness.server.governor.main import create_app


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0

    def __call__(self) -> float:
        return self.current


def write_policy(repo_root: Path) -> None:
    policy_dir = repo_root / "config" / "governor"
    policy_dir.mkdir(parents=True)
    (policy_dir / "policy.yaml").write_text(
        "version: '1'\n"
        "routes:\n"
        "  - route: coding/git\n"
        "    allowed_actions:\n"
        "      - branch_commit_push\n"
        "    rate_limit:\n"
        "      requests: 5\n"
        "      window_seconds: 3600\n"
        "  - route: coding/github\n"
        "    allowed_actions:\n"
        "      - create_pull_request\n"
        "    rate_limit:\n"
        "      requests: 1\n"
        "      window_seconds: 60\n",
        encoding="utf-8",
    )


def build_client(repo_root: Path) -> TestClient:
    fake_clock = FakeClock()
    service = IRISGovernorService(
        policy_engine=GovernorPolicyEngine(load_governor_policy(repo_root)),
        audit_logger=GovernorAuditLogger(repo_root / "data" / "audit.db"),
        rate_limiter=TokenBucketRateLimiter(clock=fake_clock),
    )
    return TestClient(create_app(governor_service=service), headers=auth_headers())


def test_guard_endpoint_requires_bearer_auth(tmp_path: Path) -> None:
    write_policy(tmp_path)
    client = build_client(tmp_path)

    denied = client.post("/guard/coding/git", json={"action": "x"}, headers={"Authorization": ""})
    probe = client.get("/healthz", headers={"Authorization": ""})

    assert denied.status_code == 401
    assert probe.status_code == 200


def test_guard_endpoint_returns_compatible_decision_payload(tmp_path: Path) -> None:
    write_policy(tmp_path)
    client = build_client(tmp_path)

    response = client.post(
        "/guard/coding/git",
        json={
            "action": "branch_commit_push",
            "metadata": {"task_id": "task-1", "branch_name": "iris/feature/test"},
        },
    )

    assert response.status_code == 200
    assert response.json()["allowed"] is True
    assert response.json()["reason"] == "Allowed"
    assert response.json()["matched_policy"] == "coding/git"


def test_guard_endpoint_preserves_retry_after_seconds_for_rate_limits(tmp_path: Path) -> None:
    write_policy(tmp_path)
    client = build_client(tmp_path)

    first = client.post(
        "/guard/coding/github",
        json={"action": "create_pull_request", "metadata": {"task_id": "task-1"}},
    )
    second = client.post(
        "/guard/coding/github",
        json={"action": "create_pull_request", "metadata": {"task_id": "task-2"}},
    )

    assert first.status_code == 200
    assert first.json()["allowed"] is True
    assert second.status_code == 200
    assert second.json()["allowed"] is False
    assert second.json()["retry_after_seconds"] == 60


def test_healthz_and_routes_report_local_governor_state(tmp_path: Path) -> None:
    write_policy(tmp_path)
    client = build_client(tmp_path)

    health = client.get("/healthz")
    routes = client.get("/routes")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["route_count"] == 2
    assert routes.status_code == 200
    assert routes.json()["routes"] == ["coding/git", "coding/github"]
