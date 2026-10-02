"""GET /health (ADR-0069 slice 3) — serves the cached HealthSnapshot. We seed the
module cache and pass a dummy runtime so no real runtime is built and nothing is
probed."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.health import service
from iris_harness.services.health.models import CheckKind, HealthCheck, HealthSnapshot, HealthState


def _seeded_snapshot() -> HealthSnapshot:
    return HealthSnapshot(
        checks=(
            HealthCheck(
                "governor",
                CheckKind.SERVICE,
                HealthState.RED,
                "unreachable",
                endpoint="http://x/healthz",
            ),
            HealthCheck("Gmail", CheckKind.CREDENTIAL, HealthState.GREEN, "connected"),
            HealthCheck(
                "Drive",
                CheckKind.CREDENTIAL,
                HealthState.GREY,
                "not connected",
                action="iris auth gdrive login",
            ),
        ),
        sampled_at="2026-06-20T00:00:00+00:00",
    )


def test_health_endpoint_serves_cached_snapshot() -> None:
    service.store_snapshot(_seeded_snapshot())
    with TestClient(
        create_app(runtime=SimpleNamespace(), auto_start_runtime=False), headers=auth_headers()
    ) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "red"  # worst across checks
    assert body["sampled_at"] == "2026-06-20T00:00:00+00:00"
    targets = {c["target"]: c for c in body["checks"]}
    assert targets["governor"]["endpoint"] == "http://x/healthz"
    assert targets["Drive"]["state"] == "grey"
    assert targets["Drive"]["action"] == "iris auth gdrive login"
    # alerts = red projection only (governor), never the grey/green checks.
    assert [a["target"] for a in body["alerts"]] == ["governor"]


# ── connectors + incidents + a watch pass (ADR-0116) ────────────────────────


def _watcher(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    from iris_harness.services.health import watch
    from iris_harness.services.health.incidents import IncidentStore
    from iris_harness.services.health.watch import HealthWatcher, WatchConfig

    sent: list[tuple[str, str]] = []
    watcher = HealthWatcher(
        store=IncidentStore(tmp_path / "health.db"),
        config=WatchConfig(confirm_ticks=1),
        notify=lambda subject, body, channels, url=None: sent.append((subject, body)),
    )
    monkeypatch.setattr(watch, "_installed", watcher)
    return watcher, sent


def _revoked_gmail() -> HealthSnapshot:
    return HealthSnapshot(
        checks=(
            HealthCheck("governor", CheckKind.SERVICE, HealthState.GREEN, "HTTP 200"),
            HealthCheck(
                "Gmail",
                CheckKind.CREDENTIAL,
                HealthState.RED,
                "a@b.com: token revoked — re-authenticate",
                action="iris auth gmail login --user a@b.com",
                subject="a@b.com",
            ),
            HealthCheck("Anthropic", CheckKind.CREDENTIAL, HealthState.GREY, "no key set"),
        ),
        sampled_at="2026-09-19T00:00:00+00:00",
    )


def _client():  # type: ignore[no-untyped-def]
    return TestClient(
        create_app(runtime=SimpleNamespace(), auto_start_runtime=False), headers=auth_headers()
    )


def test_connectors_lists_only_external_connections_with_their_incident(
    tmp_path, monkeypatch  # type: ignore[no-untyped-def]
) -> None:
    watcher, _ = _watcher(tmp_path, monkeypatch)
    snap = _revoked_gmail()
    watcher.observe(snap)
    service.store_snapshot(snap)
    with _client() as client:
        body = client.get("/health/connectors").json()

    assert body["state"] == "red" and body["live"] is False
    assert [c["target"] for c in body["connectors"]] == ["Gmail", "Anthropic"]  # no services
    gmail = body["connectors"][0]
    assert gmail["subject"] == "a@b.com"
    assert gmail["incident"]["state"] == "needs_user"
    assert body["connectors"][1]["incident"] is None


def test_connectors_live_rebuilds_with_the_network_probe(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[bool] = []

    def fake_refresh(*, net_probe=None, heartbeat_diagnostics=None):  # type: ignore[no-untyped-def]
        calls.append(bool(net_probe))
        return _revoked_gmail()

    monkeypatch.setattr(service, "refresh", fake_refresh)
    with _client() as client:
        body = client.get("/health/connectors?live=true").json()
    assert calls == [True] and body["live"] is True


def test_incidents_endpoint_and_a_watch_pass(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _, sent = _watcher(tmp_path, monkeypatch)
    monkeypatch.setattr(service, "refresh", lambda **_: _revoked_gmail())
    with _client() as client:
        run = client.post("/health/watch").json()
        listed = client.get("/health/incidents?open_only=true").json()

    assert run["state"] == "red"
    assert any(e.startswith("opened Gmail:a@b.com") for e in run["events"])
    assert [i["key"] for i in run["open"]] == ["Gmail:a@b.com"]
    assert listed["enabled"] is True and listed["count"] == 1
    assert sent and sent[0][0] == "IRIS needs your help"


def test_incidents_without_a_watch_says_so(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from iris_harness.services.health import watch

    monkeypatch.setattr(watch, "_installed", None)
    with _client() as client:
        assert client.get("/health/incidents").json() == {
            "enabled": False,
            "count": 0,
            "incidents": [],
        }
        assert client.post("/health/watch").status_code == 503


def test_a_watch_pass_is_write_gated(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _watcher(tmp_path, monkeypatch)
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    with _client() as client:
        assert client.post("/health/watch").status_code == 403
        assert client.get("/health/incidents").status_code == 200


def test_new_health_routes_need_the_bearer_token() -> None:
    with TestClient(create_app(runtime=SimpleNamespace(), auto_start_runtime=False)) as client:
        assert client.get("/health/connectors").status_code == 401
        assert client.get("/health/incidents").status_code == 401
        assert client.get("/health/doctor").status_code == 401


# ── the install preflight (OSS plan R5) ─────────────────────────────────────


def test_health_doctor_serves_the_preflight_without_reading_the_keyring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import keyring

    from iris_harness.services.system import doctor as dr

    host = dr.HostFacts("3.12.4", "Linux", "x86_64", "6.1.0", 32 * 1024**3)
    monkeypatch.setattr(dr, "host_facts", lambda: host)
    monkeypatch.setattr(dr, "disk_free_bytes", lambda p: (tmp_path, 100 * 10**9))
    monkeypatch.setattr(dr, "configured_ollama_models", lambda: ())
    monkeypatch.setattr(
        dr,
        "ollama_facts",
        lambda client, root: dr.OllamaFacts(url=root, reachable=True, models=frozenset()),
    )
    monkeypatch.delenv("IRIS_VAULT_MASTER_KEY", raising=False)

    def forbidden(service: str, username: str) -> str:
        raise AssertionError("the API read the keyring")

    monkeypatch.setattr(keyring.get_keyring(), "get_password", forbidden)
    with TestClient(
        create_app(runtime=SimpleNamespace(), auto_start_runtime=False), headers=auth_headers()
    ) as client:
        resp = client.get("/health/doctor")

    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "demo_only"  # the starter model is not pulled
    assert body["key"]["source"] == "not_read"
    rows = {c["name"]: c for c in body["checks"]}
    assert rows["Starter models"]["status"] == "fail"
    assert body["missing_models"][0]["name"] == "qwen2.5:7b-instruct"


# ── pulling a missing starter model (the System Check screen's one write) ───


class _FakeActivityNotices:
    """Just enough of IrisRuntime's own ``_activity_notices()`` seam for the route
    to submit real work to a real (tmp-path) ActivityRunner -- no mocked spine."""

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def activities(self) -> Any:
        return self._runner


def _runtime_with_activities(tmp_path: Path) -> tuple[SimpleNamespace, Any]:
    from iris_harness.services.activities import ActivityRunner, ActivityStore

    store = ActivityStore(db_path=tmp_path / "activities.db")
    store.ensure_schema()
    runner = ActivityRunner(store=store, max_workers=1)
    rt = SimpleNamespace(_activity_notices=lambda: _FakeActivityNotices(runner))
    return rt, store


def _doctor_sees_a_missing_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from iris_harness.services.system import doctor as dr

    host = dr.HostFacts("3.12.4", "Linux", "x86_64", "6.1.0", 32 * 1024**3)
    monkeypatch.setattr(dr, "host_facts", lambda: host)
    monkeypatch.setattr(dr, "disk_free_bytes", lambda p: (tmp_path, 100 * 10**9))
    monkeypatch.setattr(dr, "configured_ollama_models", lambda: ())
    monkeypatch.setattr(
        dr,
        "ollama_facts",
        lambda client, root: dr.OllamaFacts(url=root, reachable=True, models=frozenset()),
    )
    monkeypatch.delenv("IRIS_VAULT_MASTER_KEY", raising=False)


def test_pull_model_is_write_gated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _doctor_sees_a_missing_model(monkeypatch, tmp_path)
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    rt, _ = _runtime_with_activities(tmp_path)
    with TestClient(
        create_app(runtime=rt, auto_start_runtime=False), headers=auth_headers()
    ) as client:
        resp = client.post("/health/doctor/pull-model", json={"model": "qwen2.5:7b-instruct"})
    assert resp.status_code == 403


def test_pull_model_refuses_a_model_that_is_not_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _doctor_sees_a_missing_model(monkeypatch, tmp_path)
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
    rt, _ = _runtime_with_activities(tmp_path)
    with TestClient(
        create_app(runtime=rt, auto_start_runtime=False), headers=auth_headers()
    ) as client:
        resp = client.post("/health/doctor/pull-model", json={"model": "not-a-real-model"})
    assert resp.status_code == 404


def test_pull_model_submits_an_activity_and_reports_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from iris_harness.services.system import doctor as dr

    _doctor_sees_a_missing_model(monkeypatch, tmp_path)
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")

    def fake_pull_model(name: str, *, client: Any, root: str | None, on_progress: Any) -> None:
        on_progress(dr.PullProgress("pulling", 50, 100))
        on_progress(dr.PullProgress("success", 100, 100))

    monkeypatch.setattr(dr, "pull_model", fake_pull_model)
    rt, store = _runtime_with_activities(tmp_path)
    with TestClient(
        create_app(runtime=rt, auto_start_runtime=False), headers=auth_headers()
    ) as client:
        resp = client.post("/health/doctor/pull-model", json={"model": "qwen2.5:7b-instruct"})
    assert resp.status_code == 200
    activity_id = resp.json()["activity_id"]

    for _ in range(200):
        activity = store.get(activity_id)
        if activity is not None and activity.status in ("completed", "failed"):
            break
        time.sleep(0.01)
    assert activity is not None and activity.status == "completed"
    assert activity.result_summary == "pulled qwen2.5:7b-instruct"
    assert activity.progress == 1.0
