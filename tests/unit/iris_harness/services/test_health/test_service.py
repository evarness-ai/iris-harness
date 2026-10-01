"""Slice 3 of System Health (ADR-0069): snapshot cache, runtime flags, and the
health_tick heartbeat. build_snapshot/refresh are monkeypatched, so nothing here
probes services or reads credentials."""

from __future__ import annotations

import pytest

from iris_harness.services.health import service
from iris_harness.services.health.heartbeat import build_health_tick_handler
from iris_harness.services.health.models import CheckKind, HealthCheck, HealthSnapshot, HealthState
from iris_harness.services.heartbeat.diagnostics import HeartbeatDiagnostic
from iris_harness.services.heartbeat.models import HeartbeatDefinition, HeartbeatStatus


def _snap(state: HealthState = HealthState.GREEN) -> HealthSnapshot:
    return HealthSnapshot(
        checks=(HealthCheck("x", CheckKind.SERVICE, state, "detail"),),
        sampled_at="2026-06-20T00:00:00+00:00",
    )


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # The cache is a module global; reset it around every test.
    monkeypatch.setattr(service, "_cached", None)


def _defn() -> HeartbeatDefinition:
    return HeartbeatDefinition(name="health_tick", handler="health_tick", schedule="interval:60")


# ── flags ───────────────────────────────────────────────────────────────────


def test_flag_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_HEALTH_ENABLED", raising=False)
    monkeypatch.delenv("IRIS_HEALTH_NET_PROBE", raising=False)
    assert service.health_enabled() is True  # on by default
    assert service.net_probe_enabled() is False  # opt-in


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_net_probe_truthy_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("IRIS_HEALTH_NET_PROBE", value)
    assert service.net_probe_enabled() is True


def test_health_disabled_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_HEALTH_ENABLED", "0")
    assert service.health_enabled() is False


# ── cache + current_snapshot ────────────────────────────────────────────────


def test_store_and_cached_roundtrip() -> None:
    assert service.cached_snapshot() is None
    snap = _snap()
    service.store_snapshot(snap)
    assert service.cached_snapshot() is snap


def test_current_snapshot_prefers_cache_without_building(monkeypatch: pytest.MonkeyPatch) -> None:
    seeded = _snap(HealthState.YELLOW)
    service.store_snapshot(seeded)

    def _boom(**_: object) -> HealthSnapshot:
        raise AssertionError("build_snapshot must not run when cache is warm")

    monkeypatch.setattr(service, "build_snapshot", _boom)
    assert service.current_snapshot() is seeded


def test_current_snapshot_builds_and_caches_on_cold_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    built = _snap(HealthState.RED)
    monkeypatch.setattr(service, "build_snapshot", lambda **_: built)
    result = service.current_snapshot()
    assert result is built
    assert service.cached_snapshot() is built  # now published


def test_refresh_publishes_to_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    built = _snap()
    monkeypatch.setattr(service, "build_snapshot", lambda **_: built)
    assert service.refresh() is built
    assert service.cached_snapshot() is built


# ── health_tick heartbeat ───────────────────────────────────────────────────


def test_health_tick_success_publishes_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "refresh", lambda: _snap(HealthState.YELLOW))
    run = build_health_tick_handler()(_defn())
    assert run.status is HeartbeatStatus.SUCCESS
    assert "yellow" in run.output


def test_health_tick_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_HEALTH_ENABLED", "0")
    run = build_health_tick_handler()(_defn())
    assert run.status is HeartbeatStatus.SKIPPED


def test_health_tick_failure_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom() -> HealthSnapshot:
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(service, "refresh", _boom)
    run = build_health_tick_handler()(_defn())
    assert run.status is HeartbeatStatus.FAILED
    assert "probe exploded" in run.error


def test_health_tick_passes_heartbeat_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    diag = HeartbeatDiagnostic(
        name="finance_ingest_tick",
        schedule="0 7 * * *",
        reason="overdue",
        detail="missed scheduled run",
        action="iris heartbeats trigger finance_ingest_tick",
    )

    captured: dict[str, object] = {}

    def _fake_refresh(**kwargs: object) -> HealthSnapshot:
        captured.update(kwargs)
        return _snap()

    monkeypatch.setattr(service, "refresh", _fake_refresh)
    run = build_health_tick_handler(heartbeat_diagnostics_provider=lambda: [diag])(_defn())

    assert run.status is HeartbeatStatus.SUCCESS
    assert captured["heartbeat_diagnostics"] == [diag]


@pytest.fixture(autouse=True)
def _no_extra_check_providers() -> None:
    # A runtime built earlier in the same process installs the plugin-registry
    # provider; these tests assert on the bare snapshot, so start from none.
    service.clear_check_providers()


# ── health_tick hands the snapshot to the watch (ADR-0116) ──────────────────


class _FakeWatcher:
    def __init__(self, *, fail: bool = False) -> None:
        self.seen: list[HealthSnapshot] = []
        self._fail = fail

    def observe(self, snapshot: HealthSnapshot) -> list[str]:
        self.seen.append(snapshot)
        if self._fail:
            raise RuntimeError("watch exploded")
        return ["opened x: detail"]


def test_health_tick_runs_the_installed_watch(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.services.health import watch

    snap = _snap(HealthState.RED)
    fake = _FakeWatcher()
    monkeypatch.setattr(service, "refresh", lambda: snap)
    monkeypatch.setattr(watch, "_installed", fake)
    run = build_health_tick_handler()(_defn())
    assert fake.seen == [snap]
    assert run.status is HeartbeatStatus.SUCCESS
    assert run.output.endswith("| watch: opened x: detail")


def test_a_failing_watch_never_fails_the_tick(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.services.health import watch

    monkeypatch.setattr(service, "refresh", lambda: _snap())
    monkeypatch.setattr(watch, "_installed", _FakeWatcher(fail=True))
    run = build_health_tick_handler()(_defn())
    assert run.status is HeartbeatStatus.SUCCESS
    assert "watch" not in run.output


# ── a refresh's probe choice reaches plugin providers (revoked-Gmail simulation) ──


def test_refresh_net_probe_reaches_plugin_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Plugin providers read net_probe_enabled(); a live refresh must say True to them
    even with the env flag off — else ?live=true and the watch's diagnosis skip Gmail."""
    monkeypatch.delenv("IRIS_HEALTH_NET_PROBE", raising=False)
    monkeypatch.setattr(service, "build_snapshot", lambda **kw: _snap())
    seen: list[bool] = []
    service.clear_check_providers()
    service.register_check_provider("probe", lambda: seen.append(service.net_probe_enabled()) or [])
    try:
        service.refresh(net_probe=True)
        service.refresh(net_probe=False)
        monkeypatch.setenv("IRIS_HEALTH_NET_PROBE", "1")
        service.refresh()  # no choice given: the env flag decides
        service.refresh(net_probe=False)  # an explicit choice beats the flag
    finally:
        service.clear_check_providers()
    assert seen == [True, False, True, False]
    assert service.net_probe_enabled() is True  # outside a refresh: the flag again
