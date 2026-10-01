"""The system plugin installs the health watch over the runtime's channels (ADR-0116)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.services.channels.gateway import ChannelGateway
from iris_harness.services.channels.models import DeliveryReceipt, DeliveryStatus
from iris_harness.services.health import watch
from iris_harness.services.health.models import CheckKind, HealthCheck, HealthSnapshot, HealthState


class _Heartbeats:
    def __init__(self) -> None:
        self.handlers: dict[str, object] = {}

    def register_handler(self, name: str, handler: object) -> None:
        self.handlers[name] = handler

    def has_handler(self, name: str) -> bool:
        return name in self.handlers

    def trigger_by_name(self, name: str) -> None:
        return None


class _Recorder:
    def __init__(self, name: str) -> None:
        self.name = name
        self.bodies: list[str] = []

    def send(self, message) -> DeliveryReceipt:  # type: ignore[no-untyped-def]
        self.bodies.append(message.body)
        return DeliveryReceipt(channel=self.name, status=DeliveryStatus.SENT)

    def healthy(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _reset_watch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(watch, "_installed", None)
    monkeypatch.delenv("IRIS_HEALTH_WATCH_ENABLED", raising=False)


def test_setup_installs_a_watch_that_tells_every_channel(tmp_path: Path) -> None:
    from iris_harness.plugins_builtin.system import plugin

    gateway = ChannelGateway()
    telegram, web = _Recorder("telegram"), _Recorder("web")
    gateway.register(telegram)  # type: ignore[arg-type]
    gateway.register(web)  # type: ignore[arg-type]
    registry = PluginRegistry()
    registry.add_plugin(PluginRecord(name="system", source="builtin", status=PluginStatus.LOADED))
    plugin.setup(
        PluginAPI(
            plugin="system",
            services=HarnessServices(
                config_dir=tmp_path / "config",  # no health_watch.yaml → defaults
                data_dir=tmp_path / "data",
                tier_router=None,
                agent_executor=None,
                heartbeats=_Heartbeats(),
                channels=gateway,
                deterministic_reply=lambda **kw: None,
            ),
            registry=registry,
        )
    )

    installed = watch.current_watcher()
    assert installed is not None
    assert installed.store.db_path == tmp_path / "data" / "health.db"

    red = HealthSnapshot(
        checks=(HealthCheck("mystery", CheckKind.SERVICE, HealthState.RED, "unreachable"),),
        sampled_at="2026-09-19T00:00:00+00:00",
    )
    installed._diagnose = None  # no live credential probe from a unit test
    installed.observe(red)
    installed.observe(red)  # confirmed; no repairer claims "mystery" → the owner is told

    assert len(telegram.bodies) == 1 and telegram.bodies == web.bodies
    assert telegram.bodies[0].startswith("mystery is not working: unreachable")
