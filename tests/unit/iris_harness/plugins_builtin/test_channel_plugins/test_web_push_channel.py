"""The Web Push surface: a channel, plus a nudge when an approval is waiting.

#579 shipped this plugin with no tests, reaching past ``iris_harness.sdk`` into
the event bus, the approvals package and the transport — which broke gate 2 on
main. These pin what it does through the seam it is allowed to use: it
registers its connector, subscribes on the PROCESS bus (where the approval
router emits) through ``PluginAPI.subscribe``, and a failing push degrades the
plugin instead of escaping into the router.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from iris_harness.foundation.eventbus import get_default_bus, reset_default_bus
from iris_harness.plugins_builtin.web_push_channel import plugin as web_push
from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.sdk.channels import (
    APPROVAL_REQUESTED,
    ApprovalRequestedPayload,
    DeliveryReceipt,
    DeliveryStatus,
)


class _Gateway:
    def __init__(self) -> None:
        self.registered: list[object] = []

    def register(self, connector: object) -> None:
        self.registered.append(connector)


class _Store:
    pass


class _Connector:
    """Stands in for WebPushConnector: records what it was asked to send."""

    subscribed = True
    fail_with: Exception | None = None
    sent: list[Any] = []

    def __init__(self, *, store: object) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "web_push"

    def healthy(self) -> bool:
        return self.subscribed

    def send(self, message: Any) -> DeliveryReceipt:
        if self.fail_with is not None:
            raise self.fail_with
        _Connector.sent.append(message)
        return DeliveryReceipt(channel="web_push", status=DeliveryStatus.SENT)


@pytest.fixture(autouse=True)
def _fakes(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    reset_default_bus()
    _Connector.subscribed = True
    _Connector.fail_with = None
    _Connector.sent = []
    monkeypatch.setattr(web_push, "PushSubscriptionStore", _Store)
    monkeypatch.setattr(web_push, "WebPushConnector", _Connector)
    yield
    reset_default_bus()


def _mount(tmp_path: Path) -> tuple[_Gateway, PluginRegistry]:
    gateway, registry = _Gateway(), PluginRegistry()
    registry.add_plugin(
        PluginRecord(name="web_push_channel", source="builtin", status=PluginStatus.LOADED)
    )
    services = HarnessServices(
        config_dir=tmp_path,
        data_dir=tmp_path,
        tier_router=None,
        agent_executor=None,
        heartbeats=None,
        channels=gateway,
        deterministic_reply=lambda **kw: None,
    )
    web_push.setup(PluginAPI(plugin="web_push_channel", services=services, registry=registry))
    return gateway, registry


def _approval() -> ApprovalRequestedPayload:
    return ApprovalRequestedPayload(
        approval_id="a1",
        run_id="r1",
        signal="cost cap reached",
        context_summary="a long paragraph the banner should not carry",
        timeout_at="2026-09-21T12:00:00Z",
        channel="web",
    )


def test_registers_the_connector_as_a_channel(tmp_path: Path) -> None:
    gateway, _registry = _mount(tmp_path)

    assert len(gateway.registered) == 1
    assert isinstance(gateway.registered[0], _Connector)


def test_subscribes_on_the_process_bus_through_the_api(tmp_path: Path) -> None:
    _gateway, registry = _mount(tmp_path)

    assert registry.subscriptions() == [("web_push_channel", APPROVAL_REQUESTED, "process")]


def test_a_waiting_approval_pushes_the_signal_to_the_actions_tab(tmp_path: Path) -> None:
    _mount(tmp_path)

    get_default_bus().emit_sync(APPROVAL_REQUESTED, _approval())

    assert len(_Connector.sent) == 1
    message = _Connector.sent[0]
    assert message.body == "cost cap reached"  # the signal, not the summary
    assert message.metadata == {"url": "/actions", "tag": "approval"}


def test_nobody_subscribed_sends_nothing(tmp_path: Path) -> None:
    _mount(tmp_path)
    _Connector.subscribed = False

    get_default_bus().emit_sync(APPROVAL_REQUESTED, _approval())

    assert _Connector.sent == []


def test_a_failing_push_degrades_the_plugin_and_does_not_escape(tmp_path: Path) -> None:
    _gateway, registry = _mount(tmp_path)
    _Connector.fail_with = RuntimeError("push service down")

    get_default_bus().emit_sync(APPROVAL_REQUESTED, _approval())  # must not raise

    record = next(r for r in registry.describe()["plugins"] if r["name"] == "web_push_channel")
    assert record["status"] == "degraded"
