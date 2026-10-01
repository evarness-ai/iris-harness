"""Tests for ChannelGateway registration + dispatch."""

from __future__ import annotations

from dataclasses import dataclass

from iris_harness.services.channels import (
    ChannelGateway,
    ChannelMessage,
    ChannelNotFoundError,
    DeliveryReceipt,
    DeliveryStatus,
)


@dataclass
class _FakeConnector:
    name: str = "fake"
    sent: list[ChannelMessage] | None = None
    raises: bool = False

    def __post_init__(self) -> None:
        if self.sent is None:
            self.sent = []

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        if self.raises:
            raise RuntimeError("boom")
        assert self.sent is not None
        self.sent.append(message)
        return DeliveryReceipt(channel=self.name, status=DeliveryStatus.SENT, message_id="m1")

    def healthy(self) -> bool:
        return True


def test_register_and_get_connector() -> None:
    gateway = ChannelGateway()
    connector = _FakeConnector()
    gateway.register(connector)
    assert gateway.get("fake") is connector
    assert gateway.channels() == ["fake"]


def test_register_rejects_empty_name() -> None:
    gateway = ChannelGateway()
    bad = _FakeConnector(name="")
    try:
        gateway.register(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_get_unknown_channel_raises() -> None:
    gateway = ChannelGateway()
    try:
        gateway.get("nope")
    except ChannelNotFoundError:
        return
    raise AssertionError("expected ChannelNotFoundError")


def test_send_dispatches_to_correct_connector() -> None:
    gateway = ChannelGateway()
    a = _FakeConnector(name="a")
    b = _FakeConnector(name="b")
    gateway.register(a)
    gateway.register(b)

    receipt = gateway.send("a", ChannelMessage(recipient="r", body="hi"))
    assert receipt.status is DeliveryStatus.SENT
    assert a.sent == [ChannelMessage(recipient="r", body="hi")]
    assert b.sent == []


def test_send_captures_connector_exceptions() -> None:
    gateway = ChannelGateway()
    gateway.register(_FakeConnector(raises=True))
    receipt = gateway.send("fake", ChannelMessage(recipient="r", body="hi"))
    assert receipt.status is DeliveryStatus.FAILED
    assert "boom" in receipt.error


def test_broadcast_sends_to_all_registered_channels() -> None:
    gateway = ChannelGateway()
    a = _FakeConnector(name="a")
    b = _FakeConnector(name="b")
    gateway.register(a)
    gateway.register(b)

    receipts = gateway.broadcast(ChannelMessage(recipient="r", body="hi"))
    assert {r.channel for r in receipts} == {"a", "b"}
    assert all(r.status is DeliveryStatus.SENT for r in receipts)


def test_unregister_removes_connector() -> None:
    gateway = ChannelGateway()
    gateway.register(_FakeConnector())
    assert gateway.unregister("fake") is True
    assert gateway.unregister("fake") is False
    assert gateway.channels() == []
