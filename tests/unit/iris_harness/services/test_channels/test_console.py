"""Tests for ConsoleConnector."""

from __future__ import annotations

import io

from iris_harness.services.channels import ChannelMessage, DeliveryStatus
from iris_harness.services.channels.connectors import ConsoleConnector


def test_console_connector_writes_to_stream() -> None:
    buf = io.StringIO()
    connector = ConsoleConnector(stream=buf)
    receipt = connector.send(ChannelMessage(recipient="user", body="hello", subject="hi"))
    assert receipt.status is DeliveryStatus.SENT
    assert receipt.message_id
    output = buf.getvalue()
    assert "user" in output
    assert "hello" in output
    assert "hi" in output


def test_console_connector_default_name() -> None:
    connector = ConsoleConnector(stream=io.StringIO())
    assert connector.name == "console"
    assert connector.healthy() is True
