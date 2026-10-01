"""The morning note, fired in a real IRIS and delivered to a faked Telegram, offline.

Telegram is faked at the HTTP transport, not replaced: the real ``TelegramConnector``
builds and sends the Bot API request, and ``httpx.MockTransport`` answers it without a
socket, so nothing leaves the machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from morning_note import NAME, setup

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.channels import TelegramConnector
from iris_harness.sdk.types import HeartbeatStatus
from iris_harness.testing import harness, plugin

MANIFEST = Path(__file__).with_name("manifest.yaml")
CHAT_ID = "1000001"  # a made-up chat


class FakeTelegram:
    """The Bot API, answering ``sendMessage`` and remembering what it was sent."""

    def __init__(self, *, status: int = 200) -> None:
        self.sent: list[dict[str, Any]] = []
        self.status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/sendMessage")
        self.sent.append(json.loads(request.content))
        if self.status != 200:
            return httpx.Response(self.status, json={"ok": False, "description": "down"})
        return httpx.Response(200, json={"ok": True, "result": {"message_id": len(self.sent)}})

    def as_plugin(self) -> Any:
        client = httpx.Client(transport=httpx.MockTransport(self))

        def setup_channel(api: PluginAPI) -> None:
            api.register_channel(
                TelegramConnector("test-token", default_chat_id=CHAT_ID, client=client)
            )

        return plugin(setup_channel, manifest={"name": "fake_telegram", "provides": ["channel"]})


def _morning_note(keep: list[PluginAPI]) -> Any:
    def setup_with(api: PluginAPI) -> None:
        keep.append(api)
        setup(api)

    return plugin(setup_with, manifest=MANIFEST)


def test_the_note_is_scheduled_and_delivered_to_telegram() -> None:
    telegram = FakeTelegram()
    kept: list[PluginAPI] = []
    with harness(plugins=[telegram.as_plugin(), _morning_note(kept)]) as h:
        assert h.plugin_loaded("morning_note")
        [api] = kept

        # The scheduler runs it at 07:00 on weekdays; fire it now, as the owner can.
        run = api.services.heartbeats.trigger_by_name(NAME)

        assert run is not None and run.status is HeartbeatStatus.SUCCESS, run
        assert run.result == {"sent": ["telegram"]}
        [message] = telegram.sent
        assert message["chat_id"] == CHAT_ID
        assert message["text"].startswith("Good morning. It is ")
        assert "- Stand-up at 9:30." in message["text"]
        assert h.model_calls() == ()  # a template, not a model


def test_a_failed_delivery_is_a_failed_run() -> None:
    telegram = FakeTelegram(status=502)
    kept: list[PluginAPI] = []
    with harness(plugins=[telegram.as_plugin(), _morning_note(kept)]) as h:
        run = kept[0].services.heartbeats.trigger_by_name(NAME)

        assert run is not None and run.status is HeartbeatStatus.FAILED
        assert "HTTP 502" in run.error
        assert h.plugin_loaded("morning_note")
