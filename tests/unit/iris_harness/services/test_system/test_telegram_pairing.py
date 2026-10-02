"""``poll_telegram_chat_id`` / ``get_telegram_bot_username``: the getUpdates poll,
the pairing deep link's username lookup, and their own egress lines.

Moved out of ``cli/setup.py`` so the CLI module never opens a network connection
itself (``tests/security/test_cli_egress_logged.py``) -- this module owns the call
and logs it, ``kind="channel"``, like ``cli/api_client.py`` does for the harness's
own services.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest

from iris_harness.services.system.telegram_pairing import (
    get_telegram_bot_username,
    poll_telegram_chat_id,
)


class _FakeTelegram:
    """A fake Telegram ``getUpdates`` server: no updates, then one."""

    def __init__(self, delay_calls: int) -> None:
        self.calls = 0
        self.delay_calls = delay_calls

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls <= self.delay_calls:
            return httpx.Response(200, json={"ok": True, "result": []})
        return httpx.Response(
            200,
            json={
                "ok": True,
                "result": [
                    {"update_id": 1, "message": {"chat": {"id": 555}, "text": "hi"}},
                ],
            },
        )

    def client(self) -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(self.handler), base_url="https://api.telegram.org"
        )


@pytest.fixture()
def egress(caplog: pytest.LogCaptureFixture) -> Any:
    caplog.set_level(logging.INFO, logger="iris.egress")

    def lines() -> list[str]:
        return [r.getMessage() for r in caplog.records if r.name == "iris.egress"]

    return lines


def test_poll_telegram_chat_id_returns_id_once_a_message_arrives(egress: Any) -> None:
    fake = _FakeTelegram(delay_calls=2)
    sleeps: list[float] = []
    clock = {"t": 0.0}

    chat_id = poll_telegram_chat_id(
        "tok",
        client=fake.client(),
        deadline_s=100.0,
        poll_interval_s=1.0,
        sleep=sleeps.append,
        now=lambda: clock["t"],
    )
    assert chat_id == "555"
    assert fake.calls == 3
    assert sleeps == [1.0, 1.0]

    lines = egress()
    assert len(lines) == 3, lines
    for line in lines:
        assert line.startswith("EGRESS channel GET -> api.telegram.org"), line
        assert "tok" not in line  # the bot token never reaches the log line


def test_poll_telegram_chat_id_times_out_returns_none() -> None:
    fake = _FakeTelegram(delay_calls=999)
    clock = {"t": 0.0}

    def fake_sleep(seconds: float) -> None:
        clock["t"] += seconds

    chat_id = poll_telegram_chat_id(
        "tok",
        client=fake.client(),
        deadline_s=5.0,
        poll_interval_s=2.0,
        sleep=fake_sleep,
        now=lambda: clock["t"],
    )
    assert chat_id is None


def _getme_client(body: dict[str, Any], *, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="https://api.telegram.org")


def test_get_telegram_bot_username_returns_it_for_the_pairing_link(egress: Any) -> None:
    client = _getme_client({"ok": True, "result": {"id": 1, "username": "TheDreamAgentBot"}})
    username = get_telegram_bot_username("tok", client=client)
    assert username == "TheDreamAgentBot"

    lines = egress()
    assert len(lines) == 1
    assert lines[0].startswith("EGRESS channel GET -> api.telegram.org")
    assert "tok" not in lines[0]


def test_get_telegram_bot_username_returns_none_for_a_bad_token() -> None:
    client = _getme_client(
        {"ok": False, "error_code": 401, "description": "Unauthorized"}, status=401
    )
    assert get_telegram_bot_username("bad-tok", client=client) is None


def test_get_telegram_bot_username_returns_none_on_a_malformed_body() -> None:
    client = _getme_client({"ok": True, "result": []})  # result should be an object, not a list
    assert get_telegram_bot_username("tok", client=client) is None
