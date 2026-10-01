"""Tests for the `/reminders` slash command — a read of the one reminder store (D14)."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from rich.console import Console

from iris_harness.cli import commands

Call = tuple[str, str, str, dict[str, object] | None]


def _wire(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, object]
) -> tuple[io.StringIO, list[Call]]:
    output = io.StringIO()
    monkeypatch.setattr(
        "iris_harness.cli.render.console",
        Console(file=output, force_terminal=False, width=200),
    )
    calls: list[Call] = []

    def fake_api_json(
        api_url: str,
        *,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        timeout: int = 30,
    ) -> dict[str, object]:
        calls.append((api_url, method, path, payload))
        return response

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    return output, calls


_ROWS = {
    "count": 1,
    "timezone": "America/Chicago",
    "reminders": [
        {
            "id": "rem-1",
            "text": "Take out the recycling",
            "remind_at": "2026-09-28T13:00:00+00:00",
            "remind_at_local": "Mon Sep 28, 8:00 AM",
            "status": "pending",
            "recurrence": "weekly",
            "recurrence_label": "every Monday",
        }
    ],
}


@pytest.mark.parametrize(
    ("args", "path"),
    [
        ("", "/api/v1/reminders?status=open"),
        ("list", "/api/v1/reminders?status=open"),
        ("due", "/api/v1/reminders?status=open&due=true"),
        ("missed", "/api/v1/reminders?status=failed"),
        ("all", "/api/v1/reminders?status=all"),
    ],
)
def test_reminders_reads_the_one_store_api(
    monkeypatch: pytest.MonkeyPatch, args: str, path: str
) -> None:
    output, calls = _wire(monkeypatch, _ROWS)

    assert commands._cmd_reminders(SimpleNamespace(api_url="http://testserver"), args) is True

    assert calls == [("http://testserver", "GET", path, None)]
    text = output.getvalue()
    assert "Take out the recycling" in text
    assert "Mon Sep 28, 8:00 AM (every Monday)" in text
    assert "America/Chicago" in text


def test_reminders_tick_fires_the_delivery_heartbeat(monkeypatch: pytest.MonkeyPatch) -> None:
    output, calls = _wire(
        monkeypatch, {"status": "success", "output": "notification_reminder_tick fired=1"}
    )

    assert commands._cmd_reminders(SimpleNamespace(api_url="http://testserver"), "tick") is True

    assert calls == [
        ("http://testserver", "POST", "/heartbeat/trigger/notification_reminder_tick", None)
    ]
    assert "fired=1" in output.getvalue()


def test_reminders_add_is_retired(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reminders are created in chat now; the markdown store's `add` is gone."""
    output, calls = _wire(monkeypatch, {})

    assert commands._cmd_reminders(SimpleNamespace(api_url="http://x"), "add today 09:00 X")

    assert calls == []
    assert "Usage: /reminders" in output.getvalue()
