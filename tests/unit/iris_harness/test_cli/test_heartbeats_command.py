"""Tests for the `/heartbeats` slash command."""

from __future__ import annotations

import io
from types import SimpleNamespace

from rich.console import Console

from iris_harness.cli import commands


def _capture_console(monkeypatch) -> io.StringIO:
    output = io.StringIO()
    monkeypatch.setattr(
        "iris_harness.cli.render.console",
        Console(file=output, force_terminal=False, width=200),
    )
    return output


def test_heartbeats_list_calls_api_and_renders_rows(monkeypatch) -> None:
    output = _capture_console(monkeypatch)
    calls: list[tuple[str, str, str, dict[str, object] | None]] = []

    def fake_api_json(
        api_url: str,
        *,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        timeout: int = 30,
    ) -> dict[str, object]:
        calls.append((api_url, method, path, payload))
        return {
            "heartbeats": [
                {
                    "name": "notification_reminder_tick",
                    "schedule": "interval:60",
                    "enabled": True,
                    "description": "Find due reminders.",
                },
                {
                    "name": "wiki_lint",
                    "schedule": "interval:3600",
                    "enabled": False,
                    "description": "Wiki lint sweep.",
                },
            ]
        }

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    ctx = SimpleNamespace(api_url="http://testserver")

    assert commands._cmd_heartbeats(ctx, "") is True

    assert calls == [("http://testserver", "GET", "/heartbeat", None)]
    rendered = output.getvalue()
    assert "notification_reminder_tick" in rendered
    assert "interval:60" in rendered
    assert "wiki_lint" in rendered


def test_heartbeats_runs_passes_name_and_limit(monkeypatch) -> None:
    output = _capture_console(monkeypatch)
    calls: list[tuple[str, str, str, dict[str, object] | None]] = []

    def fake_api_json(
        api_url: str,
        *,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        timeout: int = 30,
    ) -> dict[str, object]:
        calls.append((api_url, method, path, payload))
        return {
            "count": 1,
            "total": 5,
            "runs": [
                {
                    "name": "notification_reminder_tick",
                    "status": "success",
                    "started_at": "2026-05-12T08:56:50",
                    "finished_at": "2026-05-12T08:56:50",
                    "output": "notification_reminder_tick due=1 delivered=2 skipped=0",
                    "error": "",
                }
            ],
        }

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    ctx = SimpleNamespace(api_url="http://testserver")

    assert commands._cmd_heartbeats(ctx, "runs notification_reminder_tick 5") is True

    assert calls == [
        (
            "http://testserver",
            "GET",
            "/heartbeat/runs?limit=5&name=notification_reminder_tick",
            None,
        )
    ]
    rendered = output.getvalue()
    assert "notification_reminder_tick" in rendered
    assert "delivered=2" in rendered
    assert "showing 1 of 5 runs" in rendered


def test_heartbeats_runs_default_limit_and_no_filter(monkeypatch) -> None:
    _capture_console(monkeypatch)
    calls: list[tuple[str, str, str, dict[str, object] | None]] = []

    def fake_api_json(
        api_url: str,
        *,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        timeout: int = 30,
    ) -> dict[str, object]:
        calls.append((api_url, method, path, payload))
        return {"count": 0, "total": 0, "runs": []}

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    ctx = SimpleNamespace(api_url="http://testserver")

    assert commands._cmd_heartbeats(ctx, "runs") is True

    assert calls == [("http://testserver", "GET", "/heartbeat/runs?limit=20", None)]


def test_heartbeats_trigger_posts_to_trigger_endpoint(monkeypatch) -> None:
    output = _capture_console(monkeypatch)
    calls: list[tuple[str, str, str, dict[str, object] | None]] = []

    def fake_api_json(
        api_url: str,
        *,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        timeout: int = 30,
    ) -> dict[str, object]:
        calls.append((api_url, method, path, payload))
        return {"status": "success", "output": "notification_reminder_tick due=0 delivered=0"}

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    ctx = SimpleNamespace(api_url="http://testserver")

    assert commands._cmd_heartbeats(ctx, "trigger notification_reminder_tick") is True

    assert calls == [
        ("http://testserver", "POST", "/heartbeat/trigger/notification_reminder_tick", None)
    ]
    assert "success" in output.getvalue()


def test_heartbeats_trigger_without_name_shows_usage(monkeypatch) -> None:
    output = _capture_console(monkeypatch)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("_api_json should not be called when usage is wrong")

    monkeypatch.setattr(commands, "_api_json", fail_if_called)
    ctx = SimpleNamespace(api_url="http://testserver")

    assert commands._cmd_heartbeats(ctx, "trigger") is True
    assert "Usage:" in output.getvalue()


def test_heartbeats_set_and_reset_call_the_edit_routes(monkeypatch) -> None:
    output = _capture_console(monkeypatch)
    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def fake_api_json(
        api_url: str,
        *,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        timeout: int = 30,
    ) -> dict[str, object]:
        calls.append((method, path, payload))
        return {"name": "email_sweep", "schedule_text": "every 5 min", "enabled": True}

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    ctx = SimpleNamespace(api_url="http://testserver")

    assert commands._cmd_heartbeats(ctx, "set email_sweep every 5m") is True
    assert commands._cmd_heartbeats(ctx, "set filemanager_photos on") is True
    assert commands._cmd_heartbeats(ctx, "set finance_monitor daily 07:30") is True
    assert commands._cmd_heartbeats(ctx, "reset email_sweep") is True

    assert calls == [
        ("PATCH", "/heartbeat/email_sweep", {"schedule": "interval:300"}),
        ("PATCH", "/heartbeat/filemanager_photos", {"enabled": True}),
        ("PATCH", "/heartbeat/finance_monitor", {"schedule": "30 7 * * *"}),
        ("DELETE", "/heartbeat/email_sweep/override", None),
    ]
    assert "every 5 min" in output.getvalue()


def test_heartbeats_set_refuses_bad_words_before_calling_the_api(monkeypatch) -> None:
    output = _capture_console(monkeypatch)
    monkeypatch.setattr(
        commands, "_api_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError("called"))
    )
    ctx = SimpleNamespace(api_url="http://testserver")

    assert commands._cmd_heartbeats(ctx, "set email_sweep every 5s") is True
    assert commands._cmd_heartbeats(ctx, "set email_sweep") is True

    assert "30 seconds" in output.getvalue()
    assert "Usage: /heartbeats" in output.getvalue()
