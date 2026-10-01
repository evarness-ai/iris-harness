"""Tests for the `/routines` slash command."""

from __future__ import annotations

import io
from types import SimpleNamespace

from rich.console import Console

from iris_harness.cli import commands


def test_routines_list_prints_full_ids(monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(
        "iris_harness.cli.render.console",
        Console(file=output, force_terminal=False, width=60),
    )
    routine_id = "routine-20260509090000-daily-repo-brief-1234abcd"

    def fake_api_json(
        api_url: str,
        *,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        timeout: int = 30,
    ) -> dict[str, object]:
        assert (api_url, method, path, payload) == ("http://testserver", "GET", "/routines", None)
        return {
            "routines": [
                {
                    "id": routine_id,
                    "title": "Daily repo brief",
                    "schedule": "daily:09:00",
                    "template": "daily_repo_brief",
                    "approval_status": "scheduled",
                    "success_count": 0,
                    "run_count": 0,
                }
            ]
        }

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    ctx = SimpleNamespace(api_url="http://testserver")

    assert commands._cmd_routines(ctx, "list") is True

    text = output.getvalue()
    assert "Full routine IDs" in text
    assert routine_id in text


def test_routine_singular_alias_is_registered() -> None:
    assert commands._REGISTRY["/routine"].handler is commands._cmd_routines


def test_routines_add_command_calls_api(monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(
        "iris_harness.cli.render.console",
        Console(file=output, force_terminal=False, width=200),
    )
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
            "routine": {
                "id": "routine-test",
                "title": "Morning brief",
                "schedule": "interval:60",
                "template": "morning_briefing",
                "approval_status": "draft",
            }
        }

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    ctx = SimpleNamespace(api_url="http://testserver")

    assert commands._cmd_routines(ctx, "add interval:60 morning_briefing Morning brief") is True

    assert calls == [
        (
            "http://testserver",
            "POST",
            "/routines",
            {
                "schedule": "interval:60",
                "template": "morning_briefing",
                "title": "Morning brief",
            },
        )
    ]
    assert "routine-test" in output.getvalue()


def test_routines_approve_command_patches_status(monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(
        "iris_harness.cli.render.console",
        Console(file=output, force_terminal=False, width=200),
    )
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
        return {"routine": {"id": "routine-test", "approval_status": "approved"}}

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    ctx = SimpleNamespace(api_url="http://testserver")

    assert commands._cmd_routines(ctx, "approve routine-test") is True

    assert calls == [
        (
            "http://testserver",
            "PATCH",
            "/routines/routine-test",
            {"approval_status": "approved"},
        )
    ]
    assert "approved" in output.getvalue()


def test_routines_tick_command_calls_routine_tick_endpoint(monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(
        "iris_harness.cli.render.console",
        Console(file=output, force_terminal=False, width=200),
    )
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
        return {"status": "success", "output": "routine_tick due=1 executed=1 success=1"}

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    ctx = SimpleNamespace(api_url="http://testserver")

    assert commands._cmd_routines(ctx, "tick") is True

    assert calls == [("http://testserver", "POST", "/routines/tick", None)]
    assert "executed=1" in output.getvalue()


def test_routines_clear_command_deletes_all_routines(monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(
        "iris_harness.cli.render.console",
        Console(file=output, force_terminal=False, width=200),
    )
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
            "deleted_count": 2,
            "deleted_routines": [
                {"id": "routine-first", "title": "Morning brief"},
                {"id": "routine-second", "title": "Daily repo brief"},
            ],
        }

    monkeypatch.setattr(commands, "_api_json", fake_api_json)
    ctx = SimpleNamespace(api_url="http://testserver")

    assert commands._cmd_routines(ctx, "clear") is True

    assert calls == [("http://testserver", "DELETE", "/routines", None)]
    text = output.getvalue()
    assert "deleted 2 routines" in text
    assert "routine-first" in text
    assert "routine-second" in text
