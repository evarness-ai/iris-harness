"""`iris plugins`: the running IRIS's plugins, read from GET /plugins."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from iris_harness.cli import plugins as plugins_mod
from iris_harness.cli.plugins import plugins_app

runner = CliRunner()

_LIST = {
    "profile": {"name": "personal-assistant"},
    "totals": {"loaded": 1, "degraded": 1, "failed": 0},
    "plugins": [
        {
            "name": "calendar",
            "status": "loaded",
            "party": "first-party",
            "registration_counts": {"intent_handler": 1, "heartbeat": 2},
            "subscription_count": 2,
            "seam_count": 3,
        },
        {
            "name": "finance_workflows",
            "status": "degraded",
            "party": "untrusted",
            "registration_counts": {},
            "subscription_count": 0,
            "seam_count": 0,
            "last_error": "learned_source:finance_paid_by_email: RuntimeError: store gone",
        },
        {
            "name": "planner",
            "status": "loaded",
            "registration_counts": {},
            "subscription_count": 0,
            "seam_count": 0,
            "degraded_reason": "optional capability mail.read unavailable (degraded)",
        },
    ],
}
_DETAIL = {
    "name": "calendar",
    "status": "loaded",
    "source": "entry_point:calendar",
    "version": "1.0.0",
    "trust": "in-process",
    "party": "untrusted",
    "registrations": [{"kind": "intent_handler", "name": "calendar", "detail": ""}],
    "subscriptions": [{"topic": "reminder.snoozed", "scope": "process"}],
    "seams": [{"seam": "api_router", "key": "reminders"}],
    "search_providers": [
        {"name": "events", "registered": True},
        {"name": "events_news", "registered": False},
    ],
    "capabilities": {
        "provides": [{"name": "calendar.events", "provided": False, "used_by": ["planner"]}],
        "uses": [{"name": "mail.read", "providers": []}],
        "requires": [{"name": "credentials.accounts", "providers": ["gmail"]}],
    },
    "drift": {"provides_not_registered": ["channel"], "tools_declared_not_registered": []},
    "manifest": {
        "webui": {
            "screens": [
                {"id": "agenda", "label": "Agenda", "route": "/agenda", "nav": True},
                {"id": "reminders", "label": "Reminder", "route": "/reminders", "nav": False},
            ]
        }
    },
}


@pytest.fixture()
def api(cli_api: Any) -> list[str]:
    seen: list[str] = []

    def _handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen.append(url)
        if url.endswith("/plugins"):
            return httpx.Response(200, json=_LIST)
        if url.endswith("/plugins/calendar"):
            return httpx.Response(200, json=_DETAIL)
        return httpx.Response(404, json={"detail": "no such plugin: nope"})

    cli_api(_handle, plugins_mod)
    return seen


def test_list_shows_status_counts_and_errors(api: list[str]) -> None:
    result = runner.invoke(plugins_app, [])
    assert result.exit_code == 0, result.stdout
    out = " ".join(result.stdout.split())  # rich wraps at the test terminal's width
    assert "personal-assistant" in out
    assert "calendar" in out
    assert "2 subscription(s)" in out and "3 seam(s)" in out
    assert "store gone" in out
    assert api == ["http://localhost:8003/plugins"]


def test_list_prints_party_beside_each_plugin(api: list[str]) -> None:
    out = " ".join(runner.invoke(plugins_app, []).stdout.split())
    assert "party=first-party" in out and "party=untrusted" in out
    assert "degraded: optional capability" in out  # the degraded line is untouched


def test_show_prints_trust_and_party(api: list[str]) -> None:
    out = " ".join(runner.invoke(plugins_app, ["show", "calendar"]).stdout.split())
    assert "trust=in-process party=untrusted" in out


def test_list_shows_the_degraded_reason_only_for_a_degraded_plugin(api: list[str]) -> None:
    result = runner.invoke(plugins_app, [])
    assert result.exit_code == 0, result.stdout
    out = " ".join(result.stdout.split())
    assert "degraded: optional capability mail.read unavailable (degraded)" in out
    assert out.count("degraded: ") == 1  # calendar (no reason) prints none


def test_show_prints_the_degraded_reason_when_present(
    api: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    result = runner.invoke(plugins_app, ["show", "calendar"])
    assert "degraded:" not in result.stdout  # healthy: None prints nothing
    detail = {**_DETAIL, "degraded_reason": "optional capability mail.read unavailable (degraded)"}
    monkeypatch.setattr(plugins_mod, "_fetch", lambda api_url, path: detail)
    result = runner.invoke(plugins_app, ["show", "calendar"])
    assert "degraded: optional capability mail.read unavailable" in " ".join(result.stdout.split())


def test_show_lists_subscriptions_with_their_bus_and_seams(api: list[str]) -> None:
    result = runner.invoke(plugins_app, ["show", "calendar"])
    assert result.exit_code == 0, result.stdout
    assert "reminder.snoozed" in result.stdout and "@process" in result.stdout
    assert "api_router: reminders" in result.stdout
    assert "provides_not_registered: channel" in result.stdout
    assert "tools_declared_not_registered" not in result.stdout


def test_show_prints_who_provides_and_uses_what(api: list[str]) -> None:
    result = runner.invoke(plugins_app, ["show", "calendar"])
    assert result.exit_code == 0, result.stdout
    out = " ".join(result.stdout.split())
    assert "capabilities (3)" in out
    assert "provides calendar.events (never provided) used by planner" in out
    assert "uses mail.read provided by none: degraded" in out
    assert "requires credentials.accounts provided by gmail" in out


def test_show_lists_the_search_providers_it_declares(api: list[str]) -> None:
    result = runner.invoke(plugins_app, ["show", "calendar"])
    assert result.exit_code == 0, result.stdout
    out = " ".join(result.stdout.split())
    assert "search providers (2)" in out
    assert "events events_news (not registered)" in out


def test_show_lists_the_console_screens_it_owns(api: list[str]) -> None:
    """OSS plan R17: the same declaration the web nav reads, on the CLI."""
    result = runner.invoke(plugins_app, ["show", "calendar"])
    assert result.exit_code == 0, result.stdout
    out = " ".join(result.stdout.split())
    assert "screens (2)" in out
    assert "/agenda Agenda" in out
    assert "/reminders Reminder (no nav entry)" in out


def test_show_unknown_plugin_says_so(api: list[str]) -> None:
    result = runner.invoke(plugins_app, ["show", "nope"])
    assert result.exit_code == 1
    assert "no such plugin: nope" in result.stdout


def test_json_is_the_api_body(api: list[str]) -> None:
    result = runner.invoke(plugins_app, ["show", "calendar", "--json"])
    assert result.exit_code == 0
    assert '"reminder.snoozed"' in result.stdout


def test_friendly_error_when_the_api_is_down(cli_api: Any) -> None:
    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    cli_api(_boom, plugins_mod)
    result = runner.invoke(plugins_app, [])
    assert result.exit_code == 1
    assert "Is the stack up?" in result.stdout
