"""``iris health`` (ADR-0116): thin renderers over the health API."""

from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from iris_harness.cli import health
from iris_harness.main import app

runner = CliRunner()

_INCIDENT = {
    "id": 7,
    "key": "Gmail:a@b.com",
    "target": "Gmail",
    "subject": "a@b.com",
    "state": "needs_user",
    "detail": "a@b.com: token revoked — re-authenticate",
    "action": "iris auth gmail login --user a@b.com",
    "opened_at": "2026-09-19T12:00:00+00:00",
    "repairs": [{"tried": "token refresh", "ok": False, "detail": "revoked", "final": True}],
    "notified_at": "2026-09-19T12:01:00+00:00",
    "notify_count": 1,
    "resolved_at": None,
    "resolution": None,
}


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    replies: dict[str, Any] = {
        "/health/connectors": {
            "state": "red",
            "live": False,
            "sampled_at": "2026-09-19T12:00:00+00:00",
            "connectors": [
                {
                    "target": "Gmail",
                    "state": "red",
                    "detail": _INCIDENT["detail"],
                    "action": _INCIDENT["action"],
                },
                {"target": "Anthropic", "state": "green", "detail": "API key present"},
            ],
        },
        "/health/incidents": {"enabled": True, "count": 1, "incidents": [_INCIDENT]},
        "/health/watch": {
            "state": "red",
            "summary": "1 red, 4 green",
            "events": ["opened Gmail:a@b.com: revoked"],
            "open": [_INCIDENT],
        },
    }

    def fake_call(api: str | None, path: str, *, method: str = "GET", timeout: float = 30) -> Any:
        seen.append((method, path))
        return replies[path.split("?")[0]]

    monkeypatch.setattr(health, "_call", fake_call)
    return seen


def test_connectors_shows_the_fix_and_exits_nonzero_when_red(calls) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(app, ["health", "connectors", "--live"])
    assert result.exit_code == 1
    assert "Gmail" in result.output and "Anthropic" in result.output
    assert "fix: iris auth gmail login --user a@b.com" in result.output
    assert calls == [("GET", "/health/connectors?live=true")]


def test_incidents_tells_the_story(calls) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(app, ["health", "incidents", "--open", "-n", "5"])
    assert result.exit_code == 0
    assert "#7 Gmail (a@b.com)" in result.output
    assert "✗ token refresh — revoked" in result.output
    assert "told you 1×" in result.output
    assert calls == [("GET", "/health/incidents?open_only=true&limit=5")]


def test_watch_posts_a_pass(calls) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(app, ["health", "watch"])
    assert result.exit_code == 0
    assert "opened Gmail:a@b.com" in result.output
    assert "1 open incident(s)" in result.output
    assert calls == [("POST", "/health/watch")]


def test_json_is_the_raw_body(calls) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(app, ["health", "incidents", "--json"])
    assert '"key": "Gmail:a@b.com"' in result.output


def test_an_unreachable_api_is_a_clean_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_API_URL", "http://127.0.0.1:9")  # nothing listens on discard
    result = runner.invoke(app, ["health", "incidents"])
    assert result.exit_code == 1
    assert "unreachable" in result.output


def test_connector_rows_line_up_whatever_the_state(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"target": t, "state": st, "detail": "d"}
        for t, st in [("Gmail", "green"), ("Drive", "grey"), ("Calendar", "red"), ("X", "yellow")]
    ]
    body = {"state": "red", "live": False, "sampled_at": "2026-09-19T12:00:00", "connectors": rows}
    monkeypatch.setattr(health, "_call", lambda *a, **k: body)
    result = runner.invoke(app, ["health", "connectors"])
    lines = [ln for ln in result.output.splitlines() if ln.rstrip().endswith(" d")]
    assert len(lines) == 4
    assert len({ln.index(r["target"]) for ln, r in zip(lines, rows, strict=True)}) == 1
