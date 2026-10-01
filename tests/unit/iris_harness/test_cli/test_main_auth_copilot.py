"""Tests for the `iris auth copilot {login|status|logout}` Typer subcommand.

These cover the user-facing entry points wired up in `iris_harness.main` so future
refactors can't silently break the device-flow login surface that humans rely
on to acquire a Copilot subscription token.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iris_harness.llm import copilot_auth
from iris_harness.llm.copilot_auth import (
    DeviceCodeResponse,
    OAuthCacheEntry,
)
from iris_harness.main import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def cache_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "oauth.json"
    monkeypatch.setattr(copilot_auth, "DEFAULT_CACHE_PATH", path)
    return path


def _write_cache_file(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"access_token": token}), encoding="utf-8")


def test_login_requires_opt_in_env_var(
    runner: CliRunner,
    cache_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("IRIS_ENABLE_COPILOT_BACKEND", raising=False)

    result = runner.invoke(app, ["auth", "copilot", "login"])

    assert result.exit_code == 2
    assert "IRIS_ENABLE_COPILOT_BACKEND" in result.output


def test_login_drives_device_flow_and_writes_cache(
    runner: CliRunner,
    cache_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_ENABLE_COPILOT_BACKEND", "1")

    captured: dict[str, Any] = {}

    class _FakeFlow:
        def __init__(self, *, cache_path: Path) -> None:
            captured["cache_path"] = cache_path

        def login(self, announce: Any) -> OAuthCacheEntry:
            announce(
                DeviceCodeResponse(
                    device_code="dev",
                    user_code="ABCD-1234",
                    verification_uri="https://github.com/login/device",
                    expires_in=900,
                    interval=5,
                )
            )
            entry = OAuthCacheEntry(access_token="gho_testtoken_1234567890")
            _write_cache_file(captured["cache_path"], entry.access_token)
            return entry

    monkeypatch.setattr(copilot_auth, "CopilotDeviceFlow", _FakeFlow)

    result = runner.invoke(app, ["auth", "copilot", "login"])

    assert result.exit_code == 0, result.output
    assert "ABCD-1234" in result.output
    assert "https://github.com/login/device" in result.output
    assert cache_path.exists()


def test_status_reports_cached_token(
    runner: CliRunner,
    cache_path: Path,
) -> None:
    _write_cache_file(cache_path, "gho_cached_token_xyz0123456789")

    result = runner.invoke(app, ["auth", "copilot", "status"])

    assert result.exit_code == 0, result.output
    assert "gho_" in result.output
    # Rich hard-wraps long paths on narrow terminals (CI runners are 80 columns
    # and the tmp path is long enough to split the filename itself), so compare
    # against the output with line breaks removed.
    assert cache_path.name in result.output.replace("\n", "")


def test_status_exits_nonzero_when_missing(
    runner: CliRunner,
    cache_path: Path,
) -> None:
    assert not cache_path.exists()

    result = runner.invoke(app, ["auth", "copilot", "status"])

    assert result.exit_code == 1
    assert "no token cached" in result.output


def test_logout_removes_cache(
    runner: CliRunner,
    cache_path: Path,
) -> None:
    _write_cache_file(cache_path, "gho_to_remove_0123456789")

    result = runner.invoke(app, ["auth", "copilot", "logout"])

    assert result.exit_code == 0, result.output
    assert "removed" in result.output
    assert not cache_path.exists()


def test_logout_is_idempotent_when_nothing_cached(
    runner: CliRunner,
    cache_path: Path,
) -> None:
    assert not cache_path.exists()

    result = runner.invoke(app, ["auth", "copilot", "logout"])

    assert result.exit_code == 0, result.output
    assert "no token to remove" in result.output
