"""Unit tests for the Copilot device-flow auth provider."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from iris_harness.llm.copilot_auth import (
    CopilotAuthError,
    CopilotBackendDisabledError,
    CopilotDeviceFlow,
    CopilotTokenProvider,
    DeviceCodeResponse,
    OAuthCacheEntry,
    _assert_copilot_enabled,
    _delete_cache,
    _mask_secret,
    _read_cache,
    _write_cache,
    build_copilot_credential_provider,
    logout,
)


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHttpClient:
    def __init__(
        self,
        *,
        post_responses: list[_FakeResponse] | None = None,
        get_responses: list[_FakeResponse] | None = None,
    ) -> None:
        self.post_responses = list(post_responses or [])
        self.get_responses = list(get_responses or [])
        self.post_calls: list[tuple[str, Any, Any]] = []
        self.get_calls: list[tuple[str, Any]] = []

    def post(self, url: str, *args: Any, **kwargs: Any) -> _FakeResponse:
        self.post_calls.append((url, kwargs.get("data"), kwargs.get("headers")))
        return self.post_responses.pop(0)

    def get(self, url: str, *args: Any, **kwargs: Any) -> _FakeResponse:
        self.get_calls.append((url, kwargs.get("headers")))
        return self.get_responses.pop(0)


# ---------- helpers ------------------------------------------------------


def test_mask_secret_handles_short_and_empty() -> None:
    assert _mask_secret("") == "<empty>"
    assert _mask_secret("abc") == "***"
    assert _mask_secret("ghu_1234567890abcdef") == "ghu_***ef"


def test_assert_copilot_enabled_requires_flag() -> None:
    with pytest.raises(CopilotBackendDisabledError):
        _assert_copilot_enabled({})
    # Must not raise
    _assert_copilot_enabled({"IRIS_ENABLE_COPILOT_BACKEND": "1"})


# ---------- cache --------------------------------------------------------


def test_write_cache_uses_0600_permissions(tmp_path: Path) -> None:
    cache = tmp_path / "oauth.json"
    _write_cache(
        cache, OAuthCacheEntry(access_token="ghu_abc", token_type="bearer", scope="read:user")
    )
    mode = cache.stat().st_mode & 0o777
    assert mode == 0o600
    assert json.loads(cache.read_text())["access_token"] == "ghu_abc"


def test_read_cache_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(CopilotAuthError):
        _read_cache(tmp_path / "oauth.json")


def test_read_cache_malformed_raises(tmp_path: Path) -> None:
    cache = tmp_path / "oauth.json"
    cache.write_text("not json", encoding="utf-8")
    with pytest.raises(CopilotAuthError):
        _read_cache(cache)


def test_delete_cache_reports_removal(tmp_path: Path) -> None:
    cache = tmp_path / "oauth.json"
    cache.write_text("{}", encoding="utf-8")
    assert _delete_cache(cache) is True
    assert _delete_cache(cache) is False


# ---------- device flow --------------------------------------------------


def test_device_flow_requests_device_code(tmp_path: Path) -> None:
    http = _FakeHttpClient(
        post_responses=[
            _FakeResponse(
                payload={
                    "device_code": "dc",
                    "user_code": "UC-1",
                    "verification_uri": "https://github.com/login/device",
                    "expires_in": 900,
                    "interval": 5,
                }
            )
        ]
    )
    flow = CopilotDeviceFlow(cache_path=tmp_path / "oauth.json", http_client=http)
    device = flow.request_device_code()
    assert isinstance(device, DeviceCodeResponse)
    assert device.user_code == "UC-1"
    assert device.verification_uri == "https://github.com/login/device"


def test_device_flow_polls_until_success(tmp_path: Path) -> None:
    http = _FakeHttpClient(
        post_responses=[
            _FakeResponse(payload={"error": "authorization_pending"}),
            _FakeResponse(payload={"error": "slow_down"}),
            _FakeResponse(
                payload={"access_token": "ghu_final", "token_type": "bearer", "scope": "read:user"}
            ),
        ]
    )
    sleeps: list[float] = []
    times = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    flow = CopilotDeviceFlow(
        cache_path=tmp_path / "oauth.json",
        http_client=http,
        now=lambda: next(times),
        sleep=sleeps.append,
    )
    device = DeviceCodeResponse(
        device_code="dc", user_code="UC", verification_uri="u", expires_in=900, interval=5
    )
    entry = flow.poll_for_access_token(device)
    assert entry.access_token == "ghu_final"
    assert (tmp_path / "oauth.json").exists()
    # slow_down should have increased the interval
    assert sleeps[-1] >= sleeps[0]


def test_device_flow_times_out(tmp_path: Path) -> None:
    http = _FakeHttpClient(post_responses=[])
    times = iter([0.0, 10_000.0])  # second call is past the deadline
    flow = CopilotDeviceFlow(
        cache_path=tmp_path / "oauth.json",
        http_client=http,
        now=lambda: next(times),
        sleep=lambda _seconds: None,
    )
    device = DeviceCodeResponse(
        device_code="dc", user_code="UC", verification_uri="u", expires_in=1, interval=5
    )
    with pytest.raises(CopilotAuthError):
        flow.poll_for_access_token(device)


def test_device_flow_propagates_hard_error(tmp_path: Path) -> None:
    http = _FakeHttpClient(post_responses=[_FakeResponse(payload={"error": "access_denied"})])
    times = iter([0.0, 1.0])
    flow = CopilotDeviceFlow(
        cache_path=tmp_path / "oauth.json",
        http_client=http,
        now=lambda: next(times),
        sleep=lambda _seconds: None,
    )
    device = DeviceCodeResponse(
        device_code="dc", user_code="UC", verification_uri="u", expires_in=900, interval=5
    )
    with pytest.raises(CopilotAuthError):
        flow.poll_for_access_token(device)


# ---------- token provider ----------------------------------------------


def _seed_oauth_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "oauth.json"
    _write_cache(
        cache, OAuthCacheEntry(access_token="ghu_abc", token_type="bearer", scope="read:user")
    )
    return cache


def test_token_provider_refreshes_and_caches(tmp_path: Path) -> None:
    cache = _seed_oauth_cache(tmp_path)
    http = _FakeHttpClient(
        get_responses=[
            _FakeResponse(payload={"token": "tok_1", "expires_at": 1_000.0}),
        ]
    )
    provider = CopilotTokenProvider(
        cache_path=cache,
        http_client=http,
        now=lambda: 0.0,
    )
    assert provider.get_token() == "tok_1"
    # Second call stays within the safety window so no extra GET
    assert provider.get_token() == "tok_1"
    assert len(http.get_calls) == 1


def test_token_provider_refreshes_after_expiry(tmp_path: Path) -> None:
    cache = _seed_oauth_cache(tmp_path)
    http = _FakeHttpClient(
        get_responses=[
            _FakeResponse(payload={"token": "tok_1", "expires_at": 100.0}),
            _FakeResponse(payload={"token": "tok_2", "expires_at": 500.0}),
        ]
    )
    clock = iter([200.0, 200.0, 200.0])
    provider = CopilotTokenProvider(
        cache_path=cache,
        http_client=http,
        now=lambda: next(clock),
    )
    assert provider.get_token() == "tok_1"
    assert provider.get_token() == "tok_2"
    assert len(http.get_calls) == 2


def test_token_provider_extra_headers_constant() -> None:
    headers = CopilotTokenProvider(cache_path=Path("/tmp/ignored")).extra_headers()
    assert headers["Copilot-Integration-Id"] == "vscode-chat"
    assert headers["Editor-Version"].startswith("iris-coding")


def test_token_provider_status_masks_secrets(tmp_path: Path) -> None:
    cache = _seed_oauth_cache(tmp_path)
    provider = CopilotTokenProvider(cache_path=cache, http_client=_FakeHttpClient())
    status = provider.status()
    assert "***" in status["oauth_token"]
    assert status["api_token"] == "<none>"
    assert status["cache_path"] == str(cache)


# ---------- module-level factory + logout -------------------------------


def test_build_copilot_credential_provider_enforces_gate(tmp_path: Path) -> None:
    from iris_harness.llm.client import CodingLLMConfig

    config = CodingLLMConfig(
        provider="copilot",
        model="gpt-4o",
        base_url="https://api.githubcopilot.com",
        api_key_env=None,
        auth_mode="copilot",
    )

    with pytest.raises(CopilotBackendDisabledError):
        build_copilot_credential_provider(config, environ={})

    provider = build_copilot_credential_provider(
        config, environ={"IRIS_ENABLE_COPILOT_BACKEND": "1"}
    )
    assert isinstance(provider, CopilotTokenProvider)


def test_logout_respects_custom_cache_path(tmp_path: Path) -> None:
    cache = tmp_path / "oauth.json"
    cache.write_text("{}", encoding="utf-8")
    assert logout(cache) is True
    assert logout(cache) is False


def test_module_registers_itself_with_credential_factory() -> None:
    from iris_harness.llm.client import _CREDENTIAL_PROVIDER_FACTORIES

    assert "copilot" in _CREDENTIAL_PROVIDER_FACTORIES
