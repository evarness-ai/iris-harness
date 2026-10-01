"""GitHub Copilot device-flow OAuth + short-lived API token provider.

This module implements the OAuth device flow used by GitHub Copilot clients
(Copilot.vim, aider, and similar community tooling) to authenticate against
the Copilot Chat endpoint at ``https://api.githubcopilot.com`` using an
existing Copilot subscription.

**Terms-of-service caveat**: GitHub does not publish a stable, documented API
SKU for Copilot Chat. This path reuses the internal token-exchange endpoint
that official editor plugins call. The :envvar:`IRIS_ENABLE_COPILOT_BACKEND`
environment variable must be explicitly set to ``1`` to opt in.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from iris_harness.llm.client import (
    ApiCredentialProvider,
    CodingLLMConfig,
    register_credential_provider_factory,
)

logger = logging.getLogger(__name__)

COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
DEVICE_CODE_URL = "https://github.com/login/device/code"
ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"  # noqa: S105
COPILOT_TOKEN_EXCHANGE_URL = "https://api.github.com/copilot_internal/v2/token"  # noqa: S105
DEFAULT_SCOPE = "read:user"
ENABLE_ENV_VAR = "IRIS_ENABLE_COPILOT_BACKEND"
EDITOR_VERSION = "iris-coding/0.1"
INTEGRATION_ID = "vscode-chat"

DEFAULT_CACHE_PATH = Path.home() / ".config" / "iris" / "copilot" / "oauth.json"

TOKEN_REFRESH_SAFETY_WINDOW_SECONDS = 60.0


def _mask_secret(secret: str) -> str:
    """Return a log-safe fingerprint of a secret string."""
    if not secret:
        return "<empty>"
    if len(secret) <= 8:
        return "***"
    return f"{secret[:4]}***{secret[-2:]}"


class CopilotAuthError(RuntimeError):
    """Raised when a Copilot authentication step fails."""


class CopilotBackendDisabledError(RuntimeError):
    """Raised when the Copilot backend is selected without the opt-in env flag."""


@dataclass(frozen=True)
class DeviceCodeResponse:
    """Parsed response from the device-code endpoint."""

    device_code: str
    user_code: str
    verification_uri: str
    expires_in: int
    interval: int


@dataclass(frozen=True)
class OAuthCacheEntry:
    """Persisted OAuth access token + minimal metadata."""

    access_token: str
    token_type: str = "bearer"  # noqa: S105 — the OAuth token *type*, not a token
    scope: str = DEFAULT_SCOPE

    def to_payload(self) -> dict[str, str]:
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "scope": self.scope,
        }


class _SupportsPost(Protocol):
    def post(self, url: str, *args: Any, **kwargs: Any) -> Any: ...


class _SupportsGet(Protocol):
    def get(self, url: str, *args: Any, **kwargs: Any) -> Any: ...


class HttpClient(_SupportsPost, _SupportsGet, Protocol):
    """Minimal subset of ``httpx.Client`` used by this module."""


NowFn = Callable[[], float]
SleepFn = Callable[[float], None]


def _assert_copilot_enabled(environ: Mapping[str, str] | None = None) -> None:
    """Guarantee the Copilot opt-in flag is set before any network calls."""
    source = os.environ if environ is None else environ
    flag = (source.get(ENABLE_ENV_VAR) or "").strip()
    if flag != "1":
        raise CopilotBackendDisabledError(
            "Copilot backend disabled. Set "
            + ENABLE_ENV_VAR
            + "=1 to opt in (note: this uses the Copilot Chat endpoint via the "
            "editor integration flow; review GitHub's Copilot terms before use)."
        )


def _default_http_client() -> httpx.Client:
    return httpx.Client(timeout=30.0)


def _write_cache(path: Path, entry: OAuthCacheEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry.to_payload(), indent=2), encoding="utf-8")
    os.chmod(path, 0o600)


def _read_cache(path: Path) -> OAuthCacheEntry:
    if not path.exists():
        raise CopilotAuthError(
            "No Copilot OAuth token found. Run `iris-coding auth copilot login` first."
        )
    payload_text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        raise CopilotAuthError(f"OAuth cache file is malformed (invalid JSON): {exc}") from exc
    if not isinstance(payload, dict):
        raise CopilotAuthError("OAuth cache file is malformed (expected JSON object)")
    token = str(payload.get("access_token") or "").strip()
    if not token:
        raise CopilotAuthError("OAuth cache file is missing an access_token")
    return OAuthCacheEntry(
        access_token=token,
        token_type=str(payload.get("token_type") or "bearer"),
        scope=str(payload.get("scope") or DEFAULT_SCOPE),
    )


def _delete_cache(path: Path) -> bool:
    if path.exists():
        path.unlink()
        return True
    return False


class CopilotDeviceFlow:
    """Drive the GitHub device-code OAuth flow and persist the access token."""

    def __init__(
        self,
        *,
        cache_path: Path = DEFAULT_CACHE_PATH,
        http_client: HttpClient | None = None,
        client_id: str = COPILOT_CLIENT_ID,
        scope: str = DEFAULT_SCOPE,
        now: NowFn | None = None,
        sleep: SleepFn | None = None,
    ) -> None:
        self._cache_path = cache_path
        self._http = http_client or _default_http_client()
        self._client_id = client_id
        self._scope = scope
        self._now = now or time.monotonic
        self._sleep = sleep or time.sleep

    def request_device_code(self) -> DeviceCodeResponse:
        """Ask GitHub for a new device code and user-facing verification URL."""
        response = self._http.post(
            DEVICE_CODE_URL,
            data={"client_id": self._client_id, "scope": self._scope},
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
        return DeviceCodeResponse(
            device_code=str(payload["device_code"]),
            user_code=str(payload["user_code"]),
            verification_uri=str(
                payload.get("verification_uri")
                or payload.get("verification_url")
                or "https://github.com/login/device"
            ),
            expires_in=int(payload.get("expires_in", 900)),
            interval=int(payload.get("interval", 5)),
        )

    def poll_for_access_token(self, device: DeviceCodeResponse) -> OAuthCacheEntry:
        """Poll GitHub for the user's access token until authorization completes."""
        deadline = self._now() + device.expires_in
        interval = float(device.interval)
        while self._now() < deadline:
            self._sleep(interval)
            response = self._http.post(
                ACCESS_TOKEN_URL,
                data={
                    "client_id": self._client_id,
                    "device_code": device.device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={"Accept": "application/json"},
            )
            if response.status_code >= 500:
                response.raise_for_status()
            payload = response.json()
            error = payload.get("error")
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval = interval + 5
                continue
            if error:
                raise CopilotAuthError("Device-flow authorization failed: " + str(error))
            access_token = payload.get("access_token")
            if not isinstance(access_token, str) or not access_token.strip():
                raise CopilotAuthError("Unexpected device-flow response: " + json.dumps(payload))
            entry = OAuthCacheEntry(
                access_token=access_token.strip(),
                token_type=str(payload.get("token_type") or "bearer"),
                scope=str(payload.get("scope") or self._scope),
            )
            _write_cache(self._cache_path, entry)
            logger.info(
                "Persisted Copilot OAuth token %s to %s",
                _mask_secret(entry.access_token),
                self._cache_path,
            )
            return entry
        raise CopilotAuthError("Device-flow authorization timed out before user approved")

    def login(
        self, announce: Callable[[DeviceCodeResponse], None] | None = None
    ) -> OAuthCacheEntry:
        """Run the full device-flow login and persist the token."""
        device = self.request_device_code()
        if announce is not None:
            announce(device)
        return self.poll_for_access_token(device)


class CopilotTokenProvider:
    """Exchange the persisted OAuth token for a short-lived Copilot API token on demand."""

    def __init__(
        self,
        *,
        cache_path: Path = DEFAULT_CACHE_PATH,
        http_client: HttpClient | None = None,
        now: NowFn | None = None,
        editor_version: str = EDITOR_VERSION,
        integration_id: str = INTEGRATION_ID,
    ) -> None:
        self._cache_path = cache_path
        self._http = http_client or _default_http_client()
        self._now = now or time.time
        self._editor_version = editor_version
        self._integration_id = integration_id
        self._cached_token: str | None = None
        self._cached_expires_at: float = 0.0

    def _ensure_oauth_entry(self) -> OAuthCacheEntry:
        entry = _read_cache(self._cache_path)
        if not entry.access_token:
            raise CopilotAuthError(
                "No Copilot OAuth token found. Run `iris-coding auth copilot login` first."
            )
        return entry

    def _needs_refresh(self) -> bool:
        if self._cached_token is None:
            return True
        return self._now() >= (self._cached_expires_at - TOKEN_REFRESH_SAFETY_WINDOW_SECONDS)

    def _refresh_api_token(self) -> None:
        oauth = self._ensure_oauth_entry()
        response = self._http.get(
            COPILOT_TOKEN_EXCHANGE_URL,
            headers={
                "Authorization": f"token {oauth.access_token}",
                "Editor-Version": self._editor_version,
                "Accept": "application/json",
            },
        )
        response.raise_for_status()
        payload = response.json()
        token = payload.get("token")
        if not isinstance(token, str) or not token.strip():
            raise CopilotAuthError("Copilot token exchange returned no token")
        expires_at = payload.get("expires_at")
        if not isinstance(expires_at, (int, float)):
            raise CopilotAuthError("Copilot token exchange returned no expires_at")
        self._cached_token = token.strip()
        self._cached_expires_at = float(expires_at)
        logger.debug(
            "Refreshed Copilot API token %s (expires_at=%s)",
            _mask_secret(self._cached_token),
            expires_at,
        )

    def get_token(self) -> str:
        if self._needs_refresh():
            self._refresh_api_token()
        assert self._cached_token is not None  # for the type checker
        return self._cached_token

    def extra_headers(self) -> Mapping[str, str]:
        return {
            "Editor-Version": self._editor_version,
            "Copilot-Integration-Id": self._integration_id,
        }

    def status(self) -> dict[str, str]:
        """Return a log-safe summary of the current auth state."""
        oauth = _read_cache(self._cache_path)
        if self._cached_token is None:
            seconds_remaining = 0
        else:
            seconds_remaining = max(0, int(self._cached_expires_at - self._now()))
        return {
            "oauth_token": _mask_secret(oauth.access_token),
            "api_token": _mask_secret(self._cached_token) if self._cached_token else "<none>",
            "api_token_expires_in_seconds": str(seconds_remaining),
            "cache_path": str(self._cache_path),
        }


def build_copilot_credential_provider(
    config: CodingLLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
    cache_path: Path = DEFAULT_CACHE_PATH,
    http_client: HttpClient | None = None,
    now: NowFn | None = None,
) -> ApiCredentialProvider:
    """Factory used by :mod:`iris_harness.llm.client` when ``auth_mode=copilot``."""
    _assert_copilot_enabled(environ)
    return CopilotTokenProvider(
        cache_path=cache_path,
        http_client=http_client,
        now=now,
    )


def logout(cache_path: Path = DEFAULT_CACHE_PATH) -> bool:
    """Delete the persisted OAuth token. Returns ``True`` if a token was removed."""
    return _delete_cache(cache_path)


# Register the factory at import time so the LLM client can lazy-load it.
register_credential_provider_factory("copilot", build_copilot_credential_provider)
