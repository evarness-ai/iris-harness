"""The pairing flow over HTTP: ``/api/v1/devices`` (ADR-0117, "Pairing flow").

Thin callers of ``kernel.governance.devices.DeviceService``: the code, the token, the
attempt counts, the throttle and the ledger rows all live there. What is decided
*here* is only what HTTP adds — which principal may call which route, and how a
browser is handed its token (an ``HttpOnly`` cookie, never the response body).

Two facts about where these routes sit in ``iris_api``'s middleware, both set up in
``main.create_app`` from the constants below:

- ``PAIR_CLAIM_PATH`` is exempt from the bearer check. It is the one data route with
  no credential: a device that is not paired yet has nothing to present.
- The three mutating routes are exempt from ``IRIS_WEBUI_ALLOW_WRITES`` and from the
  write guard's blanket read-device refusal (:func:`is_device_admin_write`). Device
  administration is authentication administration, not a console write: behind that
  switch nobody could pair on a default install, and a lost phone could not be
  revoked from a read-only console. Scope is enforced in each route instead.

The handlers are plain ``def``: FastAPI runs them in its threadpool, so the SQLite
calls underneath never block the event loop.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Literal

from fastapi import HTTPException, Request, Response
from pydantic import BaseModel

from iris_harness.foundation.auth import DEVICE_COOKIE, Principal, Scope

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import FastAPI

    from iris_harness.kernel.governance.devices import DeviceRow, DeviceService

__all__ = [
    "DEVICES_PATH",
    "DEVICE_COOKIE_MAX_AGE",
    "PAIR_CLAIM_PATH",
    "PAIR_START_PATH",
    "LazyDeviceService",
    "install_device_routes",
    "is_device_admin_write",
]

DEVICES_PATH = "/api/v1/devices"
PAIR_START_PATH = f"{DEVICES_PATH}/pair/start"
PAIR_CLAIM_PATH = f"{DEVICES_PATH}/pair/claim"

# 400 days is the ceiling browsers apply to a cookie's lifetime anyway; the token
# itself does not expire — a device stays paired until it is revoked.
DEVICE_COOKIE_MAX_AGE = 400 * 24 * 60 * 60

_READ_ONLY_DETAIL = "this device is paired read-only"
_CLAIM_REFUSED_DETAIL = "pairing code not accepted"
_THROTTLED_DETAIL = "too many pairing attempts — try again shortly"


def is_device_admin_write(method: str, path: str) -> bool:
    """True for exactly the three mutating device routes, and nothing near them.

    Exact on purpose. The write guard is deny-by-default so that a NEW mutating route
    is gated without anyone remembering to gate it; a prefix match here would quietly
    exempt whatever is added under ``/api/v1/devices/`` next.
    """
    if method == "POST":
        return path in (PAIR_START_PATH, PAIR_CLAIM_PATH)
    if method == "DELETE":
        device_id = path.removeprefix(f"{DEVICES_PATH}/")
        return device_id != path and bool(device_id) and "/" not in device_id
    return False


class LazyDeviceService:
    """The app's one ``DeviceService``, built on first use.

    One, because the failed-claim throttle lives in the service object: the auth
    middleware's verifier and the routes must share it, and so must every claim.
    Lazy, because opening the devices DB is a disk write and a request carrying the
    shared secret never needs it. The lock matters: the verifier runs in a threadpool,
    and two first requests must not build two services.
    """

    def __init__(self, service: DeviceService | None = None) -> None:
        self._service = service
        self._lock = threading.Lock()

    def get(self) -> DeviceService:
        if self._service is None:
            with self._lock:
                if self._service is None:
                    from iris_harness.kernel.governance.devices import (
                        DeviceService,
                    )

                    self._service = DeviceService()
        return self._service

    def verify(self, token: str) -> tuple[str, Scope] | None:
        """The ``DeviceVerifier`` handed to ``install_bearer_auth``."""
        return self.get().verify(token)


class DevicePairStartRequest(BaseModel):
    """Body for ``POST /api/v1/devices/pair/start``."""

    scope: Literal["read", "control"] = "control"


class DevicePairClaimRequest(BaseModel):
    """Body for ``POST /api/v1/devices/pair/claim``. ``code`` is deliberately
    unconstrained: a malformed code gets the same 400 as a wrong one."""

    code: str
    name: str
    kind: Literal["app", "browser"]


def _device_payload(row: DeviceRow, *, current_id: str | None) -> dict[str, Any]:
    # Field by field, not ``asdict``: a column added to the row later must not reach
    # a response by default.
    return {
        "device_id": row.device_id,
        "name": row.name,
        "kind": row.kind,
        "scope": row.scope,
        "created_at": row.created_at,
        "last_seen_at": row.last_seen_at,
        "revoked_at": row.revoked_at,
        "current": current_id is not None and row.device_id == current_id,
    }


def _principal(request: Request) -> Principal:
    principal = getattr(request.state, "principal", None)
    if not isinstance(principal, Principal):
        # Unreachable while the bearer middleware is installed; if it ever is not,
        # these routes fail closed instead of treating "nobody" as the service.
        raise HTTPException(status_code=401, detail="missing or invalid bearer token")
    return principal


def _actor(principal: Principal) -> str:
    return "service" if principal.kind == "service" else f"device:{principal.device_id}"


def _arrived_over_https(request: Request) -> bool:
    """Whether the browser spoke HTTPS. Behind ``tailscale serve`` the app sees http
    and the proxy says so in ``X-Forwarded-Proto``. The header is not authenticated,
    but it can only ADD ``Secure``: a spoofed value makes the cookie stricter."""
    if request.url.scheme == "https":
        return True
    forwarded = request.headers.get("x-forwarded-proto", "")
    return forwarded.split(",")[0].strip().lower() == "https"


def _cookie_attributes(request: Request) -> dict[str, Any]:
    return {
        "key": DEVICE_COOKIE,
        "path": "/",
        "httponly": True,
        "samesite": "strict",
        "secure": _arrived_over_https(request),
    }


def install_device_routes(  # one closure per route, as in create_app
    app: FastAPI, devices: Callable[[], DeviceService]
) -> None:
    """Register the five device routes. ``devices`` returns the shared service."""
    from iris_harness.kernel.governance.devices import (
        DeviceNotFoundError,
        PairingRefusedError,
        PairingThrottledError,
    )

    @app.post(PAIR_START_PATH)
    def devices_pair_start(
        http_request: Request, body: DevicePairStartRequest | None = None
    ) -> dict[str, Any]:
        """Mint a pairing code. Service or a ``control`` device only."""
        principal = _principal(http_request)
        if not principal.can_control:
            raise HTTPException(status_code=403, detail=_READ_ONLY_DETAIL)
        scope = body.scope if body is not None else "control"
        code = devices().start_pairing(scope=scope, actor=_actor(principal))
        return {"code": code.code, "scope": code.scope, "expires_at": code.expires_at}

    @app.post(PAIR_CLAIM_PATH)
    def devices_pair_claim(
        body: DevicePairClaimRequest, http_request: Request, response: Response
    ) -> dict[str, Any]:
        """Trade a pairing code for a device token. No credential: the code is it."""
        try:
            paired = devices().claim(code=body.code, name=body.name, kind=body.kind)
        except PairingThrottledError as exc:
            raise HTTPException(
                status_code=429,
                detail=_THROTTLED_DETAIL,
                headers={"Retry-After": str(exc.retry_after)},
            ) from exc
        except PairingRefusedError as exc:
            # One answer for wrong, expired, used and voided: see PairingRefusedError.
            raise HTTPException(status_code=400, detail=_CLAIM_REFUSED_DETAIL) from exc
        except ValueError as exc:  # the device name; ``kind`` is already a Literal
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        payload: dict[str, Any] = {
            "device": _device_payload(paired.device, current_id=paired.device.device_id)
        }
        if paired.device.kind == "browser":
            # The token rides only in the HttpOnly cookie: page script never holds it.
            response.set_cookie(
                value=paired.token,
                max_age=DEVICE_COOKIE_MAX_AGE,
                **_cookie_attributes(http_request),
            )
        else:
            payload["token"] = paired.token
        return payload

    @app.get(DEVICES_PATH)
    def devices_list(http_request: Request) -> dict[str, Any]:
        """Every device, newest first, revoked ones included. Any principal reads."""
        current_id = _principal(http_request).device_id
        rows = sorted(devices().list_devices(), key=lambda r: r.created_at, reverse=True)
        return {"devices": [_device_payload(r, current_id=current_id) for r in rows]}

    @app.get(f"{DEVICES_PATH}/me")
    def devices_me(http_request: Request) -> dict[str, Any]:
        """Who this request is authenticated as."""
        principal = _principal(http_request)
        row = devices().get(principal.device_id) if principal.device_id else None
        return {
            "kind": principal.kind,
            "scope": principal.scope,
            "via": principal.via,
            "device": (
                _device_payload(row, current_id=principal.device_id) if row is not None else None
            ),
        }

    @app.delete(f"{DEVICES_PATH}/{{device_id}}")
    def devices_revoke(device_id: str, http_request: Request, response: Response) -> dict[str, Any]:
        """Revoke a device. Service or ``control`` revokes any; ANY device may revoke
        itself (sign out), a ``read`` one included."""
        principal = _principal(http_request)
        is_self = principal.kind == "device" and principal.device_id == device_id
        # Scope before existence: a read device learns nothing about which IDs exist.
        if not principal.can_control and not is_self:
            raise HTTPException(status_code=403, detail=_READ_ONLY_DETAIL)
        try:
            row = devices().revoke(device_id, actor=_actor(principal))
        except DeviceNotFoundError as exc:
            raise HTTPException(status_code=404, detail="no such device") from exc
        if is_self and principal.via == "cookie":
            response.delete_cookie(**_cookie_attributes(http_request))
        return {"device": _device_payload(row, current_id=principal.device_id)}
