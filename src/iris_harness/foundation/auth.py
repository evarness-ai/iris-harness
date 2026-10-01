"""The bearer-token policy every IRIS HTTP service and client shares.

Pure functions only, and no FastAPI: this module is read by the CLI, the
evaluator client and the coding agent as well as by the servers, so it sits in
the foundation where all of them can reach it. The middleware that installs the
policy on an app is ``iris_harness.server.auth`` (OSS plan M6, decision 7).

Security-floor phase 0 (single-user, local-first): every route on the backend
services (iris_api :8003, governor :8080, evaluator :8090) requires
``Authorization: Bearer $IRIS_AUTH_SECRET``. Liveness probes stay open so the
startup script and process supervisors can poll readiness without a secret.

Fail-closed: when the secret is unset, non-exempt requests are refused with
503 (operator misconfiguration) rather than served unauthenticated. This is
deliberately code, not config — a YAML edit must not be able to widen the
authentication boundary.

Device tokens (ADR-0117): a service may also accept a paired device's token, from
the bearer header or the ``iris_device`` cookie. The shared secret stays the
service credential; :func:`resolve_principal` says which of the two a request
carried, and :func:`authorize` is unchanged for the services that accept only the
secret. The lookup itself is injected (:class:`DeviceVerifier`) so this module
stays free of storage.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit

DEFAULT_EXEMPT_PATHS = frozenset({"/healthz", "/health"})
# For a service whose ``/health`` is data rather than a liveness probe (iris_api's is
# the full System Health snapshot), only the bare probe stays open.
PROBE_ONLY_EXEMPT_PATHS = frozenset({"/healthz"})

DEVICE_COOKIE = "iris_device"
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

Scope = Literal["read", "control"]

_UNSET_DETAIL = (
    "IRIS_AUTH_SECRET is not set — this service refuses unauthenticated requests. "
    "Export IRIS_AUTH_SECRET and send 'Authorization: Bearer <secret>'."
)
_DENIED_DETAIL = "missing or invalid bearer token"
_CROSS_ORIGIN_DETAIL = "cookie-authenticated writes must come from this console's own origin"


@dataclass(frozen=True)
class Principal:
    """Who a request was authenticated as.

    ``service`` is the shared secret (the CLI, the Vite proxy, service-to-service);
    it always has ``control`` scope. ``device`` is a paired device, with the scope it
    was paired at. ``via`` records where the credential rode in, because only a
    cookie is attached by the browser on its own and so only a cookie needs the
    same-origin check.
    """

    kind: Literal["service", "device"]
    scope: Scope
    device_id: str | None = None
    via: Literal["bearer", "cookie"] = "bearer"

    @property
    def can_control(self) -> bool:
        return self.scope == "control"


SERVICE_PRINCIPAL = Principal(kind="service", scope="control")

# Token in, ``(device_id, scope)`` out — or ``None`` for an unknown or revoked token.
DeviceVerifier = Callable[[str], "tuple[str, Scope] | None"]


def expected_secret() -> str | None:
    """Return the configured shared secret, or ``None`` when unset/blank."""
    secret = os.environ.get("IRIS_AUTH_SECRET", "")
    return secret if secret.strip() else None


def bearer_token(authorization: str | None) -> str | None:
    """Extract the token from an ``Authorization: Bearer …`` header value."""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        return token or None
    return None


def authorize(
    authorization: str | None,
    path: str,
    exempt_paths: frozenset[str] = DEFAULT_EXEMPT_PATHS,
) -> tuple[int, str] | None:
    """Return ``None`` when the request may proceed, else ``(status, detail)``.

    Pure function so the services and their tests share exactly one policy:
    exempt paths always pass; an unset secret fails closed with 503; a
    missing or mismatched token gets 401 (constant-time compare).
    """
    if path in exempt_paths:
        return None
    secret = expected_secret()
    if secret is None:
        return 503, _UNSET_DETAIL
    presented = bearer_token(authorization)
    if presented is None or not hmac.compare_digest(presented.encode(), secret.encode()):
        return 401, _DENIED_DETAIL
    return None


def same_origin(origin: str | None, host: str | None) -> bool:
    """True when an ``Origin`` header names the host the request was sent to.

    Compares host and port only: behind ``tailscale serve`` the browser speaks
    https while the app sees http, so the scheme cannot be compared. A missing or
    unparseable ``Origin`` (including the literal ``null``) is not same-origin.
    """
    if not origin or not host:
        return False
    try:
        netloc = urlsplit(origin).netloc
    except ValueError:
        return False
    return bool(netloc) and netloc.lower() == host.strip().lower()


def resolve_principal(  # one early return per refusal, as in authorize()
    *,
    authorization: str | None,
    cookie_token: str | None,
    method: str,
    path: str,
    origin: str | None,
    host: str | None,
    verifier: DeviceVerifier | None,
    exempt_paths: frozenset[str] = DEFAULT_EXEMPT_PATHS,
) -> Principal | tuple[int, str] | None:
    """Authenticate a request as the service or as a paired device.

    Returns ``None`` for an exempt path (no principal, no check), a
    :class:`Principal` when the request may proceed, else ``(status, detail)``.

    Order matters. The unset-secret 503 comes first and covers devices too: a
    service with no secret is misconfigured, and a device token must not turn that
    into a working deployment. A bearer header, when present, is the only credential
    considered — a bad one is refused rather than falling back to the cookie, so a
    client never authenticates as something other than what it presented. With
    ``verifier=None`` this accepts exactly what :func:`authorize` accepts.
    """
    if path in exempt_paths:
        return None
    secret = expected_secret()
    if secret is None:
        return 503, _UNSET_DETAIL

    presented = bearer_token(authorization)
    if presented is not None:
        if hmac.compare_digest(presented.encode(), secret.encode()):
            return SERVICE_PRINCIPAL
        device = verifier(presented) if verifier is not None else None
        if device is None:
            return 401, _DENIED_DETAIL
        return Principal(kind="device", scope=device[1], device_id=device[0], via="bearer")

    if verifier is None or not cookie_token:
        return 401, _DENIED_DETAIL
    device = verifier(cookie_token)
    if device is None:
        return 401, _DENIED_DETAIL
    # The browser attaches a cookie to any request for this host, including one a
    # hostile page triggers. SameSite=Strict is the first defence; this is the one
    # that does not depend on the browser.
    if method.upper() in UNSAFE_METHODS and not same_origin(origin, host):
        return 403, _CROSS_ORIGIN_DETAIL
    return Principal(kind="device", scope=device[1], device_id=device[0], via="cookie")


def auth_headers() -> dict[str, str]:
    """Headers internal clients attach when calling IRIS services.

    Empty when the secret is unset so client construction never fails — the
    server side is what enforces (and then refuses with 503/401).
    """
    secret = expected_secret()
    if secret is None:
        return {}
    return {"Authorization": f"Bearer {secret}"}


__all__ = [
    "DEFAULT_EXEMPT_PATHS",
    "DEVICE_COOKIE",
    "PROBE_ONLY_EXEMPT_PATHS",
    "SERVICE_PRINCIPAL",
    "UNSAFE_METHODS",
    "DeviceVerifier",
    "Principal",
    "Scope",
    "auth_headers",
    "authorize",
    "bearer_token",
    "expected_secret",
    "resolve_principal",
    "same_origin",
]
