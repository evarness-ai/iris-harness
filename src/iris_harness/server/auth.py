"""Installing the shared bearer-token policy on a FastAPI app.

The policy itself is ``iris_harness.foundation.auth`` -- pure, FastAPI-free, and
read by clients as well as servers. Only this half needs a web framework, so only
this half lives in the server layer (OSS plan M6, decision 7).
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import socket
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from iris_harness.foundation.auth import (
    DEFAULT_EXEMPT_PATHS,
    DEVICE_COOKIE,
    DeviceVerifier,
    resolve_principal,
)
from iris_harness.foundation.logsafe import log_safe

if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.requests import HTTPConnection

logger = logging.getLogger(__name__)

# host[:port] and nothing else: a DNS name (underscores allowed, for compose service
# names), an IPv4 address, or a bracketed IPv6 literal. No "/", "?", "#", "@" or
# whitespace, which is what an attacker needs to smuggle a path through the header.
_VALID_HOST = re.compile(r"^(?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9._-]+)(?::[0-9]{1,5})?$")


def routed_path(request: HTTPConnection) -> str:
    """The path the router matched: the ONLY path a security decision may read.

    ``request.url`` is rebuilt from the Host header, so on starlette 1.0.0 a request
    for ``/memory/facts`` sent with ``Host: 127.0.0.1/healthz?`` had
    ``request.url.path == "/healthz"`` — an exempt path — while the router still
    served ``/memory/facts`` (PYSEC-2026-161). ``scope["path"]`` is what routing uses.
    """
    return str(request.scope["path"])


def valid_host_header(value: str | None) -> bool:
    """Whether a Host header is a plain ``host[:port]`` (a missing header is fine)."""
    return value is None or bool(_VALID_HOST.match(value))


# ── DNS rebinding: which host NAMES this server answers to ────────────────────
#
# A page on attacker.example can re-resolve its own name to 127.0.0.1 (or to the VM's
# tailnet address) and then read IRIS's answers as same-origin: the browser sends
# ``Host: attacker.example``. A well-formed Host is therefore not enough; the name has
# to be one this deployment is actually reached by. Only names are checked. An IP
# literal is always accepted: a rebinding attack needs a DNS name the attacker
# controls, and a page whose origin is the IP itself is already that IP's page.

ALLOWED_HOSTS_ENV = "IRIS_ALLOWED_HOSTS"
# Read raw here (foundation.public_url logs on a bad value, and this runs per request).
_PUBLIC_URL_ENV = "IRIS_PUBLIC_URL"

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
# Starlette's TestClient sends ``Host: testserver``. A single-label name no public DNS
# hands to an attacker, so accepting it in production costs nothing.
TEST_CLIENT_HOSTS = frozenset({"testserver"})
# The compose service names, which a sibling container calls a service by (the local
# stack's iris-api reaches ``http://governor:8080``): docker-compose.yml's services and
# the channel gateway, the one harness server that stack does not run. Pinned to
# docker-compose.yml by a test, so a renamed service cannot start answering 400. A
# deployment that calls a harness server by another name lists it in IRIS_ALLOWED_HOSTS.
COMPOSE_SERVICE_HOSTS = frozenset(
    {"governor", "iris-api", "channel-gateway", "ollama", "ollama-init"}
)

_HOST_NOT_ALLOWED_DETAIL = (
    "Host header names a host this server does not answer to. Set IRIS_PUBLIC_URL to the "
    "address you reach it at, or add the name to IRIS_ALLOWED_HOSTS."
)
_REFUSED_LOGGED: set[str] = set()
_REFUSED_LOG_CAP = 64


def host_name(value: str) -> str:
    """The host part of a ``host[:port]`` header: lowercased, no brackets, no port.

    Expects a value :func:`valid_host_header` accepted, so an unbracketed value has at
    most one ``:``. A trailing dot (``localhost.``, the fully-qualified spelling) goes.
    """
    host = value.strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[1:end] if end > 0 else host
    return host.partition(":")[0].rstrip(".")


def _entry_host(entry: str) -> str:
    """One ``IRIS_ALLOWED_HOSTS`` entry as a bare name: ``host``, ``host:port`` or a URL."""
    entry = entry.strip()
    if "://" in entry:
        try:
            return (urlsplit(entry).hostname or "").rstrip(".")
        except ValueError:
            return ""
    return host_name(entry) if entry else ""


def _with_short_name(name: str) -> set[str]:
    """``name`` plus its first label: MagicDNS (and a LAN's search domain) resolve the
    short name too, and it is what a client on the tailnet calls the machine by."""
    if not name:
        return set()
    return {name, name.split(".", 1)[0]}


@lru_cache(maxsize=1)
def _machine_names() -> frozenset[str]:
    try:
        return frozenset(_with_short_name(socket.gethostname().lower().rstrip(".")))
    except OSError:
        return frozenset()


@lru_cache(maxsize=8)
def _allowed_host_names(public_url: str, extra: str) -> frozenset[str] | None:
    entries = [part.strip() for part in extra.split(",") if part.strip()]
    if "*" in entries:
        return None
    names = set(LOOPBACK_HOSTS | TEST_CLIENT_HOSTS | COMPOSE_SERVICE_HOSTS | _machine_names())
    try:
        public_host = urlsplit(public_url.strip()).hostname or ""
    except ValueError:
        public_host = ""
    names |= _with_short_name(public_host.rstrip("."))
    names |= {_entry_host(part) for part in entries}
    names.discard("")
    return frozenset(names)


def allowed_host_names() -> frozenset[str] | None:
    """The host names this server answers to, or ``None`` when any name is allowed.

    Loopback, ``testserver``, the compose service names, this machine's hostname, the
    host of ``IRIS_PUBLIC_URL`` (the tailnet name on the cloud harness), each of those
    last two with its short form, and every entry of ``IRIS_ALLOWED_HOSTS``
    (comma-separated). ``IRIS_ALLOWED_HOSTS=*`` turns the check off, for a front this
    list cannot know about.
    """
    return _allowed_host_names(
        os.environ.get(_PUBLIC_URL_ENV, ""), os.environ.get(ALLOWED_HOSTS_ENV, "")
    )


def host_allowed(value: str | None) -> bool:
    """Whether a well-formed Host header names a host this server answers to.

    A missing header passes (no browser omits it; HTTP/1.0 tooling may), and so does
    any IP literal (see the block comment above).
    """
    if value is None:
        return True
    name = host_name(value)
    try:
        ipaddress.ip_address(name)
    except ValueError:
        pass
    else:
        return True
    allowed = allowed_host_names()
    return allowed is None or name in allowed


def host_header_ok(value: str | None) -> bool:
    """Both Host checks: well-formed (:func:`valid_host_header`) and allowed."""
    return valid_host_header(value) and host_allowed(value)


def log_refused_host(value: str) -> None:
    """One WARNING per refused name: the host only, nothing else from the request."""
    name = host_name(value)
    if name in _REFUSED_LOGGED:
        logger.debug("refused Host %r again", log_safe(name))
        return
    if len(_REFUSED_LOGGED) < _REFUSED_LOG_CAP:
        _REFUSED_LOGGED.add(name)
    logger.warning(
        "refused a request for Host %r: not a name this server answers to "
        "(set IRIS_PUBLIC_URL, or add it to IRIS_ALLOWED_HOSTS)",
        log_safe(name),
    )


def install_host_guard(app: FastAPI) -> None:
    """Register the Host-header guard: 400 for a malformed Host or an unknown name.

    :func:`install_bearer_auth` calls this itself; a service without bearer auth (the
    channel gateway) calls it directly. Register it before any logging middleware so
    the log stays outermost.
    """
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    # A malformed Host is refused before anything can build a URL from it. Defense in
    # depth over ``routed_path``: any code that reads ``request.url`` (a redirect,
    # url_for, a log line) is then fed a host that cannot carry a path. A well-formed
    # Host naming a host this server is not reached by is DNS rebinding.
    @app.middleware("http")
    async def _host_guard(request: Request, call_next: Any) -> Any:
        host = request.headers.get("host")
        if not valid_host_header(host):
            return JSONResponse(status_code=400, content={"detail": "invalid Host header"})
        if not host_allowed(host):
            log_refused_host(str(host))
            return JSONResponse(status_code=400, content={"detail": _HOST_NOT_ALLOWED_DETAIL})
        return await call_next(request)


def install_bearer_auth(
    app: FastAPI,
    *,
    exempt_paths: frozenset[str] = DEFAULT_EXEMPT_PATHS,
    device_verifier: DeviceVerifier | None = None,
    public_route: Callable[[str, str], bool] | None = None,
) -> None:
    """Register the bearer-auth middleware on a FastAPI app.

    Starlette runs the middleware registered LAST first, so call this before
    registering the ingress-log middleware — the log stays outermost and
    refused attempts still land in the ingress trail.

    ``device_verifier`` opts a service into paired-device tokens (ADR-0117), from
    the bearer header or the ``iris_device`` cookie. Without it the shared secret is
    the only credential, which is what the governor and the evaluator want: nothing
    but another IRIS service calls them. Either way an authenticated request carries
    ``request.state.principal``; an exempt path carries ``None``.

    ``public_route(method, path)`` names further requests that pass without a
    credential, asked per request because plugins declare theirs after the app is
    built (``runtime.api_routes.register_public_callback``: an OAuth redirect that
    the route authenticates itself, by its one-time ``state``). It reads the routed
    path, like everything here, and it never widens the Host guard below.

    It also installs the Host-header guard (:func:`install_host_guard`: 400 for
    anything but ``host[:port]``, or for a host name this server is not reached by),
    which runs before the bearer check.
    """
    from functools import partial

    from starlette.concurrency import run_in_threadpool
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @app.middleware("http")
    async def _bearer_auth(request: Request, call_next: Any) -> Any:
        if public_route is not None and public_route(request.method, routed_path(request)):
            request.state.principal = None
            return await call_next(request)
        resolve = partial(
            resolve_principal,
            authorization=request.headers.get("authorization"),
            cookie_token=request.cookies.get(DEVICE_COOKIE),
            method=request.method,
            path=routed_path(request),
            origin=request.headers.get("origin"),
            host=request.headers.get("host"),
            verifier=device_verifier,
            exempt_paths=exempt_paths,
        )
        # A device lookup reads SQLite, so it leaves the event loop; the secret-only
        # check is a string compare and stays on it.
        outcome = resolve() if device_verifier is None else await run_in_threadpool(resolve)
        if not isinstance(outcome, tuple):
            request.state.principal = outcome
            return await call_next(request)
        status, detail = outcome
        headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
        return JSONResponse(status_code=status, content={"detail": detail}, headers=headers)

    # Registered after the bearer check, so it runs BEFORE it.
    install_host_guard(app)


class PublicCallbackQueryRedactor(logging.Filter):
    """Keep a public callback's query string out of the server's access log.

    uvicorn's access log prints the whole request line. On a public callback (see
    ``install_bearer_auth``'s ``public_route``) the query IS the request's credential
    -- an OAuth ``state`` and a one-time authorization ``code`` -- so the line keeps
    the path and drops the query. Every other request line is left alone.
    """

    def __init__(self, public_route: Callable[[str, str], bool]) -> None:
        super().__init__()
        self._public_route = public_route

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        # uvicorn.access: (client_addr, method, full_path, http_version, status_code)
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            path, sep, _ = args[2].partition("?")
            if sep and self._public_route(str(args[1]), path):
                record.args = (*args[:2], f"{path}?<redacted>", *args[3:])
        return True


def install_callback_log_redaction(public_route: Callable[[str, str], bool]) -> None:
    """Add :class:`PublicCallbackQueryRedactor` to uvicorn's access log, once."""
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, PublicCallbackQueryRedactor) for f in access.filters):
        access.addFilter(PublicCallbackQueryRedactor(public_route))


__all__ = [
    "ALLOWED_HOSTS_ENV",
    "COMPOSE_SERVICE_HOSTS",
    "LOOPBACK_HOSTS",
    "TEST_CLIENT_HOSTS",
    "PublicCallbackQueryRedactor",
    "allowed_host_names",
    "host_allowed",
    "host_header_ok",
    "host_name",
    "install_bearer_auth",
    "install_callback_log_redaction",
    "install_host_guard",
    "log_refused_host",
    "routed_path",
    "valid_host_header",
]
