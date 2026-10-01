"""The CLI's one way to call the harness's own services (IRIS API, evaluator, ...).

Every outbound network call is logged on ``iris.egress`` (``log_egress``), and a CLI
command talking to the running IRIS is an outbound call like any other: the server logs
it as ingress, and without this module the client side was silent. So every HTTP call a
CLI module makes goes through here, and ``tests/security/test_cli_egress_logged.py``
fails the build when one does not.

Two entry points, one log line each, for the two transports the CLI uses:

- ``harness_api_client`` — an ``httpx.Client`` whose request hook logs every request it
  sends (redirects included) and which carries the service secret.
- ``harness_urlopen`` — ``urllib.request.urlopen`` with the log line in front. The
  urllib callers keep their ``urllib.error.HTTPError`` / ``OSError`` handling as is.

The line names the host (and port) only: never the path, which can carry IDs, never
the query string, and never userinfo or a header. ``purpose`` says what the call is for.
"""

from __future__ import annotations

import urllib.request
from typing import Any
from urllib.parse import urlsplit

import httpx

from iris_harness.foundation.auth import auth_headers
from iris_harness.foundation.observability.logging_setup import log_egress


def api_host(url: str) -> str:
    """``host[:port]`` of ``url``: no scheme, userinfo, path, query or fragment."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = parts.port
    except ValueError:
        return "unknown"
    if not host:
        return "unknown"
    if ":" in host:  # an IPv6 literal keeps its brackets so the port stays readable
        host = f"[{host}]"
    return f"{host}:{port}" if port else host


def _log_call(method: str, url: str, purpose: str) -> None:
    log_egress(destination=api_host(url), method=method, kind="service", purpose=purpose)


def harness_api_client(
    *,
    purpose: str,
    timeout: float,
    auth: bool = True,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """An ``httpx.Client`` for the harness's own services that logs each request.

    ``auth`` sends the service secret (``auth_headers``), which every data route needs;
    pass ``False`` only for an open probe such as ``/healthz``. ``transport`` is httpx's
    own seam (tests pass a ``MockTransport``; the request hook still runs). Use it as a
    context manager so the connection pool closes.
    """

    def _on_request(request: httpx.Request) -> None:
        _log_call(request.method, str(request.url), purpose)

    return httpx.Client(
        timeout=timeout,
        headers=auth_headers() if auth else None,
        event_hooks={"request": [_on_request]},
        transport=transport,
    )


def harness_urlopen(request: urllib.request.Request, *, purpose: str, timeout: float) -> Any:
    """``urllib.request.urlopen(request)`` for the harness's own services, logged first.

    The caller builds the ``Request`` (headers, auth, body) and handles its errors;
    this adds only the egress line. The URL is the operator's own API URL.
    """
    _log_call(request.get_method(), request.full_url, purpose)
    return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310


__all__ = ["api_host", "harness_api_client", "harness_urlopen"]
