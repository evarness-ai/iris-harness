"""Channel Gateway service — WebSocket bridge to the IRIS API.

MVP scope (see ``project-iris-prd/07-proactive-autonomy.md`` §7.4):
single FastAPI app on :8006 exposing ``/health`` and ``/ws``. The WS
handler authenticates with ``IRIS_AUTH_SECRET``, sends a 30 s
app-level ping, and proxies each ``chat`` frame to the IRIS API's
``POST /chat/stream`` (NDJSON), forwarding events back as WS frames.
When ``TELEGRAM_BOT_TOKEN`` is configured, the service also starts a
Telegram Bot API long-poller and routes inbound Telegram text through
the same IRIS API streaming endpoint before sending the final reply.

Single-instance only. Session state lives in the IRIS API; the gateway
keeps no per-session memory beyond the live WebSocket.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect, status

from iris_harness.foundation.env import env_flag
from iris_harness.foundation.paths import repo_root

from .main_types import TelegramRuntimeProtocol

_REPO_ROOT_ENV = repo_root() / ".env"
load_dotenv(_REPO_ROOT_ENV)

# Install the root log handler (IRIS_LOG_LEVEL, default INFO) so every iris.* INFO line —
# including the ingress/egress audit trail — appears in the service log instead of being
# dropped at the default WARNING level.
from iris_harness.foundation.auth import auth_headers  # noqa: E402
from iris_harness.foundation.observability.logging_setup import configure_logging  # noqa: E402
from iris_harness.server.auth import (  # noqa: E402
    host_header_ok,
    install_host_guard,
    log_refused_host,
    routed_path,
    valid_host_header,
)

configure_logging(service="channel_gateway")

logger = logging.getLogger(__name__)

# A browser WebSocket cannot set an Authorization header, so /ws also takes the shared
# secret as ``?token=``. Our own ingress line records the routed path only, but uvicorn
# logs every handshake with its query string ('"WebSocket /ws?token=..." [accepted]'),
# which put the secret in the service log. The filter rewrites the value before any
# handler sees the record.
_QUERY_TOKEN = re.compile(r"(?i)([?&](?:access_)?token=)[^&\s\"']*")


class _RedactQueryToken(logging.Filter):
    """Replace a ``token=`` query value in a log record with ``[redacted]``."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = _QUERY_TOKEN.sub(r"\1[redacted]", record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(
                _QUERY_TOKEN.sub(r"\1[redacted]", arg) if isinstance(arg, str) else arg
                for arg in record.args
            )
        return True


def _install_query_token_redaction() -> None:
    """Attach the filter to uvicorn's loggers once. uvicorn configures its logging
    before it imports this app and does not clear logger filters, so this sticks."""
    for name in ("uvicorn.error", "uvicorn.access"):
        target = logging.getLogger(name)
        if not any(isinstance(f, _RedactQueryToken) for f in target.filters):
            target.addFilter(_RedactQueryToken())


_install_query_token_redaction()

PING_INTERVAL_SECONDS = 30.0
PONG_GRACE_SECONDS = 10.0
UPSTREAM_TIMEOUT_SECONDS = 600.0  # streaming chat responses can run several minutes


def _env_flag(name: str, *, default: bool) -> bool:
    """Thin alias for the shared reader, keeping this module's semantics.

    One of six copies M6.3 found in three disagreeing variants; see
    ``iris_harness.foundation.env`` for what they disagreed about.
    """
    return env_flag(name, default=default)


def _iris_api_url() -> str:
    return os.environ.get("IRIS_API_URL", "http://localhost:8003").rstrip("/")


def _expected_auth_token() -> str | None:
    """Return the configured shared secret, or None when auth is disabled.

    The MVP requires IRIS_AUTH_SECRET to be set; without it the gateway
    refuses every connection. Tests set it via the same env var the rest
    of the suite already requires.
    """
    secret = os.environ.get("IRIS_AUTH_SECRET")
    return secret if secret else None


def _extract_bearer(websocket: WebSocket, query_token: str | None) -> str | None:
    """Pull the bearer token out of the handshake headers or query string."""
    header = websocket.headers.get("authorization")
    if header and header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return query_token or None


def _token_matches(presented: str | None, expected: str) -> bool:
    """Constant-time comparison, as ``foundation/auth.py`` does for the HTTP services.

    ``!=`` stops at the first differing byte, so response timing leaks how much of a
    guess was right."""
    return presented is not None and hmac.compare_digest(presented.encode(), expected.encode())


def create_app(
    *,
    http_client: httpx.AsyncClient | None = None,
    telegram_runtime_factory: Callable[[], TelegramRuntimeProtocol | None] | None = None,
) -> FastAPI:
    """Build the FastAPI app. ``http_client`` is injected by tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.tracer = None
        app.state.observability = None
        if _env_flag("IRIS_CHANNEL_GATEWAY_TRACING_ENABLED", default=False):
            try:
                from iris_harness.foundation.observability.tracer import setup_tracing_state

                app.state.observability = setup_tracing_state()
                app.state.tracer = app.state.observability.tracer
            except Exception:  # noqa: BLE001
                logger.warning("tracing setup failed; running without Phoenix")
        else:
            logger.info("channel gateway tracing disabled")

        app.state.http = http_client or httpx.AsyncClient(
            timeout=UPSTREAM_TIMEOUT_SECONDS, headers=auth_headers()
        )
        app.state.telegram_runtime = None
        owns_http = http_client is None
        try:
            factory = telegram_runtime_factory
            if factory is None:
                from .telegram import build_telegram_runtime_from_env

                def factory() -> TelegramRuntimeProtocol | None:
                    return build_telegram_runtime_from_env(iris_api_url=_iris_api_url())

            telegram_runtime = factory()
            if telegram_runtime is not None:
                telegram_runtime.start()
                app.state.telegram_runtime = telegram_runtime
        except Exception:
            logger.exception("telegram gateway poller failed to start")
        try:
            yield
        finally:
            telegram_runtime = getattr(app.state, "telegram_runtime", None)
            if telegram_runtime is not None:
                telegram_runtime.stop()
            if owns_http:
                await app.state.http.aclose()

    app = FastAPI(title="IRIS Channel Gateway", version="0.1.0", lifespan=lifespan)
    # No bearer middleware here (/ws authenticates itself), but the same Host guard as
    # the other services: DNS rebinding reaches /health like any open route. It covers
    # HTTP only; the /ws handshake checks the Host itself. Registered before the
    # ingress log, so the log stays outermost.
    install_host_guard(app)

    @app.middleware("http")
    async def _ingress_log(request: Request, call_next: Any) -> Any:
        # Log EVERY inbound HTTP request crossing the service boundary (no allowlist),
        # so there is a complete ingress trail. Health probes are demoted to DEBUG.
        import time as _time

        from iris_harness.foundation.observability.logging_setup import (
            ingress_logger,
            log_ingress,
        )

        start = _time.monotonic()
        path = routed_path(request)
        is_probe = path in {"/healthz", "/health"}
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            log_ingress(
                method=request.method,
                path=path,
                source=request.client.host if request.client else "",
                status=500,
                duration_ms=(_time.monotonic() - start) * 1000,
            )
            raise
        dur_ms = (_time.monotonic() - start) * 1000
        if is_probe:
            ingress_logger.debug("INGRESS %s %s status=%s", request.method, path, status_code)
        else:
            log_ingress(
                method=request.method,
                path=path,
                source=request.client.host if request.client else "",
                status=status_code,
                duration_ms=dur_ms,
            )
        return response

    @app.get("/health")
    async def health() -> dict[str, Any]:
        telegram_runtime = getattr(app.state, "telegram_runtime", None)
        return {
            "status": "ok",
            "service": "channel_gateway",
            "iris_api": _iris_api_url(),
            "auth_required": _expected_auth_token() is not None,
            "telegram": {
                "configured": telegram_runtime is not None,
                "running": bool(getattr(telegram_runtime, "running", False)),
            },
        }

    @app.websocket("/ws")
    async def ws_chat(websocket: WebSocket, token: str | None = Query(default=None)) -> None:
        await _handle_chat_socket(app, websocket, query_token=token)

    return app


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------


async def _handle_chat_socket(
    app: FastAPI, websocket: WebSocket, *, query_token: str | None
) -> None:
    # The HTTP Host guard does not see a WebSocket handshake, so the same check runs
    # here. Origin is not checked: the handshake needs the shared secret (header or
    # ?token=), never a cookie, so a hostile page has no ambient credential to ride.
    host = websocket.headers.get("host")
    if not host_header_ok(host):
        if host is not None and valid_host_header(host):
            log_refused_host(host)
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    expected = _expected_auth_token()
    presented = _extract_bearer(websocket, query_token)
    # No shared secret configured → refuse rather than silently allow.
    if expected is None or not _token_matches(presented, expected):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    from iris_harness.foundation.observability.logging_setup import log_ingress

    log_ingress(
        method="WS",
        path="/ws",
        source=websocket.client.host if websocket.client else "",
    )
    tracer = getattr(app.state, "tracer", None)
    session_id: str | None = None
    last_pong = time.monotonic()
    busy = False

    from iris_harness.foundation.observability.tracer import (
        maybe_span,  # lazy: optional dep
    )

    with maybe_span(tracer, "ws_connect") as span:
        if span is not None:
            span.set_attribute("channel", "ws")

    async def send_json(payload: dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(payload))
        with maybe_span(tracer, "ws_message_out") as s:
            if s is not None:
                s.set_attribute("type", str(payload.get("type", "")))
                if session_id:
                    s.set_attribute("session_id", session_id)

    async def ping_loop() -> None:
        nonlocal last_pong
        try:
            while True:
                await asyncio.sleep(PING_INTERVAL_SECONDS)
                await send_json({"type": "ping", "ts": time.time()})
                # If pong hasn't arrived within the grace window since the last
                # message, treat the socket as dead.
                if time.monotonic() - last_pong > PING_INTERVAL_SECONDS + PONG_GRACE_SECONDS:
                    logger.info("ws ping timeout; closing")
                    await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
                    return
        except (WebSocketDisconnect, RuntimeError):
            return

    ping_task = asyncio.create_task(ping_loop())

    try:
        while True:
            raw = await websocket.receive_text()
            last_pong = time.monotonic()  # any inbound frame proves liveness
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                await send_json({"type": "error", "error": "invalid_json"})
                continue
            ftype = frame.get("type")

            with maybe_span(tracer, "ws_message_in") as s:
                if s is not None:
                    s.set_attribute("type", str(ftype))

            if ftype == "pong":
                continue  # liveness already updated above
            if ftype != "chat":
                await send_json({"type": "error", "error": f"unknown_type:{ftype}"})
                continue
            if busy:
                await send_json({"type": "error", "error": "busy"})
                continue

            session_id = str(frame.get("session_id") or "default")
            message = frame.get("message")
            if not isinstance(message, str) or not message.strip():
                await send_json({"type": "error", "error": "missing_message"})
                continue

            busy = True
            try:
                await send_json({"type": "hello", "session_id": session_id})
                await _proxy_chat_stream(
                    app=app,
                    websocket=websocket,
                    send_json=send_json,
                    session_id=session_id,
                    message=message,
                    channel=str(frame.get("channel") or "web"),
                    preferred_model=frame.get("preferred_model"),
                    provider_profile=frame.get("provider_profile"),
                    router_model=frame.get("router_model"),
                    strict=bool(frame.get("strict", False)),
                )
            finally:
                busy = False
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.exception("ws handler crashed")
        with maybe_span(tracer, "ws_error") as s:
            if s is not None:
                s.set_attribute("error", repr(exc))
    finally:
        ping_task.cancel()
        with maybe_span(tracer, "ws_disconnect") as s:
            if s is not None and session_id:
                s.set_attribute("session_id", session_id)


async def _proxy_chat_stream(
    *,
    app: FastAPI,
    websocket: WebSocket,
    send_json: Any,
    session_id: str,
    message: str,
    channel: str = "web",
    preferred_model: Any,
    provider_profile: Any,
    router_model: Any,
    strict: bool,
) -> None:
    """POST to IRIS API ``/chat/stream`` and forward each NDJSON event."""
    payload = {
        "message": message,
        "session_id": session_id,
        "channel": channel,
        "preferred_model": preferred_model,
        "provider_profile": provider_profile,
        "router_model": router_model,
        "strict": strict,
    }
    client: httpx.AsyncClient = app.state.http
    url = f"{_iris_api_url()}/chat/stream"
    try:
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code >= 400:
                body = (await resp.aread()).decode("utf-8", errors="replace")
                await send_json(
                    {"type": "error", "error": f"upstream_{resp.status_code}", "body": body}
                )
                return
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    await send_json({"type": "error", "error": "bad_upstream_chunk"})
                    continue
                ev_type = event.pop("event", "unknown")
                event["type"] = ev_type
                await send_json(event)
    except httpx.HTTPError as exc:
        await send_json({"type": "error", "error": f"upstream_unreachable:{exc!r}"})


app = create_app()
