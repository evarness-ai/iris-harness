"""Chat: a turn (blocking and streamed), cancelling one, and model warmup.

    POST   /chat
    POST   /chat/stream
    POST   /chat/cancel
    POST   /warmup

Moved out of ``create_app`` unchanged (review item: split the god function); the route
table and OpenAPI schema are identical before and after. The write guard in ``main``
still gates the mutating routes.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterator
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from iris_harness.runtime import IrisRuntime

logger = logging.getLogger(__name__)


class ChatCancelRequest(BaseModel):
    """Body for ``POST /chat/cancel`` — stop the session's running turn."""

    session_id: str = Field(default="default", max_length=128)


class ChatRequest(BaseModel):
    """Request body for ``POST /chat``."""

    message: str = Field(..., min_length=1, max_length=8000)
    session_id: str = Field(default="default", max_length=128)
    preferred_model: str | None = Field(default=None, max_length=128)
    provider_profile: str | None = Field(default=None, max_length=64)
    router_model: str | None = Field(default=None, max_length=128)
    strict: bool = False
    # Phase C: name of the gateway the user is messaging from
    # (``console``, ``telegram``, ``web``, ``voice``, ...). New
    # gateway adapters opt in by setting this; nothing else needs
    # to change for routine authoring to default the delivery
    # channel to the origin.
    channel: str = Field(default="console", max_length=32)
    # Who reads the answer (ADR-0125): ``other`` when people besides the owner do -- the
    # Telegram gateway sends it for a group chat. ``other`` can only make the answer
    # check stricter, never looser, so a caller gains nothing by claiming it.
    audience: Literal["owner", "other"] = "owner"


_SESSION_CHANNEL_PREFIX_RE = re.compile(r"^(?P<ch>[a-z]+):")


def _resolve_models(app: FastAPI, request: ChatRequest) -> tuple[str | None, str | None]:
    """Apply this session's `/model` / `/router` choice when the body names none.

    An explicit field in the request always wins — a caller that states a model
    means it. The stored override only fills the gap, which is what makes a
    `/model` typed in web chat outlive the turn that set it.
    """
    stored_model, stored_router = app.state.model_overrides.get(request.session_id)
    return (
        (request.preferred_model or stored_model) or None,
        (request.router_model or stored_router) or None,
    )


def _origin_channel(channel: str, session_id: str) -> str:
    """Resolve the effective origin channel for a chat request.

    Gateways now declare ``channel`` explicitly (Phase 1), but in-flight
    clients may still POST without it, leaving the model default ``console``.
    As a defensive fallback, when ``channel`` is the literal default and the
    ``session_id`` carries a channel prefix (e.g. ``telegram:123456789`` or
    ``web:...``), derive the channel from that prefix. An explicitly supplied
    non-console channel always wins.
    """
    if channel and channel != "console":
        return channel
    match = _SESSION_CHANNEL_PREFIX_RE.match(session_id or "")
    if match:
        return match.group("ch")
    return channel


class ChatResponse(BaseModel):
    """Response body for ``POST /chat``."""

    response: str
    intent: str
    agent_type: str
    sources: list[str]
    has_errors: bool
    error_summary: str | None
    metadata: dict[str, Any]


class WarmupRequest(BaseModel):
    """Request body for ``POST /warmup``."""

    role: str = Field(..., pattern="^(router|executor)$")
    model: str | None = Field(default=None, max_length=128)
    provider_profile: str | None = Field(default=None, max_length=64)


class WarmupResponse(BaseModel):
    """Response body for ``POST /warmup``."""

    ok: bool
    role: str
    model: str
    latency_ms: float
    error: str | None = None


def install_chat_routes(app: FastAPI, runtime: Callable[[], Any]) -> None:
    """Register these routes. ``runtime`` returns the live runtime or raises 503."""

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        rt: IrisRuntime | None = app.state.runtime
        if rt is None:
            raise HTTPException(status_code=503, detail="runtime unavailable")
        preferred_model, router_model = _resolve_models(app, request)
        result = rt.chat(
            request.message,
            session_id=request.session_id,
            preferred_model=preferred_model,
            provider_profile=request.provider_profile or None,
            router_model=router_model,
            strict=request.strict,
            channel=_origin_channel(request.channel, request.session_id),
            audience=request.audience,
        )
        return ChatResponse(
            response=result.response,
            intent=result.intent,
            agent_type=result.agent_type,
            sources=list(result.sources),
            has_errors=result.has_errors,
            error_summary=result.error_summary,
            metadata=result.metadata,
        )

    @app.post("/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        rt: IrisRuntime | None = app.state.runtime
        if rt is None:
            raise HTTPException(status_code=503, detail="runtime unavailable")

        preferred_model, router_model = _resolve_models(app, request)

        # The turn runs on its own thread and this response only relays it: a phone that
        # backgrounds the app drops the connection, and that must not stop the turn
        # (detached_turns.py). Stop is POST /chat/cancel.
        turn = app.state.turns.start(
            request.session_id,
            lambda: rt.chat_stream(
                request.message,
                session_id=request.session_id,
                preferred_model=preferred_model,
                provider_profile=request.provider_profile or None,
                router_model=router_model,
                strict=request.strict,
                channel=_origin_channel(request.channel, request.session_id),
                audience=request.audience,
            ),
        )

        def gen() -> Iterator[bytes]:
            try:
                for evt in app.state.turns.relay(turn):
                    if evt is None:
                        # Quiet during a long model call: a keep-alive, so nothing on the
                        # way to the phone takes thinking for a dead connection. The web
                        # client ignores events it does not know.
                        yield b'{"event": "ping"}\n'
                        continue
                    if isinstance(evt, Exception):
                        raise evt
                    if evt.kind == "token":
                        payload: dict[str, Any] = {"event": "token", "text": evt.text}
                    elif evt.kind == "activity":
                        payload = {"event": "activity", "text": evt.text}
                    elif evt.kind == "trace":
                        payload = {"event": "trace", "text": evt.text}
                        if evt.payload is not None:
                            payload["payload"] = evt.payload
                    elif evt.kind == "done":
                        result = evt.result
                        if result is None:
                            payload = {"event": "error", "error": "missing result"}
                        else:
                            payload = {
                                "event": "done",
                                "response": result.response,
                                "intent": result.intent,
                                "agent_type": result.agent_type,
                                "sources": list(result.sources),
                                "has_errors": result.has_errors,
                                "error_summary": result.error_summary,
                                "metadata": result.metadata,
                            }
                    else:
                        payload = {"event": "error", "error": evt.error or "unknown"}
                    if evt.payload is not None and "payload" not in payload:
                        payload["payload"] = evt.payload
                    yield (json.dumps(payload) + "\n").encode("utf-8")
            except Exception as exc:
                logger.exception("chat_stream endpoint failed")
                yield (json.dumps({"event": "error", "error": str(exc)}) + "\n").encode("utf-8")

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    @app.post("/chat/cancel")
    def chat_cancel(request: ChatCancelRequest) -> dict[str, Any]:
        """Stop the session's running turn — the Stop button.

        Closing the stream no longer stops a turn (the phone backgrounding the app
        closes it too), so stopping is asked for. The turn ends at its next event.
        """
        return {"cancelled": app.state.turns.cancel(request.session_id)}

    @app.post("/warmup", response_model=WarmupResponse)
    def warmup(request: WarmupRequest) -> WarmupResponse:
        rt: IrisRuntime | None = app.state.runtime
        if rt is None:
            raise HTTPException(status_code=503, detail="runtime unavailable")
        result = rt.warmup(
            role=request.role,
            model=request.model or None,
            provider_profile=request.provider_profile or None,
        )
        return WarmupResponse(
            ok=result.ok,
            role=result.role,
            model=result.model,
            latency_ms=result.latency_ms,
            error=result.error,
        )
