"""Async WebSocket client for the IRIS channel gateway.

Implements the Hermes-style reconnect loop: persistent connection,
app-level pong reply on every server ping, and exponential backoff with
jitter on drop (1s → 2s → 4s → … capped at 30s). The same ``session_id``
is sent on every reconnect so the IRIS API resumes the conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import websockets
from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)

_INITIAL_BACKOFF_SECONDS = 1.0
_MAX_BACKOFF_SECONDS = 30.0
_BACKOFF_JITTER = 0.25  # ±25 %


@dataclass
class GatewayClientConfig:
    url: str  # e.g. "ws://localhost:8006/ws"
    token: str
    session_id: str = "default"


class GatewayClient:
    """Long-lived gateway client; yields server frames as they arrive.

    The client reconnects automatically. Frames yielded by :meth:`stream`
    are dicts (decoded JSON). The caller is responsible for sending chat
    requests via :meth:`send_chat`.
    """

    def __init__(self, config: GatewayClientConfig) -> None:
        self._config = config
        self._conn: ClientConnection | None = None
        self._backoff = _INITIAL_BACKOFF_SECONDS

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Yield frames from the gateway, reconnecting on drop forever."""
        while True:
            try:
                async with self._connect() as conn:
                    self._conn = conn
                    self._backoff = _INITIAL_BACKOFF_SECONDS
                    async for raw in conn:
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8", errors="replace")
                        try:
                            frame = json.loads(raw)
                        except json.JSONDecodeError:
                            logger.warning("gateway sent non-JSON frame: %r", raw[:120])
                            continue
                        if frame.get("type") == "ping":
                            await self._send(conn, {"type": "pong", "ts": frame.get("ts")})
                            continue
                        yield frame
            except (OSError, websockets.WebSocketException) as exc:
                logger.info("gateway connection lost: %r — backing off %.1fs", exc, self._backoff)
            finally:
                self._conn = None
            await asyncio.sleep(_jittered(self._backoff))
            self._backoff = min(self._backoff * 2, _MAX_BACKOFF_SECONDS)

    async def send_chat(
        self,
        message: str,
        *,
        preferred_model: str | None = None,
        provider_profile: str | None = None,
        router_model: str | None = None,
        strict: bool = False,
    ) -> None:
        """Send a chat frame on the current connection.

        Raises ``RuntimeError`` when no connection is live; callers can
        retry once :meth:`stream` reconnects.
        """
        conn = self._conn
        if conn is None:
            raise RuntimeError("gateway not connected")
        await self._send(
            conn,
            {
                "type": "chat",
                "session_id": self._config.session_id,
                "message": message,
                "preferred_model": preferred_model,
                "provider_profile": provider_profile,
                "router_model": router_model,
                "strict": strict,
            },
        )

    def _connect(self) -> Any:
        headers = [("Authorization", f"Bearer {self._config.token}")]
        return websockets.connect(
            self._config.url,
            additional_headers=headers,
            open_timeout=10,
            close_timeout=5,
        )

    @staticmethod
    async def _send(conn: ClientConnection, payload: dict[str, Any]) -> None:
        await conn.send(json.dumps(payload))


def _jittered(seconds: float) -> float:
    spread = seconds * _BACKOFF_JITTER
    return max(0.0, seconds + random.uniform(-spread, spread))  # noqa: S311
