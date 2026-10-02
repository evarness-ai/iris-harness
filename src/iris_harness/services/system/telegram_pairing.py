"""Telegram bot-token -> chat_id pairing poll, for ``iris setup``'s Telegram step.

Lives here, not in ``cli/setup.py``: a CLI module may not open its own network
connection (``tests/security/test_cli_egress_logged.py``) -- every outbound call must
go through a module that owns its own ``iris.egress`` line, the way ``services/system/
doctor.py`` already does for its Ollama calls. This one logs ``kind="channel"``, the
same kind the Telegram channel itself uses, since it genuinely is a Telegram call and
not a call to one of the harness's own services (``cli/api_client.py``'s ``kind="service"``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from urllib.parse import urlsplit

import httpx

from iris_harness.foundation.observability.logging_setup import log_egress

TELEGRAM_BASE_URL = "https://api.telegram.org"


def _host(url: str) -> str:
    """``host[:port]`` of ``url`` -- never the path, query, or bot token it carries."""
    parts = urlsplit(url)
    host = parts.hostname or "unknown"
    return f"{host}:{parts.port}" if parts.port else host


def get_telegram_bot_username(token: str, *, client: httpx.Client | None = None) -> str | None:
    """The bot's own ``@username`` (``getMe``), for the pairing deep link
    (``https://t.me/<username>?start=...``) -- ``None`` if the token is invalid or
    the call fails, so the caller can fall back to "open Telegram and message your
    bot" instructions.
    """
    own_client = client is None
    http = client or httpx.Client(base_url=TELEGRAM_BASE_URL, timeout=10.0)
    try:
        log_egress(
            destination=_host(TELEGRAM_BASE_URL),
            method="GET",
            kind="channel",
            purpose="telegram bot identity",
        )
        try:
            resp = http.get(f"/bot{token}/getMe")
            resp.raise_for_status()
            result = resp.json().get("result", {})
        except httpx.HTTPError:
            return None
        username = result.get("username") if isinstance(result, dict) else None
        return str(username) if username else None
    finally:
        if own_client:
            http.close()


def poll_telegram_chat_id(
    token: str,
    *,
    client: httpx.Client | None = None,
    deadline_s: float = 60.0,
    poll_interval_s: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> str | None:
    """Poll Telegram's ``getUpdates`` for the chat_id of the next message the user
    sends the bot, so they never have to run a manual curl. ``None`` on timeout."""
    own_client = client is None
    http = client or httpx.Client(base_url=TELEGRAM_BASE_URL, timeout=10.0)
    destination = _host(TELEGRAM_BASE_URL)
    try:
        deadline = now() + deadline_s
        while now() < deadline:
            log_egress(
                destination=destination,
                method="GET",
                kind="channel",
                purpose="telegram pairing poll",
            )
            try:
                resp = http.get(f"/bot{token}/getUpdates", params={"offset": -1, "timeout": 1})
                resp.raise_for_status()
                updates = resp.json().get("result", [])
            except httpx.HTTPError:
                updates = []
            for update in reversed(updates):
                message = update.get("message") or update.get("channel_post")
                if isinstance(message, dict):
                    chat_id = message.get("chat", {}).get("id")
                    if chat_id is not None:
                        return str(chat_id)
            sleep(poll_interval_s)
        return None
    finally:
        if own_client:
            http.close()


__all__ = ["TELEGRAM_BASE_URL", "get_telegram_bot_username", "poll_telegram_chat_id"]
