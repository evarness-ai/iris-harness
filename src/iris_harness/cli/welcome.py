"""The first-chat welcome on the CLI (ADR-0127): ask the harness, print it when new.

The harness decides whether the welcome is due (once per IRIS_HOME); the CLI only asks
``POST /chat/welcome`` when a chat starts and prints the text when this call is the one
that ran it. A welcome that cannot be fetched never stops the chat: the caller goes on.
"""

from __future__ import annotations

import logging

import httpx

from iris_harness.cli.api_client import harness_api_client

logger = logging.getLogger(__name__)


def fetch_welcome(api_url: str, *, channel: str = "console") -> str | None:
    """The welcome text when this call ran it; None when it ran before or is out of reach."""
    try:
        with harness_api_client(purpose="chat-welcome", timeout=30) as client:
            response = client.post(f"{api_url.rstrip('/')}/chat/welcome", json={"channel": channel})
            response.raise_for_status()
            body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.debug("first-chat welcome unavailable: %s", exc)
        return None
    if not isinstance(body, dict) or body.get("created") is not True:
        return None
    text = body.get("response")
    return text if isinstance(text, str) and text.strip() else None


__all__ = ["fetch_welcome"]
