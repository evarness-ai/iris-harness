"""Telegram connector — delivers messages via the Bot API.

Uses ``httpx`` synchronously so it can plug straight into ``ChannelGateway``.
Inject a custom ``client`` (``httpx.Client``) to mock in tests.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from iris_harness.foundation.observability.logging_setup import log_egress

from ..models import ChannelMessage, DeliveryReceipt, DeliveryStatus

logger = logging.getLogger(__name__)


class TelegramConnector:
    """Synchronous Telegram Bot API connector."""

    def __init__(
        self,
        bot_token: str,
        *,
        name: str = "telegram",
        default_chat_id: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 10.0,
        base_url: str = "https://api.telegram.org",
    ) -> None:
        if not bot_token:
            raise ValueError("bot_token is required")
        self.name = name
        self._token = bot_token
        self._default_chat_id = default_chat_id
        self._timeout = timeout
        self._api_base = f"{base_url.rstrip('/')}/bot{bot_token}"
        self._endpoint = f"{self._api_base}/sendMessage"
        self._client = client if client is not None else httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def __del__(self) -> None:  # best-effort cleanup
        if getattr(self, "_owns_client", False):
            try:
                self._client.close()
            except Exception:  # noqa: BLE001, S110 — best-effort GC cleanup, nothing to log
                pass

    def send(self, message: ChannelMessage) -> DeliveryReceipt:
        chat_id = message.recipient or self._default_chat_id
        if not chat_id:
            return DeliveryReceipt(
                channel=self.name,
                status=DeliveryStatus.FAILED,
                error="no chat_id provided (recipient empty and no default_chat_id)",
            )

        payload: dict[str, Any] = {"chat_id": chat_id, "text": message.body}
        parse_mode = message.metadata.get("parse_mode")
        if isinstance(parse_mode, str):
            payload["parse_mode"] = parse_mode
        # Inline buttons (ADR-0076): metadata["inline_keyboard"] is a list of rows,
        # each a list of buttons: {"text", "callback_data"} for an action, or
        # {"text", "url"} for a link (the digest's "Full digest" / "Digest settings").
        keyboard = message.metadata.get("inline_keyboard")
        if isinstance(keyboard, list) and keyboard:
            payload["reply_markup"] = {"inline_keyboard": keyboard}

        try:
            response = self._post_message(payload)
            if "reply_markup" in payload and _buttons_refused(response):
                # A button Telegram will not take (a URL it calls invalid) must not
                # sink the message itself: send it again without the buttons.
                logger.warning(
                    "telegram: buttons refused (%s); sending without them", response.text[:200]
                )
                payload.pop("reply_markup")
                response = self._post_message(payload)
        except httpx.HTTPError as exc:
            return DeliveryReceipt(
                channel=self.name,
                status=DeliveryStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
            )

        if response.status_code == 429:
            return DeliveryReceipt(
                channel=self.name,
                status=DeliveryStatus.RATE_LIMITED,
                error=response.text[:200],
            )

        if response.status_code >= 400:
            return DeliveryReceipt(
                channel=self.name,
                status=DeliveryStatus.FAILED,
                error=f"HTTP {response.status_code}: {response.text[:200]}",
            )

        try:
            data = response.json()
        except ValueError:
            data = {}
        message_id = ""
        result = data.get("result") if isinstance(data, dict) else None
        if isinstance(result, dict):
            mid = result.get("message_id")
            if mid is not None:
                message_id = str(mid)

        return DeliveryReceipt(
            channel=self.name,
            status=DeliveryStatus.SENT,
            message_id=message_id,
        )

    def _post_message(self, payload: dict[str, Any]) -> httpx.Response:
        log_egress(
            destination="api.telegram.org",
            method="POST",
            kind="channel",
            purpose="telegram.sendMessage",
        )
        return self._client.post(self._endpoint, json=payload)

    def answer_callback(self, callback_query_id: str, *, text: str = "") -> None:
        """Acknowledge an inline-button tap (clears the client's loading spinner)."""
        try:
            body: dict[str, Any] = {"callback_query_id": callback_query_id}
            if text:
                body["text"] = text
            log_egress(
                destination="api.telegram.org",
                method="POST",
                kind="channel",
                purpose="telegram.answerCallbackQuery",
            )
            self._client.post(f"{self._api_base}/answerCallbackQuery", json=body)
        except httpx.HTTPError:  # best-effort UX cleanup
            logger.debug("telegram: answerCallbackQuery failed")

    def clear_inline_keyboard(self, chat_id: str, message_id: str) -> None:
        """Remove the inline keyboard from a message once its action is resolved."""
        try:
            log_egress(
                destination="api.telegram.org",
                method="POST",
                kind="channel",
                purpose="telegram.editMessageReplyMarkup",
            )
            self._client.post(
                f"{self._api_base}/editMessageReplyMarkup",
                json={"chat_id": chat_id, "message_id": message_id, "reply_markup": {}},
            )
        except httpx.HTTPError:  # best-effort
            logger.debug("telegram: editMessageReplyMarkup failed")

    def healthy(self) -> bool:
        return bool(self._token)


def _buttons_refused(response: httpx.Response) -> bool:
    """True when Telegram rejected a message for its inline buttons (``BUTTON_*``)."""
    return response.status_code == 400 and "BUTTON_" in response.text.upper()
