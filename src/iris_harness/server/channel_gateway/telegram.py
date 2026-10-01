"""Telegram inbound bridge for the channel gateway service."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

import httpx

from iris_harness.foundation.auth import auth_headers
from iris_harness.kernel.governance.hooks.response_payload import Audience
from iris_harness.services.channels.approval_commands import (
    ApprovalNotFound,
    ApprovalSummary,
    handle_approval_command,
)
from iris_harness.services.channels.connectors.telegram import TelegramConnector
from iris_harness.services.channels.connectors.telegram_poller import (
    TelegramPoller,
    allowed_user_ids_from_env,
)
from iris_harness.services.channels.reminder_commands import (
    ChainedCommands,
    ReminderCommands,
    ReminderNotFound,
    ReminderOutcome,
    SnoozeNotUnderstood,
)

from .main_types import TelegramRuntimeProtocol

logger = logging.getLogger(__name__)

UPSTREAM_TIMEOUT_SECONDS = 600.0
DEFAULT_TELEGRAM_BASE_URL = "https://api.telegram.org"


def _optional_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def _csv_env(name: str) -> frozenset[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return frozenset()
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


def _env_enabled(name: str, *, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


class ApiApprovalBackend:
    """Answers approvals through the API, for the gateway's Telegram poller.

    The gateway is its own process; the runtime that resumes an approved run lives in
    the API, so the answer goes to ``POST /governance/approvals/{id}/respond`` — the one
    gated write the shared service secret may make (owner's decision, 2026-09-21).
    """

    def __init__(self, *, iris_api_url: str, client: httpx.Client) -> None:
        self._base = iris_api_url.rstrip("/")
        self._client = client

    def summary(self, approval_id: str) -> ApprovalSummary | None:
        response = self._client.get(f"{self._base}/governance/approvals")
        response.raise_for_status()
        for row in response.json().get("approvals", []):
            if row.get("approval_id") == approval_id:
                card = row.get("card") or {}
                return ApprovalSummary(
                    title=str(card.get("title") or row.get("signal") or "this approval"),
                    destructive=row.get("kind") == "destructive",
                )
        return None

    def respond(self, approval_id: str, status: str, actor: str) -> tuple[bool, str]:
        response = self._client.post(
            f"{self._base}/governance/approvals/{approval_id}/respond",
            json={"status": status, "actor": actor},
        )
        if response.status_code == 404:
            raise ApprovalNotFound(approval_id)
        if response.status_code == 409:
            raise ValueError(str(response.json().get("detail", "already answered")))
        response.raise_for_status()
        body = response.json()
        return bool(body.get("resumed")), str(body.get("detail", ""))


class ApiReminderBackend:
    """Done / Snooze on a reminder through the API, for the gateway's Telegram poller.

    ``POST /api/v1/reminders/{id}/done|snooze`` and the by-message pair for a reply —
    writes the shared service secret may make (PR 3b, beside the approval answer)."""

    def __init__(self, *, iris_api_url: str, client: httpx.Client) -> None:
        self._base = f"{iris_api_url.rstrip('/')}/api/v1/reminders"
        self._client = client

    def done(self, reminder_id: str, *, source: str) -> ReminderOutcome:
        return self._post(f"{self._base}/{reminder_id}/done", {"source": source})

    def snooze(self, reminder_id: str, spoken: str, *, source: str) -> ReminderOutcome:
        return self._post(f"{self._base}/{reminder_id}/snooze", {"for": spoken, "source": source})

    def done_by_message(
        self, channel: str, chat_id: str, message_id: str, *, source: str
    ) -> ReminderOutcome:
        return self._post(
            f"{self._base}/by-message/{channel}/{chat_id}/{message_id}/done", {"source": source}
        )

    def snooze_by_message(
        self, channel: str, chat_id: str, message_id: str, spoken: str, *, source: str
    ) -> ReminderOutcome:
        return self._post(
            f"{self._base}/by-message/{channel}/{chat_id}/{message_id}/snooze",
            {"for": spoken, "source": source},
        )

    def _post(self, url: str, body: dict[str, str]) -> ReminderOutcome:
        response = self._client.post(url, json=body)
        if response.status_code == 404:
            raise ReminderNotFound(url)
        if response.status_code == 409:
            raise ValueError(str(response.json().get("detail", "already ended")))
        if response.status_code == 422:
            raise SnoozeNotUnderstood(body.get("for", ""))
        response.raise_for_status()
        payload = response.json()
        nxt = payload.get("next") or {}
        reminder = payload.get("reminder") or {}
        until = None
        if (payload.get("undo") or {}).get("kind") == "snooze" and reminder.get("remind_at"):
            until = datetime.fromisoformat(reminder["remind_at"])
        next_at = datetime.fromisoformat(nxt["remind_at"]) if nxt.get("remind_at") else None
        if payload.get("already_paid"):
            entity = str((reminder.get("bill") or {}).get("entity") or "")
            return ReminderOutcome(already_paid=True, entity=entity)
        return ReminderOutcome(until=until, next_at=next_at)


class TelegramIrisChatBridge:
    """Synchronous handler used by TelegramPoller to call IRIS chat streaming."""

    def __init__(
        self,
        *,
        iris_api_url: str,
        preferred_model: str | None = None,
        provider_profile: str | None = None,
        router_model: str | None = None,
        strict: bool = False,
        client: httpx.Client | None = None,
    ) -> None:
        self._iris_api_url = iris_api_url.rstrip("/")
        self._preferred_model = preferred_model
        self._provider_profile = provider_profile
        self._router_model = router_model
        self._strict = strict
        self._client = (
            client
            if client is not None
            else httpx.Client(timeout=UPSTREAM_TIMEOUT_SECONDS, headers=auth_headers())
        )
        self._owns_client = client is None

    @property
    def client(self) -> httpx.Client:
        """The authenticated client to the API, shared with the approval backend."""
        return self._client

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def welcome(self, audience: Audience = "owner") -> str | None:
        """The first-chat welcome (ADR-0127) when this call ran it; None when it ran before.

        Raises when the API cannot answer, so the poller asks again on the next message.
        """
        response = self._client.post(
            f"{self._iris_api_url}/chat/welcome",
            json={"channel": "telegram", "audience": audience},
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict) or body.get("created") is not True:
            return None
        text = body.get("response")
        return text if isinstance(text, str) and text.strip() else None

    def reply(self, text: str, session_id: str, audience: Audience = "owner") -> str:
        payload = {
            "message": text,
            "session_id": session_id,
            "channel": "telegram",
            # Who reads the answer: ``other`` in a group chat (ADR-0125).
            "audience": audience,
            "preferred_model": self._preferred_model,
            "provider_profile": self._provider_profile,
            "router_model": self._router_model,
            "strict": self._strict,
        }
        token_parts: list[str] = []
        last_error = ""

        try:
            with self._client.stream(
                "POST",
                f"{self._iris_api_url}/chat/stream",
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", errors="replace")
                    logger.warning(
                        "telegram gateway upstream returned HTTP %d: %s",
                        response.status_code,
                        body[:200],
                    )
                    return (
                        f"IRIS API returned HTTP {response.status_code}. Please check the API logs."
                    )

                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("telegram gateway ignored malformed upstream chunk")
                        continue

                    event_type = event.get("event")
                    if event_type == "token":
                        chunk = event.get("text")
                        if isinstance(chunk, str):
                            token_parts.append(chunk)
                    elif event_type == "done":
                        reply = event.get("response")
                        if isinstance(reply, str) and reply.strip():
                            return reply.strip()
                        fallback = "".join(token_parts).strip()
                        return fallback or "IRIS did not return a response."
                    elif event_type == "error":
                        error = event.get("error")
                        last_error = error if isinstance(error, str) else "unknown error"
        except httpx.HTTPError as exc:
            logger.warning("telegram gateway could not reach IRIS API: %r", exc)
            return "IRIS API is unreachable. Please check the gateway logs."

        fallback = "".join(token_parts).strip()
        if fallback:
            return fallback
        if last_error:
            return f"IRIS returned an error: {last_error}"
        return "IRIS did not return a response."


class TelegramGatewayRuntime:
    """Lifecycle wrapper for the Telegram poller and its IRIS API client."""

    def __init__(self, *, poller: TelegramPoller, bridge: TelegramIrisChatBridge) -> None:
        self._poller = poller
        self._bridge = bridge

    @property
    def running(self) -> bool:
        return self._poller.running

    def start(self) -> None:
        self._poller.start()

    def stop(self, *, join_timeout: float = 5.0) -> None:
        self._poller.stop(join_timeout=join_timeout)
        self._bridge.close()


def build_telegram_runtime_from_env(
    *,
    iris_api_url: str,
    upstream_client: httpx.Client | None = None,
    telegram_poll_client: httpx.Client | None = None,
    telegram_send_client: httpx.Client | None = None,
) -> TelegramRuntimeProtocol | None:
    """Build Telegram polling runtime from environment, or None when disabled."""
    if not _env_enabled("IRIS_CHANNEL_GATEWAY_TELEGRAM_ENABLED", default=True):
        return None

    bot_token = _optional_env("TELEGRAM_BOT_TOKEN")
    if bot_token is None:
        logger.info("telegram gateway poller not configured: TELEGRAM_BOT_TOKEN is missing")
        return None

    default_chat_id = _optional_env("TELEGRAM_CHAT_ID")
    allowed_chat_ids = _csv_env("TELEGRAM_ALLOWED_CHAT_IDS")
    if not allowed_chat_ids and default_chat_id:
        allowed_chat_ids = frozenset({default_chat_id})
    if not allowed_chat_ids:
        # Fail closed: without an allowlist any Telegram user who finds the bot
        # would drive the full agent. Refuse to start the poller instead.
        logger.warning(
            "telegram gateway poller not started: no allowlist configured — set "
            "TELEGRAM_ALLOWED_CHAT_IDS or TELEGRAM_CHAT_ID"
        )
        return None

    telegram_base_url = _optional_env("TELEGRAM_BASE_URL") or DEFAULT_TELEGRAM_BASE_URL
    bridge = TelegramIrisChatBridge(
        iris_api_url=iris_api_url,
        preferred_model=_optional_env("TELEGRAM_PREFERRED_MODEL"),
        provider_profile=_optional_env("TELEGRAM_PROVIDER_PROFILE"),
        router_model=_optional_env("TELEGRAM_ROUTER_MODEL"),
        strict=_env_enabled("TELEGRAM_STRICT_MODE", default=False),
        client=upstream_client,
    )
    connector = TelegramConnector(
        bot_token=bot_token,
        default_chat_id=default_chat_id,
        client=telegram_send_client,
        base_url=telegram_base_url,
    )
    approvals = ApiApprovalBackend(iris_api_url=iris_api_url, client=bridge.client)
    reminders = ApiReminderBackend(iris_api_url=iris_api_url, client=bridge.client)
    # Approvals first, then a reminder's Done / Snooze (a tap, a command or a reply).
    commands = ChainedCommands(
        lambda text, user_id: handle_approval_command(text, user_id, approvals),
        ReminderCommands(reminders),
    )
    poller = TelegramPoller(
        bot_token=bot_token,
        connector=connector,
        chat_handler=bridge.reply,
        command_handler=commands,
        opener=bridge.welcome,
        allowed_chat_ids=allowed_chat_ids,
        allowed_user_ids=allowed_user_ids_from_env(),
        base_url=telegram_base_url,
        client=telegram_poll_client,
    )
    logger.info(
        "telegram gateway poller configured (allowed_chat_ids=%s)",
        sorted(allowed_chat_ids),
    )
    return TelegramGatewayRuntime(poller=poller, bridge=bridge)
