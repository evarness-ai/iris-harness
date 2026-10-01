"""Telegram inbound message poller.

Runs a daemon thread that long-polls ``getUpdates`` and dispatches each text
message through a caller-supplied handler, then sends the reply back via the
existing ``TelegramConnector``.

Each incoming message is handled in a ``ThreadPoolExecutor`` worker so the
poll loop never blocks on slow LLM calls. A "thinking…" indicator is sent
after ``thinking_delay`` seconds. A handler still running at ``handler_timeout``
seconds is not abandoned: the chat gets a "still working" note, and the answer is
posted when it lands, up to ``late_answer_limit`` seconds after the message. The
first version gave up at 90s and dropped an answer that arrived 7s later
(2026-09-25).

A text message that *replies* to one of the bot's messages goes to ``reply_handler``
first (``(text, user_id, chat_id, replied_message_id) -> CommandReply | None``; by
default the command handler's ``on_reply``): "snooze 1h" or "done" in reply to a
reminder acts on that reminder (loop-proof D14, PR 3b). ``None`` sends it to chat.

Who is served: the chat allowlist (``TELEGRAM_ALLOWED_CHAT_IDS`` / ``TELEGRAM_CHAT_ID``)
admits a *chat*, and in a group chat that is everyone in it. The optional
``TELEGRAM_ALLOWED_USER_IDS`` narrows it to people: when set, a message or a button tap
must also come from one of those Telegram user ids (``from.id``). Unset, nothing changes.

Who reads the answer: in a group chat its members do, not only the owner, so the chat
handler is told the audience (ADR-0125): ``owner`` for a private chat, ``other`` for any
other chat type (``group``, ``supergroup``, or one Telegram did not name).
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
from collections.abc import Callable
from typing import Any

import httpx

from iris_harness.kernel.governance.hooks.response_payload import Audience

from ..approval_commands import CommandHandler, CommandReply
from ..models import ChannelMessage
from ..reminder_commands import ReplyHandler
from .telegram import TelegramConnector

logger = logging.getLogger(__name__)

ALLOWED_USER_IDS_ENV = "TELEGRAM_ALLOWED_USER_IDS"

#: ``(text, session_id, audience) -> reply``: what the poller hands a message to.
ChatHandler = Callable[[str, str, Audience], str]

#: ``(audience) -> text | None``: the first-chat welcome (ADR-0127) when this call ran it.
Opener = Callable[[Audience], str | None]


def audience_of_chat(chat: object) -> Audience:
    """Who reads a reply in ``chat``: the owner only in a private chat (ADR-0125).

    A group's members read it too, and a chat whose type is missing is not assumed to be
    the owner's: an audience nobody can name is not the owner (``audience_of``).
    """
    kind = chat.get("type") if isinstance(chat, dict) else None
    return "owner" if kind == "private" else "other"


def allowed_user_ids_from_env() -> frozenset[str]:
    """The optional per-user allowlist, read from ``TELEGRAM_ALLOWED_USER_IDS`` (csv).

    One reader for both pollers (the channel gateway's and the runtime's), so the two
    cannot disagree on what the variable means. Empty means "no user check"."""
    raw = os.environ.get(ALLOWED_USER_IDS_ENV, "")
    return frozenset(item.strip() for item in raw.split(",") if item.strip())


_MAX_MESSAGE_LENGTH = 4096
_TRUNCATION_SUFFIX = "\n\n…[response truncated]"


class TelegramPoller:
    """Long-poll Telegram getUpdates and route messages to IRIS.

    ``chat_handler`` receives ``(text, session_id, audience)`` and must return the reply
    string. ``allowed_chat_ids`` restricts which chat IDs are served; an empty set
    serves nobody (fail closed). ``allowed_user_ids``, when not empty, also requires
    the sender to be one of those user ids; empty skips that check.

    Each message is dispatched to a thread-pool worker so the poll loop remains
    responsive even when the LLM takes tens of seconds to respond.
    """

    def __init__(
        self,
        bot_token: str,
        connector: TelegramConnector,
        chat_handler: ChatHandler,
        *,
        confirmation_options: Callable[[str], list[str] | None] | None = None,
        command_handler: CommandHandler | None = None,
        reply_handler: ReplyHandler | None = None,
        opener: Opener | None = None,
        allowed_chat_ids: frozenset[str] = frozenset(),
        allowed_user_ids: frozenset[str] = frozenset(),
        poll_timeout: int = 30,
        handler_timeout: float = 90.0,
        late_answer_limit: float = 540.0,
        thinking_delay: float = 10.0,
        max_concurrent: int = 4,
        base_url: str = "https://api.telegram.org",
        client: httpx.Client | None = None,
    ) -> None:
        if not bot_token:
            raise ValueError("bot_token is required")
        self._token = bot_token
        self._connector = connector
        self._handler = chat_handler
        # ADR-0076: given a session id, return the pending approve/reject options
        # (or None) so replies can carry inline buttons.
        self._confirmation_options = confirmation_options
        # Approval commands — typed, or sent by an approval message's buttons — are
        # answered here, before chat: ``(text, telegram_user_id)`` -> reply or None.
        self._command_handler = command_handler
        # A reply to one of the bot's messages (a reminder's Done / Snooze by words).
        self._reply_handler: ReplyHandler | None = reply_handler or getattr(
            command_handler, "on_reply", None
        )
        # The first-chat welcome (ADR-0127): ``(audience) -> text`` when this call ran it,
        # None when it ran before. The harness decides; asked until it has answered once.
        self._opener = opener
        self._welcome_settled = opener is None
        self._welcome_lock = threading.Lock()
        self._allowed = allowed_chat_ids
        self._allowed_users = allowed_user_ids
        self._poll_timeout = poll_timeout
        self._handler_timeout = handler_timeout
        # Below the gateway's 600s upstream timeout, so the answer can still arrive.
        self._late_answer_limit = max(late_answer_limit, handler_timeout)
        self._thinking_delay = thinking_delay
        self._base = f"{base_url.rstrip('/')}/bot{bot_token}"
        self._client = client if client is not None else httpx.Client(timeout=poll_timeout + 15)
        self._owns_client = client is None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="iris-tg-handler",
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="iris-telegram-poller",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "telegram poller started (allowed_ids=%s, handler_timeout=%.0fs, thinking_delay=%.0fs)",
            sorted(self._allowed) if self._allowed else "none — fail closed, all senders dropped",
            self._handler_timeout,
            self._thinking_delay,
        )

    def stop(self, *, join_timeout: float = 5.0) -> None:
        self._stop.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
        if self._owns_client:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001, S110
                pass
        logger.info("telegram poller stopped")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Poll loop — never blocks on LLM calls
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        offset = 0
        backoff = 1.0
        while not self._stop.is_set():
            try:
                updates = self._fetch_updates(offset)
                backoff = 1.0
                for update in updates:
                    update_id: int = update.get("update_id", 0)
                    offset = max(offset, update_id + 1)
                    if not self._stop.is_set():
                        # Non-blocking: handler runs in thread pool
                        self._executor.submit(self._handle_update, update)
            except httpx.HTTPStatusError as exc:
                code = exc.response.status_code
                if code in {401, 403}:
                    logger.error("telegram poller: auth error %d — stopping", code)
                    return
                logger.warning("telegram poller: HTTP %d — retry in %.0fs", code, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
            except httpx.HTTPError as exc:
                logger.debug("telegram poller: network error %s — retry in %.0fs", exc, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)
            except Exception:
                logger.exception("telegram poller: unexpected error — retry in %.0fs", backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60.0)

    def _fetch_updates(self, offset: int) -> list[dict[str, Any]]:
        resp = self._client.get(
            f"{self._base}/getUpdates",
            params={
                "offset": offset,
                "timeout": self._poll_timeout,
                "allowed_updates": '["message","callback_query"]',
            },
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict) or not data.get("ok"):
            return []
        result = data.get("result", [])
        return result if isinstance(result, list) else []

    # ------------------------------------------------------------------
    # Per-message handler — runs in thread pool, never blocks the loop
    # ------------------------------------------------------------------

    def _handle_update(self, update: dict[str, Any]) -> None:
        callback = update.get("callback_query")
        if isinstance(callback, dict):
            self._handle_callback(callback)
            return

        message = update.get("message")
        if not isinstance(message, dict):
            return

        text = message.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return

        chat = message.get("chat", {})
        chat_id = str(chat.get("id", ""))
        if not chat_id:
            return

        # Fail closed: an unconfigured allowlist must not mean "anyone who finds
        # the bot drives the agent" — it means nobody does.
        if not self._allowed:
            logger.warning(
                "telegram poller: dropping message from chat_id %s — no allowlist "
                "configured (set TELEGRAM_ALLOWED_CHAT_IDS or TELEGRAM_CHAT_ID)",
                chat_id,
            )
            return
        if chat_id not in self._allowed:
            logger.warning(
                "telegram poller: ignoring message from unauthorised chat_id %s", chat_id
            )
            return
        if not self._user_allowed(message):
            logger.warning(
                "telegram poller: ignoring message in chat %s from unauthorised user %s",
                chat_id,
                _user_id(message) or "(none)",
            )
            return

        logger.info("telegram poller: message from chat %s: %r", chat_id, text[:80])

        # The first chat on this install opens with IRIS's welcome, sent ahead of the
        # answer. `/start` is Telegram's "open the chat" and asks nothing more.
        if self._send_welcome_if_new(chat_id, audience_of_chat(chat)) and _is_start(text):
            return

        if self._try_command(text.strip(), chat_id, _user_id(message)):
            return
        replied = _replied_message_id(message)
        if replied and self._try_reply(text.strip(), chat_id, replied, _user_id(message)):
            return

        # Send a "thinking…" indicator if the LLM takes longer than expected.
        thinking_timer = threading.Timer(
            self._thinking_delay,
            self._send_thinking,
            args=(chat_id,),
        )
        thinking_timer.daemon = True
        thinking_timer.start()

        reply = ""
        try:
            reply = self._call_with_timeout(text.strip(), chat_id, audience_of_chat(chat))
        finally:
            thinking_timer.cancel()

        if not reply:
            return

        self._send_reply(chat_id, reply)

    def _handle_callback(self, callback: dict[str, Any]) -> None:
        """Resolve an inline-button tap (approve/reject) as a chat decision."""
        data = callback.get("data")
        cbq_id = str(callback.get("id", ""))
        msg = callback.get("message")
        chat = msg.get("chat", {}) if isinstance(msg, dict) else {}
        chat_id = str(chat.get("id", ""))
        message_id = str(msg.get("message_id", "")) if isinstance(msg, dict) else ""

        if not isinstance(data, str) or not data.strip() or not chat_id:
            if cbq_id:
                self._connector.answer_callback(cbq_id)
            return
        # Fail closed, same as the message path: no allowlist means nobody may
        # resolve approvals — not "anybody may".
        if not self._allowed or chat_id not in self._allowed:
            if not self._allowed:
                logger.warning(
                    "telegram poller: dropping callback from chat_id %s — no allowlist "
                    "configured (set TELEGRAM_ALLOWED_CHAT_IDS or TELEGRAM_CHAT_ID)",
                    chat_id,
                )
            if cbq_id:
                self._connector.answer_callback(cbq_id)
            return
        # In a group, anyone can tap a button; an approval is only the allowed users' to give.
        if not self._user_allowed(callback):
            logger.warning(
                "telegram poller: ignoring button tap in chat %s from unauthorised user %s",
                chat_id,
                _user_id(callback) or "(none)",
            )
            if cbq_id:
                self._connector.answer_callback(cbq_id)
            return

        # Remove the buttons from the original message so it can't be tapped twice.
        if message_id:
            self._connector.clear_inline_keyboard(chat_id, message_id)

        if self._command_handler is not None:
            # Stop the button's spinner now: approving resumes a run, which can take a
            # while, and the reply follows as its own message.
            if cbq_id:
                self._connector.answer_callback(cbq_id)
                cbq_id = ""
            if self._try_command(data.strip(), chat_id, _user_id(callback)):
                return

        reply = self._call_with_timeout(data.strip(), chat_id, audience_of_chat(chat))
        if cbq_id:
            self._connector.answer_callback(cbq_id)
        if reply:
            self._send_reply(chat_id, reply)

    def _send_welcome_if_new(self, chat_id: str, audience: Audience) -> bool:
        """Send the first-chat welcome when the harness says this is it; True if sent.

        Once the harness has answered (it ran now, or it ran before on another surface)
        it is not asked again for the life of this poller. An answer that never came (the
        API out of reach) leaves it to the next message; the chat goes on regardless.
        """
        if self._welcome_settled or self._opener is None:
            return False
        with self._welcome_lock:
            if self._welcome_settled:
                return False
            try:
                welcome = self._opener(audience)
            except Exception:
                logger.warning("telegram poller: first-chat welcome unavailable", exc_info=True)
                return False
            self._welcome_settled = True
        if not welcome:
            return False
        self._send_reply(chat_id, welcome)
        return True

    def _user_allowed(self, update_part: dict[str, Any]) -> bool:
        """Whether the sender passes the optional per-user allowlist.

        No list configured means no user check (the chat allowlist alone decides, as
        before). With one, a message without a sender id is refused, not waved through."""
        return not self._allowed_users or _user_id(update_part) in self._allowed_users

    def _try_command(self, text: str, chat_id: str, user_id: str) -> bool:
        """Answer an approval command; False when ``text`` is not one (it goes to chat)."""
        if self._command_handler is None:
            return False
        try:
            reply: CommandReply | None = self._command_handler(text, user_id)
        except Exception:  # a failed answer is said, never swallowed into chat
            logger.exception("telegram poller: approval command failed")
            reply = CommandReply("Could not answer that approval. Try it in Activity.")
        if reply is None:
            return False
        self._send_command_reply(chat_id, reply)
        return True

    def _try_reply(self, text: str, chat_id: str, message_id: str, user_id: str) -> bool:
        """Answer a reply to the bot's message; False when it is not one the reply
        handler acts on (it goes to chat)."""
        if self._reply_handler is None:
            return False
        try:
            reply = self._reply_handler(text, user_id, chat_id, message_id)
        except Exception:  # a failed action is said, never swallowed into chat
            logger.exception("telegram poller: reply action failed")
            reply = CommandReply("Could not do that. Try the buttons on the reminder.")
        if reply is None:
            return False
        self._send_command_reply(chat_id, reply)
        return True

    def _send_command_reply(self, chat_id: str, reply: CommandReply) -> None:
        metadata: dict[str, object] = {"inline_keyboard": reply.keyboard} if reply.keyboard else {}
        receipt = self._connector.send(
            ChannelMessage(recipient=chat_id, body=reply.text, metadata=metadata)
        )
        if receipt.error:
            logger.warning(
                "telegram poller: command reply failed for %s: %s", chat_id, receipt.error
            )

    def _inline_keyboard_for(self, session_id: str) -> list[list[dict[str, str]]] | None:
        """One row of approve/reject buttons when a confirmation is pending."""
        if self._confirmation_options is None:
            return None
        try:
            options = self._confirmation_options(session_id)
        except Exception:  # noqa: BLE001
            return None
        if not options:
            return None
        return [[{"text": o.capitalize(), "callback_data": o} for o in options]]

    def _send_reply(self, chat_id: str, reply: str) -> None:
        if len(reply) > _MAX_MESSAGE_LENGTH:
            keep = _MAX_MESSAGE_LENGTH - len(_TRUNCATION_SUFFIX)
            reply = reply[:keep] + _TRUNCATION_SUFFIX
        keyboard = self._inline_keyboard_for(f"telegram:{chat_id}")
        metadata: dict[str, object] = {"inline_keyboard": keyboard} if keyboard else {}
        receipt = self._connector.send(
            ChannelMessage(recipient=chat_id, body=reply, metadata=metadata)
        )
        if receipt.error:
            logger.warning(
                "telegram poller: delivery failed for chat %s: %s", chat_id, receipt.error
            )

    def _call_with_timeout(self, text: str, chat_id: str, audience: Audience) -> str:
        """Run the chat handler in a daemon thread with a hard timeout."""
        result: list[str] = []
        exc_holder: list[BaseException] = []

        def _run() -> None:
            try:
                result.append(self._handler(text, f"telegram:{chat_id}", audience))
            except Exception as e:  # noqa: BLE001
                exc_holder.append(e)

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(timeout=self._handler_timeout)

        if worker.is_alive():
            logger.warning(
                "telegram poller: handler still running after %.0fs for chat %s; "
                "the answer will follow",
                self._handler_timeout,
                chat_id,
            )
            follower = threading.Thread(
                target=self._deliver_late,
                args=(worker, result, exc_holder, text, chat_id),
                daemon=True,
            )
            follower.start()
            return "⏳ Still working on it — I'll send the answer here as soon as it's ready."

        return self._outcome(result, exc_holder)

    @staticmethod
    def _outcome(result: list[str], exc_holder: list[BaseException]) -> str:
        if exc_holder:
            logger.exception("telegram poller: chat handler raised", exc_info=exc_holder[0])
            return "Sorry, something went wrong. Please try again."
        return result[0] if result else ""

    def _deliver_late(
        self,
        worker: threading.Thread,
        result: list[str],
        exc_holder: list[BaseException],
        text: str,
        chat_id: str,
    ) -> None:
        """Wait out the rest of ``late_answer_limit`` and post what the handler says."""
        worker.join(timeout=self._late_answer_limit - self._handler_timeout)
        if worker.is_alive():
            logger.warning(
                "telegram poller: handler gave no answer within %.0fs for chat %s",
                self._late_answer_limit,
                chat_id,
            )
            self._send_reply(
                chat_id,
                f"⚠️ No answer after {self._late_answer_limit / 60:.0f} minutes. The model "
                "may be overloaded — please try again.",
            )
            return
        reply = self._outcome(result, exc_holder)
        if reply:
            asked = text if len(text) <= 60 else text[:57] + "…"
            self._send_reply(chat_id, f"About “{asked}”:\n\n{reply}")

    def _send_thinking(self, chat_id: str) -> None:
        try:
            self._connector.send(ChannelMessage(recipient=chat_id, body="⏳ Thinking…"))
        except Exception:  # noqa: BLE001
            logger.debug("telegram poller: failed to send thinking indicator to %s", chat_id)


def _is_start(text: str) -> bool:
    """Telegram's ``/start`` (``/start``, ``/start@SomeBot``, ``/start payload``)."""
    first = text.strip().split(maxsplit=1)[0].lower() if text.strip() else ""
    return first == "/start" or first.startswith("/start@")


def _replied_message_id(message: dict[str, Any]) -> str:
    """The id of the message this one replies to, or "" when it is not a reply."""
    replied = message.get("reply_to_message")
    if not isinstance(replied, dict):
        return ""
    value = replied.get("message_id")
    return str(value) if value is not None else ""


def _user_id(update_part: dict[str, Any]) -> str:
    """The Telegram user who sent a message or tapped a button."""
    sender = update_part.get("from")
    return str(sender.get("id", "")) if isinstance(sender, dict) else ""
