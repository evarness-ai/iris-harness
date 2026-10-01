"""Approval delivery over Telegram — the kernel's ApprovalChannel, implemented here.

Lived in ``governance/approvals/channels/`` until M6.2. The kernel decides which channel
an approval goes to and what it says; it cannot also own a transport that talks to
Telegram, because that means the bottom-ish layer importing the channels package above
it (OSS plan M6, decision 6). So the protocol stayed there and this came here, and the
composition root registers it with ``register_remote_channel``.

Reply handling is the channel gateway's, as before.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from iris_harness.kernel.governance.approvals.store import ApprovalRow
from iris_harness.services.channels.approval_commands import ApprovalSummary, approval_keyboard
from iris_harness.services.channels.models import ChannelMessage, DeliveryStatus

if TYPE_CHECKING:
    from iris_harness.services.channels.connectors.telegram import TelegramConnector

logger = logging.getLogger(__name__)


def _md(text: str) -> str:
    """Escape Telegram legacy-Markdown specials in text we did not write (an email
    subject with an underscore or an asterisk would otherwise break the message).
    Legacy mode escapes exactly these four, and only outside an entity."""
    for ch in ("_", "*", "`", "["):
        text = text.replace(ch, "\\" + ch)
    return text


def _destructive_text(approval: ApprovalRow) -> str:
    """A destructive-tool approval in plain words (ADR-0118 step 4): what will change,
    whether it can be undone, what the owner asked — then how to answer."""
    card = approval.card
    assert card is not None
    lines = "\n".join(f"• {_md(line)}" for line in card.lines)
    # Escapes are not allowed inside an entity, so the plugin- and user-written text
    # stays outside bold/italic: the title line is plain, the request is quoted.
    asked = f'\nYou asked: "{_md(card.asked)}"' if card.asked else ""
    return (
        f"*Approve?* {_md(card.title)}\n\n{lines}\n\n{_md(card.undo_sentence())}{asked}\n\n"
        f"Tap a button below. Answer by `{approval.timeout_at[:19]} UTC`."
    )


class TelegramApprovalChannel:
    """Send approval notifications via Telegram; reply handling lives in channel_gateway."""

    name = "telegram"

    def __init__(
        self,
        *,
        connector: TelegramConnector | None = None,
        chat_id: str | None = None,
        bot_token: str | None = None,
    ) -> None:
        self._connector = connector
        self._chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self._bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")

    def _get_connector(self) -> TelegramConnector | None:
        if self._connector is not None:
            return self._connector
        if not self._bot_token:
            return None
        from iris_harness.services.channels.connectors.telegram import TelegramConnector as _TC

        return _TC(self._bot_token, default_chat_id=self._chat_id)

    def is_configured(self) -> bool:
        return bool(self._connector or self._bot_token)

    def notify_timeout(self, approval: ApprovalRow) -> None:
        """The request went unanswered — and its /approve commands no longer work."""
        self._send(
            approval,
            f"\u23f0 *IRIS Approval Timed Out*\n\n"
            f"Run: `{approval.run_id}`\n"
            f"Signal: `{approval.signal}`\n"
            f"Context: {approval.context_summary}\n\n"
            f"No answer before `{approval.timeout_at[:19]} UTC`, so "
            f"{approval.lapse_consequence()}. Ask again and I'll start over.",
        )

    def notify(self, approval: ApprovalRow) -> None:
        connector = self._get_connector()
        if connector is None:
            logger.warning(
                "TelegramChannel: no bot token configured; approval %s queued only",
                approval.approval_id,
            )
            return

        if approval.card is not None:
            self._send(approval, _destructive_text(approval), buttons=True)
            return
        text = (
            f"\U0001f514 *IRIS Approval Required*\n\n"
            f"Run: `{approval.run_id}`\n"
            f"Signal: `{approval.signal}`\n"
            f"Context: {approval.context_summary}\n\n"
            f"Tap a button below. Times out `{approval.timeout_at[:19]} UTC`."
        )
        self._send(approval, text, buttons=True)

    def _send(self, approval: ApprovalRow, text: str, *, buttons: bool = False) -> None:
        connector = self._get_connector()
        if connector is None:
            logger.warning(
                "TelegramChannel: no bot token configured; approval %s not delivered",
                approval.approval_id,
            )
            return
        # One tap answers: the buttons send /approve|/reject <id> (a destructive one asks
        # "Yes / Cancel" first), which the Telegram poller routes before chat. Only a new
        # approval gets them; a timeout notice has nothing left to answer.
        metadata: dict[str, object] = {"parse_mode": "Markdown"}
        if buttons:
            title = approval.card.title if approval.card is not None else approval.signal
            metadata["inline_keyboard"] = approval_keyboard(
                approval.approval_id,
                ApprovalSummary(title=title, destructive=approval.items is not None),
            )
        msg = ChannelMessage(body=text, recipient=self._chat_id or "", metadata=metadata)
        try:
            receipt = connector.send(msg)
            if receipt.status == DeliveryStatus.FAILED:
                logger.warning("TelegramChannel send failed: %s", receipt.error)
            else:
                logger.info("approval %s notified via Telegram", approval.approval_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("TelegramChannel send error: %s", exc)
