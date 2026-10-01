"""Answering an approval from a chat surface: a tap on its buttons, or a typed command.

The approval message carries inline buttons whose callback data is a command —
``/approve <id>``, ``/reject <id>``, and for a destructive approval ``/ask <id>`` first,
which asks "Trash 3 emails? Yes / Cancel" before anything runs (the owner's
red-button-plus-confirm decision, ADR-0118 step 4). A typed command takes the same
path. Pollers try these before handing text to chat: until 2026-09-21 a typed
``/approve`` went to the chat model as ordinary text, because the handler that parsed it
had no caller.

Where the answer goes is the backend's business. The channel gateway is a separate
process, so it answers through the API, which holds the runtime that resumes the run;
the in-process runtime poller answers directly. Both share the parsing, the user check
and the wording here.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

_ID = r"([0-9a-f-]{36})"
_COMMAND = re.compile(rf"^/(approve|reject|ask|cancel)\s+{_ID}\s*$", re.IGNORECASE)

ALLOWED_USERS_PATH = Path.home() / ".config" / "iris" / "telegram-allowed-users"

Keyboard = list[list[dict[str, str]]]


@dataclass(frozen=True)
class CommandReply:
    """What to send back: text, and optionally the buttons under it."""

    text: str
    keyboard: Keyboard | None = None


@dataclass(frozen=True)
class ApprovalSummary:
    """Enough to word a reply and pick the buttons."""

    title: str
    destructive: bool


class ApprovalNotFound(LookupError):
    """No pending approval with that id (never existed, or already answered)."""


class ApprovalBackend(Protocol):
    """Where an answer goes: the API (channel gateway) or the runtime (in process)."""

    def summary(self, approval_id: str) -> ApprovalSummary | None: ...

    def respond(self, approval_id: str, status: str, actor: str) -> tuple[bool, str]:
        """Record the answer; return (resumed, detail). Raise ``ApprovalNotFound`` when
        there is nothing pending to answer, ``ValueError`` when it was already answered."""
        ...


def approval_keyboard(approval_id: str, summary: ApprovalSummary) -> Keyboard:
    """The buttons on an approval message. A destructive one names its action and asks
    once more before running; anything else is a plain Approve / Reject."""
    if summary.destructive:
        first = {"text": summary.title, "callback_data": f"/ask {approval_id}"}
    else:
        first = {"text": "Approve", "callback_data": f"/approve {approval_id}"}
    return [[first, {"text": "Reject", "callback_data": f"/reject {approval_id}"}]]


def load_allowed_users(path: Path = ALLOWED_USERS_PATH) -> frozenset[str]:
    """Telegram user ids allowed to answer, one per line; empty (the chat allowlist
    alone applies) when the file is absent."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return frozenset()
    return frozenset(line.strip() for line in lines if line.strip() and not line.startswith("#"))


def handle_approval_command(
    text: str,
    user_id: str,
    backend: ApprovalBackend,
    *,
    allowed_users: frozenset[str] | None = None,
) -> CommandReply | None:
    """Answer ``/approve|/reject|/ask|/cancel <id>``; None when ``text`` is not one.

    The caller has already checked the chat against its allowlist. ``allowed_users`` is
    the finer check within it; empty means everyone in the allowed chat.
    """
    match = _COMMAND.match(text.strip())
    if match is None:
        return None
    verb, approval_id = match.group(1).lower(), match.group(2).lower()

    users = load_allowed_users() if allowed_users is None else allowed_users
    if users and user_id not in users:
        logger.warning("telegram user %s may not answer approvals; ignoring %s", user_id, verb)
        return CommandReply("You don't have permission to answer IRIS approvals.")

    summary = backend.summary(approval_id)
    if summary is None:
        return CommandReply("That approval is not waiting any more: it was answered or it expired.")

    if verb == "ask":
        return CommandReply(
            f"{summary.title}? IRIS will run exactly the call on the card and nothing else.",
            keyboard=[
                [{"text": f"Yes: {summary.title}", "callback_data": f"/approve {approval_id}"}],
                [{"text": "Cancel", "callback_data": f"/cancel {approval_id}"}],
            ],
        )
    if verb == "cancel":
        return CommandReply(
            f"Not done. {summary.title} is still waiting for you.",
            keyboard=approval_keyboard(approval_id, summary),
        )

    status = "approved" if verb == "approve" else "rejected"
    try:
        resumed, detail = backend.respond(approval_id, status, f"telegram:{user_id}")
    except ApprovalNotFound:
        return CommandReply("That approval is not waiting any more: it was answered or it expired.")
    except ValueError as exc:
        return CommandReply(f"Already answered: {exc}")
    if status == "approved":
        head = f"Approved: {summary.title}."
    else:
        # A destructive rejection runs nothing and resumes so IRIS can say so; an
        # evaluator rejection leaves the run halted, which its detail already says.
        head = "Rejected. Nothing was changed." if summary.destructive else "Rejected."
    return CommandReply(f"{head}\n\n{detail}" if resumed and detail else f"{head} {detail}".strip())


class LocalApprovalBackend:
    """Answers in process, for the runtime's own Telegram poller."""

    def __init__(self, *, queue: Any, resumer: Any = None, executor: Any = None) -> None:
        self._queue = queue
        self._resumer = resumer
        self._executor = executor

    def summary(self, approval_id: str) -> ApprovalSummary | None:
        row = self._queue.get(approval_id)
        if row is None or row.status != "pending":
            return None
        title = row.card.title if row.card is not None else row.signal
        return ApprovalSummary(title=title, destructive=row.items is not None)

    def respond(self, approval_id: str, status: str, actor: str) -> tuple[bool, str]:
        from iris_harness.kernel.governance.approvals.service import (
            respond_to_approval,
        )
        from iris_harness.kernel.governance.approvals.store import (
            ApprovalNotFoundError,
        )

        try:
            outcome = respond_to_approval(
                approval_id,
                status=status,
                actor=actor,
                queue=self._queue,
                resumer=self._resumer,
                executor=self._executor,
            )
        except ApprovalNotFoundError as exc:
            raise ApprovalNotFound(str(exc)) from exc
        return outcome.resumed, outcome.detail


CommandHandler = Callable[[str, str], CommandReply | None]

__all__ = [
    "ApprovalBackend",
    "ApprovalNotFound",
    "ApprovalSummary",
    "CommandHandler",
    "CommandReply",
    "LocalApprovalBackend",
    "approval_keyboard",
    "handle_approval_command",
    "load_allowed_users",
]
