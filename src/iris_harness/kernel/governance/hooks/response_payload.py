"""The ``PRE_RESPONSE`` payload contract: what an answer's hook context carries.

Every producer of a ``PRE_RESPONSE`` context builds its payload here (today one: the
response curator), so a key cannot be written under one name and read under another.

Payload (``HookContext.payload``):

- ``response`` (str): the answer text.
- ``audience`` (``owner`` | ``other``): who reads it (ADR-0125). An answer to the owner
  never halts on the owner's own identity; an answer someone else reads masks it. The
  owner-PII shadow hook reads it (PR 4); no guard acts on it until PR 5.

Who reads the answer is the turn's, not the curator's, to say: the surface that took the
message knows. The turn pipeline opens :func:`audience_scope` with the request's
audience and the curator reads :func:`current_audience` when it builds the payload.

- Every chat surface (REPL, web, a private Telegram chat) answers the owner.
- A Telegram chat that is a group (``chat.type`` ``group`` / ``supergroup``) is
  ``other``: its members read the answer. The poller says so, the channel gateway
  forwards it on ``/chat`` and ``/chat/stream``, and the turn carries it.
- Still ``owner`` and still to be built: a reply or draft composed for a third party (an
  email body the owner then sends, a calendar invitation's text), when it is written as
  an answer rather than as a tool's arguments (``send_email``'s arguments are
  ``PRE_TOOL_USE``, the egress side). No surface composes one as an answer today.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Final, Literal

Audience = Literal["owner", "other"]

#: Payload keys.
RESPONSE: Final = "response"
AUDIENCE: Final = "audience"

#: The audience of every answer that does not say otherwise.
OWNER: Final[Audience] = "owner"


_AUDIENCE: ContextVar[Audience] = ContextVar("iris_turn_audience", default=OWNER)


@contextmanager
def audience_scope(audience: Audience) -> Iterator[None]:
    """Who reads this turn's answer, for as long as the block runs. Called by the harness."""
    token = _AUDIENCE.set(OWNER if audience == OWNER else "other")
    try:
        yield
    finally:
        _AUDIENCE.reset(token)


def current_audience() -> Audience:
    """Who reads the answer being built: the turn's audience, ``owner`` outside a turn."""
    return _AUDIENCE.get()


def pre_response_payload(
    response: str,
    *,
    audience: Audience = OWNER,
    audit: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """The payload of a ``PRE_RESPONSE`` context; ``audit`` is metadata for the audit row."""
    return {**(audit or {}), RESPONSE: response, AUDIENCE: audience}


def audience_of(payload: dict[str, Any]) -> Audience:
    """Who reads the answer: ``owner`` when the payload does not say, and ``other`` for any
    value but ``owner`` -- an audience nobody can name is not the owner."""
    value = payload.get(AUDIENCE, OWNER)
    return OWNER if value == OWNER else "other"


__all__ = [
    "AUDIENCE",
    "OWNER",
    "RESPONSE",
    "Audience",
    "audience_of",
    "audience_scope",
    "current_audience",
    "pre_response_payload",
]
