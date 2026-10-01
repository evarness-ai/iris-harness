"""The current turn's user message, readable from anywhere on that turn.

A ReAct tool built by the core closes over the turn's query (that is how
``search_inbox`` rejects search terms the user never typed). A tool contributed
by a **plugin** cannot: it is registered once at ``setup(api)``, long before any
turn exists. Without a seam, every extracted domain tool silently loses its
query fallback — ``finance_lookup`` called with no arguments would answer the
wrong question instead of the asked one (OSS plan M4.2).

So the harness publishes the turn's query here and hands plugins a reader on
``HarnessServices.current_query``. A ``ContextVar`` (not a module global) keeps
it correct under the concurrent requests the API serves, the same choice the
email agent already made for its own query.

The turn's session id is published beside it for the same reason: a plugin tool that
puts a question to the user (a numbered shortlist it wants picked from) must record
that question against the conversation it was asked in, and has no other way to learn
which one that is (ADR-0106 ``choice``).
"""

from __future__ import annotations

from contextvars import ContextVar

_CURRENT_QUERY: ContextVar[str] = ContextVar("iris_turn_query", default="")
_CURRENT_SESSION_ID: ContextVar[str] = ContextVar("iris_turn_session_id", default="")


def set_current_query(query: str) -> None:
    """Publish the message this turn is answering. Called by the harness."""
    _CURRENT_QUERY.set(query or "")


def current_query() -> str:
    """The message this turn is answering, or ``""`` outside a turn."""
    return _CURRENT_QUERY.get("")


def set_current_session_id(session_id: str) -> None:
    """Publish the session this turn belongs to. Called by the harness."""
    _CURRENT_SESSION_ID.set(session_id or "")


def current_session_id() -> str:
    """The session this turn belongs to, or ``""`` outside a turn."""
    return _CURRENT_SESSION_ID.get("")
