"""Deterministic replies — a templated turn with no model in the loop.

``system_chat_result`` builds the ``ChatResult`` a core intercept or a plugin answers
with when the reply is fixed text: recorded into the session, signalled and traced like
any other turn. It is ``HarnessServices.deterministic_reply``, and the confirmation turn
and the standing-instruction turn answer through it too.

Carved out of ``IrisRuntime`` at OSS plan M5.7 track C slice 14 as
``DeterministicReplies(host)``, held as ``runtime.replies``. Stateless.
:class:`RepliesHost` declares the two runtime members read; the host is read **at call
time**, not captured. It is its own collaborator rather than a method on the
confirmation turn because three callers reach it and only one of them is a
confirmation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from iris_harness.agent.intent_router import IntentResult
from iris_harness.agent.response_curator import CuratedResponse
from iris_harness.runtime.types import ChatResult

if TYPE_CHECKING:
    from iris_harness.runtime.session_memory import SessionMemory
    from iris_harness.runtime.turn_capture import TurnCapture


class RepliesHost(Protocol):
    """The two runtime members a deterministic reply reaches."""

    capture: TurnCapture
    sessions: SessionMemory


class DeterministicReplies:
    """Builds deterministic turns for one runtime. See the module docstring."""

    def __init__(self, host: RepliesHost) -> None:
        self._host = host

    def system_chat_result(
        self,
        *,
        message: str,
        session_id: str,
        response: str,
        metadata: dict[str, object],
        span: Any = None,
        intent: str = "system",
        agent_type: str | None = None,
        sources: Sequence[str] = ("system",),
        has_errors: bool = False,
        error_summary: str | None = None,
    ) -> ChatResult:
        """A deterministic, templated turn: recorded, signalled and traced like any
        other, with no model in the loop. This is ``HarnessServices.deterministic_reply``.

        ``intent`` / ``agent_type`` / ``sources`` let a domain plugin label its own
        turns (the calendar plugin answers as ``calendar`` over ``reminders``) and
        ``has_errors`` / ``error_summary`` let it report a failed action honestly;
        the defaults are the system reply every caller had before.
        """
        intent_result = IntentResult(
            intent=intent,
            agent_type=agent_type or intent,
            confidence=0.99,
            raw_query=message,
        )
        curated = CuratedResponse(
            text=response,
            sources=list(sources),
            has_errors=has_errors,
            error_summary=error_summary,
            metadata={"total_latency_ms": 0.0, **metadata},
        )
        self._host.sessions.record_turn(session_id, message, response)
        self._host.capture.record_signal(
            intent_result,
            curated,
            latency_ms=0.0,
            model="deterministic",
            provider="local",
            query=message,
        )
        if span is not None:
            try:
                span.set_attribute("output.value", response)
            except Exception:  # noqa: BLE001, S110
                pass
            span.set_attribute("intent", intent_result.intent)
            span.set_attribute("agent_type", intent_result.agent_type)
            span.set_attribute("session_id", session_id)
            span.set_attribute("model", "deterministic")
            span.set_attribute("provider", "local")
            span.set_attribute("latency_ms", 0.0)
            span.set_attribute("has_errors", has_errors)
        return ChatResult(
            response=response,
            intent=intent_result.intent,
            agent_type=intent_result.agent_type,
            sources=tuple(curated.sources),
            has_errors=has_errors,
            error_summary=error_summary,
            metadata={
                **curated.metadata,
                "is_multi_step": False,
                "plan_size": 0,
                "session_id": session_id,
                "model": "deterministic",
                "provider": "local",
                "router_model": "deterministic",
                "router_provider": "local",
            },
        )
