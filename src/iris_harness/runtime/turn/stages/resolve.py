"""Stage 3 — deterministic refinements of the classified intent.

Continuation ("yes", follow-ups), skill routing (semantic router), and the
action-escalation gate (regex). Emits ``intent_router.end``. Runs inside the
classify stage's span (see :mod:`classify`).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from iris_harness.foundation.observability.session_log import log_timeline_event
from iris_harness.foundation.observability.tracer import set_span_attributes
from iris_harness.runtime.turn.host import TurnHost
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import StreamEvent


def run(runtime: TurnHost, state: TurnState, *, span: Any = None) -> Iterator[StreamEvent]:
    message, session_id = state.message, state.session_id
    intent_result = state.intent_result
    assert intent_result is not None, "classify stage must run first"
    intent_result = runtime._resolve_continuation_intent(
        message, intent_result, session_id=session_id
    )
    intent_result = runtime._resolve_skill_intent(message, intent_result)
    intent_result = runtime._resolve_action_intent(message, intent_result)
    state.intent_result = intent_result
    set_span_attributes(
        span,
        {
            "iris.intent": intent_result.intent,
            "iris.agent_type": intent_result.agent_type,
            "iris.confidence": intent_result.confidence,
            "iris.is_multi_step": intent_result.is_multi_step,
            **runtime.tier_router.trace_metadata_for_intent(intent_result.intent),
        },
    )
    payload = {
        "intent": intent_result.intent,
        "agent_type": intent_result.agent_type,
        "confidence": intent_result.confidence,
        "is_multi_step": intent_result.is_multi_step,
    }
    log_timeline_event(
        "intent_router.end", phase="intent_router.end", payload=payload, session_id=session_id
    )
    yield StreamEvent(
        kind="trace",
        text=f"intent {intent_result.intent} -> {intent_result.agent_type}",
        payload={"name": "intent_router.end", **payload},
    )
