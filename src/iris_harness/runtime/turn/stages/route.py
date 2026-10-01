"""Stage 4 — pre-LLM audit + memory context.

The router decision is appended to the audit ledger, then the memory context
(user profile, retrieved facts, recent turns; identity files load here) is
built for the agent.
"""

from __future__ import annotations

from collections.abc import Iterator

from iris_harness.foundation.observability.session_log import log_timeline_event
from iris_harness.runtime.turn.host import TurnHost
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import StreamEvent


def run(runtime: TurnHost, state: TurnState) -> Iterator[StreamEvent]:
    message, session_id = state.message, state.session_id
    intent_result = state.intent_result
    assert intent_result is not None, "classify stage must run first"
    runtime._router_audit().record(
        session_id=session_id,
        message=message,
        result=intent_result,
        router_model=state.request.router_model
        or runtime.tier_router.model_for_intent("intent_classification"),
        channel=state.request.channel,
    )
    yield StreamEvent(kind="activity", text="building memory context")
    memory_ctx = runtime.sessions.build_memory_context(
        message, session_id=session_id, intent=intent_result.intent
    )
    state.memory_ctx = memory_ctx
    runtime._record_downstream_reuse(memory_ctx)
    payload = {
        "has_user_profile": bool(memory_ctx.user_profile),
        "has_active_context": bool(memory_ctx.active),
        "episodic_patterns": len(memory_ctx.episodic_patterns),
        "recent_turns": len(memory_ctx.recent_turns),
        "behavior_name": memory_ctx.behavior_name,
    }
    log_timeline_event(
        "memory.context", phase="memory.context", payload=payload, session_id=session_id
    )
    yield StreamEvent(
        kind="trace", text="memory context ready", payload={"name": "memory.context", **payload}
    )
