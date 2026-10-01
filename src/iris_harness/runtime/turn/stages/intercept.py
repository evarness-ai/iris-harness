"""Stage 1 — deterministic handlers (the chain from ``config/intercepts.yaml`` + plugins).

First match answers the turn: no classifier or model runs, and the answer goes to
``guard`` and ``record`` instead. The user message was logged and screened by
``screen`` (stage 0) before this runs.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from iris_harness.runtime.turn.host import TurnHost
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import StreamEvent

logger = logging.getLogger(__name__)


def run(runtime: TurnHost, state: TurnState) -> Iterator[StreamEvent]:
    message, session_id = state.message, state.session_id
    # §4.2: attribute a measured outcome (user_correction) to the prior turn now
    # that we can see how the user responded to it.
    runtime.capture.evaluate_prior_turn_outcome(session_id, message)
    # A plain "yes"/"no" answering IRIS's own "should I remember this?" is the whole
    # turn — resolve it here rather than routing it to an agent that has no idea what
    # the question was.
    answered = runtime.capture.resolve_question(session_id, message)
    if answered is not None:
        state.result = runtime.replies.system_chat_result(
            message=message,
            session_id=session_id,
            response=answered,
            metadata={"memory_question": "answered"},
            span=state.span,
        )
        state.intercepted = True
        yield StreamEvent(
            kind="trace",
            text="memory_question.end",
            payload={"name": "memory_question.end", "matched": True},
        )
        return
    # Some deterministic intercepts scan a folder + embed its files, which runs
    # synchronously below and can take several seconds. Emit a status first so
    # the stream isn't silent (the web UI renders it as the "working" line).
    hint = runtime.intercepts.activity_hint(message)
    if hint:
        yield StreamEvent(kind="activity", text=hint)
    hit = runtime.intercepts.dispatch(
        message, session_id=session_id, channel=state.request.channel, span=state.span
    )
    if hit is None:
        return
    # A reply an intercept answered did not pick from a shortlist, and did not answer a
    # question this session was asked, so either is withdrawn — the rule classify applies
    # to every other turn (ADR-0106 ``choice`` + M5.C3). Without it an inferred question
    # would survive this turn and claim the next one.
    try:
        runtime.continuations.drop_stale_inferred(session_id, state.started_at)
    except Exception:  # bookkeeping must never fail an answered turn
        logger.exception("could not withdraw a stale continuation for session %s", session_id)
    spec = hit.spec
    if spec.trace_text is not None:
        trace_payload: dict[str, Any] = {"name": f"{spec.name}.end", "matched": True}
        for field_name in spec.trace_fields:
            trace_payload[field_name] = hit.result.metadata.get(field_name, "")
        yield StreamEvent(kind="trace", text=spec.trace_text, payload=trace_payload)
    state.result = hit.result
    state.intercepted = True
    state.handler = spec.name
    state.guard_output = spec.guard_output
