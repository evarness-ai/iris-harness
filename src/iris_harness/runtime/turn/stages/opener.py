"""Stage 0 of a system-opened turn — the opener answers a turn nobody typed (ADR-0127).

A turn the system opens has no user message, so it runs neither ``screen`` (there is no
input to screen, and logging one would put words in the owner's mouth) nor
``intercept`` (whose handlers read a message). This stage takes their place: it records
the turn's start as ``turn_open``, runs the named opener (a deterministic handler from
``config/intercepts.yaml`` ``openers:``) and hands its answer to ``guard`` and
``record``, exactly as ``intercept`` hands over a handler's answer.
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from iris_harness.foundation.observability.session_log import (
    log_timeline_event,
    log_turn_open,
)
from iris_harness.runtime.turn.host import TurnHost
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import StreamEvent


def run(runtime: TurnHost, state: TurnState) -> Iterator[StreamEvent]:
    opener, session_id = state.request.opener, state.session_id
    if not opener:
        raise RuntimeError("a system-opened turn needs an opener")
    spec = runtime.intercepts.opener(opener)
    if spec is None:
        raise RuntimeError(f"no opener named {opener!r} is declared")
    log_turn_open(session_id, opener=opener, label=spec.trace_text or opener)
    log_timeline_event(
        "turn.start",
        phase="turn.start",
        payload={"message_chars": 0, "opener": opener},
        session_id=session_id,
    )
    yield StreamEvent(
        kind="trace",
        text="turn.start",
        payload={"name": "turn.start", "message_chars": 0, "opener": opener},
    )
    started = time.perf_counter()
    hit = runtime.intercepts.open(
        opener, session_id=session_id, channel=state.request.channel, span=state.span
    )
    if hit is None:
        raise RuntimeError(f"opener {opener!r} gave no answer")
    state.result = hit.result
    state.intercepted = True
    state.handler = hit.spec.name
    state.guard_output = hit.spec.guard_output
    log_timeline_event(
        "handler.end",
        phase="handler.end",
        payload={
            "handler": hit.spec.name,
            "intent": hit.result.intent,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        session_id=session_id,
    )
    if hit.spec.trace_text is not None:
        yield StreamEvent(
            kind="trace",
            text=hit.spec.trace_text,
            payload={"name": f"{hit.spec.name}.end", "matched": True},
        )
