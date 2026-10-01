"""The model-free response check for an answer a deterministic handler gave.

Every answer passes the kernel's ``PRE_RESPONSE`` check exactly once. A generated
answer passes it inside ``curate`` (``ResponseCurator``); a deterministic handler's
answer never reaches ``curate``, so it passes it here, marked ``deterministic`` with
the handler's name in the audit row (docs/architecture/deterministic-path-parity.md,
step c). A halted answer is replaced with the one refusal text, as ``curate`` does.

A handler whose answer repeats text someone else wrote (``guard_output``, decision B)
also gets the model-based output guard when it is enabled, with the same halt and the
same warning banner a generated answer gets.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from iris_harness.agent.response_curator import (
    GOVERNANCE_BLOCKED_TEXT,
    JudgeSignal,
    governance_warning_banner,
)
from iris_harness.foundation.observability.session_log import log_timeline_event
from iris_harness.foundation.observability.tracer import set_span_attributes
from iris_harness.runtime.turn.host import TurnHost
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import StreamEvent


def run(runtime: TurnHost, state: TurnState) -> Iterator[StreamEvent]:
    result = state.result
    assert result is not None, "guard runs on an answered turn"
    curator = runtime.response_curator
    started = time.perf_counter()
    signal = curator.guard(result.response, session_id=state.session_id, handler=state.handler)
    signals: list[JudgeSignal] = [signal]
    if signal.verdict != "halt" and state.guard_output:
        signals.append(
            curator.guard_output(
                result.response, session_id=state.session_id, handler=state.handler
            )
        )
    halted = next((s for s in signals if s.verdict == "halt"), None)
    set_span_attributes(
        state.span,
        {"iris.deterministic": True, "iris.deterministic_handler": state.handler or ""},
    )
    outcome: dict[str, Any] = {
        "verdict": halted.verdict if halted else signal.verdict,
        "handler": state.handler,
        "checks": [s.name for s in signals],
    }
    # On the turn's record too, so Call trace draws the check a deterministic answer
    # passed next to the audit rows the check wrote.
    log_timeline_event(
        "guard.end",
        phase="guard.end",
        payload={
            **outcome,
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        session_id=state.session_id,
    )
    yield StreamEvent(kind="trace", text="guard.end", payload={"name": "guard.end", **outcome})
    if halted is not None:
        state.result = replace(
            result,
            response=GOVERNANCE_BLOCKED_TEXT,
            has_errors=True,
            error_summary=halted.reason,
            metadata={
                **result.metadata,
                "governance_guard": "halt",
                "pattern": halted.metadata.get("pattern"),
                "halted_by": halted.name,
            },
        )
        return
    warnings = tuple(f"{s.name}: {s.reason}" for s in signals if s.verdict in {"warn", "retry"})
    if warnings:
        banner = governance_warning_banner(warnings)
        state.result = replace(
            result,
            response=f"{banner}\n\n{result.response}" if result.response else banner,
            metadata={**result.metadata, "warning_banner": banner},
        )
