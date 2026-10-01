"""One code path for a chat turn (OSS plan decision 9).

``run_turn`` drives the stages in order and yields :class:`StreamEvent`s.
``chat_stream`` yields them straight through; ``chat`` drains them with
:func:`drain` and returns the final result. The two entry points can no longer
drift because there is nothing to keep in sync.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import replace as _replace
from typing import Any

from iris_harness.foundation.activity import chat_turn
from iris_harness.foundation.observability.session_log import agent_scope
from iris_harness.kernel.governance.display_mask import StreamMasker, mask_text
from iris_harness.kernel.governance.hooks.response_payload import audience_scope
from iris_harness.kernel.governance.turn_label import turn_label_scope
from iris_harness.runtime.turn.host import TurnHost
from iris_harness.runtime.turn.stages import (
    classify,
    curate,
    execute,
    guard,
    intercept,
    plan,
    record,
    route,
    screen,
)
from iris_harness.runtime.turn.state import OnError, TurnRequest, TurnState
from iris_harness.runtime.types import ChatResult, StreamEvent

logger = logging.getLogger(__name__)

Stage = Callable[[TurnHost, TurnState], Iterator[StreamEvent]]

# The order is the design: the turn-level input screen first (every turn, whatever
# answers it), deterministic handlers next, then the classifier (with its deterministic
# refinements, stage `resolve`, inside its span), the pre-LLM audit + memory, the plan,
# the governed execution, the curator, the guard for a deterministic answer, and
# finally the record.
STAGES: tuple[tuple[str, Stage], ...] = (
    ("screen", screen.run),
    ("intercept", intercept.run),
    ("classify", classify.run),
    ("route", route.run),
    ("plan", plan.run),
    ("execute", execute.run),
    ("curate", curate.run),
    ("guard", guard.run),
    ("record", record.run),
)

# Which turns each stage serves. Nothing breaks out of the pipeline: a stage that does
# not serve this turn is skipped, and the stages every turn needs always run
# (deterministic-path parity, step c).
#   all        — every turn, however it was answered
#   generated  — only a turn no deterministic handler answered (the model path)
#   handled    — only a turn a deterministic handler answered
# A turn the screen refused runs `record` alone: it writes the refusal to the session
# log, so it reaches the transcript through the one recording path.
STAGE_AUDIENCE: dict[str, str] = {
    "screen": "all",
    "intercept": "all",
    "classify": "generated",
    "route": "generated",
    "plan": "generated",
    "execute": "generated",
    "curate": "generated",
    "guard": "handled",
    "record": "all",
}


def serves(name: str, state: TurnState) -> bool:
    """Whether stage ``name`` runs for this turn, from its audience and the turn so far."""
    if state.screened_out:
        return name == "record"
    audience = STAGE_AUDIENCE.get(name, "all")
    if audience == "generated":
        return not state.intercepted
    if audience == "handled":
        return state.intercepted
    return True


# A resume is a turn with no new user message behind it: an approved governance halt
# continuing itself. So `screen` and `intercept` are dropped — `screen` logs and screens
# the user message (a resume must not forge a second one, and the original was screened
# when it arrived) and `intercept` could hand the turn to something other than the run
# being resumed. Everything after them stays,
# deliberately: `curate` is the ResponseCurator and every safety judge, and dropping
# those to save a step is the bypass the intercept chain's 2026-07-06 red-team rule
# exists to prevent. `record` then puts the answer in the session log, which is how it
# reaches the web transcript.
RESUME_STAGES: tuple[tuple[str, Stage], ...] = tuple(
    stage for stage in STAGES if stage[0] not in {"screen", "intercept"}
)
# Who the session log credits with an LLM call made while a stage runs. The execute
# stage's name is only the default: AgentExecutor narrows it to the task's agent
# (``email``, ``system`` …), and a handler may narrow it again.
STAGE_AGENTS: dict[str, str] = {
    "screen": "governance",
    "intercept": "intercept",
    "classify": "intent_router",
    "route": "route",
    "plan": "task_planner",
    "execute": "agent_executor",
    "curate": "response_curator",
    "guard": "governance",
    "record": "turn_capture",
}


def run_turn(
    runtime: TurnHost,
    request: TurnRequest,
    *,
    span: Any = None,
    stage_spans: bool = False,
    on_error: OnError = "event",
    stages: tuple[tuple[str, Stage], ...] = STAGES,
) -> Iterator[StreamEvent]:
    """Run the stages; yield events; end with ``done`` (or ``error`` / an exception).

    ``on_error="event"`` (streaming) converts any exception into a terminal
    ``error`` event with a user-friendly message; ``"raise"`` (sync) lets it
    propagate so API callers keep their exception semantics.
    """
    # Chat comes first: background work that shares the model yields while a turn runs
    # (foundation/activity.py). The turn's data label is published for the whole turn
    # (kernel/governance/turn_label.py): ``screen`` seeds it, a governed call's result lifts
    # it, code calls are stamped with it and the loop floors its own label on it. Here, not
    # in the facades, so ``chat``, ``chat_stream`` and the resume path all get it. Who
    # reads the answer is published the same way, for the PRE_RESPONSE check (ADR-0125).
    with chat_turn(), turn_label_scope(), audience_scope(request.audience):
        yield from _run_stages(runtime, request, span, stage_spans, on_error, stages)


def _run_stages(
    runtime: TurnHost,
    request: TurnRequest,
    span: Any,
    stage_spans: bool,
    on_error: OnError,
    stages: tuple[tuple[str, Stage], ...],
) -> Iterator[StreamEvent]:
    state = TurnState(request=request, span=span, stage_spans=stage_spans)
    # Display masking runs on the way out, after `record` has written the turn, so
    # the audit ledger keeps the original text and only the screen sees the mask.
    masker = StreamMasker()
    try:
        for name, stage in stages:
            if not serves(name, state):
                continue
            with agent_scope(STAGE_AGENTS.get(name, name)):
                for event in stage(runtime, state):
                    if event.kind == "token":
                        safe = masker.feed(event.text or "")
                        if safe:
                            yield _replace(event, text=safe)
                        continue
                    yield event
        if state.result is None:
            raise RuntimeError("turn produced no result")
        tail = masker.flush()
        if tail:
            yield StreamEvent(kind="token", text=tail)
        yield StreamEvent(
            kind="done",
            result=_replace(state.result, response=mask_text(state.result.response)),
        )
    except Exception as exc:  # the boundary of the whole turn
        if on_error == "raise":
            raise
        from iris_harness.llm.errors import (
            friendly_llm_error as _friendly_llm_error,
        )

        logger.exception("chat turn failed for message=%r", request.message)
        yield StreamEvent(kind="error", error=_friendly_llm_error(exc))


def drain(events: Iterator[StreamEvent]) -> ChatResult:
    """Consume a turn's events and return the final result (the ``chat`` path)."""
    for event in events:
        if event.kind == "done" and event.result is not None:
            return event.result
        if event.kind == "error":
            raise RuntimeError(event.error or "chat turn failed")
    raise RuntimeError("turn ended without a result")


__all__ = ["RESUME_STAGES", "STAGES", "Stage", "drain", "run_turn"]
