"""Stage 8 — record the answered turn: session log, root span, optional mission proposal."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from iris_harness.foundation.observability.session_log import log_agent_response
from iris_harness.foundation.observability.tracer import set_span_attributes
from iris_harness.runtime.continuations import (
    _OFFER_TAIL_CHARS,
    ends_on_a_question,
    offered_choices,
    offers_to_proceed,
)
from iris_harness.runtime.turn.host import TurnHost
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import StreamEvent

logger = logging.getLogger(__name__)


def run(runtime: TurnHost, state: TurnState) -> Iterator[StreamEvent]:
    final = state.result
    assert final is not None, "curate stage must run first"
    total_tokens_meta = final.metadata.get("total_tokens")
    total_tokens = (
        int(total_tokens_meta)
        if isinstance(total_tokens_meta, int | float | str)
        and str(total_tokens_meta).strip().isdigit()
        else None
    )
    latency_meta = final.metadata.get("total_latency_ms", 0.0)
    total_duration_ms = float(latency_meta) if isinstance(latency_meta, int | float | str) else None
    log_agent_response(
        state.session_id,
        response=final.response,
        intent=final.intent,
        agent_type=final.agent_type,
        has_errors=final.has_errors,
        total_tokens=total_tokens,
        total_duration_ms=total_duration_ms,
    )
    set_span_attributes(
        state.span,
        {
            "iris.total_tokens": total_tokens,
            "iris.total_latency_ms": total_duration_ms,
            "iris.intent": final.intent,
            "iris.agent_type": final.agent_type,
            "iris.has_errors": final.has_errors,
        },
    )
    # Reactive mission auto-creation (opt-in IRIS_MISSION_AUTOCREATE): a multi-step
    # goal is *proposed* as a tracked mission (PENDING → Action Center). Additive —
    # the turn already answered above.
    intent_result = state.intent_result
    if (
        intent_result is not None
        and runtime.mission_proposals.mission_autocreate_enabled()
        and intent_result.is_multi_step
        and not final.has_errors
    ):
        from iris_harness.services.missions.proposer import (
            build_multi_step_mission,
        )

        runtime.mission_proposals.propose_mission(
            build_multi_step_mission(state.message, intent=intent_result.intent)
        )
    _record_continuation(runtime, state, final)
    yield from ()


def _record_continuation(runtime: TurnHost, state: TurnState, final: Any) -> None:
    """Register the question this answer leaves hanging (ADR-0106 C4).

    The incident's missing fact. The planner ended a turn with "Would you like to
    proceed with this plan?" and recorded nothing, so the user's "yes" arrived at a
    harness where 22 intercepts could each see that *an* approvable thing existed
    and none could see who had asked. Now the asker is on record, the confirmation
    intercepts are shielded from an answer that is not theirs, and classify routes
    it home.

    Only agent-answered turns register one here: a deterministic handler's turn has no
    ``intent_result`` (it skipped ``classify``), so an intercept that wants a continuation
    opens one itself through ``HarnessServices``. That is the seam, and the file organizer
    is its first consumer.
    """
    intent_result = state.intent_result
    if intent_result is None or final.has_errors:
        return
    owner = final.agent_type or intent_result.agent_type
    if not owner:
        return

    # Tier B: the loop stopped ON an `ask_user` step, so this is not a guess about
    # whether a question was asked — the agent said so, and the run is resumable from
    # the checkpoint it wrote. Carry the resume point on the continuation.
    paused_at = final.metadata.get("paused_at_step")
    if isinstance(paused_at, int):
        _ask(
            runtime,
            state,
            owner=owner,
            question=final.response.strip(),
            kind="question",
            intent=intent_result.intent,
            run_id=str(final.metadata.get("run_id") or "") or None,
            step_id=paused_at,
        )
        return

    # Tier A: no explicit pause, but the answer ends on a question. Only inferred, so it
    # never supersedes a question an owner opened explicitly during this turn — the email
    # shortlist ends "Which one should I read?", and superseding it would throw away the
    # very options the reply is about to pick from.
    #
    # The gate is the question mark, not the phrasing. Requiring an offer phrase here is
    # what left the 2026-09-16 news thread unowned: "Do you want full articles, more
    # sources, or a short briefing on any one story?" matched nothing, so every follow-up
    # was classified from scratch and the third one reached the inbox.
    if ends_on_a_question(final.response) and not _opened_this_turn(runtime, state):
        # An offer to pick from a numbered list the answer just showed is a choice, and
        # its options are what was on screen. A yes/no approval cannot take "4" for an
        # answer, so recording it as one left the pick with nothing to resolve against.
        options = offered_choices(final.response)
        if options:
            _ask(
                runtime,
                state,
                owner=owner,
                question=final.response.strip().splitlines()[-1].strip(),
                kind="choice",
                intent=intent_result.intent,
                payload={
                    "choices": [
                        {"position": n, "text": text} for n, text in enumerate(options, start=1)
                    ]
                },
            )
            return
        # An offer to act is a yes/no. Anything else the answer ends on is a real
        # question, and its answer is free text — the kind that had no path home until
        # now (ADR-0106 M5.C3).
        _ask(
            runtime,
            state,
            owner=owner,
            question=final.response[-_OFFER_TAIL_CHARS:].strip(),
            kind="approval" if offers_to_proceed(final.response) else "question",
            intent=intent_result.intent,
        )


def _opened_this_turn(runtime: TurnHost, state: TurnState) -> bool:
    """True when an owner opened this session's pending question during this turn."""
    try:
        opened = runtime.continuations.opened_since(state.session_id, state.started_at)
    except Exception:  # bookkeeping must never fail an answered turn
        logger.exception("could not read the open continuation for %s", state.session_id)
        return False
    return opened is not None


def _ask(runtime: TurnHost, state: TurnState, **kwargs: Any) -> None:
    """Record the question, never at the cost of the answer already delivered."""
    try:
        runtime.continuations.ask(state.session_id, **kwargs)
    except Exception:  # bookkeeping must never fail an answered turn
        logger.exception("could not record a continuation for session %s", state.session_id)
