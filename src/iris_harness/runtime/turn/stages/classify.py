"""Stage 2 — intent classification (keyword-first; router LLM only below the threshold).

The deterministic refinements (stage 3, :mod:`resolve`) run inside this stage's
span so the span carries the final intent, exactly as the pre-pipeline ``chat``
did.

ADR-0106 adds one override after those refinements: when this session owes an
answer to a known owner and the user just gave it, the turn routes to that owner
rather than to whatever the classifier makes of a bare "yes" in isolation.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from typing import TYPE_CHECKING

from iris_harness.foundation.observability.session_log import agent_scope, log_timeline_event
from iris_harness.runtime.continuations import reads_as_answer, reads_as_choice
from iris_harness.runtime.turn.host import TurnHost
from iris_harness.runtime.turn.stages import resolve, stage_span
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import StreamEvent

if TYPE_CHECKING:
    from iris_harness.memory.state.continuations import Continuation


def run(runtime: TurnHost, state: TurnState) -> Iterator[StreamEvent]:
    message, session_id = state.message, state.session_id
    yield StreamEvent(kind="activity", text="routing intent")
    log_timeline_event("pipeline.phase", phase="intent_router.start", session_id=session_id)
    classifier = runtime._classifier_for(state.request.router_model)
    with stage_span(runtime, state, "iris.stage.intent_router") as span:
        with agent_scope("intent_router"):
            state.intent_result = classifier.classify(
                message, context=runtime.sessions.format_recent_context(session_id)
            )
        yield from resolve.run(runtime, state, span=span)
        _honour_continuation(runtime, state)


def _honour_continuation(runtime: TurnHost, state: TurnState) -> None:
    """Route an answer back to whoever asked the question (ADR-0106).

    A bare "yes" carries no intent of its own, so classified in isolation it lands
    wherever the keywords fall. The continuation says who is waiting, so it decides.

    This **steers**, it does not answer. An intercept that answered here would skip
    the ResponseCurator and every safety judge behind it — the red-team rule the
    intercept chain documents — so the turn goes on through route → plan → execute
    → curate exactly like any other, just pointed at the right owner. Tier B resumes
    the halted loop *inside* the execute stage for the same reason: the resumed run
    re-enters at the paused step and its answer leaves through curate like any other.

    **The two tiers claim a turn differently, and Tier B claims it unconditionally.**
    Tier A reads a yes/no, so a message that is not one leaves the question open.
    Tier B's question is free text ("which of these did you mean?"), which
    ``reads_as_answer`` deliberately refuses to classify — so there is no yes/no to
    find, and the rule is instead the one the one-pending-per-session invariant
    already implies: the paused run asked this conversation the most recent question,
    so it owns the next message. A user who ignores the question and changes the
    subject is resumed too; the answer is injected as the paused step's observation
    together with an explicit instruction to abandon the old task if it does not fit
    (see ``_RESUME_OBSERVATION`` in ``core.agentic_core``). That keeps the harness
    deterministic about *who* the turn belongs to — the fact whose absence caused the
    incident — and leaves "is this an answer or a new subject" with the model, which
    is the only thing that can read it.
    """
    if state.request.resume_point is not None:
        # An approved governance halt continuing itself. The turn's owner is already
        # settled by the approval, and there is no user message here to read as an
        # answer — claiming a pending continuation would close someone else's question
        # on the strength of a reply that was never sent.
        return
    continuation = runtime.continuations.pending(state.session_id)
    if continuation is None:
        return
    if continuation.kind == "choice":
        _honour_choice(runtime, state, continuation)
        return
    decision = reads_as_answer(continuation, state.message)
    if decision is None and not continuation.is_resumable_run and continuation.kind != "question":
        return  # not an answer; the question stays open and the turn routes normally
    if continuation.kind == "question" and not continuation.is_resumable_run:
        # An inferred free-text question (ADR-0106 M5.C3). It claims this turn on the
        # same rule Tier B already uses: the owner asked this conversation the most
        # recent question, so it owns the next message, and the model decides whether
        # that message answers it or changes the subject. Before this, such a
        # continuation steered nothing — `reads_as_answer` refuses a non-approval kind
        # and there is no run to resume — so the reply was classified from scratch.
        #
        # One turn only, like `choice`: it is answered by the next reply or not at all.
        # A question that shielded the router indefinitely would be the worse bug.
        _honour_inferred_question(runtime, state, continuation)
        return
    state.continuation = continuation
    state.continuation_decision = decision
    runtime.continuations.answered(continuation.continuation_id)
    if continuation.intent and state.intent_result is not None:
        state.intent_result = replace(
            state.intent_result,
            intent=continuation.intent,
            agent_type=continuation.intent,
            source="continuation",
        )
    log_timeline_event(
        "pipeline.phase",
        phase="continuation.resumed",
        payload={
            "owner": continuation.owner,
            "decision": decision,
            "tier": "B" if continuation.is_resumable_run else "A",
            "run_id": continuation.run_id,
            "step_id": continuation.step_id,
        },
        session_id=state.session_id,
    )


def _honour_inferred_question(
    runtime: TurnHost, state: TurnState, continuation: Continuation
) -> None:
    """Point this turn at the owner that asked, and close the question.

    Steering only, like every other branch here: the turn still goes on through route →
    plan → execute → curate. ``plan`` composes the task so the owner receives the
    question alongside the reply — "Key quotes" on its own is not a task, and the
    goal-drift guard measuring a run against two words halts it.
    """
    state.continuation = continuation
    runtime.continuations.answered(continuation.continuation_id)
    if state.intent_result is not None:
        state.intent_result = replace(
            state.intent_result,
            intent=continuation.intent or state.intent_result.intent,
            agent_type=continuation.owner or state.intent_result.agent_type,
            source="continuation",
        )
    log_timeline_event(
        "pipeline.phase",
        phase="continuation.resumed",
        payload={"owner": continuation.owner, "tier": "question", "run_id": None},
        session_id=state.session_id,
    )


def _honour_choice(runtime: TurnHost, state: TurnState, continuation: Continuation) -> None:
    """Route a pick from a numbered list back to the owner that offered the list.

    A ``choice`` differs from the other kinds in one respect: its answers were
    enumerated when it was asked, so which option the reply names is decidable here
    (``reads_as_choice``) and the option itself travels to the owner — the owner acts
    on what was on screen rather than re-deriving it from the words "the 1 st one".

    **A choice is answered by the next reply or not at all.** A yes/no question can
    stay open across an unrelated turn because its vocabulary is narrow; "2" is not.
    Left open, a stale shortlist would claim a bare number the user later types in
    reply to something else entirely, so a reply that does not pick withdraws it.
    """
    index = reads_as_choice(continuation, state.message)
    if index is None:
        runtime.continuations.drop(state.session_id)
        return
    state.continuation = continuation
    state.continuation_choice = continuation.choices[index]
    runtime.continuations.answered(continuation.continuation_id)
    if state.intent_result is not None:
        state.intent_result = replace(
            state.intent_result,
            intent=continuation.intent or state.intent_result.intent,
            agent_type=continuation.owner,
            source="continuation",
        )
    log_timeline_event(
        "pipeline.phase",
        phase="continuation.resumed",
        payload={"owner": continuation.owner, "tier": "choice", "choice": index + 1},
        session_id=state.session_id,
    )
