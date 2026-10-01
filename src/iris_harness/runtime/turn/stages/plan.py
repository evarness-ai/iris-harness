"""Stage 5 — task planning: one task for the common case, a DAG for multi-step asks."""

from __future__ import annotations

from collections.abc import Iterator

from iris_harness.agent.agent_executor import AgentTask
from iris_harness.foundation.observability.session_log import log_timeline_event
from iris_harness.foundation.observability.tracer import set_span_attributes
from iris_harness.runtime.continuations import choice_request, question_request
from iris_harness.runtime.turn.host import TurnHost
from iris_harness.runtime.turn.stages import stage_span
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import StreamEvent


def task_params(state: TurnState) -> dict[str, str]:
    """The per-turn params every AgentTask carries (intent bias + model overrides)."""
    intent_result = state.intent_result
    assert intent_result is not None
    extra: dict[str, str] = {"intent": intent_result.intent}
    if intent_result.is_multi_step:
        extra["is_multi_step"] = "1"
    if state.request.preferred_model:
        extra["preferred_model"] = state.request.preferred_model
    if state.request.provider_profile:
        extra["provider_profile"] = state.request.provider_profile
    return extra


def _resume_task(runtime: TurnHost, state: TurnState) -> AgentTask | None:
    """The single task that continues a halted run, or None for ordinary planning.

    A resume is not a plan: the work was already decomposed by the run that paused,
    and its steps are in the checkpoint. Planning the *reply* instead would decompose
    "the second one" — and on a multi-step reply would fan out a wave of fresh runs
    beside the paused one, leaving the checkpoint stranded and answering a question
    the user did not ask. So the planner is skipped outright and one task carries the
    resume point down to the react handler.

    Two things get resumed here, and they differ in exactly one respect:

    - **A Tier B continuation** (ADR-0106 M5.C5c): the loop asked a question and the
      user has answered it. The answer is loop input, so it rides on ``resume_reply``
      and is injected as the observation of the step that asked.
    - **An approved governance halt**: the evaluator stopped the run and a human
      approved continuing it. There is no answer to inject — the paused step's
      observation is a real tool result — so ``resume_reply`` stays None and the loop
      simply re-enters at N+1.

    A picked ``choice`` continuation takes the same single-task path through
    :func:`_choice_task`: no run to resume, but no plan to make either.

    The explicit request point wins: it is set by a resume the harness initiated, and
    a continuation that happens to be open for the session is not what that turn is.
    """
    intent_result = state.intent_result
    assert intent_result is not None

    point = state.request.resume_point
    if point is not None:
        run_id, step_id = point
        owner = runtime._resolve_supported_agent(
            intent_result.agent_type, fallback=intent_result.agent_type
        )
        return AgentTask(
            query=state.message,
            agent_type=owner,
            memory_context=state.memory_ctx,
            session_id=state.session_id,
            params=task_params(state),
            origin_channel=state.request.channel,
            resume_run_id=run_id,
            resume_step_id=step_id,
            resume_reply=None,
        )

    continuation = state.continuation
    if continuation is not None and state.continuation_choice is not None:
        return _choice_task(runtime, state)
    if continuation is not None and not continuation.is_resumable_run:
        # An inferred free-text question classify claimed for its owner. There is no run
        # to resume, so this is a fresh task — but it carries the question, because the
        # reply alone is not one.
        return _question_task(runtime, state) if continuation.kind == "question" else None
    if continuation is None:
        return None
    # Resolve the owner like any other target. A continuation outlives the turn that
    # opened it, so its owner may no longer be a registered agent (a plugin disabled
    # between turns) — and an unresolved agent_type falls through to a handler that
    # would ignore the resume point entirely and answer the reply on its own.
    owner = runtime._resolve_supported_agent(
        continuation.owner or intent_result.agent_type, fallback=intent_result.agent_type
    )
    return AgentTask(
        query=state.message,
        agent_type=owner,
        memory_context=state.memory_ctx,
        session_id=state.session_id,
        params=task_params(state),
        origin_channel=state.request.channel,
        resume_run_id=continuation.run_id,
        resume_step_id=continuation.step_id,
        resume_reply=state.message,
    )


def _question_task(runtime: TurnHost, state: TurnState) -> AgentTask:
    """The task that hands a free-text reply back to the owner that asked the question.

    Not a plan, for the same reason a choice is not: "Key quotes" has nothing to
    decompose, and planning it hands the planner two words whose only meaning is the
    question it cannot see.
    """
    continuation = state.continuation
    intent_result = state.intent_result
    assert continuation is not None and intent_result is not None
    owner = runtime._resolve_supported_agent(
        continuation.owner or intent_result.agent_type, fallback=intent_result.agent_type
    )
    return AgentTask(
        query=question_request(continuation, state.message),
        agent_type=owner,
        memory_context=state.memory_ctx,
        session_id=state.session_id,
        params=task_params(state),
        origin_channel=state.request.channel,
    )


def _choice_task(runtime: TurnHost, state: TurnState) -> AgentTask:
    """The single task that hands a picked option back to the owner that offered it.

    Not a plan, for the same reason a resume is not: "the 1 st one" has nothing to
    decompose, and planning it would give the planner a message whose only meaning is
    the list it cannot see. The owner resolves like any other target, so an owner
    unregistered since the list was shown degrades to the classified agent.
    """
    continuation = state.continuation
    intent_result = state.intent_result
    assert continuation is not None and intent_result is not None
    owner = runtime._resolve_supported_agent(
        continuation.owner or intent_result.agent_type, fallback=intent_result.agent_type
    )
    choice = state.continuation_choice or {}
    return AgentTask(
        query=choice_request(continuation, choice, state.message),
        agent_type=owner,
        memory_context=state.memory_ctx,
        session_id=state.session_id,
        params=task_params(state),
        origin_channel=state.request.channel,
        selected_choice=state.continuation_choice,
    )


def run(runtime: TurnHost, state: TurnState) -> Iterator[StreamEvent]:
    message, session_id = state.message, state.session_id
    intent_result = state.intent_result
    assert intent_result is not None and state.memory_ctx is not None
    resuming = _resume_task(runtime, state)
    if resuming is not None:
        yield StreamEvent(kind="activity", text="continuing where we left off")
        with stage_span(runtime, state, "iris.stage.task_planner") as span:
            # `is_multi_step=False` takes the planner's deterministic single-task
            # branch — no LLM call, and a real TaskPlan for the curate stage, which
            # needs one. The task it produces is then replaced by the resuming one,
            # which is the same shape plus the resume point.
            state.plan = runtime.task_planner.plan(
                message,
                is_multi_step=False,
                allowed_agent_types=set(runtime.agent_executor.registered_agents()),
                default_agent_type=resuming.agent_type,
            )
            state.__dict__["_waves"] = [[resuming]]
            state.__dict__["_task_ids"] = ["resume"]
            state.tasks = [resuming]
            set_span_attributes(
                span,
                {
                    "iris.resume_run_id": resuming.resume_run_id,
                    "iris.resume_step_id": resuming.resume_step_id,
                    "iris.first_agent": resuming.agent_type,
                },
            )
        log_timeline_event(
            "pipeline.phase",
            phase="planner.resume",
            payload={
                "run_id": resuming.resume_run_id,
                "step_id": resuming.resume_step_id,
                "agent_type": resuming.agent_type,
                "choice": resuming.selected_choice is not None,
            },
            session_id=session_id,
        )
        return
    yield StreamEvent(kind="activity", text="planning agent steps")
    with stage_span(runtime, state, "iris.stage.task_planner") as span:
        plan = runtime.task_planner.plan(
            message,
            is_multi_step=intent_result.is_multi_step,
            allowed_agent_types=set(runtime.agent_executor.registered_agents()),
            default_agent_type=intent_result.agent_type,
        )
        state.plan = plan
        extra = task_params(state)
        # Dependency-ordered waves (ADR-0045): the execute stage streams the first
        # task and runs the rest wave by wave. Materialise every AgentTask here so
        # the plan payload and the executor see one list.
        waves: list[list[AgentTask]] = []
        sub_ids: list[str] = []
        # ADR-0111: which earlier sub-tasks each one depends on, so the execute stage
        # can hand their results downstream.
        deps: dict[str, list[str]] = {sub.id: list(sub.depends_on) for sub in plan.tasks}
        for wave in plan.execution_groups():
            wave_tasks: list[AgentTask] = []
            for sub in wave:
                target = runtime._resolve_supported_agent(
                    sub.agent_type or intent_result.agent_type, fallback=intent_result.agent_type
                )
                wave_tasks.append(
                    AgentTask(
                        query=sub.params.get("query", plan.query or message),
                        agent_type=target,
                        memory_context=state.memory_ctx,
                        session_id=session_id,
                        params={**dict(sub.params), **extra},
                        origin_channel=state.request.channel,
                    )
                )
                sub_ids.append(sub.id)
            if wave_tasks:
                waves.append(wave_tasks)
        if not waves:
            fallback_agent = runtime._resolve_supported_agent(
                intent_result.agent_type, fallback="system"
            )
            waves = [
                [
                    AgentTask(
                        query=plan.query or message,
                        agent_type=fallback_agent,
                        memory_context=state.memory_ctx,
                        session_id=session_id,
                        params=extra,
                        origin_channel=state.request.channel,
                    )
                ]
            ]
            sub_ids = ["fallback"]
        state.__dict__["_waves"] = waves
        state.__dict__["_task_ids"] = sub_ids
        state.__dict__["_deps"] = deps
        state.tasks = [t for wave in waves for t in wave]
        first_agent = state.tasks[0].agent_type
        set_span_attributes(
            span, {"iris.plan_size": len(state.tasks), "iris.first_agent": first_agent}
        )
    payload = {"plan_size": len(state.tasks), "first_agent": first_agent, "task_ids": sub_ids}
    log_timeline_event("planner.end", phase="planner.end", payload=payload, session_id=session_id)
    yield StreamEvent(
        kind="trace",
        text=f"plan ready: {len(state.tasks)} task(s), first agent {first_agent}",
        payload={"name": "planner.end", **payload},
    )
