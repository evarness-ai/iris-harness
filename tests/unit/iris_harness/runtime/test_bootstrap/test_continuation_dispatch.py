"""ADR-0106 M5.C3 — the shield and the steer.

Two mechanisms, tested against the real intercept chain:

- **Shield** (``_dispatch_intercepts``): a confirmation-resolving intercept does not
  get to answer a "yes" when the session's open question belongs to someone else.
- **Steer** (``classify``): the answer routes back to whoever asked, rather than
  being classified in isolation, where a bare "yes" lands wherever keywords fall.

Continuations are opened explicitly here. Agents and plugins opening their own is
M5.C4; this slice is the mechanism they will use.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.intent_router import IntentResult
from iris_harness.memory.retriever import MemoryContext
from iris_harness.memory.state.continuations import ContinuationStore
from iris_harness.runtime.continuations import ContinuationRegistry
from iris_harness.runtime.intercepts import InterceptSpec
from iris_harness.runtime.turn.stages import classify, plan
from iris_harness.runtime.turn.state import TurnRequest, TurnState


@pytest.fixture()
def registry(tmp_path: Path) -> ContinuationRegistry:
    return ContinuationRegistry(store=ContinuationStore(db_path=tmp_path / "checkpoints.db"))


class _FakeRuntime:
    """Just enough runtime for the two units under test."""

    def __init__(self, registry: ContinuationRegistry, chain: list[tuple[InterceptSpec, Any]]):
        self.continuations = registry
        self.chain = chain
        self.calls: list[str] = []


def _spec(name: str, *, resolves: bool) -> InterceptSpec:
    return InterceptSpec(name, f"_handle_{name}_turn", resolves_confirmation=resolves)


# ── the shield ────────────────────────────────────────────────────────────────


def _dispatch(runtime: _FakeRuntime, message: str, session_id: str) -> Any:
    """The real dispatch over the fake's chain (OSS plan M5.7 track C slice 15 moved it
    to ``InterceptDispatch``; the shield is its ``dispatch``)."""
    from iris_harness.runtime.intercept_dispatch import InterceptDispatch

    dispatch = InterceptDispatch(runtime)
    dispatch.effective_chain = lambda: runtime.chain  # the chain under test, not the YAML
    dispatch._log_chat_result = lambda *_a, **_k: None  # noise
    return dispatch.dispatch(message, session_id=session_id, channel="console", span=None)


def test_confirmation_intercept_is_shielded_when_another_owner_is_waiting(
    registry: ContinuationRegistry,
) -> None:
    """The incident, generalised. The planner is waiting on an answer; the organize
    approval intercept must not get to claim the user's "yes"."""

    def organize_handler(_message: str, **_kw: Any) -> Any:
        runtime.calls.append("organize_confirmation")
        return object()  # would have claimed the turn

    runtime = _FakeRuntime(
        registry, [(_spec("organize_confirmation", resolves=True), organize_handler)]
    )
    registry.ask("s1", "planner", question="proceed with this plan?", intent="planner")

    assert _dispatch(runtime, "yes go head with the plan", "s1") is None
    assert runtime.calls == []  # never even consulted


def test_the_owning_intercept_still_runs(registry: ContinuationRegistry) -> None:
    """The shield is about *other* owners. An intercept answering its own question
    is exactly what should happen."""
    sentinel = object()

    def organize_handler(_message: str, **_kw: Any) -> Any:
        runtime.calls.append("organize_confirmation")
        return sentinel

    runtime = _FakeRuntime(
        registry, [(_spec("organize_confirmation", resolves=True), organize_handler)]
    )
    registry.ask("s1", "organize_confirmation", question="approve the organize plan?")

    hit = _dispatch(runtime, "yes", "s1")
    assert hit is not None and hit.result is sentinel
    assert runtime.calls == ["organize_confirmation"]


def test_non_confirmation_intercepts_are_never_shielded(
    registry: ContinuationRegistry,
) -> None:
    """A pending question must not stop the user asking something unrelated."""
    sentinel = object()

    def folder_handler(_message: str, **_kw: Any) -> Any:
        runtime.calls.append("folder_files")
        return sentinel

    runtime = _FakeRuntime(registry, [(_spec("folder_files", resolves=False), folder_handler)])
    registry.ask("s1", "planner", question="proceed?")

    hit = _dispatch(runtime, "how many files in my downloads folder", "s1")
    assert hit is not None and hit.result is sentinel


def test_no_pending_continuation_leaves_the_chain_untouched(
    registry: ContinuationRegistry,
) -> None:
    sentinel = object()

    def organize_handler(_message: str, **_kw: Any) -> Any:
        return sentinel

    runtime = _FakeRuntime(
        registry, [(_spec("organize_confirmation", resolves=True), organize_handler)]
    )

    hit = _dispatch(runtime, "approve the organize plan", "s1")
    assert hit is not None and hit.result is sentinel


def test_another_sessions_pending_question_shields_nothing(
    registry: ContinuationRegistry,
) -> None:
    """Session scoping cuts both ways: another conversation's open question must
    neither claim this turn nor block it."""
    sentinel = object()

    def organize_handler(_message: str, **_kw: Any) -> Any:
        return sentinel

    runtime = _FakeRuntime(
        registry, [(_spec("organize_confirmation", resolves=True), organize_handler)]
    )
    registry.ask("other", "planner", question="proceed?")

    hit = _dispatch(runtime, "approve the organize plan", "mine")
    assert hit is not None and hit.result is sentinel


# ── the steer ─────────────────────────────────────────────────────────────────


class _SteerRuntime:
    def __init__(self, registry: ContinuationRegistry) -> None:
        self.continuations = registry


def _state(message: str, session_id: str = "s1") -> TurnState:
    state = TurnState(request=TurnRequest(message=message, session_id=session_id))
    state.intent_result = IntentResult(
        intent="general", agent_type="general", confidence=0.4, source="fallback"
    )
    return state


def test_answer_routes_back_to_the_owner(registry: ContinuationRegistry) -> None:
    registry.ask("s1", "planner", question="proceed with this plan?", intent="planner")
    state = _state("yes go head with the plan")

    classify._honour_continuation(_SteerRuntime(registry), state)

    assert state.intent_result is not None
    assert state.intent_result.intent == "planner"  # not "general"
    assert state.intent_result.source == "continuation"
    assert state.continuation_decision == "approve"
    # Answered, so the next "yes" is not claimed by a question already settled.
    assert registry.pending("s1") is None


def test_a_non_answer_leaves_the_question_open_and_the_intent_alone(
    registry: ContinuationRegistry,
) -> None:
    registry.ask("s1", "planner", question="proceed?", intent="planner")
    state = _state("what's my email digest?")
    before = replace(state.intent_result)  # type: ignore[arg-type]

    classify._honour_continuation(_SteerRuntime(registry), state)

    assert state.intent_result == before
    assert state.continuation is None
    assert registry.pending("s1") is not None  # still waiting


def test_rejection_also_routes_back(registry: ContinuationRegistry) -> None:
    registry.ask("s1", "planner", question="proceed?", intent="planner")
    state = _state("no, cancel that")

    classify._honour_continuation(_SteerRuntime(registry), state)

    assert state.continuation_decision == "reject"
    assert state.intent_result is not None
    assert state.intent_result.intent == "planner"


def test_nothing_pending_is_a_no_op(registry: ContinuationRegistry) -> None:
    state = _state("yes")
    before = replace(state.intent_result)  # type: ignore[arg-type]

    classify._honour_continuation(_SteerRuntime(registry), state)

    assert state.intent_result == before
    assert state.continuation is None


# ── Tier B: the steer claims the turn outright (M5.C5c) ───────────────────────
#
# Tier A reads a yes/no, so a message that is not one leaves the question open. Tier
# B's question is free text, which `reads_as_answer` deliberately refuses to
# classify — so the rule is the one the one-pending-per-session invariant already
# implies: the paused run asked the most recent question, so it owns the next message.


def _tier_b(registry: ContinuationRegistry, *, session_id: str = "s1") -> None:
    registry.ask(
        session_id,
        "system",
        question="Neo4j or Stardog?",
        kind="question",
        intent="system",
        run_id="run-1",
        step_id=0,
    )


def test_a_free_text_answer_claims_the_paused_run(registry: ContinuationRegistry) -> None:
    _tier_b(registry)
    state = _state("use Stardog")

    classify._honour_continuation(_SteerRuntime(registry), state)

    assert state.continuation is not None
    assert state.continuation.is_resumable_run
    assert state.continuation.run_id == "run-1"
    assert state.continuation.step_id == 0
    # No yes/no to find — Tier B carries no decision, only an owner and a run.
    assert state.continuation_decision is None
    assert state.intent_result is not None
    assert state.intent_result.source == "continuation"
    assert registry.pending("s1") is None  # answered


def test_a_change_of_subject_also_claims_it(registry: ContinuationRegistry) -> None:
    """Deliberate, and the reason the injected observation carries an escape clause:
    the harness stays deterministic about who owns the turn, and the model decides
    whether the reply answered the question."""
    _tier_b(registry)
    state = _state("actually, what's my calendar today?")

    classify._honour_continuation(_SteerRuntime(registry), state)

    assert state.continuation is not None
    assert state.continuation.is_resumable_run


def test_a_tier_a_non_answer_is_still_left_alone(registry: ContinuationRegistry) -> None:
    """The unconditional claim is Tier B's only. A re-prompt continuation with no run
    behind it keeps the shipped behaviour — a non-answer leaves the question open."""
    registry.ask("s1", "planner", question="proceed?", intent="planner")
    state = _state("what's my email digest?")

    classify._honour_continuation(_SteerRuntime(registry), state)

    assert state.continuation is None
    assert registry.pending("s1") is not None


def test_a_question_continuation_with_no_run_is_claimed_as_a_fresh_task(
    registry: ContinuationRegistry,
) -> None:
    """An inferred free-text question claims its turn (ADR-0106 M5.C3).

    This used to assert the opposite, for a reason that has since been fixed: a question
    with no checkpoint had nothing to resume into, so claiming it swallowed the turn and
    delivered nothing. `plan._question_task` is that missing half — the claim now yields
    a fresh task carrying the question, so the turn is delivered, not swallowed.

    Leaving it unclaimed is what let the 2026-09-16 news thread route a reply to the
    inbox: the owner had asked the most recent question and nothing said so.
    """
    registry.ask("s1", "system", question="which one?", kind="question", intent="system")
    state = _state("the second one")

    classify._honour_continuation(_SteerRuntime(registry), state)

    assert state.continuation is not None
    assert state.intent_result is not None
    assert state.intent_result.agent_type == "system"  # the owner that asked
    assert state.intent_result.source == "continuation"
    assert registry.pending("s1") is None  # answered; it claims one turn, not two

    # And the turn is delivered rather than swallowed — the reason the old rule existed.
    state.memory_ctx = MemoryContext()
    task = plan._resume_task(_PlanOwnerRuntime(registry), state)  # type: ignore[arg-type]
    assert task is not None
    assert "You asked: which one?" in task.query
    assert task.query.endswith("The user replied: the second one")


class _PlanOwnerRuntime(_SteerRuntime):
    def _resolve_supported_agent(self, candidate: str, *, fallback: str) -> str:
        return candidate or fallback


# ── Tier B: the plan stage continues the run instead of planning the reply ────


class _PlanRuntime:
    """Enough runtime for `plan.run`'s resume branch: a planner and an executor."""

    def __init__(self) -> None:
        self.planned: list[str] = []

        class _Planner:
            def plan(_self, query: str, **kwargs: Any) -> Any:
                self.planned.append(query)
                from iris_harness.agent.task_planner import TaskPlanner

                return TaskPlanner(llm_call=None).plan(
                    query, is_multi_step=bool(kwargs.get("is_multi_step"))
                )

        class _Executor:
            def registered_agents(_self) -> list[str]:
                return ["system", "general"]

        self.task_planner = _Planner()
        self.agent_executor = _Executor()

    def _resolve_supported_agent(self, agent_type: str, *, fallback: str = "system") -> str:
        return agent_type or fallback


def _plan_state(message: str, continuation: Any = None) -> TurnState:
    from iris_harness.memory.retriever import MemoryContext

    state = _state(message)
    state.memory_ctx = MemoryContext()
    state.continuation = continuation
    return state


def test_the_plan_stage_emits_one_resuming_task(registry: ContinuationRegistry) -> None:
    from iris_harness.runtime.turn.stages import plan

    _tier_b(registry)
    continuation = registry.pending("s1")
    state = _plan_state("use Stardog", continuation)
    runtime = _PlanRuntime()

    list(plan.run(runtime, state))

    assert len(state.tasks) == 1
    task = state.tasks[0]
    assert task.resume_run_id == "run-1"
    assert task.resume_step_id == 0
    assert task.agent_type == "system"  # the owner's
    assert task.query == "use Stardog"  # the reply; the seed supplies the task
    # curate asserts on this, so a resume must still leave a plan behind.
    assert state.plan is not None


def test_the_planner_never_decomposes_a_reply(registry: ContinuationRegistry) -> None:
    """The work was decomposed by the run that paused. Planning "use Stardog" again —
    and on a multi-step reply fanning out a wave beside the paused run — would strand
    the checkpoint and answer a question nobody asked."""
    from iris_harness.runtime.turn.stages import plan

    _tier_b(registry)
    state = _plan_state("use Stardog", registry.pending("s1"))
    state.intent_result = replace(state.intent_result, is_multi_step=True)  # type: ignore[arg-type]
    runtime = _PlanRuntime()

    list(plan.run(runtime, state))

    assert len(state.tasks) == 1  # not a wave


def test_an_ordinary_turn_still_plans(registry: ContinuationRegistry) -> None:
    from iris_harness.runtime.turn.stages import plan

    state = _plan_state("what's my calendar?", None)
    runtime = _PlanRuntime()

    list(plan.run(runtime, state))

    assert runtime.planned == ["what's my calendar?"]
    assert state.tasks and state.tasks[0].resume_run_id is None


def test_a_tier_a_continuation_does_not_trigger_a_resume(
    registry: ContinuationRegistry,
) -> None:
    from iris_harness.runtime.turn.stages import plan

    registry.ask("s1", "planner", question="proceed?", intent="planner")
    state = _plan_state("yes", registry.pending("s1"))
    runtime = _PlanRuntime()

    list(plan.run(runtime, state))

    assert state.tasks and state.tasks[0].resume_run_id is None


# ── the plan stage serves both kinds of resume ────────────────────────────────


def test_an_approved_halt_resumes_without_injecting_a_reply(
    registry: ContinuationRegistry,
) -> None:
    """The two resumes differ in exactly one respect. A Tier B turn injects the user's
    answer as the paused step's observation; an approved *governance* halt must not,
    because there that observation is a real tool result and overwriting it would
    destroy the work the resume exists to continue."""
    from iris_harness.runtime.turn.stages import plan
    from iris_harness.runtime.turn.state import TurnRequest

    state = _plan_state("pick a graph database", None)
    state.request = TurnRequest(
        message="pick a graph database",
        session_id="s1",
        channel="web",
        resume_run_id="run-1",
        resume_step_id=1,
    )
    runtime = _PlanRuntime()

    list(plan.run(runtime, state))

    assert len(state.tasks) == 1
    task = state.tasks[0]
    assert task.resume_run_id == "run-1"
    assert task.resume_step_id == 1
    assert task.resume_reply is None  # nothing to inject
    assert task.origin_channel == "web"  # a second halt goes back to the browser
    # The planner is still consulted, but only for its deterministic single-task branch:
    # `curate` asserts on a TaskPlan, so a resume has to leave one behind. What it must
    # not do is *decompose*, and the single task above is that assertion.
    assert state.plan is not None
    assert len(state.plan.tasks) == 1


def test_the_request_point_wins_over_a_pending_continuation(
    registry: ContinuationRegistry,
) -> None:
    """A continuation that happens to be open for the session is not what this turn is."""
    from iris_harness.runtime.turn.stages import plan
    from iris_harness.runtime.turn.state import TurnRequest

    _tier_b(registry)
    state = _plan_state("pick a graph database", registry.pending("s1"))
    state.request = TurnRequest(
        message="pick a graph database",
        session_id="s1",
        resume_run_id="approved-run",
        resume_step_id=4,
    )
    runtime = _PlanRuntime()

    list(plan.run(runtime, state))

    task = state.tasks[0]
    assert task.resume_run_id == "approved-run"
    assert task.resume_step_id == 4
    assert task.resume_reply is None


def test_a_tier_b_resume_still_injects_the_reply(registry: ContinuationRegistry) -> None:
    from iris_harness.runtime.turn.stages import plan

    _tier_b(registry)
    state = _plan_state("use Stardog", registry.pending("s1"))
    runtime = _PlanRuntime()

    list(plan.run(runtime, state))

    assert state.tasks[0].resume_reply == "use Stardog"
