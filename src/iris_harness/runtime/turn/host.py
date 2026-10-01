"""What the turn pipeline requires of its host (OSS plan M5.7 slice 6).

The pipeline was extracted from ``IrisRuntime`` under decision 9, but the seam it
was extracted across was never written down: every stage took ``runtime: Any`` and
reached whatever it needed. Measured, that is **21 members, 16 of them private** —
a pipeline package outside the class depending on the class's internals, with
nothing to check the dependency and nothing to read to learn what it is.

:class:`TurnHost` is that seam, declared. It is a ``Protocol``, so:

* **mypy is the enforcement, not a convention.** A stage reaching a 22nd member
  fails the type check, and ``run_turn(self, ...)`` in ``bootstrap`` is a
  structural check that ``IrisRuntime`` still supplies all 21.
* **the carve can be verified.** Splitting ``IrisRuntime`` into the facade the plan
  describes means moving these members around; the Protocol is what says whether the
  thing the pipeline is handed afterwards is still a valid host.
* **the debt is legible.** Sixteen underscored names in a published interface is a
  finding, not a style slip. They are declared exactly as they are today rather than
  renamed public, because renaming them is the carve's job and doing it here would
  mix a 16-site rename into the change that makes the rename checkable. Track C is
  retiring them as it carves: slice 12 replaced ``_evaluate_prior_turn_outcome`` with
  the public ``capture`` collaborator, and slice 13 replaced ``_mission_autocreate_enabled``
  and ``_propose_mission`` with ``mission_proposals``, and slice 15 replaced
  ``_pre_intercept_activity_hint`` and ``_dispatch_intercepts`` with ``intercepts``, and
  slice 16 replaced ``_format_recent_context`` and ``_build_memory_context`` with
  ``sessions`` — 18 members, nine private.

Two nested protocols keep ``Any`` out of the parts the stages actually use:
:class:`RouterAudit` (``_router_audit()`` is annotated ``Any`` on the runtime, but
``route`` only ever calls ``.record``) and the ``tracer``, which stays ``Any``
because it is an optional OpenTelemetry object the harness treats duck-typed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from iris_harness.agent.agent_executor import AgentExecutor, AgentResult
    from iris_harness.agent.intent_router import IIntentClassifier, IntentResult
    from iris_harness.agent.response_curator import ResponseCurator
    from iris_harness.agent.task_planner import TaskPlan, TaskPlanner
    from iris_harness.kernel.governance import GovernanceKernel
    from iris_harness.llm.tier_router import TierRouter
    from iris_harness.memory.retriever import MemoryContext
    from iris_harness.runtime.continuations import ContinuationRegistry
    from iris_harness.runtime.intercept_dispatch import InterceptDispatch
    from iris_harness.runtime.mission_proposals import MissionProposals
    from iris_harness.runtime.replies import DeterministicReplies
    from iris_harness.runtime.session_memory import SessionMemory
    from iris_harness.runtime.turn_capture import TurnCapture
    from iris_harness.runtime.types import ChatResult


class RouterAudit(Protocol):
    """The audit ledger the ``route`` stage appends the classifier decision to."""

    def record(
        self,
        *,
        session_id: str,
        message: str,
        result: IntentResult,
        router_model: str | None = None,
        channel: str | None = None,
    ) -> None: ...


class TurnHost(Protocol):
    """The 21 members a turn's stages reach on the runtime they are handed.

    Grouped by the stage that needs them. Adding one here is a deliberate widening
    of the pipeline's dependency on its host, which is the point of it being a
    declaration: the diff shows it.
    """

    # -- collaborators the stages use directly -------------------------------------
    agent_executor: AgentExecutor
    continuations: ContinuationRegistry
    task_planner: TaskPlanner
    tier_router: TierRouter
    # The OpenTelemetry tracer, or None. Duck-typed everywhere in the harness
    # (``maybe_current_span`` accepts anything, including None), so widening this
    # to a protocol would claim more than the harness actually requires.
    tracer: Any

    # -- screen --------------------------------------------------------------------
    # The governance kernel the PRE_TURN screens fire on, or None when the operator
    # disabled governance (``IRIS_GOVERNANCE_ENABLED=0``).
    governance_kernel: GovernanceKernel | None

    def _span_input(self, span: Any, message: str) -> None: ...

    # -- intercept -----------------------------------------------------------------
    # Per-turn learning capture; the stage attributes the prior turn's outcome through it.
    capture: TurnCapture

    # Deterministic replies; the stage answers IRIS's own "should I remember this?"
    # question with one, rather than routing a bare "yes" to an agent.
    replies: DeterministicReplies

    # Intercept dispatch; the stage asks it for the activity hint and runs the chain.
    intercepts: InterceptDispatch

    # -- guard ---------------------------------------------------------------------
    # The curator; ``guard`` runs its model-free response check on a deterministic answer.
    response_curator: ResponseCurator

    # -- classify / resolve / route ------------------------------------------------
    # Session memory; classify reads the routing transcript, route the memory context.
    sessions: SessionMemory

    def _classifier_for(self, router_model: str | None) -> IIntentClassifier: ...

    def _resolve_continuation_intent(
        self, message: str, intent: IntentResult, *, session_id: str
    ) -> IntentResult: ...

    def _resolve_skill_intent(self, message: str, intent: IntentResult) -> IntentResult: ...

    def _resolve_action_intent(self, message: str, intent: IntentResult) -> IntentResult: ...

    # -- route ---------------------------------------------------------------------
    def _router_audit(self) -> RouterAudit: ...

    def _record_downstream_reuse(self, memory_ctx: MemoryContext) -> None: ...

    # -- plan ----------------------------------------------------------------------
    def _resolve_supported_agent(self, candidate: str, *, fallback: str) -> str: ...

    # -- curate --------------------------------------------------------------------
    def _finalize_chat(
        self,
        *,
        message: str,
        session_id: str,
        intent_result: IntentResult,
        plan: TaskPlan,
        results: list[AgentResult],
        preferred_model: str | None,
        provider_profile: str | None,
        router_model: str | None,
        strict: bool,
        span: Any = None,
        memory_ctx: MemoryContext | None = None,
        allow_escalation: bool = False,
    ) -> ChatResult: ...

    # -- record --------------------------------------------------------------------
    # Mission proposals; the stage proposes a multi-step turn through it.
    mission_proposals: MissionProposals


__all__ = ["RouterAudit", "TurnHost"]
