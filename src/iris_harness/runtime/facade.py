"""``IrisRuntime`` -- the wired harness the API and CLI consume.

Release gate 1 says `bootstrap.py` is the composition root **only**. This class is
not composition: it is the object composition produces, and at 1,116 lines it was
by far the largest thing in that file. Moving it out is the gate's own definition
applied literally.

``build_runtime`` stays in `bootstrap.py` and still returns this; every caller
that did `from iris_harness.runtime import IrisRuntime` is unaffected, because
the package `__init__` is where that name has always come from.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iris_harness.agent.agent_executor import (
    AgentExecutor,
    AgentResult,
    AgentTask,
)
from iris_harness.agent.intent_router import (
    IIntentClassifier,
    IntentResult,
    IntentRouter,
    KeywordClassifier,
)
from iris_harness.agent.response_curator import (
    CuratedResponse,
    ResponseCurator,
)
from iris_harness.agent.task_planner import TaskPlan, TaskPlanner
from iris_harness.foundation.eventbus import EventBus
from iris_harness.foundation.observability.session_log import (
    current_turn_id,
    log_timeline_event,
    pin_context,
    session_scope,
    turn_scope,
)
from iris_harness.foundation.observability.tracer import (
    maybe_current_span,
    maybe_span,
)
from iris_harness.kernel.governance import GovernanceKernel
from iris_harness.kernel.governance.hooks.response_payload import Audience
from iris_harness.llm.errors import friendly_llm_error as _friendly_llm_error
from iris_harness.llm.tier_router import TierRouter
from iris_harness.memory.compactor import ConversationCompactor
from iris_harness.memory.knowledge.models import WikiIngestEvent
from iris_harness.memory.knowledge.wiki_engine import WikiEngine
from iris_harness.memory.retention import (
    RetentionService,
    retention_config,
)
from iris_harness.memory.retention import housekeeping_due as _housekeeping_due
from iris_harness.memory.retriever import MemoryContext, MemoryRetriever
from iris_harness.memory.semantic_index import SemanticIndex
from iris_harness.memory.store import MemoryStore
from iris_harness.memory.triage import MemoryDestination, triage_memory_item
from iris_harness.runtime.activity_notices import ActivityNotices
from iris_harness.runtime.client_config import (
    _llm_router_from_config,
    config_from_profile,
    effective_provider_profile,
    infer_provider,
)
from iris_harness.runtime.confirmations import Confirmations
from iris_harness.runtime.continuations import ContinuationRegistry
from iris_harness.runtime.escalation_actions import EscalationActions
from iris_harness.runtime.handlers.skill_brief import _build_skill_brief_handler
from iris_harness.runtime.handlers.ticks import (
    build_notification_reminder_tick_handler,
    build_pressure_tick_handler,
    build_routine_tick_handler,
    execute_routine,
)
from iris_harness.runtime.intercept_dispatch import InterceptDispatch
from iris_harness.runtime.intercepts import (
    InterceptSpec,
    load_intercept_chain,
    load_openers,
)
from iris_harness.runtime.learning_controls import LearningControls
from iris_harness.runtime.mission_proposals import MissionProposals
from iris_harness.runtime.plugin_host.registry import PluginRegistry
from iris_harness.runtime.replies import DeterministicReplies
from iris_harness.runtime.routine_authoring import (
    RoutineAuthoring,
)
from iris_harness.runtime.self_learning_loop import SelfLearningLoop
from iris_harness.runtime.session_memory import SessionMemory
from iris_harness.runtime.skill_matching import (
    best_matching_skill_package,
)
from iris_harness.runtime.tool_service import ToolService
from iris_harness.runtime.turn import TurnRequest, drain, run_turn
from iris_harness.runtime.turn_capture import (
    TurnCapture,
)
from iris_harness.runtime.types import ChatResult, StreamEvent, WarmupResult
from iris_harness.runtime.welcome import FirstChatWelcome
from iris_harness.services.channels import (
    ChannelGateway,
)
from iris_harness.services.channels.connectors.telegram_poller import TelegramPoller
from iris_harness.services.heartbeat import (
    HeartbeatScheduler,
    load_heartbeats,
    load_plugin_owners,
)
from iris_harness.services.heartbeat.models import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatStatus,
)
from iris_harness.services.heartbeat.scheduler import HeartbeatHandler
from iris_harness.services.learning.crystallizer import SkillCrystallizer
from iris_harness.services.learning.engine import LearningEngine
from iris_harness.services.learning.signals import LearningSignalCollector
from iris_harness.services.learning.store import LearningMetricsStore
from iris_harness.services.missions import MissionEngine
from iris_harness.services.routines import (
    RoutineExecutionRecord,
    RoutineStore,
)
from iris_harness.tools.skills.hot_reload import SkillHotReloader
from iris_harness.tools.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

_CLARIFY_CONFIDENCE_FLOOR = 0.35


class _WikiLintHandler:
    """Placeholder lint handler — Phase H replaces with real wiki_lint module."""

    def __call__(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SKIPPED,
            output="wiki_lint not yet implemented (Phase H)",
        )


# Helpers used only by IrisRuntime; they moved out of bootstrap.py with it, because
# a helper that only the facade calls is not composition either.


def _turn_origin(metadata: dict[str, object]) -> str | None:
    """The stored ``turn_origin`` of an answer: what the loop recorded (``"external"`` when its
    run read third-party text, ``"internal"`` when it did not), else None (unknown: the
    producer did not say). Only those two exact values count, so a stray metadata key cannot
    mark a turn."""
    value = metadata.get("turn_origin")
    return value if value in ("external", "internal") else None


def _build_router_classifier_for_model(model: str) -> IIntentClassifier:
    """Build a router classifier bound to an explicit model name.

    Used to honor the ``/router <model>`` REPL command. The provider is
    auto-inferred from the model name via ``infer_provider``.
    """
    try:
        from iris_harness.llm.client import CodingLLMConfig

        provider, base_url, api_key_env = infer_provider(model)
        cfg = CodingLLMConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=0.0,
            max_tokens=256,
            timeout_seconds=30,
            # Deliberately not "router": that key maps to governance tier_1, and a
            # /router override can point at a cloud model.
            tier_name="router_override",
        )
        return _llm_router_from_config(cfg)
    except Exception:
        logger.exception(
            "router classifier build failed for model=%r; using KeywordClassifier", model
        )
        return KeywordClassifier()


_CLARIFY_NUDGE = "If that's not quite what you were after, tell me a bit more and I'll refine."


def _clarify_warranted(
    *,
    intent_confidence: float | None,
    judge_bundle: Any,
    has_errors: bool,
    already_clarified: bool,
    agent_type: str | None,
) -> bool:
    """Decide whether to append a clarifying nudge to an answer (conservative; ADR-0072).

    Never on an errored / already-clarified turn or the clarify agent itself. Fires on
    a STRONG signal — the per-turn judge flagged the answer (a ``warn``/``retry``
    verdict in the bundle) — or a WEAK one — the router was quite unsure what the user
    meant (confidence below a low floor). Kept narrow so confident, clean answers are
    never nagged (the over-asking failure mode the ADR warns against).
    """
    if has_errors or already_clarified or agent_type == "clarify":
        return False
    if isinstance(judge_bundle, dict):
        for signal in judge_bundle.get("signals", []):
            if isinstance(signal, dict) and signal.get("verdict") in {"warn", "retry"}:
                return True
    if intent_confidence is not None and intent_confidence < _CLARIFY_CONFIDENCE_FLOOR:
        return True
    return False


_wiki_lint_handler: HeartbeatHandler = _WikiLintHandler()


@dataclass
class IrisRuntime:
    """The wired IRIS harness — single object the API + CLI consume."""

    intent_router: IntentRouter
    task_planner: TaskPlanner
    agent_executor: AgentExecutor
    response_curator: ResponseCurator
    memory_store: MemoryStore
    memory_retriever: MemoryRetriever
    semantic_index: SemanticIndex | None
    compactor: ConversationCompactor
    retention: RetentionService
    wiki: WikiEngine
    mission_engine: MissionEngine
    routine_store: RoutineStore
    heartbeats: HeartbeatScheduler
    channels: ChannelGateway
    learning_store: LearningMetricsStore
    learning_engine: LearningEngine
    signal_collector: LearningSignalCollector
    skill_registry: SkillRegistry
    skill_crystallizer: SkillCrystallizer
    skill_hot_reloader: SkillHotReloader
    config_dir: Path
    data_dir: Path
    default_channel: str = "console"
    tier_router: TierRouter = field(default_factory=TierRouter)
    poller: TelegramPoller | None = None
    tracer: Any = None  # opentelemetry.trace.Tracer — set by API layer after build
    # The kernel the turn pipeline's PRE_TURN screen fires on (``screen`` stage).
    # build_runtime sets it from ``kernel_from_env()``; None means the operator
    # disabled governance, and the screen stage then does nothing.
    governance_kernel: GovernanceKernel | None = None
    # The deterministic intercept dispatch chain, declared in config/intercepts.yaml
    # (Phase 2). chat() and chat_stream() both dispatch through this single ordered
    # list via InterceptDispatch.dispatch, so the two paths can never drift apart.
    intercept_chain: tuple[InterceptSpec, ...] = field(default_factory=load_intercept_chain)
    # The openers, by name (ADR-0127): deterministic handlers for a turn the system opens
    # with no user message (``open_turn``), declared under ``openers:`` in the same file.
    openers: dict[str, InterceptSpec] = field(default_factory=load_openers)
    # OSS plan M1: what the profile's plugins registered (intercepts, tools,
    # confirmation executors are read from here at dispatch time) + their health.
    plugin_registry: PluginRegistry = field(default_factory=PluginRegistry)
    profile: Any = (
        None  # iris_harness.runtime.plugin_host.EffectiveProfile once build_runtime mounts plugins
    )
    # ADR-0106: the session-owned continuation registry — who this conversation owes an
    # answer to, and (ADR-0108 follow-up) what runs if the answer is yes. In-chat
    # confirmation for consequential (R3) actions lives here now: it used to be
    # `_pending_confirmations`, an in-memory dict keyed by session, written by exactly
    # one feature and read before the continuation shield. Folding it in made it durable
    # and gave it an owner, so the other 21 confirmation-resolving intercepts can no
    # longer take an "approve" that was meant for it. Channel-agnostic — any channel
    # resolves it with a typed "approve"/"reject" (web/telegram may render buttons that
    # send that text). See `Confirmations.handle_confirmation_turn`.
    continuations: ContinuationRegistry = field(default_factory=ContinuationRegistry)
    # The four FileManager per-session proposal dicts (cleanup, organize, categorize,
    # in-flight move) moved to FileManagerHandlers with the intercepts that owned
    # them (OSS plan M2.4) — nothing else in the runtime ever read them.
    # The escalation action loop (ADR-0068 L3), built on first use. It owns the
    # egress classifier cache that used to live here (OSS plan M5.7 track C).
    _escalation_cache: EscalationActions | None = None
    # Context-health telemetry (ADR-0081): the last in-loop budget snapshot (transcript
    # eviction + prompt size) written by the AgenticCore budget observer; session memory's
    # `context_health` reads it. In-memory; transient by nature.
    _last_react_budget: dict[str, int] = field(default_factory=dict)
    _router_cache: dict[str, IIntentClassifier] = field(default_factory=dict)
    # Append-only audit log of intent classifier decisions (one row per
    # chat turn that reaches the classifier). Populated lazily so
    # construction stays cheap; see _router_audit() below.
    _router_audit_cache: Any = None
    # Async Activity spine (P1) + the notify leg: the collaborator that lazily builds the
    # background runner for long system jobs (FileManager categorize/cleanup/organize)
    # and delivers completion and approval-lapse notices. One per runtime — it owns the
    # runner and its bus subscriptions (OSS plan M5.7 track C). See
    # docs/architecture/async-activities-and-notifications.md.
    _activity_notices_cache: ActivityNotices | None = None
    # The learning capabilities (behavior miner, intention rollup, learning analyst,
    # self-management) — their live state, the experiment-console controls and the
    # heartbeat runs. Set by build_runtime right after construction, from the env gates
    # (OSS plan M5.7 track C).
    learning: LearningControls = field(init=False, repr=False)
    # The in-chat routine conversation (authoring, management, refinement, swaps,
    # clarifications) and its per-session state. Core by the owner's ruling; set by
    # build_runtime right after construction (OSS plan M5.7 track C).
    routines: RoutineAuthoring = field(init=False, repr=False)
    # The self-learning loop's periodic jobs (escalation priors, crystallization, sandbox
    # pre-flight, routine reflection, experiment re-measure). Stateless; set by
    # build_runtime right after construction (OSS plan M5.7 track C).
    learning_loop: SelfLearningLoop = field(init=False, repr=False)
    # Per-turn learning capture: user facts, the chat learning signal, turn metrics and
    # the prior turn's measured outcome (§4.2), with the state only it reads. Set by
    # build_runtime right after construction (OSS plan M5.7 track C).
    capture: TurnCapture = field(init=False, repr=False)
    # Mission proposals (propose-not-act, HITL): the record stage's multi-step proposal and
    # the mission_proposal_tick heartbeat. Stateless; set by build_runtime right after
    # construction (OSS plan M5.7 track C).
    mission_proposals: MissionProposals = field(init=False, repr=False)
    # Deterministic replies: the templated turn with no model in the loop, handed to
    # plugins as HarnessServices.deterministic_reply. Stateless; set by build_runtime
    # right after construction (OSS plan M5.7 track C).
    replies: DeterministicReplies = field(init=False, repr=False)
    # The in-chat confirmation turn (approve/reject, dispatched to plugin-registered
    # executors by kind) and the approvals timeout sweep, with the approval queue and
    # router only they build. Set by build_runtime right after construction (OSS plan
    # M5.7 track C).
    confirmations: Confirmations = field(init=False, repr=False)
    # Intercept dispatch: the declared + plugin chain, resolved and run first on every
    # turn through the pipeline's intercept stage. Stateless; set by build_runtime right
    # after construction (OSS plan M5.7 track C).
    intercepts: InterceptDispatch = field(init=False, repr=False)
    # The first-chat welcome (ADR-0127): whether it is due, once per IRIS_HOME, and the
    # opener that answers it. Set by build_runtime right after construction.
    welcome: FirstChatWelcome = field(init=False, repr=False)
    # Session memory: the per-session conversation window, its reload, compaction and
    # context health, with the state only it writes. Set by build_runtime right after
    # construction (OSS plan M5.7 track C).
    sessions: SessionMemory = field(init=False, repr=False)
    # The governed ReAct loop, set by build_runtime when the loop is on/shadow, and
    # handed to plugins as HarnessServices.react_handler so one can mount a persona
    # (an agent type answering on this loop) instead of supplying a handler.
    react_handler: Any = None
    react_stream_handler: Any = None
    # The harness's lesson store, handed to plugins as HarnessServices.lessons: an
    # agent plugin primes itself from prior attempts and records its outcome there
    # (OSS plan M4.6). None when lesson capture is off.
    lesson_capture: Any = None
    # Intents the governed loop answers itself. A plugin may also register a
    # deterministic handler for one of them — its domain left the core carrying it
    # (OSS plan M4.2: the finance digest). Plugins mount AFTER the core's own
    # registrations, so without this the plugin lane would silently displace the
    # loop. `_reassert_loop_intents` puts the loop back on top and keeps the
    # plugin's handler as the loop's degrade fallback, which is what it always was.
    # It also adds the intents plugins claimed with `api.register_loop_intent`.
    _loop_intents: frozenset[str] = frozenset()
    _react_fallbacks: dict[str, Any] = field(default_factory=dict)
    # This runtime's private event bus. Built EAGERLY (not with the runner) because
    # plugins subscribe during `_mount_plugins`, which runs at build time, long
    # before the first Activity is submitted — a bus created later would be a
    # different object and their handlers would never fire. Private per runtime so
    # completion subscribers don't cross-talk between runtimes in tests.
    _event_bus: EventBus = field(default_factory=EventBus)
    # Registered tools for code (``services.tools``), and the executor that runs a code
    # caller's call once the owner approves it (plugin-capabilities decision 1). Set when
    # plugins mount; every surface that answers an approval hands it to the service.
    tool_service: ToolService | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Resume in-flight missions and start the heartbeat scheduler."""
        try:
            self._reconcile_orphaned_activities()
        except Exception:  # startup must not crash the app
            logger.exception("activity reconciliation failed")
        try:
            resumed = self.mission_engine.resume_pending()
            if resumed:
                logger.info("resumed %d in-flight missions", len(resumed))
        except Exception:  # startup must not crash the app
            logger.exception("mission resume failed")
        try:
            self.skill_registry.discover()
        except Exception:
            logger.exception("skill discovery failed")
        try:
            self.skill_hot_reloader.start()
        except Exception:
            logger.exception("skill hot reloader failed to start")
        self._register_default_heartbeats()
        self._seed_core_routines()
        # Load predictive start-tier priors from accumulated escalation history so
        # routing is predictive from the start of the session (L5).
        try:
            self.learning_loop.refresh_escalation_priors()
        except Exception:  # advisory, never blocks startup
            logger.exception("initial escalation-priors load failed")
        try:
            self.heartbeats.start()
        except Exception:
            logger.exception("heartbeat scheduler failed to start")
        try:
            if self.poller is not None:
                self.poller.start()
        except Exception:
            logger.exception("telegram poller failed to start")
        self._warm_models()

    def _reconcile_orphaned_activities(self) -> None:
        """Reap ``activities.db`` rows left ``queued``/``running`` by a process
        that crashed instead of shutting down cleanly (ADR for the Activity spine's
        orphan sweep). Runs once per runtime startup, before anything can submit a
        new Activity — a per-request store (``GET /activities``) never does this
        itself, or it would race a job this same process just started."""
        from iris_harness.services.activities import ActivityStore

        store = ActivityStore(db_path=self.data_dir / "activities.db")
        store.ensure_schema()
        store.reconcile_orphaned()

    def _seed_core_routines(self) -> None:
        """Seed the ``morning-digest`` routine on first boot (ADR-0122 §3, loop-proof D4).

        Idempotent: an existing row — however the owner edited, paused or retired it —
        is left alone; the routine tick keeps its schedule/channel/sections in line with
        Settings → Digest from then on. Runs before the heartbeat scheduler starts so the
        first tick already sees it.
        """
        try:
            from iris_harness.services.digest.settings import (
                iris_timezone,
                load_digest_settings,
            )
            from iris_harness.services.routines.seeded import (
                seed_morning_digest,
            )

            settings = load_digest_settings(self.data_dir, self.config_dir)
            seeded = seed_morning_digest(self.routine_store, settings, iris_timezone())
            if seeded is not None:
                logger.info(
                    "seeded routine %s (%s %s, channel=%s)",
                    seeded.id,
                    seeded.schedule,
                    seeded.metadata.get("timezone"),
                    seeded.delivery_channel,
                )
        except Exception:  # startup must not crash the app
            logger.exception("morning-digest seed failed")

    def _warm_models(self) -> None:
        """Pre-load default router + executor models in the background.

        Ollama loads weights into memory only on first request, costing 5–30s for
        a small model and tens of seconds for a large one. We fire a tiny warm-up
        call per model in a daemon thread so the user's first turn doesn't pay
        that cost. Failures are silent — warm-up is purely an optimization.
        """
        # Tests set IRIS_DISABLE_WARMUP=1 so the suite never spawns real Ollama calls.
        if os.environ.get("IRIS_DISABLE_WARMUP"):
            return

        def _warm() -> None:
            t0 = time.monotonic()
            try:
                self.intent_router.classifier.classify("hi")
            except Exception:
                logger.debug("router warm-up skipped", exc_info=True)
            try:
                warm_task = AgentTask(
                    query="hi",
                    agent_type="system",
                    memory_context=MemoryContext(recent_turns=()),
                    session_id="__warmup__",
                    params={"intent": "general"},
                )
                self.agent_executor.execute(warm_task)
            except Exception:
                logger.debug("executor warm-up skipped", exc_info=True)
            # Warm the output-safety guard (llama-guard) so its first real call
            # doesn't blow the per-judge timeout and fail open (red-team 2a).
            try:
                if self.response_curator.warm_output_safety():
                    logger.info("output-safety guard warm-up done")
            except Exception:
                logger.debug("output-safety guard warm-up skipped", exc_info=True)
            logger.info("model warm-up done in %.0fms", (time.monotonic() - t0) * 1000)

        threading.Thread(target=_warm, name="iris-model-warmup", daemon=True).start()

    def shutdown(self) -> None:
        try:
            if self.poller is not None:
                self.poller.stop()
        except Exception:
            logger.exception("telegram poller shutdown failed")
        try:
            self.heartbeats.shutdown(wait=False)
        except Exception:
            logger.exception("heartbeat scheduler shutdown failed")
        try:
            self.skill_hot_reloader.stop()
        except Exception:
            logger.exception("skill hot reloader shutdown failed")
        governor = getattr(self.tier_router, "governor", None)
        arbiter = getattr(governor, "arbiter", None) if governor is not None else None
        http = getattr(arbiter, "http", None) if arbiter is not None else None
        if http is not None:
            try:
                http.close()
            except Exception:
                logger.exception("arbiter http client shutdown failed")

    # ------------------------------------------------------------------
    # Chat pipeline
    # ------------------------------------------------------------------

    # The time/date intercept moved to the `system` reference plugin
    # (src/iris_harness/plugins_builtin/system/plugin.py) — OSS plan M1 tracer bullet.

    def run_routine(self, routine_id: str) -> RoutineExecutionRecord | None:
        """Execute one routine now, out of schedule, and record the run.

        Returns the execution record, or ``None`` when the routine id is
        unknown. Invokes the same bound capability the schedule would, delivering
        to the routine's ``delivery_channel`` — used by ``/routines run`` and the
        ``POST /routines/{id}/run`` endpoint to verify a routine end to end.
        """
        routine = self.routine_store.load(routine_id)
        if routine is None:
            return None
        checked_at = datetime.now(UTC).replace(microsecond=0)
        return execute_routine(self, routine, checked_at=checked_at, record=True)

    def preview_routine(self, routine_id: str) -> str | None:
        """Render a routine's output for preview — no delivery, no counters.

        Returns the rendered body, or ``None`` when the routine id is unknown.
        Raises ``ValueError`` when the routine's capability is not a previewable
        brief. Used by ``/routines preview`` / ``POST /routines/{id}/preview`` and
        the in-chat ``sample`` affordance.
        """
        routine = self.routine_store.load(routine_id)
        if routine is None:
            return None
        from iris_harness.runtime.handlers.skill_brief import (  # avoid cycle
            render_routine_body,
        )

        return render_routine_body(self, routine)

    def chat(
        self,
        message: str,
        *,
        session_id: str = "default",
        preferred_model: str | None = None,
        provider_profile: str | None = None,
        router_model: str | None = None,
        strict: bool = False,
        channel: str = "console",
        audience: Audience = "owner",
    ) -> ChatResult:
        """Run the full chat pipeline for one user message and return the result.

        Same pipeline as :meth:`chat_stream` (``iris_harness.runtime.turn``); this
        path drains the stream events and returns the final ``ChatResult``.
        Exceptions propagate (API callers keep their error semantics).

        ``router_model`` overrides the LLM used by the intent classifier for this
        request only (built classifiers are cached per model name). ``channel`` is
        the gateway the user is messaging from (``"console"``, ``"telegram"``,
        ``"web"``, ``"voice"``, ...); routine authoring uses it as the default
        delivery channel. ``audience`` is who reads the answer: ``other`` when people
        besides the owner do (a Telegram group chat), for the PRE_RESPONSE check.
        """
        request = TurnRequest(
            message=message,
            session_id=session_id,
            channel=channel,
            audience=audience,
            preferred_model=preferred_model,
            provider_profile=provider_profile,
            router_model=router_model,
            strict=strict,
        )
        # Current-span variant: chat() is a plain sync method (no generator
        # boundary), so context attach/detach is safe and stage + LLM spans nest
        # under it; the audit log can then stamp the active trace_id.
        with (
            maybe_current_span(self.tracer, "iris.chat") as span,
            session_scope(session_id),
            turn_scope(),
        ):
            return drain(run_turn(self, request, span=span, stage_spans=True, on_error="raise"))

    def resume_halted_run(self, *, run_id: str, step_id: int, channel: str = "console") -> Any:
        """Continue a halted run whose approval a human has just granted.

        Satisfies ``governance.approvals.service.RunResumer``. The approvals service
        records the decision and calls this; it never reaches into the loop itself,
        because ``governance`` sits below ``runtime``.

        This runs the **turn pipeline**, minus its ``intercept`` stage
        (``RESUME_STAGES``) — not the react loop directly. That matters more than it
        looks. Calling the executor straight would skip ``curate``, and with it the
        ResponseCurator and every safety judge, on the one path whose entire premise is
        that governance stopped this run once already. Going through the pipeline also
        means ``record`` files the answer in the session log, which is how it reaches the
        web transcript with no second delivery mechanism to build.

        ``intercept`` is the only stage dropped, and for two reasons: it is where the
        user's message is logged, and there is no user message here to log — forging one
        would put words in their mouth; and it is where the turn could be claimed by
        something other than the run being resumed.

        The query and session come from the checkpoint, so the resumed turn re-enters the
        task it was actually working on, in the conversation that was waiting for it.
        ``channel`` comes from the approval row: the resumed turn can be halted again,
        and a second approval has to reach the same surface as the first.
        """
        from iris_harness.agent.agentic_core import resume_seed_from_checkpoint
        from iris_harness.kernel.governance.approvals.service import ResumedRun
        from iris_harness.memory.state import CheckpointStore
        from iris_harness.runtime.turn import RESUME_STAGES

        checkpoint = CheckpointStore().get(run_id=run_id, step_id=step_id)
        seed = resume_seed_from_checkpoint(checkpoint)
        session_id = checkpoint.session_id or "default"
        request = TurnRequest(
            message=seed.query,
            session_id=session_id,
            channel=channel,
            resume_run_id=run_id,
            resume_step_id=step_id,
        )
        with (
            maybe_current_span(self.tracer, "iris.chat.resume") as span,
            session_scope(session_id),
            turn_scope(),
        ):
            result = drain(
                run_turn(
                    self,
                    request,
                    span=span,
                    stage_spans=True,
                    on_error="raise",
                    stages=RESUME_STAGES,
                )
            )
        return ResumedRun(answer=result.response, session_id=session_id)

    def open_turn(
        self,
        opener: str,
        *,
        session_id: str,
        channel: str = "console",
        audience: Audience = "owner",
    ) -> ChatResult:
        """Run a turn the system opens, with no user message (ADR-0127).

        The named opener (``config/intercepts.yaml`` ``openers:``) answers it, and the
        pipeline runs ``OPENER_STAGES``: ``open`` logs the turn's start as ``turn_open``
        and runs the opener, ``guard`` gives the answer the same response check and audit
        row a deterministic handler's answer gets, and ``record`` files it in the session
        log. There is no input, so there is no input screen and no message is logged in
        the owner's name. The first-chat welcome (``runtime.welcome``) is the caller.
        """
        from iris_harness.runtime.turn import OPENER_STAGES

        request = TurnRequest(
            message="",
            session_id=session_id,
            channel=channel,
            audience=audience,
            opener=opener,
        )
        with (
            maybe_current_span(self.tracer, "iris.chat.open") as span,
            session_scope(session_id),
            turn_scope(),
        ):
            return drain(
                run_turn(
                    self,
                    request,
                    span=span,
                    stage_spans=True,
                    on_error="raise",
                    stages=OPENER_STAGES,
                )
            )

    def chat_stream(
        self,
        message: str,
        *,
        session_id: str = "default",
        preferred_model: str | None = None,
        provider_profile: str | None = None,
        router_model: str | None = None,
        strict: bool = False,
        channel: str = "console",
        audience: Audience = "owner",
    ) -> Iterator[StreamEvent]:
        """Streaming variant of ``chat`` — yields tokens then a final result.

        Streams the **first** task's output token-by-token; the rest of the plan
        runs wave by wave before the final ``StreamEvent(kind="done")``. Errors
        surface as a terminal ``StreamEvent(kind="error")``.

        The stream is context-pinned: a server that steps it from worker threads
        would otherwise drop the session and turn scopes after the first event,
        and with them every ``llm_call`` record of which model ran.
        """
        request = TurnRequest(
            message=message,
            session_id=session_id,
            channel=channel,
            audience=audience,
            preferred_model=preferred_model,
            provider_profile=provider_profile,
            router_model=router_model,
            strict=strict,
        )
        return pin_context(self._chat_stream_events(request))

    def _chat_stream_events(self, request: TurnRequest) -> Iterator[StreamEvent]:
        with (
            maybe_span(self.tracer, "iris.chat_stream") as span,
            session_scope(request.session_id),
            turn_scope(),
        ):
            yield from run_turn(self, request, span=span, stage_spans=False, on_error="event")

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
    ) -> ChatResult:
        """Curate results, run post-processing, and produce the final ``ChatResult``."""
        prior_turns = self.sessions.conversations.get(session_id, [])
        history_text = tuple(f"{turn.role}: {turn.content}" for turn in prior_turns)
        curated = self.response_curator.curate(
            results,
            query=message,
            strict=strict,
            session_id=session_id,
            conversation_history=history_text,
            intent=intent_result.intent,
        )
        # L3 (ADR-0068): if the escalation judge is in enforce mode and diagnosed
        # a capability gap, re-run the plan at a stronger LOCAL tier and re-curate
        # BEFORE any side effects, so signals/turn/wiki are recorded once for the
        # final (escalated) answer. Bounded + local-only + governor-vetoable.
        escalated_to: str | None = None
        if allow_escalation:
            if self._escalation_cache is None:
                self._escalation_cache = EscalationActions(self)
            results, curated, escalated_to, clarify_question = (
                self._escalation_cache.maybe_escalate(
                    results=results,
                    curated=curated,
                    plan=plan,
                    intent_result=intent_result,
                    memory_ctx=memory_ctx,
                    session_id=session_id,
                    message=message,
                    preferred_model=preferred_model,
                    provider_profile=provider_profile,
                    strict=strict,
                    history_text=history_text,
                )
            )
            if clarify_question:
                # Ambiguity: return the judge's question instead of a guess. It
                # becomes the assistant turn, so the user's reply naturally
                # resumes the conversation (D8 / design §3).
                curated = replace(
                    curated,
                    text=clarify_question,
                    has_errors=False,
                    metadata={**curated.metadata, "escalation_clarify": True},
                )
        # ADR-0072 slice 3: judge-gated clarify nudge. When the per-turn judge or the
        # router signals uncertainty (and escalation didn't already ask), append ONE
        # gentle invitation to refine — never on a clean, confident answer. Shared
        # finalizer, so this covers both sync chat and streaming + the recorded turn.
        if os.getenv("IRIS_FEEDBACK_CLARIFY") == "1" and _clarify_warranted(
            intent_confidence=intent_result.confidence,
            judge_bundle=curated.metadata.get("judge_bundle"),
            has_errors=curated.has_errors,
            already_clarified=bool(curated.metadata.get("escalation_clarify")),
            agent_type=intent_result.agent_type,
        ):
            curated = replace(
                curated,
                text=f"{curated.text.rstrip()}\n\n{_CLARIFY_NUDGE}",
                metadata={**curated.metadata, "feedback_clarify_appended": True},
            )
        self.sessions.record_turn(
            session_id,
            message,
            curated.text,
            origin=_turn_origin(curated.metadata),
        )
        self.capture.extract_and_store_facts(message, session_id)
        # Say what was learned, in the same breath. A fact IRIS keeps quietly is one the
        # user cannot correct — and a queue nobody works is not learning at all.
        learned = self.capture.take_notices()
        question = self.capture.take_question(session_id)
        if learned or question:
            tail = []
            if learned:
                tail.append("_Noted — " + "; ".join(learned) + '. Say "forget that" to undo._')
            if question:
                tail.append(f"_{question} (yes / no)_")
            curated = replace(curated, text=f"{curated.text.rstrip()}\n\n" + "\n".join(tail))
        self._ingest_to_wiki(message, curated, intent_result)
        raw_latency = curated.metadata.get("total_latency_ms", 0.0)
        latency_ms = float(raw_latency) if isinstance(raw_latency, int | float | str) else 0.0
        provider_profile = effective_provider_profile(provider_profile, preferred_model)
        if provider_profile:
            _profile = config_from_profile(provider_profile, intent=intent_result.intent)
            model = preferred_model or _profile.model
            provider = _profile.provider
        elif preferred_model:
            model = preferred_model
            provider, _, _ = infer_provider(preferred_model)
        else:
            model = self.tier_router.model_for_intent(intent_result.intent)
            provider = self.tier_router.provider_for_intent(intent_result.intent)
        if escalated_to is not None:
            # Record the tier that ACTUALLY answered (D4 — resolved, not requested).
            escalated_tier = self.tier_router.get_tier_by_name(escalated_to)
            if escalated_tier is not None:
                model = escalated_tier.model
                provider = escalated_tier.provider
        if router_model:
            effective_router_model = router_model
            effective_router_provider, _, _ = infer_provider(router_model)
        else:
            effective_router_model = self.tier_router.model_for_intent("intent_classification")
            effective_router_provider = self.tier_router.provider_for_intent(
                "intent_classification"
            )
        self.capture.record_signal(
            intent_result,
            curated,
            latency_ms=latency_ms,
            model=model,
            provider=provider,
            query=message,
        )
        if span is not None:
            try:
                span.set_attribute("output.value", curated.text)
            except Exception:  # noqa: BLE001, S110 — span attribute is optional
                pass
            span.set_attribute("intent", intent_result.intent)
            span.set_attribute("agent_type", intent_result.agent_type)
            span.set_attribute("session_id", session_id)
            span.set_attribute("model", model)
            span.set_attribute("provider", provider)
            span.set_attribute("latency_ms", latency_ms)
            span.set_attribute("has_errors", curated.has_errors)
        return ChatResult(
            response=curated.text,
            intent=intent_result.intent,
            agent_type=intent_result.agent_type,
            sources=tuple(curated.sources),
            has_errors=curated.has_errors,
            error_summary=curated.error_summary,
            metadata={
                **curated.metadata,
                "is_multi_step": intent_result.is_multi_step,
                "plan_size": len(plan.tasks),
                "session_id": session_id,
                "model": model,
                "provider": provider,
                "router_model": effective_router_model,
                "router_provider": effective_router_provider,
            },
        )

    @staticmethod
    def _span_input(span: Any, message: str) -> None:
        if span is None:
            return
        try:
            span.set_attribute("input.value", message)
        except Exception:  # noqa: BLE001, S110 — span attribute is optional
            pass

    def _router_audit(self) -> Any:
        """Return the lazily-constructed router-decision audit logger.

        Cached on the runtime instance after the first call. Constructed
        lazily so a misconfigured ``data_dir`` doesn't break bootstrap;
        audit failures inside ``.record(...)`` are swallowed with a
        warning so chat turns never break over an audit write.
        """
        if self._router_audit_cache is None:
            from iris_harness.runtime.router_audit import RouterAuditLogger

            self._router_audit_cache = RouterAuditLogger(self.data_dir / "audit.db")
        return self._router_audit_cache

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def warmup(
        self,
        *,
        role: str,
        model: str | None = None,
        provider_profile: str | None = None,
    ) -> WarmupResult:
        """Pre-load a router or executor model so the next chat call is fast.

        Used by the CLI when the user runs ``/model``, ``/router``, or
        ``/provider`` so they get a clear "loading…" indicator instead of an
        unexplained 20-second wait on the next message.
        """
        t0 = time.monotonic()
        try:
            provider_profile = effective_provider_profile(provider_profile, model)
            if role == "router":
                classifier = self._classifier_for(model or None)
                classifier.classify("hi")
                resolved = model or self.tier_router.model_for_intent("intent_classification")
            elif role == "executor":
                warm_task = AgentTask(
                    query="hi",
                    agent_type="system",
                    memory_context=MemoryContext(recent_turns=()),
                    session_id="__warmup__",
                    params={
                        "intent": "general",
                        "preferred_model": model or "",
                        "provider_profile": provider_profile or "",
                    },
                )
                self.agent_executor.execute(warm_task)
                if provider_profile:
                    profile_cfg = config_from_profile(provider_profile)
                    resolved = model or profile_cfg.model
                else:
                    resolved = model or self.tier_router.model_for_intent("general")
            else:
                return WarmupResult(
                    ok=False,
                    role=role,
                    model=model or "",
                    latency_ms=(time.monotonic() - t0) * 1000,
                    error=f"unknown role {role!r} (expected 'router' or 'executor')",
                )
            return WarmupResult(
                ok=True,
                role=role,
                model=resolved,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as exc:
            logger.exception("warmup failed for role=%s model=%r", role, model)
            return WarmupResult(
                ok=False,
                role=role,
                model=model or "",
                latency_ms=(time.monotonic() - t0) * 1000,
                error=_friendly_llm_error(exc),
            )

    def _classifier_for(self, router_model: str | None) -> IIntentClassifier:
        """Return a classifier bound to ``router_model`` or the runtime default."""
        if not router_model:
            return self.intent_router.classifier
        cached = self._router_cache.get(router_model)
        if cached is not None:
            return cached
        classifier = _build_router_classifier_for_model(router_model)
        self._router_cache[router_model] = classifier
        return classifier

    def _resolve_continuation_intent(
        self,
        message: str,
        intent: IntentResult,
        *,
        session_id: str,
    ) -> IntentResult:
        """Pin short artifact follow-ups to ``code_exec`` when recent context indicates it.

        This avoids burning the generic system tool budget for turns like
        "yes do that and use my token from .env" that are clearly continuations
        of a prior code-exec artifact workflow.
        """
        if intent.agent_type == "code_exec":
            return intent

        history = self.sessions.conversations.get(session_id, [])
        if not history:
            return intent

        lowered = message.lower().strip()
        is_short = len(lowered) <= 140
        continuation = bool(
            re.search(
                r"\b(yes|yep|yeah|do that|continue|go ahead|same|update it|refine|reformat)\b",
                lowered,
            )
        )
        env_auth_detail = bool(re.search(r"\b(env|\.env|token|pat|api key|auth)\b", lowered))
        if not is_short or not (continuation or env_auth_detail):
            return intent

        recent_assistant = "\n".join(
            turn.content for turn in history[-4:] if turn.role == "assistant"
        ).lower()
        artifact_signal = bool(
            re.search(
                r"(artifacts?:|/workspace/|\.pdf\b|\.md\b|\.json\b|generated|sandbox)",
                recent_assistant,
            )
        )
        if not artifact_signal:
            return intent

        return IntentResult(
            intent="code_exec",
            agent_type="code_exec",
            confidence=max(intent.confidence, 0.91),
            is_multi_step=intent.is_multi_step,
            raw_query=intent.raw_query or message,
        )

    # Domain intents have a dedicated agent (email/finance/calendar/planner/…) and
    # must NOT be hijacked to a skill via the general handler — that was the
    # issue-0003 misroute: "any finance emails?" classified communication/email
    # but a matching skill overrode it to general/system, which then stalled. The
    # override exists only to give a skill first chance over a code_exec scraper
    # (its documented purpose), so it must skip the dedicated-agent intents.
    _SKILL_PROTECTED_INTENTS = frozenset(
        {
            "communication",
            "finance",
            "calendar",
            "planner",
            "coding",
            "files",
            "profile_query",
            "clarify",
        }
    )

    def _resolve_skill_intent(self, message: str, intent: IntentResult) -> IntentResult:
        """Route clear promoted-skill matches to the general tool handler.

        The keyword router intentionally sends known live-data pages to
        ``code_exec``. Once a first-class skill exists for one of those pages,
        the skill should get first chance so IRIS does not regenerate a scraper.
        Domain intents with a dedicated agent are protected (issue 0003).
        """
        if intent.intent in self._SKILL_PROTECTED_INTENTS:
            return intent
        try:
            packages = self.skill_registry.discover()
        except Exception:
            logger.exception("skill intent resolution failed")
            return intent
        if best_matching_skill_package(message, packages) is None:
            return intent
        return IntentResult(
            intent="general",
            agent_type="system",
            confidence=max(intent.confidence, 0.92),
            is_multi_step=intent.is_multi_step,
            raw_query=intent.raw_query or message,
        )

    def _resolve_action_intent(self, message: str, intent: IntentResult) -> IntentResult:
        """Promote a weak-tier imperative to the ``action`` intent (ADR-0103).

        Predictive escalation: an action request ("update my briefing to add the
        outstanding dues", "remind me when I get an email from X", "create a
        notification when …") that the classifier dropped into general/system/help
        would run on the weak tier-1 model and reason/tool-call poorly. Relabel it to
        ``action`` so the tier map starts it on the capable instruct tier, where the
        agent can weigh the available tools and take a governance-gated safe action.
        Generic (not finance-specific); domain actions already on a capable tier are
        left untouched. Applied on BOTH the chat() and chat_stream() paths.
        """
        from iris_harness.agent.intent_router import promote_to_action_intent

        return promote_to_action_intent(intent, message)

    def _execute_plan(
        self,
        plan: TaskPlan,
        intent: IntentResult,
        memory_ctx: MemoryContext,
        session_id: str,
        *,
        preferred_model: str | None = None,
        provider_profile: str | None = None,
    ) -> list[AgentResult]:
        extra: dict[str, str] = {"intent": intent.intent}
        if intent.is_multi_step:
            extra["is_multi_step"] = "1"
        if preferred_model:
            extra["preferred_model"] = preferred_model
        if provider_profile:
            extra["provider_profile"] = provider_profile
        # Execute dependency-ordered waves: independent sub-tasks within a wave
        # run concurrently (AgentExecutor.execute_wave), waves run in order. A
        # single-task plan (the common case) is one wave of one → runs inline.
        results: list[AgentResult] = []
        for wave in plan.execution_groups():
            wave_tasks: list[AgentTask] = []
            for sub in wave:
                target_agent = self._resolve_supported_agent(
                    sub.agent_type or intent.agent_type,
                    fallback=intent.agent_type,
                )
                wave_tasks.append(
                    AgentTask(
                        query=sub.params.get("query", plan.query),
                        agent_type=target_agent,
                        memory_context=memory_ctx,
                        session_id=session_id,
                        params={**dict(sub.params), **extra},
                    )
                )
                log_timeline_event(
                    "agent.trace",
                    phase="agent.start",
                    text=f"agent start: {target_agent}",
                    payload={
                        "name": "agent.start",
                        "agent_type": target_agent,
                        "task_id": sub.id,
                        "query_chars": len(wave_tasks[-1].query),
                    },
                )
            wave_results = self.agent_executor.execute_wave(wave_tasks)
            for result in wave_results:
                log_timeline_event(
                    "agent.trace",
                    phase="agent.result",
                    text=f"agent result: {result.agent_type}",
                    payload={
                        "name": "agent.result",
                        "agent_type": result.agent_type,
                        "success": result.success,
                        "latency_ms": result.latency_ms,
                        "output_chars": len(result.output),
                        **result.metadata,
                    },
                )
            results.extend(wave_results)
        if not results:
            fallback_agent = self._resolve_supported_agent(
                intent.agent_type,
                fallback="system",
            )
            fallback_task = AgentTask(
                query=plan.query,
                agent_type=fallback_agent,
                memory_context=memory_ctx,
                session_id=session_id,
                params=extra,
            )
            log_timeline_event(
                "agent.trace",
                phase="agent.start",
                text=f"agent start: {fallback_agent}",
                payload={
                    "name": "agent.start",
                    "agent_type": fallback_agent,
                    "task_id": "fallback",
                    "query_chars": len(fallback_task.query),
                },
            )
            fallback_result = self.agent_executor.execute(fallback_task)
            log_timeline_event(
                "agent.trace",
                phase="agent.result",
                text=f"agent result: {fallback_result.agent_type}",
                payload={
                    "name": "agent.result",
                    "agent_type": fallback_result.agent_type,
                    "success": fallback_result.success,
                    "latency_ms": fallback_result.latency_ms,
                    "output_chars": len(fallback_result.output),
                    **fallback_result.metadata,
                },
            )
            results.append(fallback_result)
        return results

    def _resolve_supported_agent(self, candidate: str, *, fallback: str) -> str:
        """Return a registered agent name, preferring ``candidate`` then ``fallback``."""
        registered = self.agent_executor.registered_agents()
        if candidate in registered:
            return candidate
        if fallback in registered:
            return fallback
        if "system" in registered:
            return "system"
        # Final guard for degenerate runtimes with no system registration.
        return next(iter(registered), fallback)

    def _ingest_to_wiki(
        self,
        message: str,
        curated: CuratedResponse,
        intent: IntentResult,
    ) -> None:
        if not curated.text or curated.has_errors:
            return
        # Automatic wiki ingest is off by default (see WikiEngine.ingest_enabled). Check
        # it before triage so a disabled wiki costs the chat turn nothing at all — this
        # ran in-line in _finalize_chat, before the reply was returned.
        if not getattr(self.wiki, "ingest_enabled", True):
            return
        triage = triage_memory_item(
            f"{message}\n\n{curated.text}",
            signal_type="wiki_ingest",
        )
        if triage.destination != MemoryDestination.WIKI:
            logger.info(
                "wiki ingest skipped by memory triage: kind=%s destination=%s reason=%s",
                triage.kind,
                triage.destination,
                triage.reason,
            )
            return
        event = WikiIngestEvent(
            source_agent=intent.agent_type,
            source_id=f"chat:{intent.intent}",
            content=f"{message}\n\n{curated.text}",
            entities_hint=[],
        )
        try:
            self.wiki.ingest(event)
        except Exception:  # wiki failures must not break /chat
            logger.exception("wiki ingest failed for intent=%s", intent.intent)

    # ------------------------------------------------------------------
    # Fact extraction & profile export
    # ------------------------------------------------------------------

    def _handle_standing_instruction_turn(
        self,
        message: str,
        *,
        session_id: str,
        span: Any = None,
    ) -> ChatResult | None:
        """Capture a user-taught standing rule and confirm it; else fall through.

        A standing instruction binds a *trigger* (a phrase the user will say
        again) to a *response shape* — "when I ask how is my day, show my emails +
        dues + calendar". This is an early-intercept handler (peer of
        ``RoutineAuthoring.handle_routine_authoring_turn``): when a turn *teaches a rule* we
        persist it as a behavior recipe and answer with a confirmation,
        **skipping the agent pipeline entirely**. That's deliberate — the
        teaching turn's intent is to set a rule, not to get the (mis-routed)
        answer the agent would otherwise stream. The rule is then honored on the
        next matching turn through the existing ``match_behavior`` injection path,
        with no nightly mine in between. Returns ``None`` (fall through to normal
        routing) when the turn isn't teaching a rule, or on any failure — capture
        must never break a turn.

        Gated by ``IRIS_STANDING_INSTRUCTIONS`` (default on — this is an explicit
        in-chat instruction, not inferred mining; set ``0`` to disable).
        """
        if os.getenv("IRIS_STANDING_INSTRUCTIONS", "1").strip().lower() in {"0", "false", "no"}:
            return None
        from iris_harness.memory.identity import write_behavior
        from iris_harness.memory.standing_instructions import (
            extract_standing_instruction,
            looks_like_standing_instruction,
        )

        if not looks_like_standing_instruction(message):
            return None
        try:
            instruction = extract_standing_instruction(
                message, self.routines.routine_authoring_llm_caller()
            )
            if instruction is None:
                return None
            write_behavior(
                instruction.name,
                instruction.instruction,
                match_keywords=instruction.trigger_keywords,
                description=f"User-taught rule: {instruction.summary()}",
                source="taught",
            )
        except Exception:  # capture must never break a turn
            logger.exception("standing-instruction capture failed; falling through")
            return None
        logger.info("standing instruction captured: %s", instruction.name)
        return self.replies.system_chat_result(
            message=message,
            session_id=session_id,
            response=f"Got it — {instruction.summary()}. Just ask and I'll do that.",
            metadata={
                "standing_instruction_captured": True,
                "behavior_name": instruction.name,
                "trigger_keywords": list(instruction.trigger_keywords),
            },
            span=span,
        )

    def _activity_notices(self) -> ActivityNotices:
        """The runtime's one :class:`ActivityNotices`, built on first use.

        One, not one per call: it owns the Activity runner and subscribes its handlers
        to ``_event_bus`` when the runner is built, so a second instance would announce
        every completion twice.
        """
        if self._activity_notices_cache is None:
            self._activity_notices_cache = ActivityNotices(self)
        return self._activity_notices_cache

    def _record_downstream_reuse(self, memory_ctx: MemoryContext) -> None:
        """Record downstream_reuse for prior turns recalled into this turn (§4.2).

        A pure observable: an earlier turn's content was semantically retrieved
        into the current turn's context — i.e. it was reused, so it was useful.
        Correlated to the REUSED turn (the one that produced the value), not the
        observing turn. Always on; best-effort.
        """
        observer_turn = current_turn_id()
        # Pointer recall mode: what the note offered, and when the backstop pushed turns
        # instead -- so the two modes can be compared on real use. Ids and counts only.
        telemetry: list[tuple[str, float, dict[str, Any]]] = []
        if memory_ctx.recall_pointer_sessions:
            sessions = list(memory_ctx.recall_pointer_sessions)
            telemetry.append(
                ("recall_pointer_offered", float(len(sessions)), {"sessions": sessions})
            )
        if memory_ctx.recall_backstop:
            telemetry.append(("recall_backstop", float(len(memory_ctx.reused_turn_refs)), {}))
        for metric, value, meta in telemetry:
            try:
                self.signal_collector.record_metric(
                    metric_name=metric,
                    value=value,
                    success=True,
                    metadata={**meta, "observed_in_turn": observer_turn},
                    turn_id=observer_turn,
                )
            except Exception:  # recall telemetry never breaks the turn
                logger.debug("failed to record %s", metric, exc_info=True)
        refs = memory_ctx.reused_turn_refs
        if not refs:
            return
        for ref in refs:
            try:
                self.signal_collector.record_metric(
                    metric_name="downstream_reuse",
                    value=1.0,
                    success=True,
                    metadata={
                        "reused_row_id": ref.row_id,
                        "reused_from_session": ref.session_id,
                        "role": ref.role,
                        "observed_in_turn": observer_turn,
                    },
                    session_id=ref.session_id or None,
                    turn_id=ref.turn_id,
                )
            except Exception:  # reuse telemetry never breaks the turn
                logger.debug("failed to record downstream_reuse", exc_info=True)

    def _housekeeping_heartbeat(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        """The daily retention pass (config/memory/retention.yaml).

        Moves conversations hot -> cold, sweeps vectors whose row is gone, compacts the
        database and rotates logs. It never deletes a summary, a confirmed fact, or
        anything the owner wrote — those are owner actions with a preview.
        """
        config = retention_config().get("housekeeping") or {}
        if not config.get("enabled", True):
            return HeartbeatRun(
                name=definition.name,
                status=HeartbeatStatus.SKIPPED,
                output="housekeeping disabled in config/memory/retention.yaml",
            )
        # Checked hourly, run daily: an ``interval:86400`` schedule restarts its clock
        # with the process, so a stack restarted more often than daily never ran the
        # pass at all. The last run is on disk; a pass that left a closing-summary
        # backlog runs again at the next check instead of waiting a day.
        due, why = _housekeeping_due(
            self.retention.last_run(),
            min_hours=float(str(definition.params.get("min_hours_between", 23))),
        )
        if not due:
            return HeartbeatRun(name=definition.name, status=HeartbeatStatus.SKIPPED, output=why)
        report = self.retention.run()
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.FAILED if report.errors else HeartbeatStatus.SUCCESS,
            output=(
                f"cooled {report.sessions_cooled} session(s), "
                f"{report.turns_deleted} turn(s) and {report.vectors_deleted} vector(s) removed, "
                f"{report.orphan_vectors_swept} orphan vector(s) swept, "
                f"{report.sessions_closed} closed ({report.sessions_close_backlog} waiting), "
                f"logs: {report.logs_rotated} rotated / {report.logs_compressed} compressed / "
                f"{report.logs_deleted} deleted "
                f"({report.log_bytes_reclaimed // (1024 * 1024)} MB reclaimed)"
                + (f"; errors: {'; '.join(report.errors)}" if report.errors else "")
            ),
        )

    def _register_default_heartbeats(self) -> None:
        self.heartbeats.register_handler("wiki_lint", _wiki_lint_handler)
        self.heartbeats.register_handler("memory_housekeeping", self._housekeeping_heartbeat)
        self.heartbeats.register_handler("learning_tick", self.learning_engine.heartbeat_handler)
        self.heartbeats.register_handler(
            "escalation_priors_tick", self.learning_loop.escalation_priors_heartbeat
        )
        self.heartbeats.register_handler(
            "crystallize_tick", self.learning_loop.crystallize_heartbeat
        )
        self.heartbeats.register_handler(
            "routine_reflection_tick", self.learning_loop.routine_reflection_heartbeat
        )
        self.heartbeats.register_handler(
            "learning_analysis_tick", self.learning.learning_analysis_heartbeat
        )
        self.heartbeats.register_handler(
            "behavior_mining_tick", self.learning.behavior_mining_heartbeat
        )
        self.heartbeats.register_handler(
            "intention_rollup_tick", self.learning.intention_rollup_heartbeat
        )
        self.heartbeats.register_handler(
            "experiment_remeasure_tick", self.learning_loop.experiment_remeasure_heartbeat
        )
        self.heartbeats.register_handler(
            "sandbox_preflight_tick", self.learning_loop.sandbox_preflight_heartbeat
        )
        self.heartbeats.register_handler("routine_tick", build_routine_tick_handler(self))
        self.heartbeats.register_handler(
            "skill_brief",
            _build_skill_brief_handler(self),
        )
        self.heartbeats.register_handler("pressure_tick", build_pressure_tick_handler(self))
        self.heartbeats.register_handler(
            "approval_timeout_tick", self.confirmations.approval_timeout_heartbeat
        )
        # `reminder_tick` (the legacy markdown reminders) registers from the calendar
        # plugin since M5.7 track A slice 4; its schedule stays in config/heartbeats.yaml.
        # System Health's `health_tick` (ADR-0069) is registered by the `system`
        # reference plugin at build time (OSS plan M1).
        # Phase 1 Track 1D — the email-sweep heartbeat registers from the
        # email_workflows plugin as of M6.1b. M3.2 kept it core because the sweep
        # keeps email.db current and the store is core; decision 2 ends that
        # argument — the whole email library leaves for src/iris_personal, so the
        # core cannot import the handler it would register. The schedule stays in
        # config/heartbeats.yaml: an entry whose handler no one registers is
        # skipped with a warning, which is the core-only shape.
        # `finance_ingest_tick` (statement sweep + extraction) and `finance_monitor`
        # (the daily bills/spend alert, ADR-0006 / ADR-0039) are registered by the
        # `finance_workflows` plugin (OSS plan M4.2) — both are workflows over the
        # record, not maintenance of it.
        # NOTE: the finance-specific retention sweep (ADR-0061) is retired in
        # FMX1 — finance documents now live in FileManager custody, so the
        # filemanager_retention sweep owns their retention.
        #
        # All four FileManager sweeps register from the file_organizer plugin as of
        # M6.1b. M2.6 called them core because they maintain the catalog `search`
        # reads (M2.5 had moved them out; the access/management split moved them
        # back). Decision 2 settles it a third and last way: file access is not core
        # either — the whole filemanager library leaves — so both halves register
        # from the plugin. Their schedules stay in config/heartbeats.yaml.
        # apple_calendar_sync (the Calendar.app read-back) registers from the
        # calendar plugin since M5.7 track A: M3.4 kept it core as "the sweep that
        # keeps the store current", but it drives EventKit, and the owner's rule is
        # that anything talking to an external app is a plugin. The store it fills
        # stays core; what fills it moved.
        self.heartbeats.register_handler(
            "mission_proposal_tick", self.mission_proposals.mission_proposal_heartbeat
        )
        # ADR-0071 slice 1 — the semantic email index subscription (off by default,
        # IRIS_EMAIL_SEMANTIC_SEARCH=1) moved to the email_workflows plugin at
        # M6.1b, with the index it keeps current.
        # The three workflow subscriptions that used to sit here — triage on
        # email.new_arrived (Track 1G), the email→wiki translator (Track 1K) and
        # followup auto-resolution (Track 2B) — are registered by the
        # email_workflows plugin via api.subscribe(..., scope="process").
        # The one reminder store (loop-proof D14): the heartbeat sends each due
        # reminder to Telegram + web push itself and records what a channel accepted
        # (accept-then-fire), so no reminder.fired → channels subscriber any more —
        # it would send twice. `push_delivery` turns a reminder no channel accepted
        # into a red health check the watch pages on.
        from iris_harness.services.digest.settings import iris_timezone
        from iris_harness.services.health.service import register_check_provider
        from iris_harness.services.notifications.health import push_delivery_provider

        self.heartbeats.register_handler(
            "notification_reminder_tick",
            build_notification_reminder_tick_handler(self),
        )
        register_check_provider(
            "push_delivery", push_delivery_provider(self.data_dir, iris_timezone)
        )
        # `job_completed` (loop-proof D13): one row per job health_watch.yaml names, red
        # when its last scheduled slot passed its grace with no successful run.
        from iris_harness.services.health.jobs import job_completed_provider

        register_check_provider(
            "job_completed", job_completed_provider(self.heartbeats, self.config_dir)
        )
        try:
            definitions = load_heartbeats(self.config_dir / "heartbeats.yaml")
            self.heartbeats.bind_known_plugins(
                load_plugin_owners(self.config_dir / "heartbeats.yaml")
            )
        except Exception:
            logger.exception("heartbeat config load failed")
            definitions = []
        registered = self.heartbeats.register_all(definitions)
        if registered:
            logger.info("registered %d heartbeat(s)", registered)
