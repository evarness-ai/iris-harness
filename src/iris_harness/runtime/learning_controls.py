"""Learning controls — the digital-twin miners and the learning analyst, carved out of
``IrisRuntime`` and ``HeartbeatRunnersMixin`` (OSS plan M5.7 track C, slice 9).

One topic that used to live in two places: the experiment console's controls
(``learning_flags`` / ``set_learning_flag`` / ``run_learning_now`` / the dry-run previews,
ADR-0083) on the runtime, and the heartbeat runs they toggle and trigger (learning
analysis, behavior mining, intention rollup) on the mixin. Carving the controls alone
would have given this object a host that was mostly the other half of its own topic.

**The state moved with it.** ``behavior_miner``, ``intention_analyst``,
``learning_analyst``, ``learning_analyst_model`` and the self-management override are
this object's fields now, not the runtime's: the controls write them and the runs gate
on them, and the only reader outside the topic is compaction's "is the miner on" check,
which asks ``runtime.learning.behavior_miner``. :meth:`LearningControls.from_env` builds
them from the env gates, called by ``build_runtime`` where the builders always ran.

:class:`LearningHost` declares the eight runtime members read, so mypy checks the
runtime still supplies them. The host is read **at call time**, not captured. It is
also *written* once: behavior mining drains ``_compaction_archive_buffer`` by rebinding
it, exactly as it did on the mixin.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from iris_harness.foundation.env import env_flag_on
from iris_harness.services.heartbeat import HeartbeatDefinition, HeartbeatRun, HeartbeatStatus

if TYPE_CHECKING:
    from pathlib import Path

    from iris_harness.llm.tier_router import TierRouter
    from iris_harness.memory.semantic_index import SemanticIndex
    from iris_harness.memory.store import MemoryStore
    from iris_harness.runtime.session_memory import SessionMemory
    from iris_harness.services.learning.signals import LearningSignalCollector
    from iris_harness.services.learning.store import LearningMetricsStore
    from iris_harness.services.routines import RoutineStore

logger = logging.getLogger(__name__)

# Hot-toggleable learning capabilities (ADR-0083 experiment console). Each maps to a
# runtime component the heartbeats gate on, so it can be flipped at runtime without a
# restart. The env var is the *default* a restart reverts to.
_LEARNING_FLAGS = ("behavior_miner", "intention_rollup", "learning_analyst")
_LEARNING_FLAG_ENV = {
    "behavior_miner": "IRIS_BEHAVIOR_MINER",
    "intention_rollup": "IRIS_INTENTION_ROLLUP",
    "learning_analyst": "IRIS_LEARNING_ANALYST",
}


def learning_flag_env(name: str) -> str:
    """The env setting behind a learning control, so a live toggle can be saved and
    survive a restart (ADR-0120). Raises ValueError for an unknown control."""
    if name == "self_management":
        return "IRIS_AGENT_SELF_MANAGEMENT"
    try:
        return _LEARNING_FLAG_ENV[name]
    except KeyError:
        raise ValueError(f"unknown learning flag: {name}") from None


class LearningHost(Protocol):
    """The eight runtime members the learning controls and runs reach.

    ``sessions`` is session memory; behavior mining drains its compaction archive
    (OSS plan M5.7 track C slice 16).
    """

    tier_router: TierRouter
    memory_store: MemoryStore
    semantic_index: SemanticIndex | None
    learning_store: LearningMetricsStore
    signal_collector: LearningSignalCollector
    routine_store: RoutineStore
    data_dir: Path
    sessions: SessionMemory


class LearningControls:
    """The learning capabilities for one runtime: their live state, the console controls,
    and the heartbeat runs. See the module docstring."""

    def __init__(
        self,
        host: LearningHost,
        *,
        learning_analyst: Callable[[str, str], str] | None = None,
        learning_analyst_model: str = "unknown",
        behavior_miner: Callable[[str, str], str] | None = None,
        intention_analyst: Callable[[str, str], str] | None = None,
    ) -> None:
        self._host = host
        # Optional agentic learning analyst (ADR-0069 #4 slice 2): a governed LLM
        # invoke(system, user) + the model it runs on. None when IRIS_LEARNING_ANALYST
        # is off (default) — the learning_analysis_tick then no-ops.
        self.learning_analyst = learning_analyst
        self.learning_analyst_model = learning_analyst_model
        # Optional behavior-pattern miner (digital-twin layer): a governed LLM
        # invoke(system, user). None when IRIS_BEHAVIOR_MINER is off (default) — the
        # behavior_mining_tick then no-ops. Propose-only: mined patterns go to a HITL
        # review queue, never auto-written as durable episodic patterns.
        self.behavior_miner = behavior_miner
        # Optional intention rollup (digital-twin layer 3): a governed LLM invoke(system,
        # user). None when IRIS_INTENTION_ROLLUP is off (default) — the intention_rollup_tick
        # then no-ops. Propose-only: rolled-up goals go to a HITL queue; an approved one
        # becomes an ACTIVE identity item (injected into context).
        self.intention_analyst = intention_analyst
        # ADR-0086: runtime override for the agent self-management tools (None = use the env
        # gate). Flipped by the experiment console; consulted per turn in `_core_for`.
        self._self_management_override: bool | None = None

    @classmethod
    def from_env(cls, host: LearningHost) -> LearningControls:
        """Build each capability its env gate turns on, on the host's tier router."""
        analyst = _build_learning_analyst(tier_router=host.tier_router)
        return cls(
            host,
            learning_analyst=analyst[0] if analyst else None,
            learning_analyst_model=analyst[1] if analyst else "unknown",
            behavior_miner=_build_behavior_miner(tier_router=host.tier_router),
            intention_analyst=_build_intention_analyst(tier_router=host.tier_router),
        )

    def learning_flags(self) -> dict[str, dict[str, bool]]:
        """Current on/off state of the hot-toggleable learning capabilities (ADR-0083).

        ``enabled`` is the live runtime state; ``env_default`` is what a process restart
        would revert to (the env gate), so the experiment console can show "you've turned
        this on for now, but it's off in the config."
        """
        out: dict[str, dict[str, bool]] = {}
        for name in _LEARNING_FLAGS:
            out[name] = {
                "enabled": self._learning_capability_enabled(name),
                "env_default": env_flag_on(_LEARNING_FLAG_ENV[name]),
            }
        # ADR-0086: the agent self-management tools share the experiment console. Not a
        # miner (nothing to "run now") — a per-turn behaviour flag toggled live here.
        out["self_management"] = {
            "enabled": self.self_management_enabled(),
            "env_default": env_flag_on("IRIS_AGENT_SELF_MANAGEMENT"),
        }
        return out

    def self_management_enabled(self) -> bool:
        """Whether the agent self-management tools (ADR-0086) are live. A runtime override
        (set via the experiment console) wins over the ``IRIS_AGENT_SELF_MANAGEMENT`` env
        gate; ``_core_for`` consults this per turn, so a toggle takes effect on the next turn
        with no restart."""
        if self._self_management_override is not None:
            return self._self_management_override
        return env_flag_on("IRIS_AGENT_SELF_MANAGEMENT")

    def _learning_capability_enabled(self, name: str) -> bool:
        if name == "behavior_miner":
            return self.behavior_miner is not None
        if name == "intention_rollup":
            return self.intention_analyst is not None
        if name == "learning_analyst":
            return self.learning_analyst is not None
        if name == "self_management":
            return self.self_management_enabled()
        raise ValueError(f"unknown learning flag: {name}")

    def set_learning_flag(self, name: str, enabled: bool) -> bool:
        """Hot-toggle a learning capability WITHOUT a restart (ADR-0083).

        Rebuilds (enable) or nulls (disable) the runtime component; the heartbeats already
        gate on the component being present, so the change takes effect on the next tick or
        an explicit :meth:`run_learning_now`. Process-scoped — a real restart reverts to the
        env default. Returns the resulting state (may stay off if the LLM can't be built).
        """
        if name == "behavior_miner":
            self.behavior_miner = (
                _build_learning_invoke(tier_router=self._host.tier_router, label="behavior miner")
                if enabled
                else None
            )
        elif name == "intention_rollup":
            self.intention_analyst = (
                _build_learning_invoke(
                    tier_router=self._host.tier_router, label="intention analyst"
                )
                if enabled
                else None
            )
        elif name == "learning_analyst":
            if enabled:
                self.learning_analyst = _build_learning_invoke(
                    tier_router=self._host.tier_router, label="learning analyst"
                )
                try:
                    cfg = self._host.tier_router.get_llm_config("task_planning")
                    self.learning_analyst_model = str(getattr(cfg, "model", "") or "unknown")
                except Exception:  # noqa: BLE001
                    self.learning_analyst_model = "unknown"
            else:
                self.learning_analyst = None
        elif name == "self_management":
            # No component to rebuild — just flip the override the per-turn check reads.
            self._self_management_override = enabled
        else:
            raise ValueError(f"unknown learning flag: {name}")
        state = self._learning_capability_enabled(name)
        logger.info("learning flag %s -> %s (runtime override)", name, "on" if state else "off")
        return state

    def run_learning_now(self, name: str) -> int:
        """Run a learning capability immediately instead of waiting for its heartbeat
        (ADR-0083) — the fast half of the tweak loop. Returns the count it produced
        (proposals mined / rolled up / recommendations). 0 if the capability is off."""
        if name == "behavior_miner":
            return self._run_behavior_mining()
        if name == "intention_rollup":
            return self._run_intention_rollup()
        if name == "learning_analyst":
            return self._run_learning_analysis()
        if name == "self_management":
            return 0  # a behaviour flag, not a miner — nothing to run
        raise ValueError(f"unknown learning flag: {name}")

    def _learning_invoke_for_preview(self, label: str) -> Callable[[str, str], str] | None:
        """A miner ``invoke`` for an explicit preview — built even when the gate is off.

        Previewing is the user's way to eyeball proposal quality on their REAL data before
        committing to the 24h tick, so it must work without ``IRIS_BEHAVIOR_MINER`` /
        ``IRIS_INTENTION_ROLLUP`` set. Reuses the already-wired miner when present.
        """
        existing = self.behavior_miner if label == "behavior miner" else self.intention_analyst
        if existing is not None:
            return existing
        return _build_learning_invoke(tier_router=self._host.tier_router, label=label)

    def preview_behavior_mining(self) -> dict[str, Any]:
        """Mine behavior patterns from recent turns and return what WOULD be proposed,
        WITHOUT persisting anything. The dry-run validation surface for layer 1: run it on
        real history, judge the proposals, then decide whether to enable the miner."""
        invoke = self._learning_invoke_for_preview("behavior miner")
        if invoke is None:
            return {"available": False, "reason": "miner LLM unavailable", "patterns": []}
        from iris_harness.services.learning.behavior_miner import (
            dedupe_semantically,
            mine_behavior_patterns,
        )

        try:
            rows = self._host.memory_store.load_turns_since(0)
            turns = [(role, content) for (_id, _sid, role, content) in rows[-80:]]
            patterns = mine_behavior_patterns(turns, invoke=invoke)
            if patterns and self._host.semantic_index is not None:
                from iris_harness.memory.identity.loader import (
                    list_episodic_patterns,
                )

                known = [p.text for p in list_episodic_patterns()]
                for st in ("pending", "approved", "rejected"):
                    known += [
                        pr.text
                        for pr in self._host.learning_store.list_behavior_proposals(status=st)
                    ]
                patterns = dedupe_semantically(
                    patterns, known, embed=self._host.semantic_index.embed
                )
            return {
                "available": True,
                "turns_considered": len(turns),
                "count": len(patterns),
                "patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "text": p.text,
                        "confidence": p.confidence,
                        "evidence": list(p.evidence),
                    }
                    for p in patterns
                ],
            }
        except Exception:  # preview is best-effort, never raises
            logger.debug("behavior-mining preview failed", exc_info=True)
            return {"available": False, "reason": "preview failed", "patterns": []}

    def preview_intention_rollup(self) -> dict[str, Any]:
        """Roll the user's activity up into intentions and return what WOULD be proposed,
        WITHOUT persisting. The dry-run validation surface for layer 3."""
        invoke = self._learning_invoke_for_preview("intention analyst")
        if invoke is None:
            return {"available": False, "reason": "analyst LLM unavailable", "intentions": []}
        from iris_harness.services.learning.intention_rollup import (
            rollup_intentions,
        )

        try:
            context = self._gather_intention_context()
            intentions = self._dedupe_intentions_semantically(
                rollup_intentions(context, invoke=invoke)
            )
            return {
                "available": True,
                "count": len(intentions),
                "intentions": [
                    {
                        "intention_id": i.intention_id,
                        "title": i.title,
                        "summary": i.summary,
                        "supporting": list(i.supporting),
                    }
                    for i in intentions
                ],
            }
        except Exception:  # preview is best-effort, never raises
            logger.debug("intention-rollup preview failed", exc_info=True)
            return {"available": False, "reason": "preview failed", "intentions": []}

    def _run_learning_analysis(self) -> int:
        """Run the agentic learning analyst over the measured report (ADR-0069 #4 s2).

        Opt-in (``IRIS_LEARNING_ANALYST``, off by default). Measures via
        ``build_intelligence``, asks the analyst to interpret, and persists the
        latest advisory analysis. It NEVER applies a recommendation — the user (or,
        later, the experiment loop) acts on them. Best-effort; returns the number
        of recommendations produced (0 when disabled or nothing actionable).
        """
        if self.learning_analyst is None:
            return 0
        from iris_harness.services.learning.analyst import analyze_learning
        from iris_harness.services.learning.intelligence import build_intelligence

        try:
            report = build_intelligence(self._host.learning_store)
            analysis = analyze_learning(
                report, invoke=self.learning_analyst, model=self.learning_analyst_model
            )
            if analysis is None:
                return 0
            self._host.learning_store.save_analysis(analysis.as_dict())
            self._host.signal_collector.record_metric(
                metric_name="learning_analysis",
                value=float(len(analysis.recommendations)),
                success=True,
                metadata={
                    "model": analysis.model,
                    "recommendations": len(analysis.recommendations),
                },
            )
            return len(analysis.recommendations)
        except Exception:  # analysis is advisory, never fatal
            logger.debug("learning analysis failed", exc_info=True)
            return 0

    def learning_analysis_heartbeat(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        """Heartbeat adapter for the learning analyst (interpret + persist, propose-only)."""
        count = self._run_learning_analysis()
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            finished_at=datetime.now(UTC),
            output=json.dumps({"recommendations": count}, sort_keys=True),
        )

    def _run_behavior_mining(self) -> int:
        """Mine recurring behavior patterns from recent turns; queue them for review.

        Opt-in (``IRIS_BEHAVIOR_MINER``, off by default). PROPOSE-ONLY: mined patterns
        land in the HITL queue (``learning_store`` proposals); an approved pattern becomes
        a durable episodic pattern only via ``iris behaviors approve``. The proposals
        table's primary key dedups across pending/approved/rejected, so a recurring habit
        is never re-queued. Mines the recent window AND any span conversation compaction
        archived since the last run (ADR-0082), so a long session's oldest turns are mined
        before they scroll out of the recency window. Best-effort; returns NEW proposals.
        """
        if self.behavior_miner is None:
            return 0
        from iris_harness.services.learning.behavior_miner import (
            dedupe_semantically,
            mine_behavior_patterns,
        )

        try:
            rows = self._host.memory_store.load_turns_since(0)
            # last ~80 turns of cross-session activity is plenty to spot a habit.
            turns = [(role, content) for (_id, _sid, role, content) in rows[-80:]]
            patterns = mine_behavior_patterns(turns, invoke=self.behavior_miner)
            # Learn before you forget (ADR-0082): also mine the span that conversation
            # compaction archived, as its OWN pass — it may have scrolled out of the recent
            # window above, and render_activity keeps only the tail, so a combined pass would
            # truncate it away. Drain the buffer once consumed.
            archived = self._host.sessions.drain_compaction_archive()
            if archived:
                archived_turns = [(t.role, t.content) for t in archived]
                patterns = patterns + mine_behavior_patterns(
                    archived_turns, invoke=self.behavior_miner
                )
                logger.info("behavior mining drained %d archived turns", len(archived))
            # Semantic dedup: drop paraphrased duplicates of each other and of patterns
            # already known (approved episodic + any prior proposal), beyond the exact-id
            # dedup the store does. Shared MiniLM embedder; no-op if unavailable.
            if patterns and self._host.semantic_index is not None:
                from iris_harness.memory.identity.loader import (
                    list_episodic_patterns,
                )

                known = [p.text for p in list_episodic_patterns()]
                for st in ("pending", "approved", "rejected"):
                    known += [
                        pr.text
                        for pr in self._host.learning_store.list_behavior_proposals(status=st)
                    ]
                patterns = dedupe_semantically(
                    patterns, known, embed=self._host.semantic_index.embed
                )
            created = 0
            for p in patterns:
                if self._host.learning_store.propose_behavior_pattern(
                    p.pattern_id, p.text, p.confidence, list(p.evidence)
                ):
                    created += 1
            if created:
                self._host.signal_collector.record_metric(
                    metric_name="behavior_mining",
                    value=float(created),
                    success=True,
                    metadata={"proposed": created},
                )
            return created
        except Exception:  # mining is advisory, never fatal
            logger.debug("behavior mining failed", exc_info=True)
            return 0

    def behavior_mining_heartbeat(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        """Heartbeat adapter for behavior mining (propose-only; HITL review queue)."""
        count = self._run_behavior_mining()
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            finished_at=datetime.now(UTC),
            output=json.dumps({"proposed": count}, sort_keys=True),
        )

    def _run_intention_rollup(self) -> int:
        """Roll the user's tasks/routines/habits/signals up into higher-level intentions;
        queue them for review.

        Opt-in (``IRIS_INTENTION_ROLLUP``, off by default). PROPOSE-ONLY: rolled-up goals
        land in the HITL queue (``learning_store`` intentions); an approved one becomes an
        ACTIVE identity item only via ``iris intentions approve``. The intentions table PK
        dedups exact re-titles across statuses; a semantic pass additionally drops
        paraphrased re-rolls of goals the user already saw (proposed / active / dismissed)
        or that are already active in ``active.md``, so a dismissed goal doesn't return to
        the queue each cycle under a slightly different wording. Best-effort; returns the
        number of NEW proposals.
        """
        if self.intention_analyst is None:
            return 0
        from iris_harness.services.learning.intention_rollup import (
            rollup_intentions,
        )

        try:
            context = self._gather_intention_context()
            intentions = rollup_intentions(context, invoke=self.intention_analyst)
            intentions = self._dedupe_intentions_semantically(intentions)
            created = 0
            for i in intentions:
                if self._host.learning_store.propose_intention(
                    i.intention_id, i.title, i.summary, list(i.supporting)
                ):
                    created += 1
            if created:
                self._host.signal_collector.record_metric(
                    metric_name="intention_rollup",
                    value=float(created),
                    success=True,
                    metadata={"proposed": created},
                )
            return created
        except Exception:  # rollup is advisory, never fatal
            logger.debug("intention rollup failed", exc_info=True)
            return 0

    def intention_rollup_heartbeat(self, definition: HeartbeatDefinition) -> HeartbeatRun:
        """Heartbeat adapter for the intention rollup (propose-only; HITL review queue)."""
        count = self._run_intention_rollup()
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            finished_at=datetime.now(UTC),
            output=json.dumps({"proposed": count}, sort_keys=True),
        )

    def _dedupe_intentions_semantically(self, intentions: list[Any]) -> list[Any]:
        """Drop rolled-up intentions that paraphrase a goal the user already saw.

        Beyond the store's exact-title PK dedup, this filters candidates whose title is a
        semantic near-duplicate of an existing intention (proposed / active / dismissed) or
        an item already in ``active.md`` — so a dismissed goal isn't re-queued each cycle
        under a slightly different wording. Best-effort; a no-op without a semantic index.
        """
        if not intentions or self._host.semantic_index is None:
            return intentions
        try:
            from iris_harness.memory.identity.loader import list_active_items
            from iris_harness.services.learning.dedup import dedupe_by_text

            known = [
                pr.title
                for st in ("proposed", "active", "dismissed")
                for pr in self._host.learning_store.list_intentions(status=st)
            ]
            known += [it.text for it in list_active_items()]
            return dedupe_by_text(
                intentions,
                key=lambda i: i.title,
                existing_texts=known,
                embed=self._host.semantic_index.embed,
            )
        except Exception:  # noqa: BLE001 — dedup is best-effort, never blocks rollup
            logger.debug("intention semantic dedup failed; passing candidates through")
            return intentions

    def _gather_intention_context(self) -> str:
        """A compact snapshot of the user's activity for the intention analyst — open
        tasks, active routines, confirmed habits, and how they steer. Best-effort."""
        lines: list[str] = []
        try:
            from iris_harness.services.tasks.store import TaskStore

            tasks = TaskStore(db_path=self._host.data_dir / "tasks.db").list(
                status="open", limit=30
            )
            if tasks:
                lines.append("Open tasks:\n" + "\n".join(f"- {t.title}" for t in tasks))
        except Exception:
            logger.debug("intention context: tasks unavailable", exc_info=True)
        try:
            from iris_harness.services.routines.models import RoutineApprovalStatus

            routines = [
                r
                for r in self._host.routine_store.list_all()
                if r.approval_status != RoutineApprovalStatus.RETIRED
            ]
            if routines:
                lines.append(
                    "Active routines:\n"
                    + "\n".join(f"- {r.title}: {r.goal}" for r in routines[:20])
                )
        except Exception:
            logger.debug("intention context: routines unavailable", exc_info=True)
        try:
            from iris_harness.memory.identity.loader import list_episodic_patterns

            habits = [p.text for p in list_episodic_patterns()]
            if habits:
                lines.append("Confirmed habits:\n" + "\n".join(f"- {h}" for h in habits[:20]))
        except Exception:
            logger.debug("intention context: habits unavailable", exc_info=True)
        try:
            summary = self._host.learning_store.user_behavior_summary()
            if summary:
                lines.append(
                    "How the user steers (signal counts): "
                    + ", ".join(f"{k}={v}" for k, v in summary.items())
                )
        except Exception:
            logger.debug("intention context: signals unavailable", exc_info=True)
        return "\n\n".join(lines)


def _build_learning_analyst(
    *, tier_router: TierRouter
) -> tuple[Callable[[str, str], str], str] | None:
    """Build the optional agentic learning analyst (ADR-0069 #4 slice 2).

    Opt-in via ``IRIS_LEARNING_ANALYST``; default off. Returns ``(invoke, model)``
    where ``invoke(system, user) -> str`` is a governed LLM call on a capable LOCAL
    tier (the report is internal telemetry, so it stays on-box), or ``None`` when
    disabled or unbuildable. The analyst only interprets — it never acts.
    """
    enabled = os.getenv("IRIS_LEARNING_ANALYST", "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None
    try:
        from iris_harness.llm.client import CodingLLMClient

        cfg = cast(Any, tier_router.get_llm_config("task_planning"))
        cfg = cfg.model_copy(update={"temperature": 0.2, "max_tokens": 768})
        client = CodingLLMClient(cfg, governance_agent_type="chat")
        model = str(getattr(cfg, "model", "") or "unknown")
    except Exception:
        logger.debug("learning analyst disabled: could not init LLM client", exc_info=True)
        return None

    def invoke(system: str, user: str) -> str:
        return client.invoke(system_prompt=system, user_prompt=user)

    return invoke, model


def _build_learning_invoke(
    *, tier_router: TierRouter, label: str
) -> Callable[[str, str], str] | None:
    """Build a governed LOCAL-tier ``invoke(system, user) -> str`` for a learning miner.

    The user's transcript/activity is personal, so these run on-box (LOCAL tier). Shared by
    the behavior miner and intention analyst; the gate is applied by the callers so this
    helper can also build a miner UNCONDITIONALLY for an explicit, non-persisting preview
    (``IrisRuntime.preview_*``). Returns ``None`` only when the client can't be built.
    """
    try:
        from iris_harness.llm.client import CodingLLMClient

        cfg = cast(Any, tier_router.get_llm_config("task_planning"))
        cfg = cfg.model_copy(update={"temperature": 0.2, "max_tokens": 768})
        client = CodingLLMClient(cfg, governance_agent_type="chat")
    except Exception:
        logger.debug("%s disabled: could not init LLM client", label, exc_info=True)
        return None

    def invoke(system: str, user: str) -> str:
        return client.invoke(system_prompt=system, user_prompt=user)

    return invoke


def _build_behavior_miner(*, tier_router: TierRouter) -> Callable[[str, str], str] | None:
    """Build the optional behavior-pattern miner (digital-twin layer 1).

    Opt-in via ``IRIS_BEHAVIOR_MINER``; default off. Returns a governed ``invoke`` or
    ``None`` when disabled/unbuildable. Mining only PROPOSES; approval is the user's via
    ``iris behaviors``.
    """
    enabled = os.getenv("IRIS_BEHAVIOR_MINER", "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None
    return _build_learning_invoke(tier_router=tier_router, label="behavior miner")


def _build_intention_analyst(*, tier_router: TierRouter) -> Callable[[str, str], str] | None:
    """Build the optional intention-rollup analyst (digital-twin layer 3).

    Opt-in via ``IRIS_INTENTION_ROLLUP``; default off. Returns a governed ``invoke`` or
    ``None`` when disabled/unbuildable. Rollup only PROPOSES; approval is the user's via
    ``iris intentions``.
    """
    enabled = os.getenv("IRIS_INTENTION_ROLLUP", "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None
    return _build_learning_invoke(tier_router=tier_router, label="intention analyst")
