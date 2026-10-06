"""Task decomposition and dependency-aware execution planning."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from iris_harness.kernel.governance import HookContext, HookPoint, kernel_from_env
from iris_harness.kernel.governance.turn_label import apply_turn_floor
from iris_harness.llm.client import GovernedPromptCall


@dataclass
class SubTask:
    """A single executable unit in a task plan."""

    id: str
    action: str
    agent_type: str
    params: dict[str, str] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    result: str | None = None


@dataclass
class TaskPlan:
    """Ordered, dependency-aware execution plan."""

    query: str
    tasks: list[SubTask] = field(default_factory=list)

    @property
    def is_simple(self) -> bool:
        return len(self.tasks) == 1

    def execution_groups(self) -> list[list[SubTask]]:
        """Return tasks grouped by execution wave (topological sort)."""
        completed: set[str] = set()
        groups: list[list[SubTask]] = []
        remaining = list(self.tasks)
        while remaining:
            ready = [t for t in remaining if all(d in completed for d in t.depends_on)]
            if not ready:
                groups.append(remaining)
                break
            groups.append(ready)
            completed.update(t.id for t in ready)
            remaining = [t for t in remaining if t.id not in completed]
        return groups


class TaskPlanner:
    """Decomposes user queries into ordered sub-task plans."""

    def __init__(self, llm_call: Callable[[str], str] | None = None) -> None:
        self._llm = llm_call
        self._kernel = kernel_from_env()

    def plan(
        self,
        query: str,
        *,
        is_multi_step: bool = False,
        allowed_agent_types: set[str] | None = None,
        default_agent_type: str = "system",
    ) -> TaskPlan:
        if not is_multi_step or self._llm is None:
            return self._direct_plan(query, default_agent_type=default_agent_type)
        return self._llm_plan(
            query,
            allowed_agent_types=allowed_agent_types,
            default_agent_type=default_agent_type,
        )

    def _direct_plan(self, query: str, *, default_agent_type: str) -> TaskPlan:
        # Emit a concrete agent type so execution never silently falls through
        # to an unrelated fallback handler.
        return TaskPlan(
            query=query,
            tasks=[
                SubTask(
                    id="t1",
                    action="answer",
                    agent_type=default_agent_type,
                    params={"query": query},
                )
            ],
        )

    def _llm_plan(
        self,
        query: str,
        *,
        allowed_agent_types: set[str] | None,
        default_agent_type: str,
    ) -> TaskPlan:
        assert self._llm is not None
        allowed = sorted(allowed_agent_types or {default_agent_type})
        prompt = (
            "Decompose this request into numbered sub-tasks. For each task write:\n"
            "TASK: <id> | <action> | <agent_type> | <depends_on_comma_separated_or_none>\n\n"
            f"Request: {query}\n\n"
            f"Agent types: {', '.join(allowed)}"
        )
        try:
            raw = self._invoke_with_governance(prompt)
            tasks = _parse_task_lines(
                raw,
                query,
                allowed_agent_types=allowed_agent_types,
                default_agent_type=default_agent_type,
            )
            if tasks:
                return collapse_same_agent_chain(
                    TaskPlan(query=query, tasks=tasks), default_agent_type=default_agent_type
                )
        except Exception:  # noqa: BLE001
            return self._direct_plan(query, default_agent_type=default_agent_type)
        return self._direct_plan(query, default_agent_type=default_agent_type)

    def _invoke_with_governance(self, prompt: str) -> str:
        assert self._llm is not None
        # A GovernedPromptCall governs itself, at the tier it goes to (llm/client.py).
        if self._kernel is None or isinstance(self._llm, GovernedPromptCall):
            return self._llm(prompt)

        run_id = str(uuid.uuid4())
        classify_ctx = HookContext(
            hook_point=HookPoint.PRE_CLASSIFY,
            run_id=run_id,
            agent_type="task_planner",
            payload={"prompt": prompt},
        )
        classify_decision, classified_ctx = self._kernel.fire_sync(
            HookPoint.PRE_CLASSIFY, classify_ctx
        )
        if classify_decision.outcome in ("deny", "require_approval"):
            raise RuntimeError(f"governance blocked planner call: {classify_decision.reason}")

        llm_ctx = HookContext(
            hook_point=HookPoint.PRE_LLM_CALL,
            run_id=run_id,
            agent_type="task_planner",
            # Floored at the turn's label: this prompt may look tamer than the data the
            # turn holds (kernel/governance/turn_label.apply_turn_floor).
            classification=apply_turn_floor(classified_ctx.classification),
            # An opaque callable: where it sends the prompt is unknown, so the call is
            # governed as leaving the machine (fail closed), not guessed to be local.
            tier="tier_3",
            # The callable is opaque: the model behind it is not known here, and the row
            # says so rather than leaving the identity off. (A ``GovernedPromptCall``
            # governs itself and names its model.)
            payload={"prompt": prompt, "model": "unknown", "provider": "unknown"},
        )
        decision, _ = self._kernel.fire_sync(HookPoint.PRE_LLM_CALL, llm_ctx)
        if decision.outcome in ("deny", "require_approval"):
            raise RuntimeError(f"governance blocked planner call: {decision.reason}")

        return self._llm(prompt)


def _parse_task_lines(
    raw: str,
    query: str,
    *,
    allowed_agent_types: set[str] | None = None,
    default_agent_type: str = "system",
) -> list[SubTask]:
    tasks: list[SubTask] = []
    allowed = allowed_agent_types or {default_agent_type}
    for line in raw.splitlines():
        m = re.match(
            r"TASK:\s*(\S+)\s*\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|\s*(.+)?",
            line.strip(),
            re.IGNORECASE,
        )
        if not m:
            continue
        task_id, action, agent_type, deps_raw = m.groups()
        normalized_agent = agent_type.strip().lower()
        if normalized_agent not in allowed:
            normalized_agent = default_agent_type
        deps_str = (deps_raw or "").strip().lower()
        depends_on = (
            [] if deps_str in ("none", "", "-") else [d.strip() for d in deps_str.split(",")]
        )
        tasks.append(
            SubTask(
                id=task_id.strip(),
                action=action.strip(),
                agent_type=normalized_agent,
                # ADR-0111: a sub-task runs its own action; the user's sentence rides
                # along as ``request`` for context (it used to be every task's query).
                params={"query": action.strip(), "request": query},
                depends_on=depends_on,
            )
        )
    return tasks


def collapse_same_agent_chain(plan: TaskPlan, *, default_agent_type: str) -> TaskPlan:
    """ADR-0111: a pure chain on one agent is one task — the loop's own scratchpad
    carries each step's result to the next, and a split would make the later step
    blind. Cross-agent plans and parallel waves keep their shape."""
    if len(plan.tasks) <= 1:
        return plan
    agents = {t.agent_type for t in plan.tasks}
    parallel = any(len(wave) > 1 for wave in plan.execution_groups())
    if len(agents) > 1 or parallel:
        return plan
    agent = next(iter(agents)) or default_agent_type
    return TaskPlan(
        query=plan.query,
        tasks=[SubTask(id="t1", action="answer", agent_type=agent, params={"query": plan.query})],
    )
