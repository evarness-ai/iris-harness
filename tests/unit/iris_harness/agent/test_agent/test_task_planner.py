"""Behavioral tests for the task planner."""

from __future__ import annotations

from iris_harness.agent.task_planner import SubTask, TaskPlan, TaskPlanner, _parse_task_lines


def test_direct_plan_produces_single_task() -> None:
    planner = TaskPlanner()
    plan = planner.plan("What is the weather?", is_multi_step=False)

    assert plan.is_simple
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "t1"


def test_execution_groups_single_task_is_one_group() -> None:
    plan = TaskPlan(
        query="test",
        tasks=[SubTask(id="t1", action="answer", agent_type="system")],
    )
    groups = plan.execution_groups()

    assert len(groups) == 1
    assert groups[0][0].id == "t1"


def test_execution_groups_parallel_independent_tasks() -> None:
    plan = TaskPlan(
        query="test",
        tasks=[
            SubTask(id="t1", action="fetch_emails", agent_type="email"),
            SubTask(id="t2", action="check_portfolio", agent_type="finance"),
            SubTask(id="t3", action="summarize", agent_type="llm", depends_on=["t1"]),
        ],
    )
    groups = plan.execution_groups()

    assert len(groups) == 2
    first_ids = {t.id for t in groups[0]}
    assert first_ids == {"t1", "t2"}
    assert groups[1][0].id == "t3"


def test_parse_task_lines_extracts_tasks_with_dependencies() -> None:
    raw = (
        "TASK: t1 | fetch_emails | email | none\n"
        "TASK: t2 | summarize | llm | t1\n"
        "TASK: t3 | check_portfolio | finance | none\n"
    )
    tasks = _parse_task_lines(raw, "query")

    assert len(tasks) == 3
    assert tasks[1].depends_on == ["t1"]
    assert tasks[0].depends_on == []


def test_llm_planner_falls_back_to_direct_on_bad_output() -> None:
    def bad_llm(prompt: str) -> str:
        return "this is not a valid task list"

    planner = TaskPlanner(llm_call=bad_llm)
    plan = planner.plan("complex query", is_multi_step=True)

    assert plan.is_simple


def test_planner_direct_plan_uses_default_agent_type() -> None:
    planner = TaskPlanner()
    plan = planner.plan(
        "need help",
        is_multi_step=False,
        default_agent_type="code_exec",
    )

    assert plan.tasks[0].agent_type == "code_exec"


def test_parse_task_lines_normalizes_unregistered_agents_to_default() -> None:
    raw = "TASK: t1 | do work | filemanager | none\n"
    tasks = _parse_task_lines(
        raw,
        "query",
        allowed_agent_types={"system", "code_exec"},
        default_agent_type="code_exec",
    )

    assert len(tasks) == 1
    assert tasks[0].agent_type == "code_exec"


# --- ADR-0111: a sub-task runs its own action; a same-agent chain collapses -----------


def test_parsed_subtasks_carry_their_action_and_the_request() -> None:
    raw = "TASK: t1 | find the dues | finance | none\nTASK: t2 | remind me for each | calendar | t1"
    tasks = _parse_task_lines(
        raw, "find the dues, after this remind me", allowed_agent_types={"finance", "calendar"}
    )
    assert [t.params["query"] for t in tasks] == ["find the dues", "remind me for each"]
    assert {t.params["request"] for t in tasks} == {"find the dues, after this remind me"}


def _plan(raw: str, *allowed: str) -> TaskPlan:
    from iris_harness.agent.task_planner import collapse_same_agent_chain

    tasks = _parse_task_lines(
        raw, "q", allowed_agent_types=set(allowed), default_agent_type="system"
    )
    return collapse_same_agent_chain(TaskPlan(query="q", tasks=tasks), default_agent_type="system")


def test_a_chain_on_one_agent_collapses_to_one_task() -> None:
    plan = _plan("TASK: t1 | find | system | none\nTASK: t2 | remind | system | t1", "system")
    assert len(plan.tasks) == 1 and plan.tasks[0].agent_type == "system"
    assert plan.tasks[0].params["query"] == "q"  # the loop gets the whole request


def test_cross_agent_plans_keep_their_shape() -> None:
    plan = _plan(
        "TASK: t1 | inbox | email | none\nTASK: t2 | calendar | calendar | none",
        "email",
        "calendar",
    )
    assert [t.agent_type for t in plan.tasks] == ["email", "calendar"]


def test_parallel_same_agent_waves_keep_their_shape() -> None:
    plan = _plan("TASK: t1 | a | system | none\nTASK: t2 | b | system | none", "system")
    assert len(plan.tasks) == 2
    assert len(plan.execution_groups()[0]) == 2


def test_llm_plan_applies_the_collapse() -> None:
    planner = TaskPlanner(
        llm_call=lambda p: "TASK: t1 | find | system | none\nTASK: t2 | remind | system | t1"
    )
    planner._kernel = None
    plan = planner.plan(
        "q", is_multi_step=True, allowed_agent_types={"system"}, default_agent_type="system"
    )
    assert len(plan.tasks) == 1
