"""ADR-0111: a dependent sub-task receives its dependencies' results."""

from __future__ import annotations

from iris_harness.agent.agent_executor import AgentTask
from iris_harness.runtime.turn.stages.execute import with_upstream


def test_a_dependent_task_sees_the_request_its_step_and_upstream_outputs() -> None:
    task = AgentTask(
        query="schedule a reminder for each due",
        agent_type="calendar",
        params={
            "query": "schedule a reminder for each due",
            "request": "find the dues, then remind me",
        },
    )
    bound = with_upstream(task, upstream=[("t1", "Wingtip 3,150.40 due 2026-09-30\n")])
    assert bound.query.splitlines() == [
        "Overall request: find the dues, then remind me",
        "Your step: schedule a reminder for each due",
        "Results from the earlier steps this one depends on:",
        "[t1] Wingtip 3,150.40 due 2026-09-30",
    ]
    assert bound.agent_type == "calendar" and bound.params == task.params


def test_a_task_with_nothing_upstream_is_untouched() -> None:
    task = AgentTask(query="q", agent_type="system", params={"query": "q"})
    assert with_upstream(task, upstream=[]) is task
