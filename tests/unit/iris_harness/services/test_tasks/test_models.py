"""Tests for the Task / Goal / WaitFor Pydantic models (ADR-0005)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from iris_harness.services.tasks import Goal, Task, WaitFor


def _now() -> datetime:
    return datetime.now(UTC)


# ─── Task ─────────────────────────────────────────────────────────────


def test_task_minimal_uses_safe_defaults() -> None:
    t = Task(id="t1", title="Reply to Bob")

    assert t.status == "open"
    assert t.priority == 0
    assert t.source_kind is None
    assert t.parent_task_id is None
    assert t.dedup_key is None
    assert t.calendar_visible is False
    assert t.related_wikilinks == ()
    assert t.completed_at is None


def test_task_requires_title() -> None:
    with pytest.raises(ValidationError):
        Task(id="t1", title="")


def test_task_done_requires_completed_at() -> None:
    """status='done' must have completed_at set (model_validator)."""
    with pytest.raises(ValidationError):
        Task(id="t1", title="x", status="done")


def test_task_completed_at_forbidden_when_not_done() -> None:
    """completed_at may not be set when status is not 'done'."""
    with pytest.raises(ValidationError):
        Task(id="t1", title="x", status="open", completed_at=_now())


def test_task_dropped_does_not_set_completed_at() -> None:
    """'dropped' is terminal but uses status only, not completed_at."""
    t = Task(id="t1", title="x", status="dropped")
    assert t.status == "dropped"
    assert t.completed_at is None


def test_task_bad_status_rejected() -> None:
    with pytest.raises(ValidationError):
        Task(id="t1", title="x", status="bogus")  # type: ignore[arg-type]


def test_task_bad_source_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        Task(id="t1", title="x", source_kind="not-real")  # type: ignore[arg-type]


def test_task_wait_for_roundtrip() -> None:
    wf = WaitFor(kind="reply_from", payload={"address": "bob@example.com"})
    t = Task(id="t1", title="Reply followup", wait_for=wf)

    assert t.wait_for is not None
    assert t.wait_for.kind == "reply_from"
    assert t.wait_for.payload == {"address": "bob@example.com"}


def test_task_bad_wait_for_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        WaitFor(kind="not_a_kind")  # type: ignore[arg-type]


# ─── Goal ─────────────────────────────────────────────────────────────


def test_goal_minimal_uses_safe_defaults() -> None:
    g = Goal(id="g1", title="Save $5K")

    assert g.status == "active"
    assert g.target_date is None
    assert g.success_criteria == ""
    assert g.completed_at is None


def test_goal_achieved_requires_completed_at() -> None:
    with pytest.raises(ValidationError):
        Goal(id="g1", title="x", status="achieved")


def test_goal_completed_at_forbidden_when_not_achieved() -> None:
    with pytest.raises(ValidationError):
        Goal(id="g1", title="x", status="active", completed_at=_now())


def test_goal_bad_status_rejected() -> None:
    with pytest.raises(ValidationError):
        Goal(id="g1", title="x", status="bogus")  # type: ignore[arg-type]


# ─── TaskAction (ADR-0073) ────────────────────────────────────────────


def test_copy_command_action_requires_command() -> None:
    import pytest

    from iris_harness.services.tasks import TaskAction

    with pytest.raises(ValueError):
        TaskAction(kind="copy_command", label="Set password")  # no command


def test_copy_command_action_cannot_be_safe() -> None:
    import pytest

    from iris_harness.services.tasks import TaskAction

    with pytest.raises(ValueError):
        TaskAction(kind="copy_command", label="x", command="do it", safe=True)


def test_safe_action_ok() -> None:
    from iris_harness.services.tasks import TaskAction

    a = TaskAction(kind="re_extract", label="Re-extract", target_id="s1", safe=True)
    assert a.safe and a.kind == "re_extract"
