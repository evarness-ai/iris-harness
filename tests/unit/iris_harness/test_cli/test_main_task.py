"""Tests for ``iris task`` and ``iris goal`` (Phase 2 Track 2A — ADR-0005)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.main import app
from iris_harness.services.tasks.store import TaskStore


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "tasks.db"


def _ids(output: str) -> list[str]:
    """Extract 8-char id prefixes from Rich-rendered table rows."""
    return [
        line.strip().split()[1]
        for line in output.splitlines()
        if line.strip().startswith("│ ")
        and len(line.strip().split()) >= 2
        and all(c in "0123456789abcdef" for c in line.strip().split()[1])
    ]


# ---------------------------------------------------------------------------
# task add / list / show
# ---------------------------------------------------------------------------


def test_task_add_creates_task_and_persists(runner: CliRunner, db: Path) -> None:
    result = runner.invoke(
        app,
        ["task", "add", "Buy groceries", "--priority", "5", "--tasks-db", str(db)],
    )
    assert result.exit_code == 0, result.output
    assert "created task" in result.output

    store = TaskStore(db_path=db)
    store.ensure_schema()
    tasks = store.list(status=None)
    assert len(tasks) == 1
    assert tasks[0].title == "Buy groceries"
    assert tasks[0].priority == 5
    assert tasks[0].source_kind == "manual"


def test_task_list_default_shows_only_open(runner: CliRunner, db: Path) -> None:
    runner.invoke(app, ["task", "add", "Open A", "--tasks-db", str(db)])
    runner.invoke(app, ["task", "add", "Open B", "--tasks-db", str(db)])

    list_result = runner.invoke(app, ["task", "list", "--tasks-db", str(db)])
    assert list_result.exit_code == 0
    ids = _ids(list_result.output)
    assert len(ids) == 2

    # Complete one and verify it disappears from the default (open) view.
    runner.invoke(app, ["task", "complete", ids[0], "--tasks-db", str(db)])
    list2 = runner.invoke(app, ["task", "list", "--tasks-db", str(db)])
    assert len(_ids(list2.output)) == 1

    # Status=all surfaces both.
    list_all = runner.invoke(app, ["task", "list", "--status", "all", "--tasks-db", str(db)])
    assert len(_ids(list_all.output)) == 2


def test_task_show_renders_detail(runner: CliRunner, db: Path) -> None:
    runner.invoke(
        app,
        [
            "task",
            "add",
            "Pay rent",
            "--description",
            "due first of the month",
            "--due",
            "2026-07-01",
            "--priority",
            "3",
            "--tasks-db",
            str(db),
        ],
    )
    list_result = runner.invoke(app, ["task", "list", "--tasks-db", str(db)])
    task_id = _ids(list_result.output)[0]

    show = runner.invoke(app, ["task", "show", task_id, "--tasks-db", str(db)])
    assert show.exit_code == 0, show.output
    assert "Pay rent" in show.output
    assert "due first of the month" in show.output
    assert "2026-07-01" in show.output
    assert "priority:" in show.output


# ---------------------------------------------------------------------------
# task complete / drop / update
# ---------------------------------------------------------------------------


def test_task_complete_marks_done_and_sets_completed_at(runner: CliRunner, db: Path) -> None:
    runner.invoke(app, ["task", "add", "X", "--tasks-db", str(db)])
    list_result = runner.invoke(app, ["task", "list", "--tasks-db", str(db)])
    tid = _ids(list_result.output)[0]

    result = runner.invoke(app, ["task", "complete", tid, "--tasks-db", str(db)])
    assert result.exit_code == 0
    assert "completed" in result.output

    store = TaskStore(db_path=db)
    [task] = store.list(status=None)
    assert task.status == "done"
    assert task.completed_at is not None


def test_task_drop_marks_dropped_without_completed_at(runner: CliRunner, db: Path) -> None:
    runner.invoke(app, ["task", "add", "Y", "--tasks-db", str(db)])
    tid = _ids(runner.invoke(app, ["task", "list", "--tasks-db", str(db)]).output)[0]

    result = runner.invoke(app, ["task", "drop", tid, "--tasks-db", str(db)])
    assert result.exit_code == 0
    assert "dropped" in result.output

    store = TaskStore(db_path=db)
    [task] = store.list(status=None)
    assert task.status == "dropped"
    assert task.completed_at is None  # ADR-0014 decision #2


def test_task_update_changes_title_and_priority(runner: CliRunner, db: Path) -> None:
    runner.invoke(app, ["task", "add", "Old", "--tasks-db", str(db)])
    tid = _ids(runner.invoke(app, ["task", "list", "--tasks-db", str(db)]).output)[0]

    result = runner.invoke(
        app,
        [
            "task",
            "update",
            tid,
            "--title",
            "New",
            "--priority",
            "9",
            "--tasks-db",
            str(db),
        ],
    )
    assert result.exit_code == 0

    store = TaskStore(db_path=db)
    [task] = store.list(status=None)
    assert task.title == "New"
    assert task.priority == 9


def test_task_update_rejects_terminal_status_via_update(runner: CliRunner, db: Path) -> None:
    """ADR-0014 decision #8 — terminal transitions go through complete/drop."""
    runner.invoke(app, ["task", "add", "Z", "--tasks-db", str(db)])
    tid = _ids(runner.invoke(app, ["task", "list", "--tasks-db", str(db)]).output)[0]

    result = runner.invoke(
        app,
        ["task", "update", tid, "--status", "done", "--tasks-db", str(db)],
    )
    assert result.exit_code == 2
    assert "complete" in result.output


def test_task_update_clear_due_removes_due_at(runner: CliRunner, db: Path) -> None:
    runner.invoke(app, ["task", "add", "DueOne", "--due", "2026-07-01", "--tasks-db", str(db)])
    tid = _ids(runner.invoke(app, ["task", "list", "--tasks-db", str(db)]).output)[0]

    result = runner.invoke(app, ["task", "update", tid, "--clear-due", "--tasks-db", str(db)])
    assert result.exit_code == 0

    store = TaskStore(db_path=db)
    [task] = store.list(status=None)
    assert task.due_at is None


# ---------------------------------------------------------------------------
# task group
# ---------------------------------------------------------------------------


def test_task_group_creates_parent_and_reparents_children(runner: CliRunner, db: Path) -> None:
    runner.invoke(app, ["task", "add", "Child A", "--tasks-db", str(db)])
    runner.invoke(app, ["task", "add", "Child B", "--tasks-db", str(db)])
    ids = _ids(runner.invoke(app, ["task", "list", "--tasks-db", str(db)]).output)
    assert len(ids) == 2

    result = runner.invoke(
        app,
        [
            "task",
            "group",
            ids[0],
            ids[1],
            "--under",
            "Errands",
            "--tasks-db",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "grouped 2" in result.output

    store = TaskStore(db_path=db)
    tasks = store.list(status=None)
    titles = {t.title for t in tasks}
    assert "Errands" in titles
    parent = next(t for t in tasks if t.title == "Errands")
    children = [t for t in tasks if t.parent_task_id == parent.id]
    assert len(children) == 2


def test_task_group_requires_two_ids(runner: CliRunner, db: Path) -> None:
    runner.invoke(app, ["task", "add", "Solo", "--tasks-db", str(db)])
    [tid] = _ids(runner.invoke(app, ["task", "list", "--tasks-db", str(db)]).output)

    result = runner.invoke(
        app,
        ["task", "group", tid, "--under", "One", "--tasks-db", str(db)],
    )
    assert result.exit_code == 2
    assert "at least two" in result.output


# ---------------------------------------------------------------------------
# id prefix resolution
# ---------------------------------------------------------------------------


def test_prefix_too_short_is_rejected(runner: CliRunner, db: Path) -> None:
    runner.invoke(app, ["task", "add", "T", "--tasks-db", str(db)])
    result = runner.invoke(app, ["task", "complete", "ab", "--tasks-db", str(db)])
    assert result.exit_code == 2
    assert "at least 4" in result.output


def test_prefix_no_match_returns_exit_1(runner: CliRunner, db: Path) -> None:
    runner.invoke(app, ["task", "add", "T", "--tasks-db", str(db)])
    result = runner.invoke(app, ["task", "complete", "deadbeef", "--tasks-db", str(db)])
    assert result.exit_code == 1
    assert "no task matching" in result.output


# ---------------------------------------------------------------------------
# goal commands
# ---------------------------------------------------------------------------


def test_goal_add_and_list(runner: CliRunner, db: Path) -> None:
    runner.invoke(
        app,
        [
            "goal",
            "add",
            "Get fit",
            "--target",
            "2026-12-31",
            "--criteria",
            "5k under 25min",
            "--tasks-db",
            str(db),
        ],
    )
    list_result = runner.invoke(app, ["goal", "list", "--tasks-db", str(db)])
    assert list_result.exit_code == 0
    assert "Get fit" in list_result.output
    assert "2026-12-31" in list_result.output


def test_goal_show_lists_linked_tasks(runner: CliRunner, db: Path) -> None:
    runner.invoke(app, ["goal", "add", "Fitness", "--tasks-db", str(db)])
    goal_id = _ids(runner.invoke(app, ["goal", "list", "--tasks-db", str(db)]).output)[0]

    runner.invoke(app, ["task", "add", "Run", "--goal", goal_id, "--tasks-db", str(db)])
    runner.invoke(app, ["task", "add", "Stretch", "--goal", goal_id, "--tasks-db", str(db)])

    show = runner.invoke(app, ["goal", "show", goal_id, "--tasks-db", str(db)])
    assert show.exit_code == 0, show.output
    assert "tasks under this goal: 2" in show.output
    assert "Run" in show.output
    assert "Stretch" in show.output


def test_goal_achieve_marks_status(runner: CliRunner, db: Path) -> None:
    runner.invoke(app, ["goal", "add", "Done goal", "--tasks-db", str(db)])
    goal_id = _ids(runner.invoke(app, ["goal", "list", "--tasks-db", str(db)]).output)[0]

    result = runner.invoke(app, ["goal", "achieve", goal_id, "--tasks-db", str(db)])
    assert result.exit_code == 0
    assert "achieved" in result.output

    store = TaskStore(db_path=db)
    [g] = store.list_goals(status="achieved")
    assert g.title == "Done goal"
    assert g.completed_at is not None
