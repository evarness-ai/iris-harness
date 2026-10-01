"""Tests for `iris reminder` CLI (Phase 2 Track 2D — ADR-0005 delivery-only)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.main import app
from iris_harness.services.notifications.store import ReminderStore


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


def _create_task(runner: CliRunner, db: Path, title: str = "T") -> str:
    runner.invoke(app, ["task", "add", title, "--tasks-db", str(db)])
    return _ids(runner.invoke(app, ["task", "list", "--tasks-db", str(db)]).output)[0]


def _create_goal(runner: CliRunner, db: Path, title: str = "G") -> str:
    runner.invoke(app, ["goal", "add", title, "--tasks-db", str(db)])
    return _ids(runner.invoke(app, ["goal", "list", "--tasks-db", str(db)]).output)[0]


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


def test_add_resolves_task_target_automatically(runner: CliRunner, db: Path) -> None:
    task_id = _create_task(runner, db)
    result = runner.invoke(
        app,
        [
            "reminder",
            "add",
            task_id,
            "--at",
            "2026-12-01T09:00",
            "--note",
            "ping",
            "--tasks-db",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "created reminder" in result.output

    store = ReminderStore(db_path=db)
    store.ensure_schema()
    [reminder] = store.list()
    assert reminder.target_kind == "task"
    assert reminder.target_id.startswith(task_id)
    assert reminder.note == "ping"


def test_add_resolves_goal_when_only_goal_matches(runner: CliRunner, db: Path) -> None:
    goal_id = _create_goal(runner, db)
    result = runner.invoke(
        app,
        ["reminder", "add", goal_id, "--at", "2026-12-01", "--tasks-db", str(db)],
    )
    assert result.exit_code == 0, result.output

    store = ReminderStore(db_path=db)
    store.ensure_schema()
    [reminder] = store.list()
    assert reminder.target_kind == "goal"


def test_add_requires_explicit_kind_when_prefix_matches_both(runner: CliRunner, db: Path) -> None:
    """A task id and a goal id sharing a 4-char prefix should force --kind."""
    # Seed many tasks and goals so we can find a shared prefix; statistically
    # certain to fail without disambiguation. Instead just inject directly.
    from iris_harness.services.tasks.store import TaskStore

    store = TaskStore(db_path=db)
    store.ensure_schema()
    # Force the same first 8 chars by providing the id explicitly.
    shared = "ab12cd34"
    store.create(id=f"{shared}-task", title="T")
    store.create_goal(id=f"{shared}-goal", title="G")

    result = runner.invoke(
        app,
        ["reminder", "add", shared, "--at", "2026-12-01", "--tasks-db", str(db)],
    )
    assert result.exit_code == 1
    assert "both task and goal" in result.output


def test_add_with_explicit_kind_disambiguates(runner: CliRunner, db: Path) -> None:
    from iris_harness.services.tasks.store import TaskStore

    store = TaskStore(db_path=db)
    store.ensure_schema()
    shared = "cafe1234"
    store.create(id=f"{shared}-task", title="T")
    store.create_goal(id=f"{shared}-goal", title="G")

    result = runner.invoke(
        app,
        [
            "reminder",
            "add",
            shared,
            "--kind",
            "task",
            "--at",
            "2026-12-01",
            "--tasks-db",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output

    rstore = ReminderStore(db_path=db)
    [reminder] = rstore.list()
    assert reminder.target_kind == "task"


def test_add_rejects_unknown_target(runner: CliRunner, db: Path) -> None:
    result = runner.invoke(
        app,
        ["reminder", "add", "deadbeef", "--at", "2026-12-01", "--tasks-db", str(db)],
    )
    assert result.exit_code == 1
    assert "no task or goal" in result.output


def test_add_rejects_bill_kind_until_phase_3(runner: CliRunner, db: Path) -> None:
    task_id = _create_task(runner, db)
    result = runner.invoke(
        app,
        [
            "reminder",
            "add",
            task_id,
            "--kind",
            "bill",
            "--at",
            "2026-12-01",
            "--tasks-db",
            str(db),
        ],
    )
    assert result.exit_code == 2
    assert "not yet supported" in result.output


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_defaults_to_pending(runner: CliRunner, db: Path) -> None:
    task_id = _create_task(runner, db)
    runner.invoke(
        app,
        ["reminder", "add", task_id, "--at", "2026-12-01", "--tasks-db", str(db)],
    )
    runner.invoke(
        app,
        ["reminder", "add", task_id, "--at", "2020-01-01", "--tasks-db", str(db)],
    )
    runner.invoke(app, ["reminder", "tick", "--tasks-db", str(db)])

    # After tick, one reminder has fired; default list should show only pending.
    result = runner.invoke(app, ["reminder", "list", "--tasks-db", str(db)])
    assert result.exit_code == 0
    assert result.output.count("pending") == 1
    assert "fired" not in result.output

    # With --include-fired both surface.
    with_fired = runner.invoke(app, ["reminder", "list", "--include-fired", "--tasks-db", str(db)])
    assert "fired" in with_fired.output


def test_list_filters_by_kind(runner: CliRunner, db: Path) -> None:
    task_id = _create_task(runner, db)
    goal_id = _create_goal(runner, db)
    runner.invoke(
        app,
        [
            "reminder",
            "add",
            task_id,
            "--kind",
            "task",
            "--at",
            "2026-12-01",
            "--tasks-db",
            str(db),
        ],
    )
    runner.invoke(
        app,
        [
            "reminder",
            "add",
            goal_id,
            "--kind",
            "goal",
            "--at",
            "2026-12-01",
            "--tasks-db",
            str(db),
        ],
    )

    out = runner.invoke(app, ["reminder", "list", "--kind", "task", "--tasks-db", str(db)])
    assert out.exit_code == 0
    assert out.output.count("│ task") == 1
    assert "│ goal" not in out.output


# ---------------------------------------------------------------------------
# cancel
# ---------------------------------------------------------------------------


def test_cancel_marks_dismissed(runner: CliRunner, db: Path) -> None:
    task_id = _create_task(runner, db)
    runner.invoke(
        app,
        ["reminder", "add", task_id, "--at", "2026-12-01", "--tasks-db", str(db)],
    )
    rid = _ids(runner.invoke(app, ["reminder", "list", "--tasks-db", str(db)]).output)[0]

    result = runner.invoke(app, ["reminder", "cancel", rid, "--tasks-db", str(db)])
    assert result.exit_code == 0
    assert "cancelled" in result.output

    store = ReminderStore(db_path=db)
    [reminder] = store.list(include_dismissed=True)
    assert reminder.dismissed_at is not None


def test_cancel_on_fired_reminder_is_no_op(runner: CliRunner, db: Path) -> None:
    task_id = _create_task(runner, db)
    runner.invoke(
        app,
        ["reminder", "add", task_id, "--at", "2020-01-01", "--tasks-db", str(db)],
    )
    runner.invoke(app, ["reminder", "tick", "--tasks-db", str(db)])
    rid = _ids(
        runner.invoke(app, ["reminder", "list", "--include-fired", "--tasks-db", str(db)]).output
    )[0]

    result = runner.invoke(app, ["reminder", "cancel", rid, "--tasks-db", str(db)])
    assert result.exit_code == 0
    assert "already fired" in result.output


# ---------------------------------------------------------------------------
# tick
# ---------------------------------------------------------------------------


def test_tick_fires_due_reminders(runner: CliRunner, db: Path) -> None:
    task_id = _create_task(runner, db)
    runner.invoke(
        app,
        ["reminder", "add", task_id, "--at", "2020-01-01", "--tasks-db", str(db)],
    )
    runner.invoke(
        app,
        ["reminder", "add", task_id, "--at", "2099-01-01", "--tasks-db", str(db)],
    )

    result = runner.invoke(app, ["reminder", "tick", "--tasks-db", str(db)])
    assert result.exit_code == 0
    assert "fired 1" in result.output


def test_tick_with_nothing_due_reports_empty(runner: CliRunner, db: Path) -> None:
    task_id = _create_task(runner, db)
    runner.invoke(
        app,
        ["reminder", "add", task_id, "--at", "2099-01-01", "--tasks-db", str(db)],
    )
    result = runner.invoke(app, ["reminder", "tick", "--tasks-db", str(db)])
    assert result.exit_code == 0
    assert "no due reminders" in result.output
