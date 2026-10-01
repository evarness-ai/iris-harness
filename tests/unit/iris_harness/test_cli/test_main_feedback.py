"""Tests for the generic ``iris feedback`` surface-feedback command (issue 0028)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.main import app
from iris_harness.services.learning.suppression import SurfaceFeedbackStore, encode_ref
from iris_harness.services.tasks.models import WaitFor
from iris_harness.services.tasks.store import TaskStore


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_feedback_on_fb_ref_records_generic_verdict(runner: CliRunner) -> None:
    ref = encode_ref("system", "health_alert", {"kind": "credential", "target": "sfbtest-probe"})
    result = runner.invoke(app, ["feedback", ref, "--not-useful"])
    assert result.exit_code == 0, result.output

    store = SurfaceFeedbackStore()
    store.ensure_schema()
    assert store.should_suppress(
        "system", "health_alert", {"kind": "credential", "target": "sfbtest-probe"}
    )


def test_feedback_on_followup_task_drops_and_suppresses(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "tasks.db"
    store = TaskStore(db_path=db)
    store.ensure_schema()
    task = store.create(
        title="Reply to Anita Rao <anita@quant-academy.example>: ...",
        source_kind="email",
        wait_for=WaitFor(
            kind="reply_from",
            payload={
                "thread_id": "t1",
                "account_id": "gmail:u@gmail.com",
                "provider": "gmail",
                "from": "Anita Rao <anita@quant-academy.example>",
            },
        ),
    )

    result = runner.invoke(app, ["feedback", task.id, "--not-useful", "--tasks-db", str(db)])
    assert result.exit_code == 0, result.output
    assert "quant-academy.example" in result.output

    # Task dropped...
    assert store.get(task.id).status == "dropped"
    # ...and the sender suppressed for future detection.
    fb = SurfaceFeedbackStore()
    fb.ensure_schema()
    assert fb.should_suppress(
        "email",
        "followup",
        {"account": "gmail:u@gmail.com", "from_domain": "quant-academy.example"},
    )


def test_feedback_useful_keeps_followup_task(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "tasks.db"
    store = TaskStore(db_path=db)
    store.ensure_schema()
    task = store.create(
        title="Reply to a real colleague",
        source_kind="email",
        wait_for=WaitFor(
            kind="reply_from",
            payload={"account_id": "gmail:u@gmail.com", "from": "boss@company.com"},
        ),
    )

    result = runner.invoke(app, ["feedback", task.id, "--useful", "--tasks-db", str(db)])
    assert result.exit_code == 0, result.output
    assert store.get(task.id).status == "open"  # kept


def test_feedback_unknown_task_errors(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "tasks.db"
    TaskStore(db_path=db).ensure_schema()
    result = runner.invoke(app, ["feedback", "no-such-id", "--tasks-db", str(db)])
    assert result.exit_code == 3


def test_feedback_rejects_both_flags(runner: CliRunner) -> None:
    ref = encode_ref("email", "followup", {"from_domain": "x.com"})
    result = runner.invoke(app, ["feedback", ref, "--useful", "--not-useful"])
    assert result.exit_code == 2
