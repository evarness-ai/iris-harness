"""Tests for the `iris run inspect <run_id>` Typer subcommand (story 12.gov-3.5)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.main import app
from iris_harness.memory.state import CheckpointStore


@pytest.fixture()
def isolated_stores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[AuditLog, CheckpointStore]]:
    """Force both default DB paths into tmp_path so the CLI talks to fresh stores."""
    import iris_harness.kernel.governance.audit.log as audit_mod
    import iris_harness.memory.state.store as cp_mod

    audit_path = tmp_path / "audit.db"
    cp_path = tmp_path / "checkpoints.db"
    monkeypatch.setattr(audit_mod, "DEFAULT_AUDIT_DB_PATH", audit_path)
    monkeypatch.setattr(cp_mod, "DEFAULT_CHECKPOINT_DB_PATH", cp_path)

    yield AuditLog(db_path=audit_path), CheckpointStore(db_path=cp_path)


def _seed_run(audit: AuditLog, *, run_id: str = "r1") -> None:
    """Three audit rows that look like a halted run."""
    audit.record(
        run_id=run_id,
        step_id=0,
        agent_type="chat",
        hook_point="pre_llm_call",
        plugin="egress_gate",
        decision="allow",
        severity="info",
        reason="egress_gate: tier=local class=public ok",
    )
    audit.record(
        run_id=run_id,
        step_id=1,
        agent_type="chat",
        hook_point="post_step",
        plugin="evaluator",
        decision="allow",
        severity="info",
        reason="evaluator: action_repeat -> ok: 1/3",
        payload={"signals": [{"name": "action_repeat", "verdict": "ok"}]},
    )
    audit.record(
        run_id=run_id,
        step_id=2,
        agent_type="chat",
        hook_point="post_step",
        plugin="evaluator",
        decision="deny",
        severity="warn",
        reason="evaluator: action_repeat -> halt: too many repeats",
        payload={"signals": [{"name": "action_repeat", "verdict": "halt"}]},
    )


def test_inspect_unknown_run_id_returns_nonzero(
    isolated_stores: tuple[AuditLog, CheckpointStore],
) -> None:
    result = CliRunner().invoke(app, ["run", "inspect", "does-not-exist"])
    assert result.exit_code == 1
    assert "no audit rows" in result.output.lower()


def test_inspect_text_renders_header_and_trace(
    isolated_stores: tuple[AuditLog, CheckpointStore],
) -> None:
    audit, checkpoints = isolated_stores
    _seed_run(audit)
    checkpoints.write(
        run_id="r1",
        step_id=2,
        agent_type="chat",
        payload={"query": "loop?", "iteration": 2},
        signal="halt",
    )

    result = CliRunner().invoke(app, ["run", "inspect", "r1"])
    assert result.exit_code == 0, result.output
    # Header — these tokens are short enough to survive Rich's 80-col wrap
    # under CliRunner. JSON test below covers full data fidelity.
    assert "r1" in result.output
    assert "chat" in result.output
    assert "steps" in result.output
    # Trace table rendered (title) + final decision visible in header
    assert "trace" in result.output
    assert "deny" in result.output
    assert "halt" in result.output
    # Checkpoint footer
    assert "checkpoint" in result.output.lower()


def test_inspect_json_emits_stable_shape(
    isolated_stores: tuple[AuditLog, CheckpointStore],
) -> None:
    audit, checkpoints = isolated_stores
    _seed_run(audit)
    checkpoints.write(
        run_id="r1",
        step_id=2,
        agent_type="chat",
        payload={"query": "loop?", "iteration": 2},
        signal="halt",
    )

    result = CliRunner().invoke(app, ["run", "inspect", "r1", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert set(data.keys()) == {"run", "steps", "checkpoint"}

    run = data["run"]
    assert run["run_id"] == "r1"
    assert run["agent_type"] == "chat"
    assert run["total_rows"] == 3
    assert run["final_decision"] == "deny"
    assert run["final_severity"] == "warn"

    steps = data["steps"]
    assert len(steps) == 3
    assert steps[0]["plugin"] == "egress_gate"
    assert steps[-1]["decision"] == "deny"
    # Payload JSON is decoded into a dict by the renderer.
    assert isinstance(steps[1]["payload"], dict)
    assert steps[1]["payload"]["signals"][0]["name"] == "action_repeat"

    cp = data["checkpoint"]
    assert cp is not None
    assert cp["run_id"] == "r1"
    assert cp["step_id"] == 2
    assert cp["signal"] == "halt"


def test_inspect_json_with_no_checkpoint(
    isolated_stores: tuple[AuditLog, CheckpointStore],
) -> None:
    audit, _ = isolated_stores
    _seed_run(audit, run_id="r2")

    result = CliRunner().invoke(app, ["run", "inspect", "r2", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["checkpoint"] is None
    assert data["run"]["run_id"] == "r2"


def test_inspect_limit_truncates_steps(
    isolated_stores: tuple[AuditLog, CheckpointStore],
) -> None:
    audit, _ = isolated_stores
    _seed_run(audit, run_id="r3")

    result = CliRunner().invoke(app, ["run", "inspect", "r3", "--format", "json", "--limit", "2"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data["steps"]) == 2
    assert data["run"]["total_rows"] == 2


def test_inspect_invalid_format(
    isolated_stores: tuple[AuditLog, CheckpointStore],
) -> None:
    result = CliRunner().invoke(app, ["run", "inspect", "r", "--format", "yaml"])
    assert result.exit_code == 2
    assert "unknown --format" in result.output.lower()


def test_inspect_with_only_checkpoint_synthesizes_header(
    isolated_stores: tuple[AuditLog, CheckpointStore],
) -> None:
    """A pinned checkpoint with no audit history still renders cleanly."""
    _, checkpoints = isolated_stores
    checkpoints.write(
        run_id="orphan",
        step_id=4,
        agent_type="chat",
        payload={"query": "x"},
        signal="require_approval",
    )
    result = CliRunner().invoke(app, ["run", "inspect", "orphan", "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["run"]["total_rows"] == 0
    assert data["run"]["final_decision"] == "require_approval"
    assert data["checkpoint"]["step_id"] == 4
