from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.main import app
from iris_harness.memory.state import CheckpointStore


@pytest.fixture()
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[CheckpointStore]:
    """Force iris_harness.memory.state.store to point at tmp_path."""
    import iris_harness.memory.state.store as store_mod

    db_path = tmp_path / "checkpoints.db"
    monkeypatch.setattr(store_mod, "DEFAULT_CHECKPOINT_DB_PATH", db_path)
    yield CheckpointStore(db_path=db_path)


def test_list_empty(isolated_store: CheckpointStore) -> None:
    result = CliRunner().invoke(app, ["checkpoint", "list"])
    assert result.exit_code == 0, result.output
    assert "no checkpoints" in result.output


def test_list_shows_rows(isolated_store: CheckpointStore) -> None:
    isolated_store.write(
        run_id="r1", step_id=2, agent_type="chat", payload={"q": "hi"}, signal="halt"
    )
    result = CliRunner().invoke(app, ["checkpoint", "list"])
    assert result.exit_code == 0
    assert "r1" in result.output
    assert "halt" in result.output
    assert "chat" in result.output


def test_show_outputs_json_payload(isolated_store: CheckpointStore) -> None:
    isolated_store.write(
        run_id="r1",
        step_id=3,
        agent_type="chat",
        payload={"query": "weather?", "iteration": 3},
        signal="require_approval",
    )
    result = CliRunner().invoke(app, ["checkpoint", "show", "r1"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["run_id"] == "r1"
    assert data["step_id"] == 3
    assert data["signal"] == "require_approval"
    assert data["payload"]["query"] == "weather?"


def test_show_missing_run(isolated_store: CheckpointStore) -> None:
    result = CliRunner().invoke(app, ["checkpoint", "show", "nope"])
    assert result.exit_code == 1
    assert "no checkpoints" in result.output.lower()


def test_pin_unpin_roundtrip(isolated_store: CheckpointStore) -> None:
    isolated_store.write(run_id="r", step_id=1, agent_type="chat", payload={})
    runner = CliRunner()
    pin = runner.invoke(app, ["checkpoint", "pin", "r"])
    assert pin.exit_code == 0
    assert "pinned" in pin.output
    assert isolated_store.get(run_id="r", step_id=1).pinned is True

    unpin = runner.invoke(app, ["checkpoint", "unpin", "r"])
    assert unpin.exit_code == 0
    assert isolated_store.get(run_id="r", step_id=1).pinned is False


def test_pin_missing_returns_nonzero(isolated_store: CheckpointStore) -> None:
    result = CliRunner().invoke(app, ["checkpoint", "pin", "missing"])
    assert result.exit_code == 1


def test_remove(isolated_store: CheckpointStore) -> None:
    isolated_store.write(run_id="r", step_id=1, agent_type="chat", payload={})
    isolated_store.write(run_id="r", step_id=2, agent_type="chat", payload={})
    result = CliRunner().invoke(app, ["checkpoint", "remove", "r"])
    assert result.exit_code == 0
    assert "2 checkpoint(s)" in result.output
    assert isolated_store.list() == ()
