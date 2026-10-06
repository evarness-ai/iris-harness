"""``iris run resume`` reads the process's shared side-effect ledger (issue #102).

It used to build a ``SideEffectLedger`` of its own, a second handle on a database the kernels
already share through ``shared_side_effect_ledger``, so it could not see that a kernel in the
same process had just re-created a removed file.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from iris_harness.kernel.governance import side_effects
from iris_harness.kernel.governance.side_effects import NO_PROBE, shared_side_effect_ledger
from iris_harness.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def _checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    class Store:
        def get_latest(self, run_id: str) -> SimpleNamespace:
            return SimpleNamespace(step_id=1, signal=None)

    monkeypatch.setattr("iris_harness.memory.state.store.CheckpointStore", Store)


def test_resume_lists_a_pending_row_through_the_shared_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "ledger.db"
    shared = shared_side_effect_ledger(db)
    shared.open()
    shared.record(run_id="run-1", step_id=0, tool="trash_email", verification_probe=NO_PROBE)

    def not_a_second_handle(*args: object, **kwargs: object) -> None:
        raise AssertionError("resume built a ledger of its own")

    monkeypatch.setattr(side_effects, "SideEffectLedger", not_a_second_handle)

    result = runner.invoke(app, ["run", "resume", "run-1", "--db", str(db), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "trash_email" in result.output


def test_resume_says_so_when_the_ledger_database_cannot_be_opened(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("a file where the ledger's folder should be")

    result = runner.invoke(
        app, ["run", "resume", "run-1", "--db", str(blocked / "ledger.db"), "--dry-run"]
    )

    assert result.exit_code != 0
