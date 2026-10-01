"""ADR-0106 M5.C5a — the checkpoint spine is actually connected.

The Phase-3 spine shipped complete and disconnected: `CheckpointStore`,
`ChatCheckpointPayload`, `_write_checkpoint` and `resume_from_checkpoint` all
existed, but nothing in production ever handed the core a store, so
`_write_checkpoint` returned on its first line and no evaluator halt was ever
resumable. Every unit test passed throughout, because each one constructs the
core itself and passes a store.

So these tests are about the *joint*, not the parts: a core built without a store
writes nothing (the old production behaviour, pinned so the degradation stays
graceful), a core built with one writes a row, and that row carries the session.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.agent.agentic_core import AgenticCore, ReactStep, ReactTrace
from iris_harness.memory.state import CheckpointNotFoundError, CheckpointStore


def _trace() -> ReactTrace:
    trace = ReactTrace(query="organize my downloads", run_id="run-1")
    trace.steps.append(ReactStep(thought="looking", action="list_files", action_input={}))
    trace.halt_reason = "evaluator: unsafe"
    return trace


def test_a_core_without_a_store_writes_nothing(tmp_path: Path) -> None:
    """Production's behaviour before C5a — kept working, just silent."""
    core = AgenticCore(session_id="s1")
    core._write_checkpoint(  # must not raise
        trace=_trace(), iteration=0, memory_context=None, signal="halt"
    )


def test_a_wired_core_writes_a_resumable_checkpoint(tmp_path: Path) -> None:
    store = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    core = AgenticCore(checkpoint_store=store, session_id="s1")
    trace = _trace()

    core._write_checkpoint(trace=trace, iteration=0, memory_context=None, signal="halt")

    cp = store.get_latest("run-1")
    assert cp.signal == "halt"
    assert cp.payload["query"] == "organize my downloads"
    assert cp.payload["iteration"] == 1  # resume re-enters at N+1
    assert trace.checkpoint_id == "run-1:0"


def test_the_checkpoint_carries_its_session(tmp_path: Path) -> None:
    """C1 added the column; this is what finally populates it, so `by_session`
    can answer "what was this conversation doing?"."""
    store = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    core = AgenticCore(checkpoint_store=store, session_id="s1")

    core._write_checkpoint(trace=_trace(), iteration=0, memory_context=None, signal="halt")

    assert store.get_latest("run-1").session_id == "s1"
    assert [cp.run_id for cp in store.by_session("s1")] == ["run-1"]
    assert store.by_session("someone-else") == ()


def test_a_sessionless_run_still_checkpoints(tmp_path: Path) -> None:
    """CLI runs and heartbeats have no conversation behind them; the column is
    nullable precisely so those keep working."""
    store = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    core = AgenticCore(checkpoint_store=store)

    core._write_checkpoint(trace=_trace(), iteration=0, memory_context=None, signal="halt")

    assert store.get_latest("run-1").session_id is None
    assert store.by_session("s1") == ()


def test_a_failing_store_never_breaks_the_run(tmp_path: Path) -> None:
    """A halted run has already produced its user-facing message. Losing the
    checkpoint costs resumability, not the answer."""

    class _Broken(CheckpointStore):
        def write(self, **_kw: object) -> object:  # type: ignore[override]
            raise RuntimeError("disk gone")

    core = AgenticCore(
        checkpoint_store=_Broken(db_path=tmp_path / "checkpoints.db"), session_id="s1"
    )
    trace = _trace()

    core._write_checkpoint(trace=trace, iteration=0, memory_context=None, signal="halt")

    assert trace.checkpoint_id is None  # no resume point, but the run stood
    with pytest.raises(CheckpointNotFoundError):
        CheckpointStore(db_path=tmp_path / "checkpoints.db").get_latest("run-1")
