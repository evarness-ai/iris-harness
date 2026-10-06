"""The side-effect ledger: one handle per database, and no key collision goes unnoticed (#102).

* Every ``kernel_from_env()`` / ``build_default_kernel()`` used to build its own ledger, each
  creating the schema on its own first use; one process builds a kernel at about seven sites.
  The ledger is now shared per resolved database path (``shared_side_effect_ledger``).
* ``record`` was ``INSERT OR IGNORE``: a key that already had a row made the call "recorded"
  with nothing written. A write-ahead row of a high-risk call that collides now denies the
  call; a post-call record that collides is reported, never confirmed; a row that is already
  settled is not overwritten by another call's settle.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.agent.tool_runner import GovernedToolRunner, ToolCall
from iris_harness.kernel.governance import GovernanceKernel, HookPoint, kernel_from_env
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.hooks.tool_payload import SIDE_EFFECT_ID
from iris_harness.kernel.governance.plugins.post_tool_use_ledger import PostToolUseLedgerHook
from iris_harness.kernel.governance.plugins.pre_tool_use_ledger import PreToolUseLedgerHook
from iris_harness.kernel.governance.side_effects import (
    DeferredSideEffectLedger,
    SideEffectKeyExists,
    SideEffectLedger,
    shared_side_effect_ledger,
)
from iris_harness.kernel.governance.side_effects.probes import NO_PROBE
from iris_harness.kernel.governance.wiring import (
    _open_side_effect_ledger,
    build_default_kernel,
    register_side_effect_ledger,
)

from .test_pre_tool_use_ledger import KEY, _post, _pre

# ------------------------------------------------------------------ record(): exclusive


def _seed(ledger: SideEffectLedger, status: str = "completed", **meta: Any) -> None:
    ledger.record(
        side_effect_id=KEY,
        run_id="run-1",
        step_id=2,
        tool="earlier",
        verification_probe=NO_PROBE,
        probe_metadata=meta,
    )
    ledger.set_status(KEY, status=status)


def test_an_exclusive_record_of_a_taken_key_raises_and_leaves_the_row(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "s.db")
    _seed(ledger)
    with pytest.raises(SideEffectKeyExists):
        ledger.record(
            side_effect_id=KEY,
            run_id="run-1",
            step_id=2,
            tool="later",
            verification_probe=NO_PROBE,
            exclusive=True,
        )
    row = ledger.get(KEY)
    assert row is not None and (row.tool, row.status) == ("earlier", "completed")


def test_a_plain_record_of_a_taken_key_is_still_a_no_op(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "s.db")
    _seed(ledger)
    assert (
        ledger.record(
            side_effect_id=KEY, run_id="run-1", step_id=2, tool="later", verification_probe=NO_PROBE
        )
        == KEY
    )
    row = ledger.get(KEY)
    assert row is not None and row.tool == "earlier"


# ------------------------------------------------------------------ the pre-record


async def test_a_pre_record_on_a_taken_key_denies_the_call(tmp_path: Path) -> None:
    """Unfixed: allowed, with the old row confirmed as this call's own record."""
    ledger = SideEffectLedger(tmp_path / "s.db")
    _seed(ledger, status="completed")
    decision = await PreToolUseLedgerHook(ledger)(_pre())
    assert decision.outcome == "deny"
    assert SIDE_EFFECT_ID not in decision.audit_metadata
    row = ledger.get(KEY)
    assert row is not None and (row.tool, row.status) == ("earlier", "completed")


def test_two_threads_pre_recording_one_key_allow_exactly_one(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "s.db")
    hook = PreToolUseLedgerHook(ledger)
    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def go() -> None:
        ctx = _pre()
        barrier.wait(timeout=10)
        decision = asyncio.run(hook(ctx))
        with lock:
            outcomes.append(decision.outcome)

    threads = [threading.Thread(target=go) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert sorted(outcomes) == ["allow", "deny"]


# ------------------------------------------------------------------ the post record


async def test_a_post_record_that_collides_is_reported_not_confirmed(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "s.db")
    _seed(ledger, status="pending")  # a plain row of an earlier call, not a pre-record
    decision = await PostToolUseLedgerHook(ledger)(_post("note", effect="write"))
    assert decision.severity == "warn"
    assert "side_effect_id" not in decision.audit_metadata
    row = ledger.get(KEY)
    assert row is not None and row.tool == "earlier"


async def test_a_settled_pre_row_is_not_settled_again_by_another_call(tmp_path: Path) -> None:
    ledger = SideEffectLedger(tmp_path / "s.db")
    await PreToolUseLedgerHook(ledger)(_pre())
    first = await PostToolUseLedgerHook(ledger)(_post())
    assert first.audit_metadata["status"] == "completed"
    again = await PostToolUseLedgerHook(ledger)(_post(error="RuntimeError"))
    assert again.severity == "warn"
    row = ledger.get(KEY)
    assert row is not None and (row.status, row.error) == ("completed", None)


async def test_a_streams_later_items_and_end_are_not_collisions(tmp_path: Path) -> None:
    """A stream is one call under one key: its items after the first find the row there."""
    ledger = SideEffectLedger(tmp_path / "s.db")
    hook = PostToolUseLedgerHook(ledger)
    decisions = [
        await hook(_post("note", effect="write", stream_item=0)),
        await hook(_post("note", effect="write", stream_item=1)),
        await hook(_post("note", effect="write", stream_end=True)),
    ]
    assert all(d.severity != "warn" for d in decisions)
    assert [r.tool for r in ledger.list_by_run("run-1")] == ["note"]


# ------------------------------------------------------------------ the runner mints fresh keys


def test_two_approved_calls_of_one_tool_in_one_step_leave_two_rows(tmp_path: Path) -> None:
    """The runner mints a call id per call, so an approved resume never reuses a key."""
    ledger = SideEffectLedger(tmp_path / "s.db")
    kernel = GovernanceKernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
    register_side_effect_ledger(kernel, ledger)
    kernel.init_lock()
    tool = ToolSpec("wipe", "wipe", lambda args: "done", effect="destructive", confirm="approval")
    runner = GovernedToolRunner(kernel=kernel, agent_type="chat")
    call = ToolCall(run_id="run-1", step_id=2, approved_by="appr-1", caller="model:chat")
    assert runner.execute(tool, {}, call).status == "ran"
    assert runner.execute(tool, {}, call).status == "ran"
    rows = ledger.list_by_run("run-1")
    assert len(rows) == 2 and {r.status for r in rows} == {"completed"}


# ------------------------------------------------------------------ one handle per database


def test_one_database_path_gets_one_ledger_however_it_is_spelled(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    a = shared_side_effect_ledger(tmp_path / "s.db")
    b = shared_side_effect_ledger(tmp_path / "sub" / ".." / "s.db")
    c = shared_side_effect_ledger(tmp_path / "other.db")
    assert a is b and a is not c
    assert isinstance(a, DeferredSideEffectLedger) and not (tmp_path / "s.db").exists()


def test_the_default_path_follows_iris_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "home-a"))
    a = shared_side_effect_ledger()
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "home-b"))
    b = shared_side_effect_ledger()
    assert a is not b and a.db_path != b.db_path


def _count_schema_inits(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    inits: list[Path] = []
    real = SideEffectLedger._init_schema

    def counting(self: SideEffectLedger) -> None:
        inits.append(self.db_path)
        real(self)

    monkeypatch.setattr(SideEffectLedger, "_init_schema", counting)
    return inits


def test_kernels_built_from_the_environment_open_the_ledger_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER_DB_PATH", str(tmp_path / "s.db"))
    monkeypatch.delenv("IRIS_GOVERNANCE_SIDE_EFFECT_LEDGER", raising=False)
    inits = _count_schema_inits(monkeypatch)
    kernels = [kernel_from_env() for _ in range(3)]
    for kernel in kernels:
        assert kernel is not None
        kernel.fire_sync(HookPoint.POST_TOOL_USE, _post())
    assert len(inits) == 1


def test_default_kernels_for_one_path_share_one_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inits = _count_schema_inits(monkeypatch)
    db = tmp_path / "s.db"
    for i in range(2):
        kernel = build_default_kernel(
            audit_log=AuditLog(db_path=tmp_path / f"a{i}.db"), side_effect_ledger_db_path=db
        )
        kernel.fire_sync(HookPoint.POST_TOOL_USE, _post())
    assert len(inits) == 1


def test_opening_the_ledger_eagerly_twice_runs_the_schema_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inits = _count_schema_inits(monkeypatch)
    first = _open_side_effect_ledger(tmp_path / "s.db")
    second = _open_side_effect_ledger(tmp_path / "s.db")
    assert first is second and first is not None and (tmp_path / "s.db").exists()
    assert len(inits) == 1


def test_a_shared_ledger_whose_file_was_removed_creates_its_schema_again(
    tmp_path: Path,
) -> None:
    db = tmp_path / "s.db"
    ledger = shared_side_effect_ledger(db)
    ledger.record(side_effect_id="k1", run_id="r", step_id=1, tool="t", verification_probe="")
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    ledger.record(side_effect_id="k2", run_id="r", step_id=1, tool="t", verification_probe="")
    assert ledger.get("k2") is not None


def test_recreating_a_removed_ledger_database_warns_exactly_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    db = tmp_path / "s.db"
    ledger = shared_side_effect_ledger(db)
    with caplog.at_level(logging.WARNING, logger="iris_harness.kernel.governance.side_effects"):
        ledger.record(side_effect_id="k1", run_id="r", step_id=1, tool="t", verification_probe="")
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        for suffix in ("", "-wal", "-shm"):
            Path(str(db) + suffix).unlink(missing_ok=True)
        ledger.record(side_effect_id="k2", run_id="r", step_id=1, tool="t", verification_probe="")
        ledger.record(side_effect_id="k3", run_id="r", step_id=1, tool="t", verification_probe="")
    hits = [r for r in caplog.records if "created again, empty" in r.getMessage()]
    assert len(hits) == 1 and hits[0].levelno == logging.WARNING
    assert str(db) in hits[0].getMessage()
    assert "write-ahead rows" in hits[0].getMessage()


def test_a_shared_ledger_that_would_not_open_is_tried_again(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    assert _open_side_effect_ledger(blocker / "s.db") is None
    blocker.unlink()
    assert _open_side_effect_ledger(blocker / "s.db") is not None
