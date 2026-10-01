"""Tests for SideEffectLedger (story 12.gov-4.10)."""

from __future__ import annotations

from pathlib import Path

from iris_harness.kernel.governance.side_effects import SideEffectLedger, SideEffectRow


def _make_ledger(tmp_path: Path) -> SideEffectLedger:
    return SideEffectLedger(db_path=tmp_path / "side_effects.db")


# ---------------------------------------------------------------------------
# record + get
# ---------------------------------------------------------------------------


def test_record_returns_uuid(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    sid = ledger.record(
        run_id="run-1",
        step_id=3,
        tool="git_commit",
        verification_probe="git_commit",
        probe_metadata={"repo_path": "/repo"},
    )
    assert sid and len(sid) == 36  # UUID str


def test_get_returns_row(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    sid = ledger.record(
        run_id="run-1",
        step_id=3,
        tool="git_commit",
        verification_probe="git_commit",
    )
    row = ledger.get(sid)
    assert row is not None
    assert isinstance(row, SideEffectRow)
    assert row.side_effect_id == sid
    assert row.run_id == "run-1"
    assert row.step_id == 3
    assert row.tool == "git_commit"
    assert row.verification_probe == "git_commit"
    assert row.status == "pending"
    assert row.completed_at is None
    assert row.error is None


def test_get_unknown_returns_none(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    assert ledger.get("nonexistent-id") is None


def test_probe_metadata_roundtrip(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    meta = {"repo_path": "/tmp/repo", "commit_sha": "abc123"}
    sid = ledger.record(
        run_id="r1",
        step_id=1,
        tool="git_push",
        verification_probe="git_push",
        probe_metadata=meta,
    )
    row = ledger.get(sid)
    assert row is not None
    assert row.probe_metadata == meta


# ---------------------------------------------------------------------------
# pending
# ---------------------------------------------------------------------------


def test_pending_excludes_completed(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    sid1 = ledger.record(run_id="r1", step_id=1, tool="git_commit", verification_probe="git_commit")
    sid2 = ledger.record(run_id="r1", step_id=2, tool="git_push", verification_probe="git_push")
    ledger.set_status(sid1, status="completed")

    pending = ledger.pending("r1")
    assert len(pending) == 1
    assert pending[0].side_effect_id == sid2


def test_pending_returns_all_non_completed(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    ledger.record(run_id="r1", step_id=1, tool="git_commit", verification_probe="git_commit")
    ledger.record(run_id="r1", step_id=2, tool="write_file", verification_probe="write_file")

    pending = ledger.pending("r1")
    assert len(pending) == 2


def test_pending_filters_by_run_id(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    ledger.record(run_id="r1", step_id=1, tool="git_commit", verification_probe="git_commit")
    ledger.record(run_id="r2", step_id=1, tool="git_commit", verification_probe="git_commit")

    assert len(ledger.pending("r1")) == 1
    assert len(ledger.pending("r2")) == 1
    assert len(ledger.pending("r3")) == 0


# ---------------------------------------------------------------------------
# set_status
# ---------------------------------------------------------------------------


def test_set_status_completed_sets_completed_at(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    sid = ledger.record(run_id="r1", step_id=1, tool="git_commit", verification_probe="git_commit")
    ledger.set_status(sid, status="completed")
    row = ledger.get(sid)
    assert row is not None
    assert row.status == "completed"
    assert row.completed_at is not None


def test_set_status_error_stores_message(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    sid = ledger.record(run_id="r1", step_id=1, tool="write_file", verification_probe="write_file")
    ledger.set_status(sid, status="error", error="permission denied")
    row = ledger.get(sid)
    assert row is not None
    assert row.status == "error"
    assert row.error == "permission denied"


def test_set_status_ambiguous(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    sid = ledger.record(run_id="r1", step_id=1, tool="git_push", verification_probe="git_push")
    ledger.set_status(sid, status="ambiguous")
    row = ledger.get(sid)
    assert row is not None
    assert row.status == "ambiguous"
    assert row.completed_at is None  # only completed sets this


# ---------------------------------------------------------------------------
# list_by_run
# ---------------------------------------------------------------------------


def test_list_by_run_returns_all(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    ledger.record(run_id="r1", step_id=1, tool="git_commit", verification_probe="git_commit")
    ledger.record(run_id="r1", step_id=2, tool="git_push", verification_probe="git_push")
    ledger.record(run_id="r2", step_id=1, tool="write_file", verification_probe="write_file")

    r1_rows = ledger.list_by_run("r1")
    assert len(r1_rows) == 2
    assert r1_rows[0].step_id == 1
    assert r1_rows[1].step_id == 2


# ---------------------------------------------------------------------------
# DB file permissions
# ---------------------------------------------------------------------------


def test_db_file_created_with_restricted_perms(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    ledger.record(run_id="r1", step_id=1, tool="git_commit", verification_probe="git_commit")
    db = tmp_path / "side_effects.db"
    assert db.exists()
    mode = db.stat().st_mode & 0o777
    assert mode == 0o600


def test_a_caller_key_is_kept_and_recording_it_again_is_a_no_op(tmp_path: Path) -> None:
    """The ledger hook keys a row by the call (``<run>:<step>:<call>``); a capability
    stream fires POST_TOOL_USE per item, and one call must stay one row."""
    ledger = _make_ledger(tmp_path)
    for _ in range(3):
        sid = ledger.record(
            side_effect_id="run-1:2:abc",
            run_id="run-1",
            step_id=2,
            tool="add_note",
            verification_probe="",
            probe_metadata={"subject": "note-9"},
        )
        assert sid == "run-1:2:abc"
    (row,) = ledger.list_by_run("run-1")
    assert row.side_effect_id == "run-1:2:abc"
    assert row.probe_subject == "note-9"


def test_the_probe_subject_defaults_to_the_key(tmp_path: Path) -> None:
    ledger = _make_ledger(tmp_path)
    ledger.record(
        side_effect_id="run-1:0:x", run_id="run-1", step_id=0, tool="t", verification_probe=""
    )
    (row,) = ledger.list_by_run("run-1")
    assert row.probe_subject == "run-1:0:x"
