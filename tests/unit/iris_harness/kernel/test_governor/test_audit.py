"""Unit tests for append-only IRIS governor audit logging."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from iris_harness.kernel.governor.audit import GovernorAuditLogger
from iris_harness.kernel.governor.models import GovernorGuardDecision


def test_audit_logger_is_append_only(tmp_path: Path) -> None:
    audit_logger = GovernorAuditLogger(tmp_path / "data" / "audit.db")
    audit_logger.record_decision(
        GovernorGuardDecision(
            route="coding/git",
            action="branch_commit_push",
            allowed=True,
            reason="Allowed",
            matched_policy="coding/git",
            metadata={"task_id": "task-1"},
        )
    )

    entries = audit_logger.list_entries()
    assert len(entries) == 1
    assert entries[0].route == "coding/git"
    assert entries[0].metadata == {"task_id": "task-1"}

    with sqlite3.connect(audit_logger.db_path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM governor_guard_audit WHERE id = 1")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE governor_guard_audit SET reason = 'mutated' WHERE id = 1")


def test_building_a_logger_writes_nothing(tmp_path: Path) -> None:
    """The API builds a governor at startup even with MCP off; that must not create a
    file (it used to leave data/audit.db wherever the app was imported)."""
    db = tmp_path / "data" / "audit.db"
    GovernorAuditLogger(db)
    assert not db.exists()
    assert not db.parent.exists()


def test_the_first_read_creates_an_empty_append_only_store(tmp_path: Path) -> None:
    audit_logger = GovernorAuditLogger(tmp_path / "data" / "audit.db")
    assert audit_logger.list_entries() == ()
    assert audit_logger.db_path.exists()
    with sqlite3.connect(audit_logger.db_path) as connection:
        triggers = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")
        }
    assert triggers == {"governor_guard_audit_no_update", "governor_guard_audit_no_delete"}
