from __future__ import annotations

from pathlib import Path

from iris_harness.runtime.handlers.local_skills import record_skill_pending_actions
from iris_harness.services.tasks import TaskStore


def _task_store(tmp_path: Path) -> TaskStore:
    ts = TaskStore(db_path=tmp_path / "tasks.db")
    ts.ensure_schema()
    return ts


def test_records_rag_ingest_pending_action(tmp_path: Path) -> None:
    payload = {
        "kind": "rag_ingest",
        "proposal": {
            "resolved_path": "/tmp/notes.md",
            "content_sha": "abc123",
            "reason": "add notes.md to the document index",
        },
    }

    recorded = record_skill_pending_actions(payload, data_dir=tmp_path)

    assert recorded == 1
    rows = _task_store(tmp_path).list(source_kind="rag-ingest", has_action=True, limit=10)
    assert len(rows) == 1
    # A resolved path is present → the action is executable (Approve & run), and
    # carries the path so the approval surfaces can re-scan + ingest it.
    assert rows[0].action is not None and rows[0].action.kind == "execute"
    assert rows[0].action.target_id == "/tmp/notes.md"
    assert rows[0].dedup_key == "rag-ingest:abc123"


def test_records_photos_album_pending_action(tmp_path: Path) -> None:
    payload = {
        "kind": "photos_albums",
        "plan": {
            "created_at": "2026-07-03T00:00:00+00:00",
            "groups": [{"name": "Screenshots 2026"}, {"name": "Receipt"}],
        },
    }

    recorded = record_skill_pending_actions(payload, data_dir=tmp_path)

    assert recorded == 1
    rows = _task_store(tmp_path).list(source_kind="photos-albums", has_action=True, limit=10)
    assert len(rows) == 1
    assert rows[0].action is not None and rows[0].action.kind == "execute"
    assert rows[0].title == "Approve proposed photo albums"


def test_records_vault_quarantine_pending_action(tmp_path: Path) -> None:
    payload = {
        "kind": "filemanager_vault",
        "file_id": "fm_123",
        "quarantine_proposal": {
            "kind": "filemanager-quarantine",
            "path": "/tmp/secrets.txt",
            "vault_file_id": "fm_123",
            "reason": "original vaulted as fm_123",
        },
    }

    recorded = record_skill_pending_actions(payload, data_dir=tmp_path)

    assert recorded == 1
    rows = _task_store(tmp_path).list(
        source_kind="filemanager-quarantine", has_action=True, limit=10
    )
    assert len(rows) == 1
    # The original path is present → executable, carrying the path to quarantine.
    assert rows[0].action is not None and rows[0].action.kind == "execute"
    assert rows[0].action.target_id == "/tmp/secrets.txt"
    assert rows[0].dedup_key == "filemanager-quarantine:fm_123"


def test_ignores_unknown_payload_shapes(tmp_path: Path) -> None:
    assert record_skill_pending_actions({"kind": "unknown"}, data_dir=tmp_path) == 0
    assert record_skill_pending_actions("not-a-dict", data_dir=tmp_path) == 0
