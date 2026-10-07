"""Opening an older database from several processes at once is safe (issue #201).

Each store that migrates its table used to read ``PRAGMA table_info`` and then run an ``ALTER
TABLE ... ADD COLUMN`` on its own connection; two processes opening the same older file both saw
the column missing and the loser's ``ALTER`` raised ``duplicate column name``. Measured at 12 of
12 rounds for every store here with 6 interpreters. They now call
``foundation.persistence.sqlite.ensure_columns`` (``BEGIN IMMEDIATE`` on its own connection).

Every database is built the way a release made it: the store creates its current schema, then
the columns it migrates are dropped (and the indexes over them), so the file is exactly what an
older version left. Then twelve real interpreters open it at once (each has imported and said ready before any starts).
"""

from __future__ import annotations

import ast
import importlib
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import iris_harness
from iris_harness.foundation.persistence import sqlite as persistence

SRC = Path(iris_harness.__file__).resolve().parents[1]

# store -> (module:class, expression run with S=class and P=path, [(table, [migrated columns])])
STORES: dict[str, tuple[str, str, list[tuple[str, list[str]]]]] = {
    "checkpoints": (
        "iris_harness.memory.state.store:CheckpointStore",
        "S(P)",
        [("checkpoints", ["session_id"])],
    ),
    "continuations": (
        "iris_harness.memory.state.continuations:ContinuationStore",
        "S(P)",
        [("continuations", ["executor_kind", "payload_json"])],
    ),
    "tasks": (
        "iris_harness.services.tasks.store:TaskStore",
        "S(P).ensure_schema()",
        [("tasks", ["action", "closed_reason"])],
    ),
    "activities": (
        "iris_harness.services.activities.store:ActivityStore",
        "S(P).ensure_schema()",
        [("activities", ["owner_pid"])],
    ),
    "signals": (
        "iris_harness.services.learning.store:LearningMetricsStore",
        "S(db_path=P).ensure_schema()",
        [
            (
                "signals",
                ["session_id", "turn_id", "trace_id", "span_id", "resolved_tier", "resolved_agent"],
            )
        ],
    ),
    "rag": (
        "iris_harness.services.rag.store:DocumentStore",
        "S(P).ensure_schema()",
        [
            ("document_sources", ["tags", "links", "mtime", "classification"]),
            ("document_chunks", ["page", "classification"]),
        ],
    ),
    "reminders": (
        "iris_harness.services.notifications.store:ReminderStore",
        "S(P).ensure_schema()",
        [],  # filled from the store's own list of added columns
    ),
    "email": (
        "iris_personal.email.store:EmailStore",
        "S(P).ensure_schema()",
        [("emails", ["triage_state", "classified_source"]), ("trashed", ["labels_before"])],
    ),
    "onboarding": (
        "iris_personal.plugins.email_workflows.onboarding:OnboardingStore",
        "S(P).get('x')",
        [("email_onboarding", ["activity_id"])],
    ),
    "approvals": (
        "iris_harness.kernel.governance.approvals.store:ApprovalStore",
        "S(P)",
        [
            (
                "approval_queue",
                ["session_id", "items_json", "card_json", "caller", "executed_at", "call_id"],
            )
        ],
    ),
}


def _load(spec: str) -> Any:
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)


def _drops(name: str) -> list[tuple[str, list[str]]]:
    if name == "reminders":
        from iris_harness.services.notifications import store as notifications

        return [("notification_reminders", [c for c, _ in notifications._ADDED_COLUMNS])]
    return STORES[name][2]


def _release_shape(db: Path, drops: list[tuple[str, list[str]]]) -> None:
    """Make ``db`` what an older release left: drop the migrated columns and their indexes."""
    conn = sqlite3.connect(db)
    for table, columns in drops:
        for column in columns:
            for index in conn.execute(f"PRAGMA index_list({table})").fetchall():
                if index[1].startswith("sqlite_autoindex"):
                    continue
                if any(r[2] == column for r in conn.execute(f"PRAGMA index_info({index[1]})")):
                    conn.execute(f"DROP INDEX {index[1]}")
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
    conn.commit()
    conn.close()


def _columns(db: Path, table: str) -> list[str]:
    conn = sqlite3.connect(db)
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


_CHILD = """
import importlib, os, pathlib, sys, time
module, name = sys.argv[1].split(":")
S = getattr(importlib.import_module(module), name)
P = pathlib.Path(sys.argv[2]); go = sys.argv[3]
print("ready", flush=True)
while not os.path.exists(go):
    pass
try:
    eval(sys.argv[4], {"S": S, "P": P})
    print("ok")
except Exception as exc:
    print("ERR", type(exc).__name__, str(exc)[:160])
"""


def _race(db: Path, spec: str, call: str, n: int = 12) -> list[str]:
    """``n`` interpreters run ``call`` on the store at ``db`` at the same instant.

    Each imports first and says ``ready``; only when all have does the parent create the ``go``
    file they poll for, so the open itself is what overlaps however loaded the machine is."""
    go = db.parent / f"{db.name}.go"
    procs = [
        subprocess.Popen(  # noqa: S603 - this interpreter, fixed code
            [sys.executable, "-c", _CHILD, spec, str(db), str(go), call],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"PYTHONPATH": str(SRC), "IRIS_AUTH_SECRET": "x", "PATH": "/usr/bin:/bin"},
        )
        for _ in range(n)
    ]
    for proc in procs:
        assert proc.stdout is not None and proc.stdout.readline().strip() == "ready"
    go.touch()
    out = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=120)
        out.append(stdout.strip() if proc.returncode == 0 else f"FAILED: {stderr.strip()[-200:]}")
    return out


@pytest.mark.parametrize("name", list(STORES))
def test_twelve_interpreters_opening_an_older_database_at_once_all_succeed(
    name: str, tmp_path: Path
) -> None:
    spec, call, _ = STORES[name]
    drops = _drops(name)
    for round_no in range(3):  # a race is probabilistic: three rounds of twelve, red before the fix
        db = tmp_path / f"{name}-{round_no}.db"
        eval(call, {"S": _load(spec), "P": db})  # noqa: S307 - a constant of this module
        _release_shape(db, drops)
        assert all(not set(cols) & set(_columns(db, table)) for table, cols in drops)

        outcomes = _race(db, spec, call)

        assert outcomes == ["ok"] * len(outcomes) and len(outcomes) == 12, outcomes
        for table, cols in drops:  # every column is back, exactly once
            have = _columns(db, table)
            assert set(cols) <= set(have) and len(have) == len(set(have))


# ----------------------------------------------------------------- the helper itself
def test_ensure_columns_looks_first_and_takes_the_write_lock_only_when_it_must(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT)")
    conn.commit()
    conn.close()
    calls: list[str] = []
    real = persistence.add_columns_if_missing

    def spy(*args: Any, **kwargs: Any) -> list[str]:
        calls.append("write")
        return real(*args, **kwargs)

    monkeypatch.setattr(persistence, "add_columns_if_missing", spy)

    assert persistence.ensure_columns(db, "t", {"a": "TEXT"}) == []  # nothing to add
    assert calls == []  # no write lock taken
    assert persistence.ensure_columns(db, "t", {"a": "TEXT", "b": "INTEGER"}) == ["b"]
    assert calls == ["write"]
    assert persistence.ensure_columns(db, "t", {"a": "TEXT", "b": "INTEGER"}) == []
    assert calls == ["write"]  # the second call looked and found nothing
    assert persistence.ensure_columns(db, "missing", {"x": "TEXT"}) == []  # no such table


def test_ensure_columns_makes_the_index_when_only_the_index_is_missing(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, a TEXT)")
    conn.commit()
    conn.close()
    index = "CREATE INDEX IF NOT EXISTS idx_t_a ON t(a)"

    persistence.ensure_columns(db, "t", {"a": "TEXT"}, indexes=(index,))

    conn = sqlite3.connect(db)
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'idx_t_a'").fetchone()
    conn.close()


def test_a_database_removed_and_created_again_at_the_same_path_is_migrated_again(
    tmp_path: Path,
) -> None:
    db = tmp_path / "t.db"
    for _ in range(2):
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        assert persistence.ensure_columns(db, "t", {"a": "TEXT"}) == ["a"]
        db.unlink()


# ------------------------------------------------- the backfills run once, where they belong
def test_the_rag_classification_backfill_runs_in_the_process_that_added_the_column(
    tmp_path: Path,
) -> None:
    from iris_harness.services.rag.store import DocumentStore

    db = tmp_path / "rag.db"
    DocumentStore(db).ensure_schema()
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO document_sources(id, path, kind, title, content_sha, added_at, "
        "last_synced_at, classification) VALUES ('s1', '/a', 'md', 'A', 'x', 't', 't', NULL)"
    )
    for cid, label in (("c1", "public"), ("c2", "secret")):
        conn.execute(
            "INSERT INTO document_chunks(id, source_id, source_path, title, chunk_index, text, "
            "classification) VALUES (?, 's1', '/a', 'A', 0, 'text', ?)",
            (cid, label),
        )
    conn.commit()
    conn.close()
    _release_shape(db, [("document_sources", ["classification"])])  # chunks keep their labels

    outcomes = _race(db, "iris_harness.services.rag.store:DocumentStore", "S(P).ensure_schema()")

    assert outcomes == ["ok"] * len(outcomes) and len(outcomes) == 12, outcomes
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT classification FROM document_sources").fetchall() == [("secret",)]
    # Not re-run by a later open: an app change to the label stays.
    conn.execute("UPDATE document_sources SET classification = 'personal' WHERE id = 's1'")
    conn.commit()
    conn.close()
    DocumentStore(db).ensure_schema()
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT classification FROM document_sources").fetchall() == [("personal",)]
    conn.close()


def test_the_notification_lifecycle_backfill_runs_once_and_never_again(tmp_path: Path) -> None:
    from iris_harness.services.notifications import store as notifications

    db = tmp_path / "reminders.db"
    notifications.ReminderStore(db).ensure_schema()
    conn = sqlite3.connect(db)
    rows = [
        ("dismissed", "2026-10-01T09:00:00+00:00", None, "2026-10-01T10:00:00+00:00"),
        ("fired", "2026-10-01T09:00:00+00:00", "2026-10-01T09:01:00+00:00", None),
        ("waiting", "2026-10-01T09:00:00+00:00", None, None),
    ]
    for rid, remind_at, fired, dismissed in rows:
        conn.execute(
            "INSERT INTO notification_reminders(id, target_kind, target_id, remind_at, "
            "created_at, fired_at, dismissed_at) VALUES (?, 'task', ?, ?, 't', ?, ?)",
            (rid, rid, remind_at, fired, dismissed),
        )
    conn.commit()
    conn.close()
    _release_shape(db, _drops("reminders"))

    outcomes = _race(
        db, "iris_harness.services.notifications.store:ReminderStore", "S(P).ensure_schema()"
    )

    assert outcomes == ["ok"] * len(outcomes) and len(outcomes) == 12, outcomes
    conn = sqlite3.connect(db)
    got = dict(conn.execute("SELECT id, status FROM notification_reminders").fetchall())
    assert got == {"dismissed": "cancelled", "fired": "sent", "waiting": "pending"}
    # The app moves a row on; a later open must not run the backfill again and undo it.
    conn.execute("UPDATE notification_reminders SET status = 'pending' WHERE id = 'fired'")
    conn.commit()
    conn.close()
    notifications.ReminderStore(db).ensure_schema()
    conn = sqlite3.connect(db)
    assert conn.execute(
        "SELECT status FROM notification_reminders WHERE id = 'fired'"
    ).fetchone() == ("pending",)
    conn.close()


def test_the_email_source_backfill_stamps_old_classifications_once_and_never_again(
    tmp_path: Path,
) -> None:
    from iris_personal.email.store import EmailStore

    db = tmp_path / "email.db"
    EmailStore(db).ensure_schema()
    conn = sqlite3.connect(db)
    for mid, category in (("classified", "bills"), ("unclassified", None)):
        conn.execute(
            "INSERT INTO emails(id, provider, account_id, from_address, received_at, "
            "classified_category, created_at, updated_at) VALUES (?, 'p', 'a', 'x@y.z', 't', ?, 't', 't')",
            (mid, category),
        )
    conn.commit()
    conn.close()
    _release_shape(db, [("emails", ["classified_source"])])  # rows keep their categories

    outcomes = _race(db, "iris_personal.email.store:EmailStore", "S(P).ensure_schema()")

    assert outcomes == ["ok"] * len(outcomes) and len(outcomes) == 12, outcomes
    conn = sqlite3.connect(db)
    got = dict(conn.execute("SELECT id, classified_source FROM emails").fetchall())
    assert got == {
        "classified": "iris",
        "unclassified": None,
    }  # stamped once, only where classified
    # A vendor category that arrives later must not be re-stamped by a later open.
    conn.execute("UPDATE emails SET classified_source = 'vendor' WHERE id = 'classified'")
    conn.commit()
    conn.close()
    EmailStore(db).ensure_schema()
    conn = sqlite3.connect(db)
    assert conn.execute(
        "SELECT classified_source FROM emails WHERE id = 'classified'"
    ).fetchone() == ("vendor",)
    conn.close()


# ----------------------------------------------------------------------- the guard
# ``ALTER TABLE`` belongs in the helper. memris keeps its own versioned upgrade path (a schema
# version row, each upgrade ALTER already tolerating "duplicate column", connections with a 30 s
# timeout): measured with 12 interpreters x 15 rounds opening a v4-shaped file, zero errors, so it
# is not this bug (#201).
_ALTER_ALLOWED = {
    "src/iris_harness/foundation/persistence/sqlite.py",
    "src/memris/store/sqlite.py",
}


def _runs_alter_table(source: str) -> bool:
    """Whether the module builds an ``ALTER TABLE`` statement (a string that starts with it,
    including the head of an f-string); a docstring or comment that mentions it does not count."""
    for node in ast.walk(ast.parse(source)):
        head: str | None = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            head = node.value
        elif isinstance(node, ast.JoinedStr) and node.values:
            first = node.values[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                head = first.value
        if head is not None and re.match(r"\s*ALTER\s+TABLE\b", head, re.IGNORECASE):
            return True
    return False


def test_no_store_runs_alter_table_outside_the_helper() -> None:
    root = SRC.parent
    offenders = []
    for path in sorted((root / "src").rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if rel in _ALTER_ALLOWED:
            continue
        if _runs_alter_table(path.read_text(encoding="utf-8")):
            offenders.append(rel)
    assert offenders == [], (
        "a migration that runs ALTER TABLE itself is the read-then-ALTER race of #201: "
        f"call foundation.persistence.sqlite.ensure_columns instead ({offenders})"
    )
