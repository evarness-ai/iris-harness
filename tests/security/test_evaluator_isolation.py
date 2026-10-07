"""Isolation property test for the out-of-process evaluator (story 12.gov-3.9 / AC-5).

Design §9.3: the evaluator's SQLite DB is **read-only from the
agent's process credentials**. In production this is enforced by
running the evaluator as a separate UNIX user/group with the DB owned
by that user. CI runs as a single user, so the test asserts the
weaker (but still load-bearing) property: when the DB file is chmod
read-only, the agent process cannot write to it — neither via a
normal SQLite write nor via the URI ``mode=ro`` connection helper.

This test would catch a regression where someone bypasses the
hardening (e.g. by re-opening the DB with write privileges from a
helper path).
"""

from __future__ import annotations

import os
import sqlite3
import stat
import sys
from pathlib import Path

import pytest

from iris_harness.kernel.governance.evaluator.client import isolation_db_open_readonly

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX chmod semantics; Windows ACL is out of scope for this property test",
)


@pytest.fixture()
def readonly_db(tmp_path: Path) -> Path:
    """Create a tiny SQLite DB, chmod it 0o400, return the path."""
    db_path = tmp_path / "evaluator.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            "CREATE TABLE signal_state (k TEXT PRIMARY KEY, v TEXT);"
            "INSERT INTO signal_state(k, v) VALUES ('init', 'ok');"
        )
        conn.commit()
    finally:
        conn.close()
    os.chmod(db_path, 0o400)
    return db_path


def test_readonly_db_rejects_normal_write(readonly_db: Path) -> None:
    """A vanilla sqlite3.connect call to a 0o400 file must fail on write.

    On most POSIX setups the connect itself fails (the journal can't be
    created); on others the SELECT works but any INSERT raises. Either
    is acceptable — the property is "no successful write".
    """
    wrote = False
    try:
        conn = sqlite3.connect(readonly_db)
        try:
            conn.execute("INSERT INTO signal_state(k, v) VALUES ('x', 'y')")
            conn.commit()
            wrote = True
        finally:
            conn.close()
    except (sqlite3.OperationalError, sqlite3.DatabaseError, PermissionError):
        wrote = False
    assert wrote is False, "agent process succeeded in writing to a 0o400 evaluator DB"


# rw for owner, group and other, built from flags (the point is a too-wide mode).
_WORLD_WRITABLE = (
    stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH
)


def test_isolation_helper_returns_readonly_connection(readonly_db: Path) -> None:
    """The defense-in-depth helper opens the DB in ``mode=ro`` even if
    the FS perms are misconfigured. A SELECT must succeed; an INSERT
    must raise.
    """
    conn = isolation_db_open_readonly(str(readonly_db))
    try:
        rows = list(conn.execute("SELECT k, v FROM signal_state"))
        assert rows == [("init", "ok")]
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO signal_state(k, v) VALUES ('x', 'y')")
    finally:
        conn.close()


def test_isolation_helper_works_even_with_writable_file(tmp_path: Path) -> None:
    """``mode=ro`` is the second line of defense; even when the FS perms
    are wrong, the SQLite URI keeps writes out.
    """
    db_path = tmp_path / "wide_open.db"
    sqlite3.connect(db_path).executescript("CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1);")
    # Intentionally world-writable to prove the URI mode=ro guard works
    # even when the filesystem layer is misconfigured.
    os.chmod(db_path, _WORLD_WRITABLE)

    conn = isolation_db_open_readonly(str(db_path))
    try:
        assert list(conn.execute("SELECT * FROM t")) == [(1,)]
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO t VALUES (2)")
    finally:
        conn.close()


def test_chmod_remains_after_test_completes(readonly_db: Path) -> None:
    """Sanity-check the fixture: the file really is 0o400."""
    mode = readonly_db.stat().st_mode & 0o777
    assert mode == 0o400, f"expected 0o400, got {oct(mode)}"
