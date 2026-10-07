"""Several interpreters opening one memris database at the same instant all succeed.

``SQLiteGraphStore.__init__`` read the schema version and then inserted it (or upgraded the
tables) inside a plain transaction, so on a database with no version row two processes could both
read "none" and both insert (``UNIQUE constraint failed: memris_meta.key``); on an older database
both could upgrade. The processes here are real interpreters released by one clock, on the three
shapes a process meets: no file at all, the current version, and an older version.
"""

# S603: the subprocesses run this repo's own interpreter on fixed code.
# ruff: noqa: S603

from __future__ import annotations

import re
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

import memris
from memris.store import SQLiteGraphStore
from memris.store import sqlite as store_module

SRC = str(Path(memris.__file__).resolve().parents[1])
SCHEMA_VERSION = store_module.SCHEMA_VERSION
PROCESSES = 6

_CHILD = textwrap.dedent("""
    import sys, time
    from memris.store import SQLiteGraphStore
    path, start = sys.argv[1], float(sys.argv[2])
    while time.time() < start:
        time.sleep(0.001)
    SQLiteGraphStore(path)
    print("ok")
""")


def _downgrade(path: Path, version: int) -> None:
    """Make a current database look like one at ``version``: drop what the upgrades from
    ``version`` onward add, and write that version."""
    conn = sqlite3.connect(path)
    for v in range(SCHEMA_VERSION - 1, version - 1, -1):
        for sql in store_module._UPGRADES[v]:
            match = re.match(r"ALTER TABLE (\w+) ADD COLUMN (\w+)", sql)
            assert match, sql
            conn.execute(f"ALTER TABLE {match[1]} DROP COLUMN {match[2]}")
    conn.execute("UPDATE memris_meta SET value = ? WHERE key = 'schema_version'", (str(version),))
    conn.commit()
    conn.close()


def _race(path: Path) -> list[str]:
    start = time.time() + 5.0
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(path), str(start)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"},
        )
        for _ in range(PROCESSES)
    ]
    outcomes = []
    for proc in procs:
        out, err = proc.communicate(timeout=120)
        outcomes.append(out.strip() if proc.returncode == 0 else f"FAILED: {err.strip()[-300:]}")
    return outcomes


def _version(path: Path) -> int:
    conn = sqlite3.connect(path)
    try:
        (value,) = conn.execute(
            "SELECT value FROM memris_meta WHERE key = 'schema_version'"
        ).fetchone()
        return int(value)
    finally:
        conn.close()


def _columns(path: Path, table: str) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


@pytest.mark.parametrize("round_", range(3))
def test_opening_a_database_that_does_not_exist_yet_at_once(tmp_path: Path, round_: int) -> None:
    """The missing case: no file, so every process sees no version row and inserts one."""
    path = tmp_path / f"fresh-{round_}.db"

    assert _race(path) == ["ok"] * PROCESSES
    assert _version(path) == SCHEMA_VERSION


def test_opening_a_current_database_at_once(tmp_path: Path) -> None:
    path = tmp_path / "current.db"
    SQLiteGraphStore(path)

    assert _race(path) == ["ok"] * PROCESSES
    assert _version(path) == SCHEMA_VERSION


@pytest.mark.parametrize("behind", [1, 2])
def test_opening_an_older_database_at_once(tmp_path: Path, behind: int) -> None:
    """One version behind (and two): both processes upgrade; the columns appear once."""
    path = tmp_path / "older.db"
    SQLiteGraphStore(path)
    _downgrade(path, SCHEMA_VERSION - behind)

    assert _race(path) == ["ok"] * PROCESSES
    assert _version(path) == SCHEMA_VERSION
    columns = _columns(path, "memris_entities")
    assert columns.count("removed_at") == 1 and columns.count("removed_statements") == 1
    SQLiteGraphStore(path)  # and it opens again
