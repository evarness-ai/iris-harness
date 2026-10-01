"""The app-tier stores open SQLite through the hardened helper and close every handle.

Each store here used to call ``sqlite3.connect`` itself: no WAL, and several through
``with sqlite3.connect(...)``, which commits but never closes. Building each one runs its
schema through its own connection path, so that is what these tests watch.

The email domain's two stores (categories, email accounts) left the core with their
modules (core/SDK boundary plan, PR 2); their cases are in
``tests/unit/iris_personal/test_email/test_store_connections.py``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from iris_harness.kernel.governor.audit import GovernorAuditLogger
from iris_harness.memory.state.continuations import ContinuationStore
from iris_harness.memory.state.store import CheckpointStore
from iris_harness.runtime.router_audit import RouterAuditLogger
from iris_harness.services.channels.web_push.store import PushSubscriptionStore
from iris_harness.services.digests.store import DigestStore

STORES: dict[str, Callable[[Path], Any]] = {
    "digests": DigestStore,
    "web_push": PushSubscriptionStore,
    "checkpoints": lambda p: CheckpointStore(db_path=p),
    "continuations": lambda p: ContinuationStore(db_path=p),
    # Creates its database on first use, not on construction: build it, then read once.
    "governor_audit": lambda p: GovernorAuditLogger(p).list_entries(),
    "router_audit": RouterAuditLogger,
}


@pytest.mark.parametrize("name", sorted(STORES))
def test_the_store_is_wal_and_leaves_no_handle_open(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def watching(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", watching)
    db = tmp_path / f"{name}.db"
    STORES[name](db)
    monkeypatch.undo()

    assert opened, "the store opened no connection"
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")
    check = sqlite3.connect(db)
    try:
        assert check.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        check.close()
