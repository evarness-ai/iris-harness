"""Where a plugin keeps its data.

A plugin that owns a table or a file resolves it with `data_path(name)` (or the
directory itself with `data_dir()`), never a bare ``Path("data/x.db")``: both honour
``$IRIS_DATA_DIR`` and ``$IRIS_HOME``, so a test, an eval run or a sandbox instance
never reads the owner's real data, and an installed ``iris`` never writes into the
directory it happens to run from. `connect` / `sqlite_conn`
open SQLite in WAL mode with a busy timeout (`sqlite_conn` also commits and closes);
`with_locked_retry` retries a write that still meets a lock; `collection_kwargs` pins the one process-wide embedder on a ChromaDB collection
so each plugin's index does not load its own copy of the model.

`ensure_columns(db_path, table, {column: declaration})` adds the columns a table lacks, safely
when several processes open an older database at once (the API, the CLI, a heartbeat): a plugin that
reads ``PRAGMA table_info`` and then runs ``ALTER TABLE ... ADD COLUMN`` itself loses that race
with ``duplicate column name``. It looks first, then takes the write lock on a connection of its own
(``BEGIN IMMEDIATE``) and decides again inside it; an ``on_added`` callback runs in the same
transaction, only in the process that added a column, for a one-time backfill. Declare nullable
columns without a default unless old rows must read a value.
"""

from __future__ import annotations

from iris_harness.foundation.persistence import (
    connect,
    data_dir,
    data_path,
    sqlite_conn,
    with_locked_retry,
)
from iris_harness.foundation.persistence.embedding import collection_kwargs
from iris_harness.foundation.persistence.sqlite import ensure_columns

__all__ = [
    "collection_kwargs",
    "connect",
    "data_dir",
    "data_path",
    "ensure_columns",
    "sqlite_conn",
    "with_locked_retry",
]
