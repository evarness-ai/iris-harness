"""Where a plugin keeps its data.

A plugin that owns a table or a file resolves it with `data_path(name)` (or the
directory itself with `data_dir()`), never a bare ``Path("data/x.db")``: both honour
``$IRIS_DATA_DIR`` and ``$IRIS_HOME``, so a test, an eval run or a sandbox instance
never reads the owner's real data, and an installed ``iris`` never writes into the
directory it happens to run from. `connect` / `sqlite_conn`
open SQLite in WAL mode with a busy timeout (`sqlite_conn` also commits and closes);
`with_locked_retry` retries a write that still meets a lock; `collection_kwargs` pins the one process-wide embedder on a ChromaDB collection
so each plugin's index does not load its own copy of the model.
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

__all__ = [
    "collection_kwargs",
    "connect",
    "data_dir",
    "data_path",
    "sqlite_conn",
    "with_locked_retry",
]
