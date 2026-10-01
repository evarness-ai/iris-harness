"""Shared persistence primitives.

Exposes a hardened SQLite connection helper used by the app-tier stores, plus
``data_path`` for resolving local data files under the runtime data dir.
See ``sqlite.py``.
"""

from __future__ import annotations

from pathlib import Path

from iris_harness.foundation.paths import data_dir
from iris_harness.foundation.persistence.sqlite import connect, sqlite_conn, with_locked_retry

__all__ = ["connect", "data_dir", "data_path", "sqlite_conn", "with_locked_retry"]


def data_path(name: str) -> Path:
    """Resolve a local data file (e.g. ``"calendar.db"``) under the runtime data dir.

    The directory is ``paths.data_dir()``: ``$IRIS_DATA_DIR`` (a throwaway temp dir
    under pytest, an isolated dir for eval/sandbox instances), else ``$IRIS_HOME/data``,
    else the checkout's ``data/``, else ``~/.iris/data`` -- never the current directory.
    Store ``db_path`` defaults MUST use this rather
    than a bare ``Path("data/x.db")``: a hardcoded relative path ignores the data-dir
    override, so a bare ``Store()`` reads the developer's REAL data. That is exactly
    how tests came to read the real calendar. Mirrors the IRIS_HOME/IRIS_DATA_DIR rule
    in the root CLAUDE.md.
    """
    return data_dir() / name
