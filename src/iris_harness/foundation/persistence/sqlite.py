"""Hardened SQLite connection helper for the app-tier stores.

Why this exists
---------------
The governance stores already open SQLite in WAL mode with autocommit
(``governance/audit/log.py``), but the app-tier stores (``memory.db`` and friends)
historically opened a raw ``sqlite3.connect(path)`` per call with **no WAL, no busy
timeout, and no ``close()``** — the bare ``with sqlite3.connect(...)`` commits but leaks
the handle. With background heartbeats writing concurrently with the request path, that
risks ``database is locked`` errors that silently drop a write (e.g. a learned user fact).

``sqlite_conn`` centralizes the safe settings; ``with_locked_retry`` adds bounded retry +
**lock telemetry** (the logging is deliberate — it surfaces contention that was previously
invisible).

This is intentionally the memory-store's first adopter; other app stores migrate in a
follow-up so the blast radius stays small.
"""

from __future__ import annotations

import logging
import random
import re
import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, overload

logger = logging.getLogger(__name__)

# 5s aligns with PRAGMA busy_timeout below; both throttle writer-vs-writer contention.
_DEFAULT_TIMEOUT_S = 5.0
_BUSY_TIMEOUT_MS = 5000

F = TypeVar("F", bound=Callable[..., Any])


def connect(
    db_path: str | Path,
    *,
    timeout: float = _DEFAULT_TIMEOUT_S,
    row_factory: Any = None,
) -> sqlite3.Connection:
    """Open a hardened raw SQLite connection (WAL + busy_timeout + synchronous=NORMAL).

    For the app-tier stores that manage their own connection lifecycle through a
    ``_connect()`` method — route that method through here to get WAL + busy_timeout
    without touching every call site. The CALLER owns commit/close (same contract as the
    raw ``sqlite3.connect`` it replaces). For new code prefer the ``sqlite_conn`` context
    manager, which also commits + closes.

    WAL persists on the DB file once set; busy_timeout makes concurrent writers
    block-and-retry at the C level (the real fix for ``database is locked`` under
    background-heartbeat + request-path contention) instead of failing instantly; NORMAL
    is the WAL-appropriate durability/perf trade-off.
    """
    conn = sqlite3.connect(str(db_path), timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA synchronous=NORMAL")
    if row_factory is not None:
        conn.row_factory = row_factory
    return conn


@contextmanager
def sqlite_conn(
    db_path: str | Path,
    *,
    timeout: float = _DEFAULT_TIMEOUT_S,
    row_factory: Any = None,
) -> Iterator[sqlite3.Connection]:
    """Open a hardened SQLite connection as a context manager.

    Mirrors the commit/rollback semantics of ``with sqlite3.connect(...)`` (commit on
    clean exit, rollback on exception) but ALSO closes the handle in ``finally`` — fixing
    the connection leak in the old call sites. Hardening (WAL + busy_timeout +
    synchronous=NORMAL) comes from ``connect`` above.
    """
    conn = connect(db_path, timeout=timeout, row_factory=row_factory)
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def add_columns_if_missing(
    db_path: str | Path,
    table: str,
    columns: Mapping[str, str],
    *,
    indexes: Sequence[str] = (),
    on_added: Callable[[sqlite3.Connection, list[str]], None] | None = None,
    timeout: float = _DEFAULT_TIMEOUT_S,
) -> list[str]:
    """Add the nullable ``columns`` ``table`` lacks, safely while other processes open it.

    The one way a store migrates a database a released version created (issue #134, stage 3).
    It runs on its OWN connection under ``BEGIN IMMEDIATE``, never inside a store's own
    transaction: a read followed by an ``ALTER`` in a deferred transaction fails with
    ``database is locked`` the moment another process commits in between, and the busy
    timeout does not help (``SQLITE_BUSY_SNAPSHOT`` skips the busy handler). Taking the write
    lock first makes the process that gets it do the work and the rest wait, then find nothing
    to add.

    ``columns`` maps a name to its declaration (``"TEXT"``, ``"INTEGER"``): nullable, no
    default, so no existing row is rewritten. ``indexes`` are ``CREATE INDEX IF NOT EXISTS``
    statements run after the columns exist. ``on_added`` runs inside the same transaction,
    only in the process that actually added columns, so a boundary recorded there is written
    exactly once. Returns the names added (empty when the table is absent or already
    current), so a second run is a no-op.
    """
    for name in (table, *columns):
        if not _IDENTIFIER.fullmatch(name):
            raise ValueError(f"not a plain SQL identifier: {name!r}")
    conn = sqlite3.connect(str(db_path), timeout=timeout, isolation_level=None)
    try:
        conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
        conn.execute("BEGIN IMMEDIATE")
        try:
            have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            added: list[str] = []
            if have:
                for name, declaration in columns.items():
                    if name in have:
                        continue
                    try:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
                    except sqlite3.OperationalError as exc:
                        if "duplicate column" not in str(exc).lower():
                            raise
                    else:
                        added.append(name)
                for statement in indexes:
                    conn.execute(statement)
                if added and on_added is not None:
                    on_added(conn, added)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
        return added
    finally:
        conn.close()


@overload
def with_locked_retry(func: F) -> F: ...


@overload
def with_locked_retry(*, attempts: int = ..., base_delay: float = ...) -> Callable[[F], F]: ...


def with_locked_retry(
    func: F | None = None,
    *,
    attempts: int = 5,
    base_delay: float = 0.05,
) -> F | Callable[[F], F]:
    """Retry a method on ``sqlite3.OperationalError("database is locked")``.

    Belt-and-suspenders on top of ``busy_timeout``: if a write still surfaces a lock, retry
    with bounded exponential backoff + jitter, **logging each occurrence** so contention is
    observable (previously it was silent). Re-raises any non-lock OperationalError
    immediately, and re-raises the lock error after the final attempt.

    Decorate the *whole* write method (not the connection) — the retry must re-run the
    entire transaction, which a context manager cannot do on its own.
    """

    def decorate(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = base_delay
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except sqlite3.OperationalError as exc:
                    if "database is locked" not in str(exc).lower() or attempt == attempts:
                        raise
                    logger.warning(
                        "sqlite: database is locked in %s (attempt %d/%d); retrying in ~%.0fms",
                        fn.__name__,
                        attempt,
                        attempts,
                        delay * 1000,
                    )
                    # S311 is fine here: jitter for backoff, not a security/crypto context.
                    time.sleep(delay + random.uniform(0, base_delay))  # noqa: S311
                    delay *= 2
            # Unreachable: the loop either returns or raises on the final attempt.
            raise AssertionError(
                "with_locked_retry exhausted without returning"
            )  # pragma: no cover

        return wrapper  # type: ignore[return-value]

    # Support both @with_locked_retry and @with_locked_retry(attempts=...)
    if func is not None:
        return decorate(func)
    return decorate
