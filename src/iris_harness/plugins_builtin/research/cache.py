"""A small, local, privacy-first TTL cache for research results.

Keyed by a stable hash of the normalized request (query + lens + limits), value is
the JSON-serialized engine payload plus an insertion timestamp. Entries past
``ttl_seconds`` are treated as misses and lazily/explicitly purged.

Stored in a single SQLite file under the data dir (WAL, short busy timeout). Every
public method swallows errors and degrades to a miss/no-op — a flaky cache must
never break a research turn.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from iris_harness.sdk.persistence import data_dir

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS research_cache (
    key        TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL
)
"""

_DEFAULT_REDIS_URL = "redis://localhost:6379/0"


def _make_cache_key(query: str, search_type: str, max_results: int, fetch_content: bool) -> str:
    """Stable sha256 hex of the normalized request tuple.

    Shared by every cache backend so a key computed for the SQLite cache resolves
    to the same entry in the Redis cache and vice versa.
    """
    normalized = json.dumps(
        [query.strip().lower(), search_type, int(max_results), bool(fetch_content)],
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class ResearchCache:
    """SQLite-backed TTL cache for ``ResearchResult`` payloads."""

    def __init__(self, db_path: Path | None = None, *, ttl_seconds: int = 3600) -> None:
        if db_path is None:
            db_path = data_dir() / "research_cache.db"
        self.db_path = db_path
        self.ttl_seconds = ttl_seconds
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def make_key(query: str, search_type: str, max_results: int, fetch_content: bool) -> str:
        """Stable sha256 hex of the normalized request tuple."""
        return _make_cache_key(query, search_type, max_results, fetch_content)

    def get(self, key: str) -> dict[str, object] | None:
        """Return the cached payload if present and within TTL, else ``None``."""
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT payload, created_at FROM research_cache WHERE key = ?",
                    (key,),
                ).fetchone()
            if row is None:
                return None
            payload, created_at = row
            if time.time() - float(created_at) > self.ttl_seconds:
                return None
            result = json.loads(payload)
            return result if isinstance(result, dict) else None
        except Exception:  # a broken cache is a miss, never an error
            logger.debug("ResearchCache.get failed for key %r", key, exc_info=True)
            return None

    def put(self, key: str, payload: dict[str, object]) -> None:
        """Upsert ``payload`` under ``key`` with the current timestamp."""
        try:
            blob = json.dumps(payload)
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO research_cache (key, payload, created_at) "
                    "VALUES (?, ?, ?)",
                    (key, blob, time.time()),
                )
        except Exception:  # failing to cache is non-fatal
            logger.debug("ResearchCache.put failed for key %r", key, exc_info=True)

    def purge_expired(self) -> int:
        """Delete entries older than the TTL; return the number removed."""
        try:
            cutoff = time.time() - self.ttl_seconds
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM research_cache WHERE created_at < ?",
                    (cutoff,),
                )
                return cur.rowcount or 0
        except Exception:  # purge is best-effort
            logger.debug("ResearchCache.purge_expired failed", exc_info=True)
            return 0


class RedisCache:
    """Redis-backed TTL cache for ``ResearchResult`` payloads.

    Mirrors :class:`ResearchCache`'s public interface, but leans on Redis key
    expiry for the TTL (no manual timestamp check). ``redis`` is an optional
    extra, imported lazily in ``__init__``; a missing package or an unreachable
    server raises ``RuntimeError`` so the factory can fall back to SQLite.
    """

    def __init__(self, *, ttl_seconds: int = 3600, url: str | None = None) -> None:
        self.ttl_seconds = ttl_seconds
        self.url = url or os.environ.get("IRIS_REDIS_URL") or _DEFAULT_REDIS_URL
        try:
            import redis  # type: ignore[import-not-found]  # optional extra

            self._client = redis.Redis.from_url(self.url)
            self._client.ping()
        except Exception as exc:  # surface a clear, catchable failure
            raise RuntimeError(f"redis backend unavailable: {exc}") from exc

    @staticmethod
    def make_key(query: str, search_type: str, max_results: int, fetch_content: bool) -> str:
        """Stable sha256 hex of the normalized request tuple (shared with SQLite)."""
        return _make_cache_key(query, search_type, max_results, fetch_content)

    def get(self, key: str) -> dict[str, object] | None:
        """Return the cached payload if the key is live, else ``None``."""
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            result = json.loads(raw)
            return result if isinstance(result, dict) else None
        except Exception:  # a broken cache is a miss, never an error
            logger.debug("RedisCache.get failed for key %r", key, exc_info=True)
            return None

    def put(self, key: str, payload: dict[str, object]) -> None:
        """Store ``payload`` under ``key`` with a TTL via Redis key expiry."""
        try:
            self._client.set(key, json.dumps(payload), ex=self.ttl_seconds)
        except Exception:  # failing to cache is non-fatal
            logger.debug("RedisCache.put failed for key %r", key, exc_info=True)

    def purge_expired(self) -> int:
        """No-op: Redis auto-expires keys. Always returns 0."""
        return 0


def build_cache(*, ttl_seconds: int = 3600) -> ResearchCache | RedisCache | None:
    """Construct the configured research cache backend. Never raises.

    Honors ``IRIS_RESEARCH_CACHE``: ``"off"``/``"none"`` disables caching; ``"redis"``
    uses Redis, falling back to SQLite if Redis is missing or unreachable; anything
    else (incl. ``"sqlite"``) uses the SQLite ``ResearchCache``.
    """
    backend = (os.environ.get("IRIS_RESEARCH_CACHE") or "sqlite").strip().lower()
    if backend in {"off", "none"}:
        return None
    if backend == "redis":
        try:
            return RedisCache(ttl_seconds=ttl_seconds)
        except Exception as exc:  # noqa: BLE001 - fall back to SQLite, never raise
            logger.warning("redis cache unavailable, falling back to sqlite: %s", exc)
            return ResearchCache(ttl_seconds=ttl_seconds)
    return ResearchCache(ttl_seconds=ttl_seconds)


__all__ = ["RedisCache", "ResearchCache", "build_cache"]
