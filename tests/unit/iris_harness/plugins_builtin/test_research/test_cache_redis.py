"""Unit tests for the research cache factory and Redis backend fallback.

Hermetic: ``redis`` is an optional extra that isn't installed in the test env, so
``IRIS_RESEARCH_CACHE=redis`` must transparently fall back to the SQLite cache. The
sqlite fallback is pinned to ``tmp_path`` so it never touches the real data dir.
"""

from __future__ import annotations

import pytest

from iris_harness.plugins_builtin.research.cache import (
    RedisCache,
    ResearchCache,
    _make_cache_key,
    build_cache,
)


@pytest.fixture(autouse=True)
def _isolate_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("IRIS_RESEARCH_CACHE", raising=False)


def test_build_cache_defaults_to_sqlite() -> None:
    cache = build_cache()
    assert isinstance(cache, ResearchCache)


def test_build_cache_off_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_RESEARCH_CACHE", "off")
    assert build_cache() is None
    monkeypatch.setenv("IRIS_RESEARCH_CACHE", "none")
    assert build_cache() is None


def test_build_cache_redis_falls_back_to_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    # redis isn't installed -> RedisCache.__init__ raises -> factory falls back, no crash.
    monkeypatch.setenv("IRIS_RESEARCH_CACHE", "redis")
    cache = build_cache()
    assert isinstance(cache, ResearchCache)


def test_build_cache_sqlite_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_RESEARCH_CACHE", "sqlite")
    assert isinstance(build_cache(), ResearchCache)


def test_redis_cache_unavailable_raises_runtime_error() -> None:
    # No redis package / server in the test env -> a clear, catchable RuntimeError.
    with pytest.raises(RuntimeError, match="redis backend unavailable"):
        RedisCache(ttl_seconds=60)


def test_make_key_identical_across_backends() -> None:
    args = ("Python JSON", "web", 5, True)
    shared = _make_cache_key(*args)
    assert ResearchCache.make_key(*args) == shared
    assert RedisCache.make_key(*args) == shared


def test_make_key_stable_and_normalized() -> None:
    a = ResearchCache.make_key("Python JSON", "web", 5, True)
    b = ResearchCache.make_key("python json", "web", 5, True)
    assert a == b
