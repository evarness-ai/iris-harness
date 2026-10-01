"""Unit tests for the research TTL cache."""

from __future__ import annotations

import time

from iris_harness.plugins_builtin.research.cache import ResearchCache


def test_put_get_round_trip(tmp_path) -> None:
    cache = ResearchCache(tmp_path / "rc.db", ttl_seconds=3600)
    payload = {"query": "q", "results": [{"title": "t"}], "n": 3}
    cache.put("k1", payload)
    assert cache.get("k1") == payload


def test_get_missing_key_returns_none(tmp_path) -> None:
    cache = ResearchCache(tmp_path / "rc.db")
    assert cache.get("nope") is None


def test_get_expired_returns_none(tmp_path) -> None:
    cache = ResearchCache(tmp_path / "rc.db", ttl_seconds=0)
    cache.put("k", {"a": 1})
    # ttl_seconds=0 means anything inserted in the past is already stale.
    time.sleep(0.01)
    assert cache.get("k") is None


def test_make_key_stable_and_distinct() -> None:
    a = ResearchCache.make_key("Python JSON", "web", 5, True)
    b = ResearchCache.make_key("python json", "web", 5, True)  # normalized -> same
    assert a == b
    c = ResearchCache.make_key("python json", "news", 5, True)
    d = ResearchCache.make_key("python json", "web", 10, True)
    e = ResearchCache.make_key("python json", "web", 5, False)
    assert len({a, c, d, e}) == 4


def test_purge_expired_removes_stale_rows(tmp_path) -> None:
    cache = ResearchCache(tmp_path / "rc.db", ttl_seconds=0)
    cache.put("k1", {"a": 1})
    cache.put("k2", {"b": 2})
    time.sleep(0.01)
    removed = cache.purge_expired()
    assert removed == 2
    assert cache.get("k1") is None


def test_purge_keeps_fresh_rows(tmp_path) -> None:
    cache = ResearchCache(tmp_path / "rc.db", ttl_seconds=3600)
    cache.put("fresh", {"a": 1})
    assert cache.purge_expired() == 0
    assert cache.get("fresh") == {"a": 1}
