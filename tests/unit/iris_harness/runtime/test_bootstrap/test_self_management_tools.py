"""Tests for the agent self-management tools (ADR-0086)."""

from __future__ import annotations

from typing import Any

from iris_harness.runtime.handlers.react import _self_management_tools
from iris_harness.runtime.react_tools import agent_self_management_enabled


class _FakeStore:
    def list_behavior_proposals(self, status: str = "pending") -> list[Any]:
        return [object()] if status in ("approved", "pending") else []

    def list_intentions(self, status: str = "proposed") -> list[Any]:
        return []


class _FakeLearning:
    def learning_flags(self) -> dict[str, dict[str, bool]]:
        return {
            "behavior_miner": {"enabled": True, "env_default": False},
            "intention_rollup": {"enabled": False, "env_default": False},
        }


class _FakeSessions:
    def context_health(self, session_id: str) -> dict[str, Any]:
        return {
            "window": {
                "fill_pct": 0.85,
                "current_tokens": 5200,
                "budget_tokens": 6144,
                "near_full": True,
                "last_compaction": {"archived_count": 6},
            },
            "budgets": {"transcript_budget": 2764},
            "suppression": {"active_suppressions": 2},
        }

    def compact_now(self, session_id: str) -> dict[str, Any]:
        return {
            "compacted": True,
            "archived_count": 6,
            "tokens_before": 5200,
            "tokens_after": 2400,
        }


class _FakeRuntime:
    learning_store = _FakeStore()
    sessions = _FakeSessions()
    learning = _FakeLearning()


def _tools(holder: list[Any], session_id: str = "s1") -> dict[str, Any]:
    return {t.name: t for t in _self_management_tools(holder, session_id)}


def test_three_tools_built() -> None:
    t = _tools([_FakeRuntime()])
    assert set(t) == {"context_health", "compact_context", "learning_status"}


def test_context_health_tool_reports_fill_and_pressure() -> None:
    t = _tools([_FakeRuntime()])
    out = t["context_health"].call({})
    assert "85%" in out and "NEAR FULL" in out
    assert "archived 6 turns" in out


def test_compact_tool_reports_reclaimed_tokens() -> None:
    t = _tools([_FakeRuntime()])
    out = t["compact_context"].call({})
    assert "summarized 6 older turns" in out
    assert "2800" in out  # 5200 - 2400 reclaimed


def test_learning_status_tool_reports_miners_and_tallies() -> None:
    t = _tools([_FakeRuntime()])
    out = t["learning_status"].call({})
    assert "behavior_miner" in out  # the on miner is named
    assert "accepted" in out


def test_tools_degrade_without_runtime() -> None:
    t = _tools([])  # holder not filled yet
    assert "unavailable" in t["context_health"].call({}).lower()
    assert "cannot compact" in t["compact_context"].call({}).lower()


def test_flag_off_by_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("IRIS_AGENT_SELF_MANAGEMENT", raising=False)
    assert agent_self_management_enabled() is False
    monkeypatch.setenv("IRIS_AGENT_SELF_MANAGEMENT", "1")
    assert agent_self_management_enabled() is True
