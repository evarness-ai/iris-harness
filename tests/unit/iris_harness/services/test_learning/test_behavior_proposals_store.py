"""Tests for the behavior-pattern proposals store (HITL review queue)."""

from __future__ import annotations

from pathlib import Path

from iris_harness.services.learning.store import LearningMetricsStore


def _store(tmp_path: Path) -> LearningMetricsStore:
    s = LearningMetricsStore(db_path=tmp_path / "learning.db")
    s.ensure_schema()
    return s


def test_propose_and_list(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.propose_behavior_pattern("p1", "Asks for inbox each morning", "high", ["mon", "tue"])
    pending = s.list_behavior_proposals(status="pending")
    assert len(pending) == 1
    assert pending[0].text == "Asks for inbox each morning"
    assert pending[0].evidence == ("mon", "tue")
    assert pending[0].confidence == "high"


def test_propose_dedups_same_id(tmp_path: Path) -> None:
    s = _store(tmp_path)
    assert s.propose_behavior_pattern("p1", "x", "low", []) is True
    assert s.propose_behavior_pattern("p1", "x", "low", []) is False  # already queued
    assert len(s.list_behavior_proposals()) == 1


def test_resolve_moves_out_of_pending(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.propose_behavior_pattern("p1", "x", "low", [])
    assert s.resolve_behavior_proposal("p1", "approved") is True
    assert s.list_behavior_proposals(status="pending") == []
    assert len(s.list_behavior_proposals(status="approved")) == 1
    assert s.get_behavior_proposal("p1").status == "approved"


def test_resolve_missing_returns_false(tmp_path: Path) -> None:
    assert _store(tmp_path).resolve_behavior_proposal("nope", "approved") is False
