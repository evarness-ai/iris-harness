"""Tests for digital-twin proposal-quality measurement."""

from __future__ import annotations

from iris_harness.services.learning.proposal_quality import build_proposal_quality


class _FakeStore:
    def __init__(self, behaviors: dict[str, int], intentions: dict[str, int]) -> None:
        self._b = behaviors
        self._i = intentions

    def list_behavior_proposals(self, status: str = "pending") -> list[object]:
        return [object()] * self._b.get(status, 0)

    def list_intentions(self, status: str = "proposed") -> list[object]:
        return [object()] * self._i.get(status, 0)


def test_counts_and_acceptance_rate() -> None:
    store = _FakeStore(
        behaviors={"pending": 2, "approved": 3, "rejected": 1},
        intentions={"proposed": 1, "active": 1, "dismissed": 3},
    )
    q = build_proposal_quality(store)

    assert (q.behaviors.awaiting, q.behaviors.accepted, q.behaviors.rejected) == (2, 3, 1)
    assert q.behaviors.reviewed == 4
    assert q.behaviors.acceptance_rate == 0.75
    assert q.intentions.acceptance_rate == 0.25


def test_no_reviews_is_zero_not_error() -> None:
    store = _FakeStore(behaviors={"pending": 5}, intentions={"proposed": 2})
    q = build_proposal_quality(store)

    assert q.behaviors.reviewed == 0
    assert q.behaviors.acceptance_rate == 0.0


def test_as_dict_rounds_and_includes_derived() -> None:
    store = _FakeStore(behaviors={"approved": 1, "rejected": 2}, intentions={})
    d = build_proposal_quality(store).as_dict()

    assert d["behaviors"]["reviewed"] == 3
    assert d["behaviors"]["acceptance_rate"] == 0.333
    assert d["intentions"]["acceptance_rate"] == 0.0
