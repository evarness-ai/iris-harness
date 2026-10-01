"""Tests for the cost ledger SQLite store (story 12.gov-3.7 / AC-1)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.kernel.governance.cost import CostStore


@pytest.fixture()
def store(tmp_path: Path) -> CostStore:
    return CostStore(db_path=tmp_path / "cost-ledger.db")


def _record(
    store: CostStore,
    *,
    run_id: str = "r1",
    tier: str = "tier_3",
    cost: float = 0.10,
    user_id: str = "local",
    ts: datetime | None = None,
) -> int:
    return store.record(
        run_id=run_id,
        agent_type="chat",
        tier=tier,
        prompt_tokens=100,
        completion_tokens=200,
        cost_usd=cost,
        user_id=user_id,
        ts=ts,
    )


def test_record_returns_row_id(store: CostStore) -> None:
    first = _record(store)
    second = _record(store)
    assert second > first


def test_sum_today_sums_only_user_rows(store: CostStore) -> None:
    _record(store, user_id="local", cost=0.10)
    _record(store, user_id="local", cost=0.25)
    _record(store, user_id="other", cost=99.00)
    total = store.sum_today(user_id="local")
    assert total == pytest.approx(0.35)


def test_sum_today_excludes_yesterday(store: CostStore) -> None:
    yesterday = datetime.now(UTC) - timedelta(days=1, hours=2)
    _record(store, user_id="local", cost=1.00, ts=yesterday)
    _record(store, user_id="local", cost=0.20)
    assert store.sum_today(user_id="local") == pytest.approx(0.20)


def test_sum_since_with_explicit_date(store: CostStore) -> None:
    two_days_ago = datetime.now(UTC) - timedelta(days=2)
    yesterday = datetime.now(UTC) - timedelta(days=1)
    _record(store, cost=10.0, ts=two_days_ago)
    _record(store, cost=2.0, ts=yesterday)
    _record(store, cost=0.5)
    # `since=today` floors to midnight UTC today — yesterday and earlier
    # rows fall before the cutoff, so only the row from today survives.
    today = datetime.now(UTC).date()
    total = store.sum_since(user_id="local", since=today)
    assert total == pytest.approx(0.5)


def test_sum_by_tier_since(store: CostStore) -> None:
    _record(store, tier="tier_1", cost=0.0)
    _record(store, tier="tier_3", cost=0.10)
    _record(store, tier="tier_3", cost=0.05)
    breakdown = store.sum_by_tier_since(user_id="local", since=datetime.now(UTC).date())
    assert breakdown == {
        "tier_1": pytest.approx(0.0),
        "tier_3": pytest.approx(0.15),
    }


def test_list_since_orders_ascending(store: CostStore) -> None:
    earlier = datetime.now(UTC) - timedelta(minutes=10)
    _record(store, cost=0.20, ts=earlier)
    _record(store, cost=0.30)
    # Query since the earlier record's date — `now().date()` drops it when the
    # 10-minute window straddles UTC midnight (flaky just after 00:00 UTC).
    entries = store.list_since(user_id="local", since=earlier.date())
    assert len(entries) == 2
    assert entries[0].cost_usd == pytest.approx(0.20)
    assert entries[1].cost_usd == pytest.approx(0.30)
    assert entries[0].ts <= entries[1].ts


def test_count_increments_per_record(store: CostStore) -> None:
    assert store.count() == 0
    _record(store)
    _record(store)
    assert store.count() == 2


def test_empty_store_sums_zero(store: CostStore) -> None:
    assert store.sum_today(user_id="local") == 0.0
    assert store.sum_by_tier_since(user_id="local", since=datetime.now(UTC).date()) == {}
