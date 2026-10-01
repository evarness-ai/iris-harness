"""The read side of the cost ledger (Track 2 PR 6, plan decision 30).

The behaviour worth protecting here is not the arithmetic — it is that a
summary from a ledger nobody writes cannot be mistaken for a summary that
happens to be zero. ``CostLimiter`` is opt-in and is the only writer, so
``recording`` travels with the numbers.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_harness.kernel.governance.cost import CostStore
from iris_harness.kernel.governance.cost.summary import (
    COST_CAP_ENV,
    COST_ENFORCE_ENV,
    COST_LIMITER_ENV,
    ENABLE_HINT,
    cost_summary,
    enforcing_enabled,
    recording_enabled,
)

NOW = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
ON = {COST_LIMITER_ENV: "1"}
OFF: dict[str, str] = {}


@pytest.fixture()
def store(tmp_path: Path) -> CostStore:
    return CostStore(db_path=tmp_path / "cost-ledger.db")


def _spend(store: CostStore, *, ts: datetime, tier: str = "tier_3", usd: float = 0.10) -> None:
    store.record(
        run_id="r1",
        agent_type="chat",
        tier=tier,
        prompt_tokens=100,
        completion_tokens=50,
        cost_usd=usd,
        user_id="local",
        ts=ts,
    )


# ── the distinction the whole module exists for ────────────────────────────


def test_zero_because_nobody_is_counting_is_not_zero_because_nothing_was_spent(
    store: CostStore,
) -> None:
    off = cost_summary(store=store, now=NOW, env=OFF)
    assert off.recording is False
    assert off.enable_hint == ENABLE_HINT

    _spend(store, ts=NOW, usd=0.25)
    on = cost_summary(store=store, now=NOW, env=ON)
    assert on.recording is True
    assert on.enable_hint is None
    # Both can report a number; only `recording` says whether to believe a 0.
    assert on.today_usd == pytest.approx(0.25)


def test_history_is_still_reported_after_recording_is_switched_off(store: CostStore) -> None:
    # Hiding what the ledger holds because the flag is off now would be its own
    # kind of lying: the spend happened.
    _spend(store, ts=NOW, usd=0.40)
    summary = cost_summary(store=store, now=NOW, env=OFF)
    assert summary.recording is False
    assert summary.today_usd == pytest.approx(0.40)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("true", True),
        ("YES", True),
        ("on", True),
        ("0", False),
        ("", False),
        ("no", False),
        ("maybe", False),
    ],
)
def test_recording_enabled_reads_the_flag(value: str, expected: bool) -> None:
    assert recording_enabled({COST_LIMITER_ENV: value}) is expected


def test_recording_is_off_when_the_flag_is_absent() -> None:
    assert recording_enabled({}) is False


# ── the rollup ─────────────────────────────────────────────────────────────


def test_today_and_month_are_different_windows(store: CostStore) -> None:
    _spend(store, ts=datetime(2026, 9, 2, 9, 0, tzinfo=UTC), usd=1.00)  # this month
    _spend(store, ts=datetime(2026, 9, 20, 9, 0, tzinfo=UTC), usd=0.25)  # today
    summary = cost_summary(store=store, now=NOW, env=ON)
    assert summary.today_usd == pytest.approx(0.25)
    assert summary.month_usd == pytest.approx(1.25)


def test_last_month_is_not_counted_as_this_month(store: CostStore) -> None:
    _spend(store, ts=datetime(2026, 8, 31, 23, 59, tzinfo=UTC), usd=9.99)
    summary = cost_summary(store=store, now=NOW, env=ON)
    assert summary.month_usd == pytest.approx(0.0)
    assert summary.entries == 1  # the row exists; it is just not this month's


def test_month_starts_on_the_first_not_thirty_days_ago(store: CostStore) -> None:
    # A rolling 30-day window against a calendar-month budget would report a
    # number the owner's Azure bill never agrees with.
    _spend(store, ts=datetime(2026, 8, 25, 12, 0, tzinfo=UTC), usd=5.00)
    _spend(store, ts=datetime(2026, 9, 1, 0, 0, tzinfo=UTC), usd=0.50)
    summary = cost_summary(store=store, now=NOW, env=ON)
    assert summary.month_usd == pytest.approx(0.50)


def test_per_tier_split_shows_where_the_money_went(store: CostStore) -> None:
    _spend(store, ts=NOW, tier="tier_1", usd=0.0)
    _spend(store, ts=NOW, tier="tier_3", usd=0.75)
    summary = cost_summary(store=store, now=NOW, env=ON)
    assert summary.by_tier_usd == {"tier_1": pytest.approx(0.0), "tier_3": pytest.approx(0.75)}


def test_another_users_spend_is_not_this_users_spend(store: CostStore) -> None:
    store.record(
        run_id="r2",
        agent_type="chat",
        tier="tier_3",
        prompt_tokens=10,
        completion_tokens=10,
        cost_usd=7.00,
        user_id="someone-else",
        ts=NOW,
    )
    summary = cost_summary(store=store, user_id="local", now=NOW, env=ON)
    assert summary.today_usd == pytest.approx(0.0)


def test_summary_serialises_for_the_api(store: CostStore) -> None:
    _spend(store, ts=NOW, usd=0.123456789)
    payload = cost_summary(store=store, now=NOW, env=ON).as_dict()
    assert set(payload) == {
        "recording",
        "enforcing",
        "daily_cap_usd",
        "user_id",
        "today_usd",
        "month_usd",
        "by_tier_usd",
        "entries",
        "ledger_db",
        "enable_hint",
    }
    # Rounded, so the card does not render 0.12345678900000001.
    assert payload["today_usd"] == 0.123457


# ── it feeds a status card, so it must not raise ───────────────────────────


def test_a_broken_ledger_reads_as_zeros_not_an_exception(tmp_path: Path) -> None:
    class Exploding(CostStore):
        def sum_today(self, **_: object) -> float:  # type: ignore[override]
            raise RuntimeError("database is locked")

    summary = cost_summary(store=Exploding(db_path=tmp_path / "c.db"), now=NOW, env=ON)
    assert summary.today_usd == 0.0
    assert summary.month_usd == 0.0
    assert summary.recording is True  # the flag is still readable


def test_an_unopenable_ledger_still_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("iris_harness.kernel.governance.cost.summary._open_store", lambda: None)
    summary = cost_summary(now=NOW, env=OFF)
    assert summary.today_usd == 0.0
    assert summary.enable_hint == ENABLE_HINT


# ── recording vs enforcing are separate switches ───────────────────────────
#
# The owner's call: keep enforcement, but behind its own flag. The ledger is a
# mechanism, the cap is a policy, and the VM needed one without the other.


def test_enforcing_defaults_on_so_recording_never_removes_a_guard(store: CostStore) -> None:
    # An install that already had the limiter must not silently lose its cap
    # because recording became separable from enforcing.
    summary = cost_summary(store=store, now=NOW, env={COST_LIMITER_ENV: "1"})
    assert summary.recording is True
    assert summary.enforcing is True
    assert summary.daily_cap_usd == pytest.approx(5.00)


def test_recording_without_enforcing(store: CostStore) -> None:
    summary = cost_summary(
        store=store,
        now=NOW,
        env={COST_LIMITER_ENV: "1", COST_ENFORCE_ENV: "0"},
    )
    assert summary.recording is True
    assert summary.enforcing is False
    # No cap is in force, so reporting one would be a number that does nothing.
    assert summary.daily_cap_usd is None


def test_nothing_is_enforced_while_nothing_is_recorded(store: CostStore) -> None:
    # Enforcement runs inside CostLimiter, which is not registered at all when
    # recording is off — so ENFORCE=1 alone must not claim a cap is in force.
    summary = cost_summary(store=store, now=NOW, env={COST_ENFORCE_ENV: "1"})
    assert summary.recording is False
    assert summary.enforcing is False
    assert summary.daily_cap_usd is None


def test_the_cap_reported_is_the_cap_configured(store: CostStore) -> None:
    summary = cost_summary(
        store=store,
        now=NOW,
        env={COST_LIMITER_ENV: "1", COST_CAP_ENV: "1.50"},
    )
    assert summary.daily_cap_usd == pytest.approx(1.50)


def test_an_unparseable_cap_reports_the_one_actually_in_force(store: CostStore) -> None:
    # The wiring falls back to the package default and logs; the card must show
    # that number, not the junk the operator typed.
    summary = cost_summary(
        store=store,
        now=NOW,
        env={COST_LIMITER_ENV: "1", COST_CAP_ENV: "five dollars"},
    )
    assert summary.daily_cap_usd == pytest.approx(5.00)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("on", True), ("0", False), ("no", False), ("", False)],
)
def test_enforcing_enabled_reads_the_flag(value: str, expected: bool) -> None:
    assert enforcing_enabled({COST_ENFORCE_ENV: value}) is expected
