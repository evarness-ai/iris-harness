"""Did the job's slot run? Slot math in IRIS_TZ over the kept runs (loop-proof D13)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from iris_harness.services.heartbeat import HeartbeatDefinition, HeartbeatRun, HeartbeatStatus
from iris_harness.services.heartbeat.run_store import HeartbeatRunStore
from iris_harness.services.heartbeat.slots import evaluate_job, slots_between, tally

CT = ZoneInfo("America/Chicago")
SWEEP = HeartbeatDefinition(name="email_sweep", handler="h", schedule="15 6,12,18 * * *")
GRACE = timedelta(minutes=30)


def _ct(day: int, hour: int, minute: int = 0, month: int = 9) -> datetime:
    return datetime(2026, month, day, hour, minute, tzinfo=CT)


@pytest.fixture
def store(tmp_path: Path) -> HeartbeatRunStore:
    store = HeartbeatRunStore(db_path=tmp_path / "heartbeat_runs.db")
    store.note_job("email_sweep", now=_ct(20, 0))
    return store


def _run(
    store: HeartbeatRunStore,
    at: datetime,
    status: HeartbeatStatus = HeartbeatStatus.SUCCESS,
    *,
    output: str = "23 new",
    error: str = "",
) -> None:
    store.record(
        HeartbeatRun(
            name="email_sweep",
            status=status,
            started_at=at,
            finished_at=at + timedelta(seconds=20),
            output=output,
            error=error,
        )
    )


def _eval(store: HeartbeatRunStore, now: datetime, definition=SWEEP):  # type: ignore[no-untyped-def]
    return evaluate_job(definition, store, now=now, tz=CT, grace=GRACE)


def test_ran_after_the_slot_is_green_with_its_summary(store: HeartbeatRunStore) -> None:
    _run(store, _ct(26, 6, 15))
    status = _eval(store, _ct(26, 9))
    assert (status.state, status.detail) == ("green", "ran 06:15 ✓ (23 new)")
    assert not status.missed


def test_inside_the_grace_the_slot_before_decides(store: HeartbeatRunStore) -> None:
    _run(store, _ct(26, 6, 15))
    status = _eval(store, _ct(26, 12, 30))  # 12:15 due, grace to 12:45
    assert status.state == "green" and status.detail.startswith("ran 06:15")


def test_slot_plus_grace_without_a_run_is_red_naming_the_slot(store: HeartbeatRunStore) -> None:
    _run(store, _ct(26, 6, 15))
    status = _eval(store, _ct(26, 12, 46))
    assert status.state == "red" and status.missed
    assert status.detail == "Missed 12:15 — last success 06:15"
    assert status.slot == _ct(26, 12, 15)


def test_a_late_success_after_the_slot_is_green(store: HeartbeatRunStore) -> None:
    _run(store, _ct(26, 6, 15))
    _run(store, _ct(26, 13, 2), output="4 new")  # the owner pressed Run now
    status = _eval(store, _ct(26, 13, 5))
    assert (status.state, status.detail) == ("green", "ran 13:02 ✓ (4 new)")


def test_a_failed_run_is_red_with_its_error(store: HeartbeatRunStore) -> None:
    _run(store, _ct(26, 6, 15))
    _run(store, _ct(26, 12, 15), HeartbeatStatus.FAILED, error="No Gmail credentials")
    status = _eval(store, _ct(26, 12, 20))
    assert status.state == "red"
    assert status.detail == "failed 12:15: No Gmail credentials — last success 06:15"
    assert status.last_error == "No Gmail credentials"


def test_missed_names_the_last_failure(store: HeartbeatRunStore) -> None:
    _run(store, _ct(25, 18, 15))
    _run(store, _ct(26, 6, 15), HeartbeatStatus.FAILED, error="token revoked")
    status = _eval(store, _ct(26, 13))
    assert status.detail == "Missed 12:15 — last success Sep 25 18:15 · last error: token revoked"


def test_a_skipped_run_is_yellow(store: HeartbeatRunStore) -> None:
    _run(store, _ct(26, 12, 15), HeartbeatStatus.SKIPPED, error="judge unreachable")
    status = _eval(store, _ct(26, 12, 20))
    assert status.state == "yellow" and "judge unreachable" in status.detail


def test_grey_before_the_first_slot_on_a_fresh_install(tmp_path: Path) -> None:
    store = HeartbeatRunStore(db_path=tmp_path / "r.db")
    store.note_job("email_sweep", now=_ct(26, 7))  # installed after 06:15
    status = _eval(store, _ct(26, 9))
    assert status.state == "grey"
    assert status.detail == "first run due 12:15"
    # …and red once its first real slot is missed.
    assert _eval(store, _ct(26, 12, 50)).state == "red"


def test_grey_while_off_or_unscheduled(store: HeartbeatRunStore) -> None:
    off = HeartbeatDefinition(
        name="email_sweep", handler="h", schedule=SWEEP.schedule, enabled=False
    )
    assert _eval(store, _ct(26, 13), off).state == "grey"
    assert evaluate_job(None, store, now=_ct(26, 13), tz=CT, grace=GRACE, name="x").state == "grey"
    assert (
        evaluate_job(SWEEP, store, now=_ct(26, 13), tz=CT, grace=GRACE, unavailable="needs darwin")
    ).state == "grey"


def test_an_interval_job_is_red_after_one_interval_plus_grace(tmp_path: Path) -> None:
    store = HeartbeatRunStore(db_path=tmp_path / "r.db")
    store.note_job("email_sweep", now=_ct(26, 5))
    every10 = HeartbeatDefinition(name="email_sweep", handler="h", schedule="interval:600")
    assert _eval(store, _ct(26, 5, 5), every10).state == "grey"
    _run(store, _ct(26, 6))
    assert _eval(store, _ct(26, 6, 30), every10).state == "green"
    status = _eval(store, _ct(26, 6, 41), every10)
    assert status.state == "red" and status.detail == "Missed 06:10 — last success 06:00"


@pytest.mark.parametrize(
    ("month", "day", "utc_hours"),
    [
        (9, 26, [11, 17, 23]),  # CDT, UTC-5
        (11, 1, [12, 18, 0]),  # fall back at 02:00: 06:15 is CST, UTC-6
        (3, 8, [11, 17, 23]),  # spring forward at 02:00: 06:15 is CDT
    ],
)
def test_slots_are_wall_clock_in_iris_tz_on_dst_days(
    month: int, day: int, utc_hours: list[int]
) -> None:
    start = datetime(2026, month, day, tzinfo=CT)
    slots = slots_between(SWEEP.schedule, start, start + timedelta(days=1), CT)
    assert [s.astimezone(CT).strftime("%H:%M") for s in slots] == ["06:15", "12:15", "18:15"]
    assert [s.astimezone(UTC).hour for s in slots] == utc_hours


def test_a_missed_slot_on_the_fall_back_day(tmp_path: Path) -> None:
    store = HeartbeatRunStore(db_path=tmp_path / "r.db")
    store.note_job("email_sweep", now=_ct(30, 0, month=10))
    _run(store, datetime(2026, 11, 1, 12, 15, tzinfo=UTC))  # 06:15 CST
    now = datetime(2026, 11, 1, 18, 50, tzinfo=UTC)  # 12:50 CST
    status = _eval(store, now)
    assert status.detail == "Missed 12:15 — last success 06:15"


def test_tally_counts_served_slots_and_lists_the_missed(store: HeartbeatRunStore) -> None:
    start, end = _ct(25, 0), _ct(26, 0)
    for hour in (6, 12, 18):
        _run(store, _ct(25, hour, 15))
    assert tally(SWEEP.schedule, store, "email_sweep", start, end, CT).ran == 3

    start, end = _ct(26, 0), _ct(27, 0)
    _run(store, _ct(26, 6, 15))
    _run(store, _ct(26, 18, 16), HeartbeatStatus.SKIPPED, error="unreachable")
    counted = tally(SWEEP.schedule, store, "email_sweep", start, end, CT)
    assert (counted.ran, counted.expected) == (1, 3)
    assert counted.missed == (_ct(26, 12, 15),)  # the skip ran, so it is not "missed"
