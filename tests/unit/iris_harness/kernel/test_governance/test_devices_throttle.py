"""The global failed-claim throttle — the window itself, then the service around it
(ADR-0117, "Pairing flow")."""

from __future__ import annotations

import sqlite3
import threading
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from iris_harness.kernel.governance.devices import (
    CLAIM_FAILURE_LIMIT,
    CLAIM_FAILURE_WINDOW,
    PAIRING_MAX_ATTEMPTS,
    ClaimThrottle,
    DeviceService,
    DeviceStore,
    PairingRefusedError,
    PairingThrottledError,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 500.0

    def __call__(self) -> float:
        return self.now


class _Ledger:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **row: Any) -> int:
        self.rows.append(row)
        return len(self.rows)

    def count(self, decision: str) -> int:
        return sum(1 for r in self.rows if r["decision"] == decision)


@pytest.fixture
def clock() -> _Clock:
    return _Clock()


# ── the window ───────────────────────────────────────────────────────────────


def test_the_defaults_are_ten_failures_in_sixty_seconds() -> None:
    assert CLAIM_FAILURE_LIMIT == 10
    assert timedelta(seconds=60) == CLAIM_FAILURE_WINDOW
    throttle = ClaimThrottle()
    assert throttle.limit == 10 and throttle.window == timedelta(seconds=60)


def test_open_until_the_limit_then_closed(clock: _Clock) -> None:
    throttle = ClaimThrottle(limit=3, clock=clock)
    assert throttle.retry_after() is None
    assert [throttle.record_failure() for _ in range(2)] == [False, False]
    assert throttle.retry_after() is None  # limit - 1 failures: still open
    assert throttle.record_failure() is True  # the one that closes it
    assert throttle.retry_after() == 60


def test_retry_after_counts_down_and_is_never_zero(clock: _Clock) -> None:
    throttle = ClaimThrottle(limit=2, window=timedelta(seconds=10), clock=clock)
    throttle.record_failure()
    throttle.record_failure()
    clock.now += 4
    assert throttle.retry_after() == 6
    clock.now += 5.5
    assert throttle.retry_after() == 1  # 0.5s left rounds UP
    clock.now += 0.5
    assert throttle.retry_after() is None  # a full window old: gone


def test_the_window_slides_rather_than_resets(clock: _Clock) -> None:
    """Failures age out one by one. A fixed window that reset on the minute would
    allow 2x the limit across the boundary."""
    throttle = ClaimThrottle(limit=3, window=timedelta(seconds=60), clock=clock)
    throttle.record_failure()  # t=0
    clock.now += 30
    throttle.record_failure()  # t=30
    throttle.record_failure()  # t=30 — closed
    assert throttle.retry_after() == 30  # until the t=0 failure ages out
    clock.now += 30
    assert throttle.retry_after() is None  # two left in the window
    throttle.record_failure()  # t=60 — closed again by ONE failure, not three
    assert throttle.retry_after() == 30


def test_a_trip_is_reported_once_until_the_window_reopens(clock: _Clock) -> None:
    throttle = ClaimThrottle(limit=2, clock=clock)
    assert [throttle.record_failure() for _ in range(4)] == [False, True, False, False]
    clock.now += 61
    assert throttle.retry_after() is None
    assert [throttle.record_failure() for _ in range(2)] == [False, True]


def test_retry_after_with_an_overshoot_waits_for_enough_to_age_out(clock: _Clock) -> None:
    """Concurrent claims admitted at limit-1 can push the count past the limit; the
    wait is then until the count drops UNDER it, not until the oldest ages out."""
    throttle = ClaimThrottle(limit=2, window=timedelta(seconds=10), clock=clock)
    throttle.record_failure()  # t=0
    clock.now += 5
    throttle.record_failure()  # t=5
    throttle.record_failure()  # t=5 (overshoot)
    assert throttle.retry_after() == 10  # t=0 ageing out (at t=10) still leaves two: wait for t=15
    clock.now += 5  # t=10: the t=0 failure is gone, two remain
    assert throttle.retry_after() == 5


@pytest.mark.parametrize("kwargs", [{"limit": 0}, {"window": timedelta(0)}])
def test_nonsense_settings_are_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ClaimThrottle(**kwargs)


def test_concurrent_failures_are_all_counted(clock: _Clock) -> None:
    throttle = ClaimThrottle(limit=10_000, clock=clock)
    trips: list[bool] = []

    def hammer() -> None:
        for _ in range(500):
            trips.append(throttle.record_failure())

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(throttle._failures) == 4000
    assert trips.count(True) == 0

    tight = ClaimThrottle(limit=100, clock=clock)
    trips.clear()

    def hammer_tight() -> None:
        for _ in range(50):
            trips.append(tight.record_failure())

    threads = [threading.Thread(target=hammer_tight) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert trips.count(True) == 1  # one trip, however many threads crossed the line


def test_the_window_is_read_and_written_under_one_lock() -> None:
    """Deterministic, unlike a hammer test: thread A is parked INSIDE the critical
    section (the clock is read under the lock), and B must not get in until A leaves.
    With the lock B cannot enter, so this never flakes; without it B walks straight in."""
    inside, release = threading.Event(), threading.Event()
    calls: list[str] = []

    def clock() -> float:
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            inside.set()
            assert release.wait(timeout=5)
        return 0.0

    throttle = ClaimThrottle(limit=5, clock=clock)
    a = threading.Thread(target=throttle.record_failure, name="A")
    b = threading.Thread(target=throttle.record_failure, name="B")
    a.start()
    assert inside.wait(timeout=5)
    b.start()
    b.join(timeout=0.3)  # B's chance to barge in
    entered_while_a_held_it = list(calls)
    release.set()
    a.join(timeout=5)
    b.join(timeout=5)
    assert entered_while_a_held_it == ["A"]
    assert calls == ["A", "B"]
    assert len(throttle._failures) == 2


# ── the service around it ────────────────────────────────────────────────────


@pytest.fixture
def ledger() -> _Ledger:
    return _Ledger()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "devices.db"


@pytest.fixture
def service(db_path: Path, ledger: _Ledger, clock: _Clock) -> DeviceService:
    return DeviceService(
        store=DeviceStore(db_path=db_path),
        ledger=ledger,
        throttle=ClaimThrottle(limit=3, clock=clock),
    )


def _fail(service: DeviceService, times: int) -> None:
    for _ in range(times):
        with pytest.raises(PairingRefusedError):
            service.claim(code="ZZZZ-ZZZZ", name="x", kind="app")


def _attempts(db_path: Path) -> list[int]:
    with sqlite3.connect(db_path) as conn:
        return [r[0] for r in conn.execute("SELECT attempts FROM pairing_codes ORDER BY rowid")]


def test_a_service_has_a_default_throttle(db_path: Path, ledger: _Ledger) -> None:
    service = DeviceService(store=DeviceStore(db_path=db_path), ledger=ledger)
    _fail(service, CLAIM_FAILURE_LIMIT)
    with pytest.raises(PairingThrottledError) as refused:
        service.claim(code="ZZZZ-ZZZZ", name="x", kind="app")
    assert 1 <= refused.value.retry_after <= 60


def test_throttled_is_not_a_refusal(service: DeviceService) -> None:
    """A surface that answers PairingRefusedError with "wrong code" must not answer
    a throttled claim that way: the code was never looked at."""
    assert not issubclass(PairingThrottledError, PairingRefusedError)
    assert not issubclass(PairingThrottledError, ValueError)


def test_while_throttled_the_code_is_not_checked_and_no_attempt_is_charged(
    service: DeviceService, clock: _Clock, db_path: Path
) -> None:
    _fail(service, 3)
    code = service.start_pairing(scope="control", actor="service")
    assert _attempts(db_path) == [0]

    for guess in ("ZZZZ-ZZZZ", code.code):
        with pytest.raises(PairingThrottledError) as refused:
            service.claim(code=guess, name="x", kind="app")
        assert refused.value.retry_after == 60
    assert _attempts(db_path) == [0]
    assert service.list_devices() == []

    clock.now += 60
    assert service.claim(code=code.code, name="x", kind="app").device.scope == "control"


def test_throttled_comes_before_any_look_at_the_request(service: DeviceService) -> None:
    _fail(service, 3)
    with pytest.raises(PairingThrottledError):
        service.claim(code="ZZZZ-ZZZZ", name="", kind="toaster")


def test_a_malformed_claim_is_not_a_failed_claim(service: DeviceService) -> None:
    for _ in range(5):
        with pytest.raises(ValueError, match="kind"):
            service.claim(code="ZZZZ-ZZZZ", name="x", kind="toaster")
    _fail(service, 1)  # still open: PairingRefusedError, not PairingThrottledError


def test_a_successful_claim_is_not_a_failure(service: DeviceService) -> None:
    for _ in range(5):
        code = service.start_pairing(scope="read", actor="service")
        service.claim(code=code.code, name="x", kind="app")
    _fail(service, 1)


def test_the_per_code_void_from_pr2_still_applies(
    db_path: Path, ledger: _Ledger, clock: _Clock
) -> None:
    service = DeviceService(
        store=DeviceStore(db_path=db_path), ledger=ledger, throttle=ClaimThrottle(clock=clock)
    )
    code = service.start_pairing(scope="control", actor="service")
    _fail(service, PAIRING_MAX_ATTEMPTS)
    assert ledger.count("pairing_code_voided") == 1
    with pytest.raises(PairingRefusedError):
        service.claim(code=code.code, name="x", kind="app")


def test_one_ledger_row_per_trip_with_no_code_in_it(
    service: DeviceService, ledger: _Ledger, clock: _Clock
) -> None:
    _fail(service, 3)
    for _ in range(4):
        with pytest.raises(PairingThrottledError):
            service.claim(code="QQQQ-QQQQ", name="x", kind="app")
    assert ledger.count("pairing_throttled") == 1
    row = next(r for r in ledger.rows if r["decision"] == "pairing_throttled")
    assert row["severity"] == "warn"
    assert row["run_id"] == "device:pairing"
    assert row["payload"] == {"limit": 3, "window_seconds": 60}
    assert "ZZZZ" not in repr(ledger.rows) and "QQQQ" not in repr(ledger.rows)

    clock.now += 61
    _fail(service, 3)
    assert ledger.count("pairing_throttled") == 2
