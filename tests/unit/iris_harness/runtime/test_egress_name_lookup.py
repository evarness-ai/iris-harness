"""The governed client's name lookup is inside the request's deadline (issue #173).

``getaddrinfo`` has no timeout, so a resolver that hangs used to hang the request before the
deadline even started. The lookup now runs on a daemon worker thread and the request waits for
it at most what is left of the deadline. A thread blocked in the C resolver cannot be cancelled,
so what is pinned here is what can be promised: the request is cut off on time, the stuck thread
ends by itself as soon as the resolver returns and never keeps the process alive, and no more
than ``MAX_STUCK_LOOKUPS`` of them can exist, however many requests are made.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

from iris_harness.runtime import egress_transport
from iris_harness.runtime.egress_transport import (
    MAX_STUCK_LOOKUPS,
    BlockedAddress,
    DeadlineExceeded,
    PinnedBackend,
)

NAME = "hang.test"


class _Hang:
    """A resolver that blocks until a test lets it go."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Semaphore(0)

    def __call__(self, host: str, port: int, **_: Any) -> list[tuple[Any, ...]]:
        self.started.release()
        self.release.wait(timeout=30)  # never longer than the test run, whatever happens
        return [(2, 1, 6, "", ("93.184.216.34", port))]


def _lookup_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "egress-resolve"]


def _wait_until(condition: Any, seconds: float = 3.0) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture()
def hang(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Hang]:
    resolver = _Hang()
    monkeypatch.setattr(egress_transport, "_resolve", resolver)
    try:
        yield resolver
    finally:
        resolver.release.set()  # let every parked lookup finish, then check none is left
        assert _wait_until(lambda: not _lookup_threads()), "a lookup thread outlived its test"


def test_a_lookup_that_hangs_is_cut_off_at_the_deadline(hang: _Hang) -> None:
    backend = PinnedBackend(0.4)

    started = time.monotonic()
    with pytest.raises(DeadlineExceeded):
        backend.connect_tcp(NAME, 80)

    assert time.monotonic() - started < 2  # not 30 s, the resolver's own wait


def test_the_stuck_thread_ends_by_itself_and_is_a_daemon(hang: _Hang) -> None:
    with pytest.raises(DeadlineExceeded):
        PinnedBackend(0.2).connect_tcp(NAME, 80)

    (stuck,) = _lookup_threads()
    assert stuck.daemon  # it can never keep the interpreter alive
    hang.release.set()
    assert _wait_until(lambda: not _lookup_threads())  # and it is not leaked


def test_stuck_lookups_are_capped_and_the_next_one_fails_closed_at_once(hang: _Hang) -> None:
    for _ in range(MAX_STUCK_LOOKUPS):
        with pytest.raises(DeadlineExceeded):
            PinnedBackend(0.1).connect_tcp(NAME, 80)
    assert len(_lookup_threads()) == MAX_STUCK_LOOKUPS

    backend = PinnedBackend(5)
    started = time.monotonic()
    with pytest.raises(BlockedAddress, match="backed up"):
        backend.connect_tcp(NAME, 80)

    assert time.monotonic() - started < 2  # no wait (a real one is 5 s), no extra thread
    assert backend.refused is not None and "backed up" in backend.refused
    assert len(_lookup_threads()) == MAX_STUCK_LOOKUPS


def test_the_slots_come_back_when_the_resolver_returns(hang: _Hang) -> None:
    for _ in range(MAX_STUCK_LOOKUPS):
        with pytest.raises(DeadlineExceeded):
            PinnedBackend(0.1).connect_tcp(NAME, 80)
    hang.release.set()
    assert _wait_until(lambda: not _lookup_threads())

    # A lookup that answers promptly works again: the cap is a ceiling, not a one-way valve.
    egress_transport._resolve = lambda host, port, **_: [(2, 1, 6, "", ("93.184.216.34", port))]
    assert egress_transport.resolve_checked("ok.test", 443, 1.0) == ["93.184.216.34"]


def test_a_prompt_lookup_is_unchanged_and_leaves_no_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        egress_transport,
        "_resolve",
        lambda host, port, **_: [(2, 1, 6, "", ("93.184.216.34", port))],
    )

    assert egress_transport.resolve_checked("ok.test", 443, 1.0) == ["93.184.216.34"]
    assert _wait_until(lambda: not _lookup_threads())


def test_a_resolver_error_and_an_internal_answer_are_still_what_they_were(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(host: str, port: int, **_: Any) -> list[tuple[Any, ...]]:
        raise OSError("no such host")

    monkeypatch.setattr(egress_transport, "_resolve", broken)
    with pytest.raises(BlockedAddress, match="did not resolve"):
        egress_transport.resolve_checked("gone.test", 443, 1.0)

    monkeypatch.setattr(
        egress_transport, "_resolve", lambda host, port, **_: [(2, 1, 6, "", ("127.0.0.1", port))]
    )
    with pytest.raises(BlockedAddress, match="inside this machine"):
        egress_transport.resolve_checked("local.test", 443, 1.0)


def test_without_a_timeout_the_lookup_is_the_plain_call(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def resolver(host: str, port: int, **_: Any) -> list[tuple[Any, ...]]:
        calls.append(threading.current_thread().name)
        return [(2, 1, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(egress_transport, "_resolve", resolver)

    assert egress_transport.resolve_checked("ok.test", 443) == ["93.184.216.34"]
    assert calls == [threading.current_thread().name]  # no worker thread
