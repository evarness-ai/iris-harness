"""The ULID generator: format, uniqueness across threads, sortability (issue #134)."""

from __future__ import annotations

import os
import re
import threading
import time
from unittest import mock

import pytest

from iris_harness.foundation import ids
from iris_harness.foundation.ids import ULID_LENGTH, is_ulid, new_ulid

_SHAPE = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")  # Crockford base32: no I, L, O, U


def test_format_is_26_crockford_characters() -> None:
    value = new_ulid()
    assert len(value) == ULID_LENGTH == 26
    assert _SHAPE.match(value)
    assert is_ulid(value)
    assert value[0] <= "7"  # 48 bits of milliseconds fit in 10 characters


def test_is_ulid_rejects_other_shapes() -> None:
    for bad in ("", "abc", "0" * 25, "0" * 27, "I" * 26, "u" * 26, None, 12, "a1b2c3d4e5f6"):
        assert not is_ulid(bad)


def test_a_hundred_thousand_ids_never_collide_and_ascend() -> None:
    values = [new_ulid() for _ in range(100_000)]
    assert len(set(values)) == len(values)
    assert values == sorted(values)  # strictly increasing within the process, same millisecond too


def test_ids_sort_by_creation_time() -> None:
    first = new_ulid()
    time.sleep(0.003)
    second = new_ulid()
    assert first < second


def test_unique_across_threads() -> None:
    per_thread, workers = 5_000, 8
    out: list[list[str]] = [[] for _ in range(workers)]
    start = threading.Barrier(workers)

    def work(slot: int) -> None:
        start.wait()
        out[slot] = [new_ulid() for _ in range(per_thread)]

    threads = [threading.Thread(target=work, args=(i,)) for i in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    flat = [v for chunk in out for v in chunk]
    assert len(flat) == per_thread * workers
    assert len(set(flat)) == len(flat)
    for chunk in out:  # each thread saw its own ids ascend
        assert chunk == sorted(chunk)


def test_same_millisecond_counts_up_not_down() -> None:
    with mock.patch.object(ids.time, "time", return_value=1_700_000_000.123):
        a, b, c = new_ulid(), new_ulid(), new_ulid()
    assert a[:10] == b[:10] == c[:10]
    assert a < b < c


def test_a_clock_that_steps_back_never_sorts_before_an_earlier_id() -> None:
    with mock.patch.object(ids.time, "time", return_value=4_000_000_000.0):
        late = new_ulid()
    with mock.patch.object(ids.time, "time", return_value=1_000_000_000.0):
        after_step_back = new_ulid()
    assert late < after_step_back


def test_the_random_part_overflow_rolls_into_the_next_millisecond() -> None:
    with mock.patch.object(ids.time, "time", return_value=1_800_000_000.0):
        prev = new_ulid()
        ids._last_random = ids._RANDOM_MAX
        nxt = new_ulid()
    assert nxt > prev and nxt[:10] != prev[:10]


def test_threads_in_one_millisecond_still_get_distinct_ids() -> None:
    """The counter path (a frozen clock) under forced thread switches: the lock matters."""
    import sys

    workers, per_thread = 8, 20_000
    out: list[list[str]] = [[] for _ in range(workers)]
    start = threading.Barrier(workers)

    def work(slot: int) -> None:
        start.wait()
        out[slot] = [new_ulid() for _ in range(per_thread)]

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        with mock.patch.object(ids.time, "time", return_value=1_900_000_000.0):
            threads = [threading.Thread(target=work, args=(i,)) for i in range(workers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
    finally:
        sys.setswitchinterval(old)
    flat = [v for chunk in out for v in chunk]
    assert len(set(flat)) == len(flat) == workers * per_thread


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_a_forked_child_never_mints_an_id_the_parent_also_mints() -> None:
    """The child inherited the parent's counter, so both counted up from the same value."""
    new_ulid()  # leave the parent mid-millisecond state behind
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child
        os.close(read_fd)
        try:
            os.write(write_fd, "\n".join(new_ulid() for _ in range(50)).encode())
        finally:
            os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as pipe:
        child_ids = pipe.read().decode().split("\n")
    os.waitpid(pid, 0)
    parent_ids = [new_ulid() for _ in range(50)]
    assert len(child_ids) == 50 and not set(child_ids) & set(parent_ids)
    assert all(is_ulid(i) for i in child_ids)
