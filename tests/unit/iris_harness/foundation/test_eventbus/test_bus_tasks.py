"""emit_sync keeps the async handler tasks it schedules alive until they finish."""

from __future__ import annotations

import asyncio
import gc
import logging
import threading
import warnings

import pytest

from iris_harness.foundation.eventbus.bus import EventBus


async def test_a_scheduled_handler_is_held_until_it_finishes() -> None:
    bus = EventBus()
    seen: list[str] = []
    release = asyncio.Event()

    async def handler(payload: str) -> None:
        await release.wait()
        seen.append(payload)

    bus.on("topic", handler)
    bus.emit_sync("topic", "x")
    assert len(bus._pending) == 1
    gc.collect()  # nobody else holds the task; the bus must
    release.set()
    await asyncio.gather(*bus._pending)
    await asyncio.sleep(0)  # let the done-callback run
    assert seen == ["x"]
    assert bus._pending == set()


async def test_a_failing_scheduled_handler_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    bus = EventBus()

    async def handler(payload: object) -> None:
        raise RuntimeError("boom")

    bus.on("topic", handler)
    with caplog.at_level(logging.ERROR, logger="iris_harness.foundation.eventbus.bus"):
        bus.emit_sync("topic")
        await asyncio.gather(*bus._pending, return_exceptions=True)
        await asyncio.sleep(0)
    assert "async handler task raised" in caplog.text
    assert bus._pending == set()


def test_with_no_loop_running_the_handler_runs_in_place() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(payload: str) -> None:
        seen.append(payload)

    bus.on("topic", handler)
    result: list[BaseException] = []

    def worker() -> None:  # a thread with no event loop at all
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                bus.emit_sync("topic", "y")
        except BaseException as exc:  # noqa: BLE001 -- surfaced below
            result.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert result == []
    assert seen == ["y"]
    assert bus._pending == set()
