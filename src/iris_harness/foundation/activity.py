"""Whether a chat turn is running in this process — so background work can yield.

The cloud VM and the owner's Mac share one local model. On 2026-09-25 the hourly
finance sweep kept that model busy with ~50 calls in two minutes; a Telegram question
that needed four of them waited behind the rest, took 97 seconds, and the gateway gave
up at 90. Chat comes first: ``run_turn`` marks every turn with :func:`chat_turn`, and a
background loop about to call a model asks :func:`chat_in_progress` and, if a person is
waiting, stops and leaves the rest for its next run.

Process-local on purpose: heartbeats and chat turns run in the same API process, and
every channel (web, Telegram, the CLI) reaches the turn through ``run_turn``.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

from iris_harness.foundation.process_state import track_globals

#: After a turn ends, background work waits this long before starting a model call:
#: an owner reading an answer often asks a follow-up at once.
QUIET_AFTER_SECONDS = 20.0

_lock = threading.Lock()
_running = 0
_last_ended = float("-inf")


@contextmanager
def chat_turn() -> Iterator[None]:
    """Mark a chat turn as running for as long as the block runs."""
    global _running, _last_ended
    with _lock:
        _running += 1
    try:
        yield
    finally:
        with _lock:
            _running -= 1
            _last_ended = time.monotonic()


def chat_in_progress(*, quiet_after: float = QUIET_AFTER_SECONDS) -> bool:
    """True while a chat turn runs, or ended less than ``quiet_after`` seconds ago."""
    with _lock:
        return _running > 0 or time.monotonic() - _last_ended < quiet_after


def _reset_for_tests() -> None:
    global _running, _last_ended
    with _lock:
        _running = 0
        _last_ended = float("-inf")


__all__ = ["QUIET_AFTER_SECONDS", "chat_in_progress", "chat_turn"]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_running", "_last_ended")
