"""The global failed-claim throttle (ADR-0117, pairing flow).

``POST /api/v1/devices/pair/claim`` is the one data route that takes no credential,
so it is the one a guesser can hammer. PR 2 already voids a code after five wrong
guesses; that protects *a* code. This protects the claim path as a whole: failed
claims from every caller are counted in one sliding window, and while the window is
full every claim is refused **before the code is looked at** — no oracle, and no
attempt charged to the owner's live code by a flood the throttle is already refusing.

Global, not per address: behind ``tailscale serve`` every request arrives from the
proxy, so a per-address count would be one bucket anyway, and one bucket cannot be
sidestepped by rotating addresses.

In memory, per process. ``iris_api`` is one process and holds one ``DeviceService``;
a restart forgets the window, which costs a guesser nothing they keep — the per-code
attempt count is in the DB and survives it.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import timedelta

__all__ = ["CLAIM_FAILURE_LIMIT", "CLAIM_FAILURE_WINDOW", "ClaimThrottle"]

CLAIM_FAILURE_LIMIT = 10
CLAIM_FAILURE_WINDOW = timedelta(seconds=60)


class ClaimThrottle:
    """A sliding window over failed claims. Thread-safe; the clock is injectable.

    ``clock`` returns seconds and only differences are used, so the default is the
    monotonic clock: a wall-clock step must not open (or jam) the window.
    """

    def __init__(
        self,
        *,
        limit: int = CLAIM_FAILURE_LIMIT,
        window: timedelta = CLAIM_FAILURE_WINDOW,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if window <= timedelta(0):
            raise ValueError("window must be positive")
        self.limit = limit
        self.window = window
        self._window_s = window.total_seconds()
        self._clock = clock
        self._failures: deque[float] = deque()
        self._tripped = False
        self._lock = threading.Lock()

    def _expire(self, now: float) -> None:
        # A failure stops counting once it is a full window old.
        while self._failures and now - self._failures[0] >= self._window_s:
            self._failures.popleft()
        if len(self._failures) < self.limit:
            self._tripped = False

    def retry_after(self) -> int | None:
        """Whole seconds until a claim will be looked at again, or ``None`` when
        claims are open. Never 0: a refused caller is told to wait at least a second."""
        with self._lock:
            now = self._clock()
            self._expire(now)
            if len(self._failures) < self.limit:
                return None
            # The window reopens when enough of the oldest failures age out to bring
            # the count under the limit. Concurrent claims that were all admitted at
            # limit-1 can overshoot, hence the index rather than "the oldest".
            reopening = self._failures[len(self._failures) - self.limit]
            return max(1, math.ceil(reopening + self._window_s - now))

    def record_failure(self) -> bool:
        """Count one failed claim. True exactly when this failure closed the window
        — once per trip, so the caller can write one ledger row per trip."""
        with self._lock:
            now = self._clock()
            self._expire(now)
            self._failures.append(now)
            if len(self._failures) >= self.limit and not self._tripped:
                self._tripped = True
                return True
            return False
