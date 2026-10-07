"""A chat turn that outlives the connection that started it.

The web chat streams a turn over one HTTP response. It used to run *inside* that
response: the web server pulls events from the turn's generator only while the client
is connected, so when a phone backgrounds the app (iOS suspends a home-screen web app
and drops its connections) the server stopped pulling and the turn stopped mid-way,
silently. The owner hit it on 2026-09-21: "trash the promo emails", "yes", switch to
another app, and the resumed run never reached its approval.

Now each turn runs to completion on its own thread; the HTTP response only relays its
events through a queue. A dropped connection stops the relay, not the turn, and the
pipeline's ``record`` stage files the answer in the session log as it always did, so
the chat shows it when the app comes back (``/api/chat-status`` says
``turn_in_progress``, and the app polls the session until it is not).

Because disconnecting no longer stops a turn, stopping is explicit: ``cancel`` sets a
flag the runner checks between events, and closes the turn's generator — exactly what
a disconnect used to do, now only when the owner presses Stop.
"""

from __future__ import annotations

import contextvars
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from iris_harness.foundation.logsafe import log_safe

logger = logging.getLogger(__name__)

_END = object()


@dataclass
class _Turn:
    session_id: str
    events: queue.Queue[Any] = field(default_factory=queue.Queue)
    cancel: threading.Event = field(default_factory=threading.Event)
    started_at: float = field(default_factory=time.time)
    # Set when the relay has gone away; the runner stops queueing (nobody is reading)
    # but keeps running the turn.
    detached: threading.Event = field(default_factory=threading.Event)


class DetachedTurns:
    """The turns in flight, one runner thread each, keyed by session."""

    def __init__(self, *, keepalive_s: float = 15.0) -> None:
        # Also how long a relay's worker thread may outlive its client: it notices the
        # disconnect at its next keep-alive at the latest.
        self._keepalive_s = keepalive_s
        self._lock = threading.Lock()
        self._turns: dict[str, _Turn] = {}

    def start(self, session_id: str, events: Callable[[], Iterator[Any]]) -> _Turn:
        turn = _Turn(session_id=session_id)
        with self._lock:
            self._turns[session_id] = turn
        # Run in a copy of the request's context, as the web server's worker thread did,
        # so whatever the request scoped (the calling principal, a test's overrides)
        # still holds for the turn.
        ctx = contextvars.copy_context()
        thread = threading.Thread(
            target=ctx.run,
            args=(self._run, turn, events),
            name=f"iris-turn-{session_id}",
            daemon=True,
        )
        thread.start()
        return turn

    def _run(self, turn: _Turn, events: Callable[[], Iterator[Any]]) -> None:
        gen = events()
        try:
            for evt in gen:
                if turn.cancel.is_set():
                    logger.info("turn for %s cancelled by the owner", log_safe(turn.session_id))
                    close = getattr(gen, "close", None)
                    if close is not None:
                        close()
                    break
                if not turn.detached.is_set():
                    turn.events.put(evt)
        except Exception as exc:  # reported to the relay, and logged
            logger.exception("chat turn for %s failed", log_safe(turn.session_id))
            if not turn.detached.is_set():
                turn.events.put(exc)
        finally:
            turn.events.put(_END)
            with self._lock:
                if self._turns.get(turn.session_id) is turn:
                    del self._turns[turn.session_id]
            if turn.detached.is_set():
                logger.info(
                    "turn for %s finished after its client left; the answer is in the session",
                    turn.session_id,
                )

    def relay(self, turn: _Turn, *, keepalive_s: float | None = None) -> Iterator[Any]:
        """Yield the turn's events as they come; ``None`` while it is quiet.

        ``None`` lets the caller send a keep-alive during a long model call, so idle
        timeouts on the way to the phone do not mistake thinking for a dead link. An
        exception raised by the turn is yielded, not raised, for the caller to report.
        Leaving the loop early (the client went away) detaches the turn: it keeps
        running, and stops queueing events nobody will read.
        """
        wait = self._keepalive_s if keepalive_s is None else keepalive_s
        try:
            while True:
                try:
                    item = turn.events.get(timeout=wait)
                except queue.Empty:
                    yield None
                    continue
                if item is _END:
                    return
                yield item
        finally:
            turn.detached.set()

    def running(self, session_id: str) -> _Turn | None:
        with self._lock:
            return self._turns.get(session_id)

    def cancel(self, session_id: str) -> bool:
        turn = self.running(session_id)
        if turn is None:
            return False
        turn.cancel.set()
        return True


__all__ = ["DetachedTurns"]
