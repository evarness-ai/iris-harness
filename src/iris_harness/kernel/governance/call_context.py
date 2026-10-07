"""Which governed call a call is nested in, and which attempt it is (issue #134, stage 2).

The runner mints a ``call_id`` for every governed call. This module is how a call learns
the id of the call it runs inside, so its audit rows can name that parent:

* ``call_scope`` makes a call the *current* call for the code it runs (a plugin tool's body,
  a capability provider). A governed call started inside reads :func:`current_call_id` as
  its parent.
* ``register_call`` records a call's lineage (parent, attempt, ``replay_of``) the moment it
  is minted; the kernel looks it up by the id on each row's own metadata
  (:func:`lineage_of`), so every row of the call carries it, including the rows a stream
  writes after the call returned.
* ``mark_run_resumed`` records that a halted run was re-entered; the kernel stamps
  ``resumed_from_run`` on the rows that run writes afterwards.

Nothing here is a caller's to set: ``ToolCall`` and the tool arguments have no such field,
and the kernel reads lineage only from this module, never from a payload. A lineage the
registry no longer holds (it is bounded) simply leaves the fields off the row; nothing is
invented.

Context propagation: a ``ContextVar`` follows ``await`` and ``asyncio.to_thread`` /
``contextvars.copy_context().run``, but NOT a bare ``threading.Thread`` or
``loop.run_in_executor`` (neither copies the context). A governed call started in such a
thread has no parent here, the same limit the egress scope has.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

_CURRENT: ContextVar[str | None] = ContextVar("iris_current_call_id", default=None)

# Bounded: a long-lived process makes many calls. 8192 lineages is a few hundred KB.
_MAX_CALLS = 8192
_MAX_RUNS = 1024


@dataclass(frozen=True)
class CallLineage:
    """What the harness knows about one call beyond its own id."""

    parent_call_id: str | None
    attempt: int
    replay_of: str | None


_lock = threading.Lock()
_calls: OrderedDict[str, CallLineage] = OrderedDict()
_resumed_runs: OrderedDict[str, None] = OrderedDict()


def current_call_id() -> str | None:
    """The governed call the running code is inside, or None."""
    return _CURRENT.get()


@contextmanager
def call_scope(call_id: str) -> Iterator[None]:
    """Make ``call_id`` the current call for the code inside the block.

    Restored in ``finally`` (an exception, a cancellation, a ``GeneratorExit`` all leave the
    previous value back). Save/restore rather than ``Token.reset``, as ``turn_scope`` does:
    a token cannot be reset from a different context.
    """
    previous = _CURRENT.get()
    _CURRENT.set(call_id)
    try:
        yield
    finally:
        _CURRENT.set(previous)


def register_call(call_id: str, *, held_call_id: str | None = None) -> CallLineage:
    """Record the lineage of a call being minted now, and return it.

    The parent is the call the running code is inside. A call that re-executes a held,
    now-approved attempt is attempt 2 and names the held attempt in ``replay_of``; every
    other call is attempt 1. (A retry through ``iris run resume`` or the loop's repeated-
    action guard is not stamped here: those paths have no stable key for "the same effect".)
    """
    lineage = CallLineage(
        parent_call_id=_CURRENT.get(),
        attempt=2 if held_call_id else 1,
        replay_of=held_call_id or None,
    )
    with _lock:
        _calls[call_id] = lineage
        _calls.move_to_end(call_id)
        while len(_calls) > _MAX_CALLS:
            _calls.popitem(last=False)
    return lineage


def lineage_of(call_id: str | None) -> CallLineage | None:
    """The recorded lineage of ``call_id``, or None (an unknown or evicted call)."""
    if not call_id:
        return None
    with _lock:
        return _calls.get(call_id)


def mark_run_resumed(run_id: str) -> None:
    """Record that ``run_id`` was re-entered after a halt, in this process."""
    if not run_id:
        return
    with _lock:
        _resumed_runs[run_id] = None
        _resumed_runs.move_to_end(run_id)
        while len(_resumed_runs) > _MAX_RUNS:
            _resumed_runs.popitem(last=False)


def is_run_resumed(run_id: str | None) -> bool:
    if not run_id:
        return False
    with _lock:
        return run_id in _resumed_runs


__all__ = [
    "CallLineage",
    "call_scope",
    "current_call_id",
    "is_run_resumed",
    "lineage_of",
    "mark_run_resumed",
    "register_call",
]
