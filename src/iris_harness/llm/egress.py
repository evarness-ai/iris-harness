"""Whether an LLM request may leave the owner's machines (decision 13).

A deployment may put a failover proxy in front of the local model server (a sidecar
the harness does not ship) that can hand a request to a cloud endpoint when the local
machine is asleep or slow. The model id says nothing about what a prompt holds: on
2026-09-28, 41 of 103 Azure-served prompts carried the owner's email and finance text,
and 39 their USER.md profile. So the proxy now fails over only a request marked
data-free with ``X-IRIS-May-Leave: 1``, and a request is marked only inside
:func:`data_free_call` — an explicit opt-in by a call site whose prompt provably holds
no profile, memory, conversation or tool result. Unmarked means private: deny by
default, so a call site that forgets fails safe.

No call site opts in today: the router sees the conversation, the agent the profile and
memory, the curator the answer. A future one opts in with a test proving its prompt clean.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

MAY_LEAVE_HEADER = "X-IRIS-May-Leave"

_DATA_FREE: ContextVar[bool] = ContextVar("iris_llm_data_free", default=False)


@contextmanager
def data_free_call() -> Iterator[None]:
    """Mark the LLM calls made inside as holding none of the owner's data."""
    token = _DATA_FREE.set(True)
    try:
        yield
    finally:
        _DATA_FREE.reset(token)


def egress_headers() -> dict[str, str]:
    """The proxy's mark for the current call: present only inside :func:`data_free_call`."""
    return {MAY_LEAVE_HEADER: "1"} if _DATA_FREE.get() else {}


__all__ = ["MAY_LEAVE_HEADER", "data_free_call", "egress_headers"]
