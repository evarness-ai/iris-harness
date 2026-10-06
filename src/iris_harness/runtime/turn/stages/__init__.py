"""The stages of a turn, one module each (OSS plan decision 9).

    screen → intercept → classify → resolve → route → plan → execute → curate → guard → notice → record

Each stage serves some turns (``pipeline.STAGE_AUDIENCE``): a deterministic handler's
answer skips the model path and goes to ``guard`` and ``record``; nothing breaks out.

Each module exposes ``run(runtime, state) -> Iterator[StreamEvent]``. A stage
yields stream events (tokens, activity, trace) as it works and writes its
product onto the :class:`~iris_harness.runtime.turn.state.TurnState`. Setting
``state.result`` ends the turn. ``screen`` fires the kernel's ``PRE_TURN`` hooks on
every turn's raw message; the other hook points fire inside the subsystems the
stages call (classifier, executor, curator), so the stage boundaries are also the
audit boundaries.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Any

from iris_harness.foundation.observability.tracer import maybe_current_span

if TYPE_CHECKING:
    from collections.abc import Iterator

    from iris_harness.runtime.turn.host import TurnHost
    from iris_harness.runtime.turn.state import TurnState


@contextmanager
def stage_span(runtime: TurnHost, state: TurnState, name: str) -> Iterator[Any]:
    """A context-attached stage span on the sync path; nothing on the stream path.

    ``maybe_current_span`` binds an OpenTelemetry token to the current context,
    which is only safe where enter and exit happen in the same context — true
    when ``chat()`` drains the pipeline inline, not inside an ASGI-driven
    generator. The streaming path therefore gets no per-stage spans, exactly as
    before the pipeline existed.
    """
    if state.stage_spans:
        with maybe_current_span(runtime.tracer, name) as span:
            yield span
    else:
        with nullcontext(None) as span:
            yield span


__all__ = ["stage_span"]
