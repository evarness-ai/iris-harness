"""The turn's data label: what a code call is governed as, and the floor under the loop's.

The ReAct loop feeds its own label into every ``PRE_LLM_CALL`` and every tool call it makes.
A plugin's code calls -- ``api.tools`` (``runtime/tool_service.py``) and ``api.capability``
(``runtime/plugin_host/registry.py``) -- had no label at all. Now the turn pipeline
publishes the turn's label here for the duration of the turn (``turn_label_scope``, entered
by ``runtime/turn/pipeline.run_turn``, which ``chat``, ``chat_stream`` and the resume path
all run), the ``screen`` stage seeds it from the ``PRE_TURN`` classification, and the
harness stamps it on every code call it builds. A plugin cannot pass or override it.

It is a floor. A governed call's ``POST_TOOL_USE`` may raise the label (the output
classifier); the runner lifts the turn's label with it (``lift_turn_label``, the stricter
of the two, never lower), and the loop reads the lifted label before each model call
(``AgenticCore._governance_pre_llm``), so data a plugin's code pulled into the turn
governs the model calls that follow.

The scope holds one mutable holder, not the label itself: a lift made on a worker thread
(the executor runs tasks through ``bind_context``, a copy of the turn's context) reaches
the turn, because the copy shares the holder. Save/restore, not ``Token.reset``, for the
same generator-safety reason as ``session_scope``: the streaming path steps the turn from
a pinned context (``pin_context``).

It is the floor under EVERY model call in the turn, not only the loop's: each governed
``PRE_LLM_CALL`` passes the label its own prompt classified as through ``apply_turn_floor``
-- the loop (``AgenticCore._governance_pre_llm``), the shared client
(``CodingLLMClient._governance_pre_llm``: the curator's judges, narration, synthesis, fact
capture, the general handlers) and the callers that fire the hook themselves (task
planner, intent router, conversation compactor, entity extractor). A call made while the
turn holds personal data cannot egress as internal because its own prompt looked tamer.

One declared exemption, and only one: a call that carries stripped public content (the
cloud search-synthesis client; the search loop strips the user's stored personal data
before it builds that prompt, so "personal" there is PII inside fetched public web
content). Its floored label reads personal as internal. ``secret`` is never lowered, for
it or anyone.

Outside a turn (a heartbeat, a CLI command, the approved-call executor) there is no
holder: nothing is stamped, a lift is a no-op and the floor is the call's own label;
``POST_TOOL_USE`` still raises the label of the call itself.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from iris_harness.kernel.governance.hooks.types import DataClassification
from iris_harness.kernel.governance.plugins.output_classifier import more_restrictive


class _TurnLabel:
    """One turn's label. Only ever raised."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._label: DataClassification | None = None

    def get(self) -> DataClassification | None:
        with self._lock:
            return self._label

    def lift(self, label: DataClassification | None) -> None:
        with self._lock:
            self._label = more_restrictive(self._label, label)


_TURN_LABEL: ContextVar[_TurnLabel | None] = ContextVar("iris_turn_label", default=None)


@contextmanager
def turn_label_scope() -> Iterator[None]:
    """A turn's label for as long as the block runs; starts unset. Called by the harness."""
    prev = _TURN_LABEL.get()
    _TURN_LABEL.set(_TurnLabel())
    try:
        yield
    finally:
        _TURN_LABEL.set(prev)


def current_turn_label() -> DataClassification | None:
    """The turn's label, or None outside a turn (or before anything labelled it)."""
    holder = _TURN_LABEL.get()
    return holder.get() if holder is not None else None


def lift_turn_label(label: DataClassification | None) -> None:
    """Raise the turn's label to ``label`` if stricter; never lowers it. No-op outside a turn."""
    holder = _TURN_LABEL.get()
    if holder is not None:
        holder.lift(label)


def apply_turn_floor(
    label: DataClassification | None, *, stripped_public_content: bool = False
) -> DataClassification | None:
    """The label a model call is governed as: its own, floored at the turn's.

    The one rule every ``PRE_LLM_CALL`` goes through. ``stripped_public_content`` is the
    declaration a caller makes when its prompt carries public content with the user's
    personal data stripped upstream (the cloud search-synthesis client, and nothing
    else): personal -- its prompt's or the turn's -- is read as internal. ``secret`` is
    never lowered. Outside a turn the floor is the call's own label.
    """
    floored = more_restrictive(label, current_turn_label())
    if stripped_public_content and floored == "personal":
        return "internal"
    return floored


__all__ = ["apply_turn_floor", "current_turn_label", "lift_turn_label", "turn_label_scope"]
