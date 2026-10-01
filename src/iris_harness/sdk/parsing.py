"""Parsing helpers a plugin's handler shares with the loop it answers on.

A plugin handler that streams a model's output has to read the same shapes the
harness's own ReAct loop reads -- a tool call, a truncated tool call, a
degenerate repetition. Re-implementing that is how a plugin and the core come to
disagree about what the model said.

The names are underscore-prefixed because they were private to the runtime when
the code-exec plugin first needed them. That is a wart, not a warning: publishing
them here is the decision that they are part of the plugin contract, and the
spelling should lose its underscore the next time the ReAct format changes.
"""

from __future__ import annotations

from iris_harness.runtime.nlu_parsing import _deterministic_time_date_reply
from iris_harness.runtime.tool_call_parsing import (
    _PROSE_CUTOFF_LOOKBACK,
    _find_degenerate_repetition,
    _find_first_cutoff,
    _looks_truncated_tool_call,
    _one_line_preview,
    _parse_tool_call,
    _sanitize_prose,
)

__all__ = [
    "_PROSE_CUTOFF_LOOKBACK",
    "_deterministic_time_date_reply",
    "_find_degenerate_repetition",
    "_find_first_cutoff",
    "_looks_truncated_tool_call",
    "_one_line_preview",
    "_parse_tool_call",
    "_sanitize_prose",
]
