"""The PostStep payload carries a bounded args text beside the hash (loop plan, decision 5)."""

from __future__ import annotations

import json

from iris_harness.agent.agentic_core import (
    _ACTION_INPUT_TEXT_CHARS,
    AgenticCoreConfig,
    _hash_action_input,
    _render_action_input,
)


def test_render_action_input_matches_hash_encoding_order() -> None:
    a = _render_action_input({"b": 1, "a": "x"})
    b = _render_action_input({"a": "x", "b": 1})
    assert a == b == json.dumps({"a": "x", "b": 1}, sort_keys=True)
    assert _hash_action_input({"b": 1, "a": "x"}) == _hash_action_input({"a": "x", "b": 1})


def test_render_action_input_is_bounded() -> None:
    text = _render_action_input({"doc": "x" * 5000})
    assert len(text) == _ACTION_INPUT_TEXT_CHARS


def test_render_action_input_keeps_non_ascii() -> None:
    assert "₹3150.40" in _render_action_input({"note": "₹3150.40"})


def test_default_iteration_cap_is_ten() -> None:
    assert AgenticCoreConfig().max_iterations == 10
