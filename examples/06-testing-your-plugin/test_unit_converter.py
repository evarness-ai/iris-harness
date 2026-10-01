"""How to test a plugin, from the fastest test to the most complete.

1. The plugin's own logic, as plain functions: no IRIS at all.
2. The plugin in a real governed IRIS (``harness``) on a scripted model: a chat turn,
   its tool call, its audit rows -- offline.
3. The guarantees you can assert on: nothing reached the network, every model call and
   every answer was audited, the plugin imports only the stable API.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from unit_converter import convert, convert_units, setup

from iris_harness.sdk import PluginAPI
from iris_harness.testing import (
    NetworkBlockedError,
    Script,
    check_stable_imports,
    harness,
    no_network,
    plugin,
)

HERE = Path(__file__).parent
MANIFEST = HERE / "manifest.yaml"
SCRIPT = Script.load(HERE / "fake_model.yaml")
QUESTION = "How many miles is 10 km?"


# -- 1. the logic, without IRIS ----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "a", "b", "expected"),
    [
        (10, "km", "mi", 6.2137),
        (100, "c", "f", 212.0),
        (0, "k", "c", -273.15),
        (3, "ft", "m", 0.9144),
    ],
)
def test_convert(value: float, a: str, b: str, expected: float) -> None:
    assert convert(value, a, b) == pytest.approx(expected, abs=1e-4)


def test_the_tool_reports_an_error_instead_of_raising() -> None:
    assert convert_units({"value": 1, "from_unit": "km", "to_unit": "c"}).startswith("error:")
    assert convert_units({"value": "lots"}).startswith("error:")


# -- 2. the plugin in a governed IRIS -----------------------------------------------------


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_turn_calls_the_tool_and_answers_from_it(entry: str) -> None:
    # Test both entry points: the REPL and the web stream (`chat_stream`) are what people
    # use, and a behaviour that only works on `chat` is a false green.
    with harness(plugins=[plugin(setup, manifest=MANIFEST)], fake_model=SCRIPT) as h:
        assert h.plugin_loaded("unit_converter")

        result = h.chat(QUESTION) if entry == "chat" else h.chat_stream(QUESTION)

        assert result.answered, result.error
        assert result.text == "10 km = 6.21 mi."
        if entry == "chat_stream":
            assert result.events[-1].kind == "done"
        # The scripted calls ran in order: the router, then the two loop steps.
        assert [call.rule for call in h.model_calls()] == [
            "route",
            "call convert_units",
            "answer from the tool",
        ]
        # The tool call went through governance: checked before and after it ran.
        tool_rows = h.audit_rows(hook_point="pre_tool_use", session_id=result.session_id)
        assert {row.tool for row in tool_rows} == {"convert_units"}
        assert all(row.decision == "allow" for row in tool_rows)


def test_an_undeclared_tool_is_refused_and_the_plugin_shows_degraded() -> None:
    def registers_an_extra_tool(api: PluginAPI) -> None:
        setup(api)
        api.register_tool("secret_tool", "Not in the manifest.", lambda args: "x")

    extra = plugin(registers_an_extra_tool, manifest=MANIFEST)
    with harness(plugins=[extra], fake_model=SCRIPT) as h:
        status, _error = h.plugins()["unit_converter"]
        assert status == "degraded"


# -- 3. the guarantees ------------------------------------------------------------------


def test_every_model_call_and_every_answer_is_audited() -> None:
    with harness(plugins=[plugin(setup, manifest=MANIFEST)], fake_model=SCRIPT) as h:
        h.chat(QUESTION)
        assert h.audit_gaps() == []
        # Each model call's audit rows, one key per call: (run, step).
        calls = h.audit_rows(hook_point="pre_llm_call")
        assert len({(row.run_id, row.step_id) for row in calls}) >= len(h.model_calls())


def test_nothing_reaches_the_network() -> None:
    # The harness refuses the network for its whole life; `no_network` does the same for
    # any block of your own, and lists what was attempted.
    with no_network() as attempted:
        with harness(plugins=[plugin(setup, manifest=MANIFEST)], fake_model=SCRIPT) as h:
            h.chat(QUESTION)
    assert attempted == []

    import socket

    with no_network(), pytest.raises(NetworkBlockedError):
        socket.create_connection(("192.0.2.1", 443), timeout=1)


def test_the_plugin_imports_only_the_stable_api() -> None:
    assert check_stable_imports([HERE / "unit_converter.py"]) == []
