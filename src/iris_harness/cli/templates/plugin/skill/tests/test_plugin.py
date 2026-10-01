"""__tmpl_title's tests: the functions on their own, then the skill in a real IRIS.

The skill is a declarative plugin: ``manifest.yaml`` declares each tool and binds it to a
function in ``functions.py``. ``plugin(manifest=...)`` mounts it in-process with no
``setup`` -- exactly as IRIS mounts it once installed -- and ``harness`` builds the runtime
``iris`` runs, in a throwaway home, on the scripted model in ``model_script.yaml``, with
the network refused. A manifest IRIS refuses (a bad ``impl``, an argument the function
cannot take) leaves the plugin unloaded, which the first assertion of each test catches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from iris_harness.testing import harness, plugin

from __tmpl_package import functions

NAME = "__tmpl_name"
MANIFEST = Path(functions.__file__).with_name("manifest.yaml")
SCRIPT = Path(__file__).with_name("model_script.yaml")


def test_the_function_converts() -> None:
    assert functions.convert_length(1, "mi", "km")["value"] == 1.609
    assert functions.convert_length(1, "mi", "km", digits=1)["text"] == "1 mi = 1.6 km"


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_the_model_uses_the_skill_through_the_governed_loop(entry: str) -> None:
    with harness(plugins=[plugin(manifest=MANIFEST)], fake_model=SCRIPT) as h:
        assert h.plugin_loaded(NAME), h.plugins()[NAME]
        ask = h.chat if entry == "chat" else h.chat_stream
        result = ask("How many miles is 5 km?")
        assert result.answered, result.error
        assert result.text == "5 km is 3.107 miles."
        assert [call.rule for call in h.model_calls()] == ["call the tool", "answer from the tool"]
        assert h.audit_rows(hook_point="pre_tool_use")
        assert h.audit_gaps() == []


def test_a_bad_argument_is_an_observation_and_the_function_never_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    real = functions.convert_length

    def counted(**kwargs: Any) -> dict[str, object]:
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(functions, "convert_length", counted)
    with harness(plugins=[plugin(manifest=MANIFEST)], fake_model=SCRIPT) as h:
        assert h.plugin_loaded(NAME), h.plugins()[NAME]
        result = h.chat("How many miles is 5 parsecs?")
        assert result.answered, result.error
        assert result.text == "I cannot convert parsecs."
        assert [call.rule for call in h.model_calls()] == [
            "call it with a bad unit",
            "report the refusal",
        ]
        assert calls == []
        assert h.audit_gaps() == []
