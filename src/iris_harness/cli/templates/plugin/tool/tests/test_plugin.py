"""__tmpl_title's tests: the tool on its own, then in a real, governed IRIS.

``harness`` builds the runtime ``iris`` runs, in a throwaway home, on the scripted model
in ``model_script.yaml``, with the network refused; ``plugin`` mounts this plugin
in-process with its own manifest, so the manifest checks apply exactly as when it is
installed. Nothing here needs a model server or a network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from iris_harness.testing import harness, plugin

from __tmpl_package import plugin as this_plugin

NAME = "__tmpl_name"
MANIFEST = Path(this_plugin.__file__).with_name("manifest.yaml")
SCRIPT = Path(__file__).with_name("model_script.yaml")


def test_the_tool_counts() -> None:
    result = json.loads(this_plugin.run({"text": "One two. Three four!"}))
    assert result == {"words": 4, "sentences": 2, "characters": 20}


def test_bad_arguments_are_an_observation_not_an_exception() -> None:
    assert this_plugin.run({}).startswith("error:")


def test_the_plugin_mounts_with_its_manifest() -> None:
    with harness(plugins=[plugin(this_plugin.setup, manifest=MANIFEST)]) as h:
        assert h.plugin_loaded(NAME), h.plugins()[NAME]


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_the_model_calls_the_tool_through_the_governed_loop(entry: str) -> None:
    mounted = plugin(this_plugin.setup, manifest=MANIFEST)
    with harness(plugins=[mounted], fake_model=SCRIPT) as h:
        ask = h.chat if entry == "chat" else h.chat_stream
        result = ask("How many words are in: the quick brown fox")
        assert result.answered, result.error
        assert result.text == "That text has 4 words."
        assert [call.rule for call in h.model_calls()] == ["call the tool", "answer from the tool"]
        # The call went through the kernel, and every model call and answer was audited.
        assert h.audit_rows(hook_point="pre_tool_use")
        assert h.audit_gaps() == []
