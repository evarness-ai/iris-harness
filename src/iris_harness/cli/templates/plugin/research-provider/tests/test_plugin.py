"""__tmpl_title's tests: the search on its own, then serving IRIS's ``research`` tool.

``harness`` builds the runtime ``iris`` runs, in a throwaway home, on the scripted model
in ``model_script.yaml``, with the network refused and no search key set; the ``default``
profile mounts the built-in ``research`` plugin, and ``plugin`` mounts this one beside it,
in-process. The model calls ``research``; the chain reaches this provider before the
keyless DuckDuckGo floor, so the answer can only have come from here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from iris_harness.sdk.research import SearchProvider
from iris_harness.testing import harness, plugin

from __tmpl_package import plugin as this_plugin

NAME = "__tmpl_name"
MANIFEST = Path(this_plugin.__file__).with_name("manifest.yaml")
SCRIPT = Path(__file__).with_name("model_script.yaml")


def test_the_backend_is_a_search_provider() -> None:
    backend = this_plugin.Backend()
    assert isinstance(backend, SearchProvider)
    assert backend.is_available()


def test_the_search_returns_at_most_max_results() -> None:
    hits = this_plugin.Backend().search("tide tables", max_results=1)
    assert [hit.title for hit in hits] == ["About tide tables"]
    assert hits[0].url.startswith("https://")


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_research_call_reaches_the_provider(entry: str) -> None:
    mounted = plugin(this_plugin.setup, manifest=MANIFEST)
    with harness(profile="default", plugins=[mounted], fake_model=SCRIPT) as h:
        assert h.plugin_loaded(NAME), h.plugins()[NAME]
        assert h.plugin_loaded("research"), h.plugins()["research"]
        ask = h.chat if entry == "chat" else h.chat_stream
        result = ask("Search the web for tide tables")
        assert result.answered, result.error
        assert result.text == "The first result is About tide tables."
        assert [call.rule for call in h.model_calls()] == ["search", "answer from the results"]
        # The research call and its external results both passed the kernel.
        assert {row.tool for row in h.audit_rows(hook_point="pre_tool_use")} == {"research"}
        assert h.audit_rows(hook_point="post_tool_use")
        assert h.audit_gaps() == []
