"""A plugin's search provider serves the ``research`` tool with the tool's guards applied.

The provider joins the chain through ``PluginAPI.register_search_provider``; nothing in
the research plugin names it. These run the real tool callable over the real chain (the
engine resolves it per call, no injection), so what the provider sees is exactly what
would leave the machine.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from iris_harness.plugins_builtin.research import tool as tool_mod
from iris_harness.plugins_builtin.research import tools
from iris_harness.plugins_builtin.research.engine import ResearchEngine
from iris_harness.plugins_builtin.research.models import SearchResult
from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.sdk.research import SearchHit


class _Provider:
    def __init__(self, *, raises: bool = False) -> None:
        self.queries: list[str] = []
        self.raises = raises

    def is_available(self) -> bool:
        return True

    def search(self, query: str, *, max_results: int, **_: object) -> list[SearchHit]:
        self.queries.append(query)
        if self.raises:
            raise RuntimeError("down")
        return [
            SearchHit(
                title="Tide tables explained", url="https://tides.test/a", snippet="High water."
            )
        ]


def _api(registry: PluginRegistry, name: str) -> PluginAPI:
    registry.add_plugin(PluginRecord(name=name, source="test", status=PluginStatus.LOADED))
    services = HarnessServices(
        config_dir=Path("/nonexistent"),
        data_dir=Path("/nonexistent"),
        tier_router=None,
        agent_executor=None,
        heartbeats=None,
        channels=None,
        deterministic_reply=lambda **kw: None,
    )
    return PluginAPI(plugin=name, services=services, registry=registry)


@pytest.fixture
def engine(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A fresh engine with no cache, so each call reaches the chain."""
    monkeypatch.setattr(tool_mod, "_ENGINE", ResearchEngine(cache=None))
    # No keyed built-in may be configured from the developer's environment.
    for var in ("IRIS_SEARXNG_URL", "TAVILY_API_KEY", "EXA_API_KEY", "BRAVE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    # The keyless floor must never run here: it is real network.
    monkeypatch.setattr(
        "iris_harness.plugins_builtin.research.providers.duckduckgo.DuckDuckGoProvider.search",
        lambda self, *a, **k: pytest.fail("the chain fell through to DuckDuckGo"),
    )
    yield


def test_a_research_call_reaches_the_plugin_provider_before_the_floor(engine: None) -> None:
    provider = _Provider()
    _api(PluginRegistry(), "tides").register_search_provider("tides", provider)

    out = tools.build_research_call()({"query": "tide tables", "fetch_content": False})

    assert provider.queries == ["tide tables"]
    assert "provider: tides" in out and "Tide tables explained" in out


def test_the_owner_identifiers_are_stripped_before_the_plugin_provider_sees_them(
    engine: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tools, "owner_identity_identifiers", lambda: ["Jane Example"])
    provider = _Provider()
    _api(PluginRegistry(), "tides").register_search_provider("tides", provider)

    tools.build_research_call()(
        {"query": "Jane Example jane@example.org tide tables", "fetch_content": False}
    )

    (sent,) = provider.queries
    assert "Jane" not in sent and "jane@example.org" not in sent
    assert "tide tables" in sent


def test_a_personal_finance_query_never_reaches_the_plugin_provider(engine: None) -> None:
    provider = _Provider()
    _api(PluginRegistry(), "tides").register_search_provider("tides", provider)

    out = tools.build_research_call()({"query": "what are my insurance dues"})

    assert provider.queries == []
    assert "won't web-search your personal finances" in out


def test_a_failing_plugin_provider_is_charged_to_it_and_the_chain_moves_on(
    engine: None,
) -> None:
    registry = PluginRegistry()
    broken = _Provider(raises=True)
    _api(registry, "broken").register_search_provider("broken", broken, priority=450)
    working = _Provider()
    _api(registry, "tides").register_search_provider("tides", working, priority=460)

    out = tools.build_research_call()({"query": "tide tables", "fetch_content": False})

    assert broken.queries == ["tide tables"] and working.queries == ["tide tables"]
    assert "provider: tides" in out
    assert registry.get("broken").failure_count == 1  # type: ignore[union-attr]


def test_a_provider_returning_the_engines_result_type_is_refused_and_the_chain_moves_on(
    engine: None,
) -> None:
    """Before 0.1.0 a provider returned the engine's mutable ``SearchResult``. That type is
    the engine's own now; a provider still returning it is refused on the call (charged to
    its plugin, which shows degraded) rather than adapted, and the next provider answers."""

    class _Legacy(_Provider):
        def search(self, query: str, *, max_results: int, **_: object) -> list[SearchHit]:
            self.queries.append(query)
            return [SearchResult(title="Old shape", url="https://old.test/a")]  # type: ignore[list-item]

    registry = PluginRegistry()
    legacy = _Legacy()
    _api(registry, "legacy").register_search_provider("legacy", legacy, priority=450)
    _api(registry, "tides").register_search_provider("tides", _Provider(), priority=460)

    out = tools.build_research_call()({"query": "tide tables", "fetch_content": False})

    assert legacy.queries == ["tide tables"]
    assert "provider: tides" in out and "Old shape" not in out
    record = registry.get("legacy")
    assert record is not None and record.status is PluginStatus.DEGRADED
    assert "returned a SearchResult hit, not a SearchHit" in (record.last_error or "")
