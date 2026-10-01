"""The search-provider chain: one registry, ordered by config, owned per plugin.

Every provider -- the research plugin's built-ins and any other plugin's -- joins through
``PluginAPI.register_search_provider``. These tests pin the order (config priority, then
the registered priority, then ``default_priority``; ties by registration), the ownership
rule, the fault boundary, and that a provider leaves the chain with its plugin.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from iris_harness.foundation.process_state import restore_process_state, snapshot_process_state
from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.sdk.research import SearchHit, SearchProvider, search_provider_chain
from iris_harness.services.research import providers as chain


class _Fake:
    """A provider that records what it was asked."""

    def __init__(self, *, available: bool = True, hits: int = 1, raises: bool = False) -> None:
        self.available = available
        self.hits = hits
        self.raises = raises
        self.queries: list[str] = []

    def is_available(self) -> bool:
        return self.available

    def search(self, query: str, *, max_results: int, **_: object) -> list[SearchHit]:
        self.queries.append(query)
        if self.raises:
            raise RuntimeError("backend down")
        return [SearchHit(title=f"hit {n}", url=f"https://x.test/{n}") for n in range(self.hits)]


@pytest.fixture(autouse=True)
def _clean_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    snapshot = snapshot_process_state()
    chain._entries.clear()
    monkeypatch.setenv("IRIS_CONFIG_DIR", str(tmp_path / "config"))
    (tmp_path / "config").mkdir()
    yield
    restore_process_state(snapshot)


def _config(tmp_path: Path, text: str) -> None:
    (tmp_path / "config" / chain.CONFIG_FILE).write_text(text, encoding="utf-8")


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


def _names() -> list[str]:
    return [link.name for link in search_provider_chain()]


def test_the_fake_is_a_search_provider() -> None:
    assert isinstance(_Fake(), SearchProvider)


def test_the_shipped_config_orders_the_builtins_and_puts_a_plugin_before_the_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The shipped file (an override dir without one reads it).
    registry = PluginRegistry()
    research = _api(registry, "research")
    for name in ("searxng", "tavily", "exa", "brave", "ddg"):
        research.register_search_provider(name, _Fake())
    _api(registry, "mine").register_search_provider("mine", _Fake())

    assert _names() == ["searxng", "tavily", "exa", "brave", "mine", "ddg"]


def test_config_priority_beats_the_registered_one_and_can_turn_a_provider_off(
    tmp_path: Path,
) -> None:
    _config(
        tmp_path,
        "default_priority: 50\n" "providers:\n" "  a: {priority: 30}\n" "  b: {enabled: false}\n",
    )
    api = _api(PluginRegistry(), "p")
    api.register_search_provider("a", _Fake(), priority=1)
    api.register_search_provider("b", _Fake(), priority=0)
    api.register_search_provider("c", _Fake(), priority=10)
    api.register_search_provider("d", _Fake())  # default_priority
    api.register_search_provider("e", _Fake())  # tie with d: registration order

    assert _names() == ["c", "a", "d", "e"]


def test_an_unavailable_provider_is_skipped(tmp_path: Path) -> None:
    _config(tmp_path, "providers: {}\n")
    api = _api(PluginRegistry(), "p")
    api.register_search_provider("off", _Fake(available=False))
    api.register_search_provider("on", _Fake())
    assert _names() == ["on"]
    assert [e.name for e in chain.registered_search_providers()] == ["off", "on"]


def test_a_malformed_config_is_an_error_not_the_default_order(tmp_path: Path) -> None:
    _config(tmp_path, "providers:\n  a: {priority: first}\n")
    _api(PluginRegistry(), "p").register_search_provider("a", _Fake())
    with pytest.raises(ValueError, match="search provider config invalid"):
        search_provider_chain()


def test_a_name_another_plugin_holds_is_refused_and_charged_to_the_newcomer() -> None:
    registry = PluginRegistry()
    _api(registry, "first").register_search_provider("shared", _Fake())
    late = _Fake()
    _api(registry, "second").register_search_provider("shared", late)

    (link,) = search_provider_chain()
    assert link.provider is not late
    record = registry.get("second")
    assert record is not None and record.status is PluginStatus.DEGRADED
    assert "already registered by plugin:first" in (record.last_error or "")


def test_the_same_plugin_registering_again_replaces_its_own() -> None:
    api = _api(PluginRegistry(), "p")
    first, second = _Fake(), _Fake()
    api.register_search_provider("x", first)
    api.register_search_provider("x", second)
    search_provider_chain()[0].search("q", max_results=1)
    assert (first.queries, second.queries) == ([], ["q"])


def test_something_that_is_not_a_provider_is_refused() -> None:
    registry = PluginRegistry()
    _api(registry, "p").register_search_provider("x", object())  # type: ignore[arg-type]
    assert search_provider_chain() == []
    record = registry.get("p")
    assert record is not None and "not a SearchProvider" in (record.last_error or "")


def test_a_bad_name_is_refused() -> None:
    registry = PluginRegistry()
    _api(registry, "p").register_search_provider("Bad Name", _Fake())
    assert search_provider_chain() == []
    assert "not valid" in (registry.get("p").last_error or "")  # type: ignore[union-attr]


def test_a_raising_provider_is_charged_to_its_plugin_and_reraised() -> None:
    registry = PluginRegistry()
    _api(registry, "p").register_search_provider("x", _Fake(raises=True))
    (link,) = search_provider_chain()
    with pytest.raises(RuntimeError, match="backend down"):
        link.search("q", max_results=1)
    record = registry.get("p")
    assert record is not None and record.failure_count == 1
    assert (record.last_error or "").startswith("search_provider:x.search:")


def test_a_raising_availability_check_leaves_the_provider_out() -> None:
    class Broken(_Fake):
        def is_available(self) -> bool:
            raise RuntimeError("no config")

    registry = PluginRegistry()
    api = _api(registry, "p")
    api.register_search_provider("broken", Broken())
    api.register_search_provider("fine", _Fake())
    assert _names() == ["fine"]
    assert registry.get("p").failure_count == 1  # type: ignore[union-attr]


def test_the_provider_leaves_the_chain_when_its_plugin_is_unmounted() -> None:
    registry = PluginRegistry()
    _api(registry, "p").register_search_provider("x", _Fake())
    assert _names() == ["x"]

    record = registry.get("p")
    assert record is not None
    record.status = PluginStatus.FAILED  # e.g. setup raised after registering
    assert _names() == []

    # Once its owner is gone, the name is free for another plugin.
    _api(registry, "q").register_search_provider("x", _Fake())
    assert _names() == ["x"]


def test_the_registration_is_recorded_as_a_seam() -> None:
    registry = PluginRegistry()
    _api(registry, "p").register_search_provider("x", _Fake())
    assert registry.describe()["seams"] == ["p:search_provider:x"]


def test_unregister_removes_only_the_owners_entry() -> None:
    _api(PluginRegistry(), "p").register_search_provider("x", _Fake())
    assert chain.unregister_search_provider("x", owner="plugin:q") is False
    assert chain.unregister_search_provider("x", owner="plugin:p") is True
    assert _names() == []


# ─── The manifest declares the providers a plugin registers ─────────────────


def _declaring(registry: PluginRegistry, name: str, providers: list[str]) -> PluginAPI:
    """An API for a plugin whose manifest declares ``providers`` under search_providers."""
    api = _api(registry, name)
    record = registry.get(name)
    assert record is not None
    record.manifest = PluginManifest.model_validate({"name": name, "search_providers": providers})
    return api


def test_a_declared_provider_joins_the_chain() -> None:
    registry = PluginRegistry()
    _declaring(registry, "p", ["mine"]).register_search_provider("mine", _Fake())
    assert _names() == ["mine"]
    record = registry.get("p")
    assert record is not None and record.status is PluginStatus.LOADED


def test_an_undeclared_provider_is_refused_and_the_plugin_degraded() -> None:
    registry = PluginRegistry()
    api = _declaring(registry, "p", ["mine"])
    api.register_search_provider("other", _Fake())

    assert _names() == []
    record = registry.get("p")
    assert record is not None and record.status is PluginStatus.DEGRADED
    assert "not declared under 'search_providers:'" in (record.last_error or "")
    assert registry.seams() == []


def test_a_manifest_with_no_providers_declares_none() -> None:
    registry = PluginRegistry()
    _declaring(registry, "p", []).register_search_provider("mine", _Fake())
    assert _names() == []
    assert registry.get("p").status is PluginStatus.DEGRADED  # type: ignore[union-attr]


@pytest.mark.parametrize("bad", ["Bad Name", "1st", ""])
def test_the_manifest_holds_provider_names_to_the_chain_rule(bad: str) -> None:
    with pytest.raises(ValueError, match="not a provider name"):
        PluginManifest.model_validate({"name": "p", "search_providers": [bad]})


def test_the_manifest_dedupes_provider_names() -> None:
    manifest = PluginManifest.model_validate({"name": "p", "search_providers": ["a", "b", "a"]})
    assert manifest.search_providers == ("a", "b")


# ─── SearchHit is the whole return contract ──────────────────────────────────


def test_a_hit_is_frozen_keyword_only_and_its_extra_read_only() -> None:
    labels = {"engine": "bing"}
    hit = SearchHit(url="https://x.test", title="T", extra=labels)
    labels["engine"] = "changed"
    assert hit.extra == {"engine": "bing"}
    with pytest.raises(AttributeError):
        hit.title = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        hit.extra["engine"] = "x"  # type: ignore[index]
    with pytest.raises(TypeError):
        SearchHit("https://x.test", "T")  # type: ignore[misc]
    # ``extra`` stays out of the hash, so a hit is hashable whatever its labels.
    assert hash(hit) == hash(SearchHit(url="https://x.test", title="T"))


class _OldStyle(_Fake):
    """A provider written against the pre-0.1.0 contract: it returns another hit type."""

    def __init__(self, hit: object) -> None:
        super().__init__()
        self.hit = hit

    def search(self, query: str, *, max_results: int, **_: object) -> list[SearchHit]:
        self.queries.append(query)
        return [self.hit]  # type: ignore[list-item]


@dataclass
class _LegacyResult:
    """The shape the old stable ``SearchResult`` had: mutable, with engine-filled fields."""

    title: str
    url: str
    score: float = 0.0


@pytest.mark.parametrize(
    "hit", [_LegacyResult(title="T", url="https://x.test"), {"title": "T", "url": "u"}]
)
def test_a_provider_returning_anything_but_hits_is_refused_and_charged(hit: object) -> None:
    registry = PluginRegistry()
    _api(registry, "p").register_search_provider("x", _OldStyle(hit))
    (link,) = search_provider_chain()

    with pytest.raises(TypeError, match=r"search provider 'x' returned a \w+ hit, not a SearchHit"):
        link.search("q", max_results=1)
    record = registry.get("p")
    assert record is not None and record.status is PluginStatus.DEGRADED
    assert (record.last_error or "").startswith("search_provider:x.search: TypeError")


def test_a_provider_returning_a_non_list_is_refused() -> None:
    class _Gen(_Fake):
        def search(self, query: str, *, max_results: int, **_: object) -> list[SearchHit]:
            return iter([SearchHit(url="https://x.test", title="T")])  # type: ignore[return-value]

    _api(PluginRegistry(), "p").register_search_provider("x", _Gen())
    with pytest.raises(TypeError, match="returned list_iterator, not a list of SearchHit"):
        search_provider_chain()[0].search("q", max_results=1)


def test_the_chain_holds_a_directly_registered_provider_to_the_contract_too() -> None:
    """The core's own registration path (no plugin, no fault boundary) is checked as well."""
    chain.register_search_provider(
        "direct", _OldStyle(_LegacyResult(title="T", url="u")), owner="core:test"
    )
    with pytest.raises(TypeError, match="search provider 'direct'"):
        search_provider_chain()[0].search("q", max_results=1)
