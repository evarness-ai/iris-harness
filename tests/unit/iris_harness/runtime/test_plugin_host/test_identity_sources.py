"""A plugin's owner-identity source: caller-bound and governed by its manifest (ADR-0125).

A plugin that knows an address of the owner's hands it to the guards through
``api.register_owner_identity_source``. The source is ``plugin:<name>`` -- the harness's
stamp, never the plugin's word -- and it may supply only the kinds the manifest declares
under ``identity: provides``. Anything else is dropped and charged to the plugin; a
registration with nothing declared is refused, as an undeclared tool is.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus


def _api(registry: PluginRegistry, *, identity: Any = None, name: str = "mail") -> PluginAPI:
    manifest = None
    if identity is not False:
        raw: dict[str, Any] = {"name": name}
        if identity is not None:
            raw["identity"] = identity
        manifest = PluginManifest.model_validate(raw)
    registry.add_plugin(
        PluginRecord(name=name, source="test", status=PluginStatus.LOADED, manifest=manifest)
    )
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
def seam(owner_identity_seam: Any) -> Any:
    owner_identity_seam.register_identity_text_provider(list)
    return owner_identity_seam


def _account(**literals: list[str]) -> Any:
    def provider() -> Mapping[str, Iterable[str]]:
        return literals

    return provider


def test_a_declared_source_reaches_the_corpus_under_the_plugins_name(seam: Any) -> None:
    registry = PluginRegistry()
    api = _api(registry, identity={"provides": ["email"]})
    api.register_owner_identity_source(_account(email=["robin@mail.example"]))
    assert seam.owner_identity_sources() == ("documents", "plugin:mail")
    assert seam.owner_identity().of("email") == {"robin@mail.example"}
    assert ("mail", "owner_identity", "mail") in registry.seams()
    record = registry.get("mail")
    assert record is not None and record.status is PluginStatus.LOADED


def test_an_undeclared_kind_is_dropped_and_charged_to_the_plugin(seam: Any) -> None:
    registry = PluginRegistry()
    api = _api(registry, identity={"provides": ["email"]})
    api.register_owner_identity_source(
        _account(email=["robin@mail.example"], name=["Robin"], never_match=["Robin Example"])
    )
    corpus = seam.owner_identity()
    assert corpus.of("email") == {"robin@mail.example"}
    assert corpus.of("name") == frozenset()
    record = registry.get("mail")
    assert record is not None and record.status is PluginStatus.DEGRADED
    assert record.last_error is not None
    assert "name, never_match" in record.last_error and "identity: provides" in record.last_error


@pytest.mark.parametrize("identity", [None, {"provides": []}, False])
def test_registering_without_a_declaration_is_refused(seam: Any, identity: Any) -> None:
    registry = PluginRegistry()
    api = _api(registry, identity=identity)
    api.register_owner_identity_source(_account(email=["robin@mail.example"]))
    assert seam.owner_identity_sources() == ("documents",)
    assert seam.owner_identity().of("email") == frozenset()
    record = registry.get("mail")
    assert record is not None and record.status is PluginStatus.DEGRADED
    assert "declares no kinds under 'identity: provides'" in (record.last_error or "")


def test_a_failing_source_is_charged_to_the_plugin_and_costs_only_its_literals(
    seam: Any,
) -> None:
    registry = PluginRegistry()
    api = _api(registry, identity={"provides": ["email"]})

    def boom() -> Mapping[str, Iterable[str]]:
        raise RuntimeError("token expired")

    api.register_owner_identity_source(boom)
    assert seam.owner_identity().of("email") == frozenset()
    record = registry.get("mail")
    assert record is not None and "owner_identity:mail" in (record.last_error or "")


def test_a_plugin_cannot_name_its_source(seam: Any) -> None:
    """The source name is the harness's stamp: two plugins, two sources, each its own."""
    registry = PluginRegistry()
    _api(registry, identity={"provides": ["email"]}, name="mail").register_owner_identity_source(
        _account(email=["a@mail.example"])
    )
    _api(registry, identity={"provides": ["handle"]}, name="code").register_owner_identity_source(
        _account(handle=["robin-gh"], email=["b@mail.example"])
    )
    assert seam.owner_identity_sources() == ("documents", "plugin:code", "plugin:mail")
    assert seam.owner_identity().of("email") == {"a@mail.example"}


def test_a_fingerprint_is_passed_through(seam: Any) -> None:
    registry = PluginRegistry()
    api = _api(registry, identity={"provides": ["email"]})
    probes: list[int] = []

    def fingerprint() -> int:
        probes.append(1)
        return 1

    api.register_owner_identity_source(_account(email=["a@mail.example"]), fingerprint=fingerprint)
    seam.owner_identity()
    assert probes == [1]


@pytest.mark.parametrize("kind", ["secret", "link", "shoe_size"])
def test_the_manifest_refuses_an_unprovidable_kind(kind: str) -> None:
    with pytest.raises(ValueError, match="not a kind a plugin may provide"):
        PluginManifest.model_validate({"name": "mail", "identity": {"provides": [kind]}})


def test_the_manifest_refuses_unknown_identity_keys() -> None:
    with pytest.raises(ValueError):
        PluginManifest.model_validate({"name": "mail", "identity": {"unmask": ["email"]}})


def test_the_default_manifest_provides_no_identity() -> None:
    assert PluginManifest.model_validate({"name": "mail"}).identity.provides == ()
