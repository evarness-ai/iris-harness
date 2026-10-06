"""``egress:``: the hosts a plugin may contact, declared, compiled and shown (issue #103)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from iris_harness.kernel.governance.plugin_egress import PluginEgressPolicy
from iris_harness.runtime.egress_access import compile_egress_policy
from iris_harness.runtime.plugin_host import PluginRegistry
from iris_harness.runtime.plugin_host.dump import dump_config, render_text
from iris_harness.runtime.plugin_host.manifest import PluginManifest, load_manifest
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginStatus

_CONFIG = Path(__file__).resolve().parents[5] / "config"


def _manifest(tmp_path: Path, body: str) -> PluginManifest:
    path = tmp_path / "manifest.yaml"
    path.write_text(f"name: demo\n{body}", encoding="utf-8")
    return load_manifest(path)


def test_a_manifest_that_declares_hosts_is_accepted(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        "egress:\n"
        "  hosts:\n"
        "    - api.open-meteo.com\n"
        "    - host: geocoding-api.open-meteo.com\n"
        "      data: personal\n"
        "    - host: '*.example.org'\n"
        "      ports: [443, 8443]\n",
    )
    hosts = {h.host: h for h in manifest.egress.hosts}
    assert hosts["api.open-meteo.com"].data == "internal"  # the shorthand's default
    assert hosts["api.open-meteo.com"].schemes == ("https",)
    assert hosts["geocoding-api.open-meteo.com"].data == "personal"
    assert hosts["*.example.org"].ports == (443, 8443)


def test_no_egress_block_declares_no_host(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, "")
    assert manifest.egress.hosts == () and not manifest.egress.declared


@pytest.mark.parametrize(
    "body, message",
    [
        ("egress:\n  hostz: [a.com]\n", "Extra inputs are not permitted"),
        ("egress:\n  hosts: [{host: a.com, color: red}]\n", "Extra inputs are not permitted"),
        ("egress:\n  hosts: ['*']\n", "open_web"),
        ("egress:\n  hosts: ['*com']\n", "wildcard"),
        ("egress:\n  hosts: ['https://a.com/x']\n", "not a host"),
        ("egress:\n  hosts: ['a.com:8443']\n", "not a host"),
        ("egress:\n  hosts: [{host: a.com, data: secret}]\n", "data"),
        ("egress:\n  hosts: [{host: a.com, schemes: [ftp]}]\n", "schemes"),
        ("egress:\n  hosts: [{host: a.com, schemes: []}]\n", "schemes"),
        ("egress:\n  hosts: [{host: a.com, ports: [0]}]\n", "not a port"),
        ("egress:\n  hosts: [a.com, A.com]\n", "twice"),
        ("egress:\n  open_web: true\n  hosts: [a.com]\n", "drop `hosts`"),
    ],
)
def test_an_unknown_shape_is_refused_with_a_clear_error(
    tmp_path: Path, body: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _manifest(tmp_path, body)


def test_external_service_needs_a_declared_destination(tmp_path: Path) -> None:
    tool = "provides: [tool]\ntools:\n  forecast:\n    sends_to: external_service\n"
    with pytest.raises(ValueError, match="declares no `egress`"):
        _manifest(tmp_path, tool)
    manifest = _manifest(tmp_path, tool + "egress:\n  hosts: [api.open-meteo.com]\n")
    assert manifest.tools["forecast"].sends_to == "external_service"
    # ``search_engine`` keeps working on its own, as the research plugin declares it.
    _manifest(tmp_path, tool.replace("external_service", "search_engine"))


def test_sends_to_still_refuses_an_unknown_destination(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sends_to"):
        _manifest(tmp_path, "provides: [tool]\ntools:\n  t:\n    sends_to: nowhere\n")


def _registry(**plugins: tuple[PluginStatus, str]) -> PluginRegistry:
    registry = PluginRegistry()
    for name, (status, body) in plugins.items():
        manifest = PluginManifest.model_validate({"name": name, **_yaml(body)})
        registry.add_plugin(
            PluginRecord(name=name, source="test", status=status, manifest=manifest)
        )
    return registry


def _yaml(body: str) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(body) or {}


def test_every_mounted_manifest_compiles_into_one_policy() -> None:
    registry = _registry(
        weather=(PluginStatus.LOADED, "egress: {hosts: [api.open-meteo.com]}"),
        quiet=(PluginStatus.LOADED, ""),
        broken=(PluginStatus.FAILED, "egress: {hosts: [evil.example]}"),
        off=(PluginStatus.DISABLED, "egress: {hosts: [evil.example]}"),
    )
    policy = compile_egress_policy(registry)
    assert isinstance(policy, PluginEgressPolicy)
    assert set(policy.plugins) == {"weather", "quiet"}  # a plugin that did not mount has no door
    ok = policy.decide("weather", scheme="https", host="api.open-meteo.com", port=443)
    assert ok.allowed and ok.rule == "api.open-meteo.com"
    assert not policy.decide("quiet", scheme="https", host="api.open-meteo.com", port=443).allowed
    assert not policy.decide("broken", scheme="https", host="evil.example", port=443).allowed


def test_the_dump_config_tree_shows_declared_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_PLUGINS_DISABLE", raising=False)
    monkeypatch.delenv("IRIS_PLUGINS_ENABLE", raising=False)
    tree = dump_config(_CONFIG, profile_name="default")
    rows = {r["name"]: r for r in tree["plugins"]}
    assert rows["research"]["egress"]["open_web"] is True
    assert rows["system"]["egress"] == {"open_web": False, "hosts": []}
    text = render_text(tree)
    assert "egress: any host (open_web)" in text
    assert "egress: none declared (raw network calls by this plugin are not governed)" in text


def test_the_shipped_gmail_manifest_names_its_google_hosts() -> None:
    root = Path(__file__).resolve().parents[5]
    manifest = load_manifest(root / "src/iris_personal/plugins/gmail/manifest.yaml")
    assert {h.host for h in manifest.egress.hosts} >= {
        "gmail.googleapis.com",
        "oauth2.googleapis.com",
    }
    assert all(h.data == "personal" for h in manifest.egress.hosts)


def test_hosts_per_plugin_are_capped() -> None:
    from iris_harness.kernel.governance.plugin_egress import MAX_HOSTS_PER_PLUGIN
    from iris_harness.runtime.plugin_host.manifest import PluginEgressDecl

    def decl(n: int) -> PluginEgressDecl:
        return PluginEgressDecl(hosts=tuple(f"h{i}.example.com" for i in range(n)))  # type: ignore[arg-type]

    assert len(decl(MAX_HOSTS_PER_PLUGIN).hosts) == MAX_HOSTS_PER_PLUGIN
    with pytest.raises(ValueError):
        decl(MAX_HOSTS_PER_PLUGIN + 1)


def test_iris_plugins_list_and_show_carry_the_declared_egress() -> None:
    from iris_harness.cli.plugins import egress_lines
    from iris_harness.runtime.plugin_host.inventory import plugin_detail, plugins_inventory

    registry = _registry(
        weather=(
            PluginStatus.LOADED,
            "egress: {hosts: [api.open-meteo.com, {host: geo.example.org, data: personal, "
            "ports: [8443]}]}",
        ),
        fetcher=(PluginStatus.LOADED, "egress: {open_web: true}"),
        quiet=(PluginStatus.LOADED, ""),
    )
    rows = {p["name"]: p for p in plugins_inventory(registry, None)["plugins"]}
    assert rows["quiet"]["egress"] == {"open_web": False, "hosts": []}
    assert rows["fetcher"]["egress"]["open_web"] is True
    detail = plugin_detail(registry, None, "weather")
    assert detail is not None and detail["egress"] == rows["weather"]["egress"]
    assert detail["manifest"]["egress"]["hosts"][0]["host"] == "api.open-meteo.com"
    lines = egress_lines(detail["egress"])
    first, second = (urlsplit(line.split()[0]) for line in lines[:2])
    assert (first.scheme, first.hostname, first.port) == ("https", "api.open-meteo.com", None)
    assert (second.scheme, second.hostname, second.port) == ("https", "geo.example.org", 8443)
    assert "data: internal" in lines[0] and "data: personal" in lines[1]
    assert "open_web" in egress_lines(rows["fetcher"]["egress"])[0]
    assert egress_lines(rows["quiet"]["egress"]) == []
