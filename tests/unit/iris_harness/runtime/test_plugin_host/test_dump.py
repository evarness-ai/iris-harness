"""`iris --dump-config` tree + text rendering over the shipped profiles."""

from __future__ import annotations

import json
from pathlib import Path

from iris_harness.runtime.plugin_host.dump import dump_config, render_json, render_text

_SHIPPED_CONFIG = Path(__file__).resolve().parents[5] / "config"


def test_shipped_profiles_all_resolve_the_system_plugin(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.delenv("IRIS_PROFILE", raising=False)
    monkeypatch.delenv("IRIS_PLUGINS_DISABLE", raising=False)
    monkeypatch.delenv("IRIS_PLUGINS_ENABLE", raising=False)
    # Every profile this tree ships: the public one (OSS plan R1) has no
    # `personal-assistant`, which mounts the private domains.
    shipped = sorted(p.stem for p in (_SHIPPED_CONFIG / "profiles").glob("*.yaml"))
    assert {"minimal", "default", "email"} <= set(shipped)
    for name in shipped:
        tree = dump_config(_SHIPPED_CONFIG, profile_name=name)
        assert tree["profile"]["name"] == name
        rows = {r["name"]: r for r in tree["plugins"]}
        assert rows["system"]["source"] == "builtin:iris_harness.plugins_builtin.system"
        assert set(rows["system"]["provides"]) == {"intercept", "tool", "heartbeat"}
        assert set(tree["available_profiles"]) == set(shipped)


def test_render_text_and_json(tmp_path: Path) -> None:
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "default.yaml").write_text(
        "description: demo\nplugins:\n  - name: system\n  - name: ghost\nintercept_order: [x]\n",
        encoding="utf-8",
    )
    # Name the profile: the suite runs as `personal-assistant` (see tests/conftest.py),
    # and this test is about rendering the profile it just wrote.
    tree = dump_config(tmp_path, profile_name="default")
    text = render_text(tree)
    assert "profile: default" in text and "demo" in text
    assert "[on ] system" in text and "builtin:" in text
    assert "[on ] ghost" in text and "NOT FOUND" in text
    assert "intercept order override: x" in text
    assert json.loads(render_json(tree))["profile"]["name"] == "default"


def test_the_search_providers_a_plugin_declares_are_shown(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Before anything boots: the research plugin's manifest names its five providers."""
    monkeypatch.delenv("IRIS_PLUGINS_DISABLE", raising=False)
    monkeypatch.delenv("IRIS_PLUGINS_ENABLE", raising=False)
    tree = dump_config(_SHIPPED_CONFIG, profile_name="default")
    rows = {r["name"]: r for r in tree["plugins"]}
    assert rows["research"]["search_providers"] == ["searxng", "tavily", "exa", "brave", "ddg"]
    assert rows["system"]["search_providers"] == []
    assert "search providers: searxng, tavily, exa, brave, ddg" in render_text(tree)
