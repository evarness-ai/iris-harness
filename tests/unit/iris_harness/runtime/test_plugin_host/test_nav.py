"""The web console's nav (OSS plan R17): core screens + every mounted plugin's, nothing else.

Core-only: the plugins here are synthetic, so this runs in the public export too. The
shipped domain plugins' screens, per profile, are pinned beside them in
``tests/unit/iris_personal/plugins/test_personal_profile/test_webui_nav.py`` -- including
``GET /api/v1/webui/nav`` served by a runtime built on the real ``email`` profile and on
``default`` (no email plugins), with the email screens' APIs mounted only on the first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from iris_harness.foundation.paths import default_config_dir
from iris_harness.runtime.plugin_host import (
    PluginRecord,
    PluginRegistry,
    PluginStatus,
    discover_plugin,
    load_profile,
)
from iris_harness.runtime.plugin_host.manifest import PluginManifest, PluginWebUI, WebScreen
from iris_harness.runtime.plugin_host.nav import CoreNav, web_nav

CONFIG = default_config_dir()

CORE_NAV = [
    "/chat",
    "/overview",
    "/agents",
    "/routines",
    "/heartbeats",
    "/playground",
    "/actions",
    "/digest",
    "/activity",
    "/sessions",
    "/calltrace",
    "/memory",
    "/learning",
    "/twin",
    "/health",
    "/system-check",
    "/onboarding",
    "/governance",
    "/devices",
    "/settings",
]


def _nav_routes(nav: dict[str, Any]) -> list[str]:
    return [item["route"] for group in nav["groups"] for item in group["items"]]


def _available(nav: dict[str, Any]) -> set[str]:
    return set(_nav_routes(nav)) | {s["route"] for s in nav["off_nav"]}


def _plugin(
    name: str, *screens: dict[str, Any], status: PluginStatus = PluginStatus.LOADED
) -> PluginRecord:
    manifest = PluginManifest.model_validate({"name": name, "webui": {"screens": list(screens)}})
    return PluginRecord(name=name, source=f"home:{name}", status=status, manifest=manifest)


def _registry(*records: PluginRecord) -> PluginRegistry:
    registry = PluginRegistry()
    for record in records:
        registry.add_plugin(record)
    return registry


NOTES = {"id": "notes", "label": "Notes", "route": "/notes"}


# ── the core's screens ─────────────────────────────────────────────────────


def test_default_profile_is_the_core_screens() -> None:
    """The shipped default profile mounts only reference plugins, which own no screens."""
    registry = PluginRegistry()
    for ref in load_profile(CONFIG, "default", env={}).plugins:
        source = discover_plugin(ref.name)
        assert source is not None, ref.name
        registry.add_plugin(
            PluginRecord(
                name=ref.name,
                source=source.label,
                status=PluginStatus.LOADED,
                manifest=source.manifest,
            )
        )
    nav = web_nav(registry, config_dir=CONFIG)
    assert _nav_routes(nav) == CORE_NAV
    assert {s["route"] for s in nav["off_nav"]} == {"/more", "/knowledge", "/pair"}
    assert all(item["plugin"] is None for g in nav["groups"] for item in g["items"])
    assert nav["problems"] == []


def test_no_registry_is_the_core_nav() -> None:
    nav = web_nav(None, config_dir=CONFIG, installed=[])
    assert _nav_routes(nav) == CORE_NAV
    assert nav["unavailable"] == []


def test_mobile_tabs_are_the_cores_four_pins_in_slot_order() -> None:
    nav = web_nav(None, config_dir=CONFIG, installed=[])
    assert nav["mobile_tabs"] == [
        {"route": "/chat", "label": "Chat"},
        {"route": "/health", "label": "Pulse"},
        {"route": "/actions", "label": "Activity"},
        {"route": "/overview", "label": "Control"},
    ]


def test_an_override_config_dir_without_the_file_reads_the_shipped_one(tmp_path: Path) -> None:
    nav = web_nav(None, config_dir=tmp_path, installed=[])
    assert _nav_routes(nav) == CORE_NAV


# ── plugin screens follow the plugin ───────────────────────────────────────


def test_a_mounted_plugins_screen_is_in_the_nav() -> None:
    nav = web_nav(_registry(_plugin("notes", NOTES)), config_dir=CONFIG, installed=[])
    apps = next(g for g in nav["groups"] if g["id"] == "apps")
    (item,) = apps["items"]
    assert (item["route"], item["plugin"], item["status"]) == ("/notes", "notes", "loaded")
    # Defaults: the header is the label, the icon generic, and no phone pin.
    assert (item["title"], item["icon"]) == ("Notes", "puzzle")
    assert "mobile_tab" not in item


@pytest.mark.parametrize(
    "status", [PluginStatus.FAILED, PluginStatus.DISABLED, PluginStatus.UNSUPPORTED]
)
def test_a_plugin_that_did_not_mount_takes_its_screens_with_it(status: PluginStatus) -> None:
    record = _plugin("notes", NOTES, status=status)
    record.load_error = "setup raised" if status is PluginStatus.FAILED else None
    assert record.manifest is not None
    installed = [("notes", record.manifest)]
    nav = web_nav(_registry(record), config_dir=CONFIG, installed=installed)
    assert "/notes" not in _available(nav)
    # No plugin screen left -> the Apps group is not drawn at all.
    assert "apps" not in {g["id"] for g in nav["groups"]}
    (entry,) = nav["unavailable"]
    assert (entry["route"], entry["plugin"]) == ("/notes", "notes")
    expected = {
        PluginStatus.FAILED: "failed: setup raised",
        PluginStatus.DISABLED: "disabled in the profile",
        PluginStatus.UNSUPPORTED: "unsupported",
    }[status]
    assert entry["reason"] == expected


def test_an_installed_plugin_outside_the_profile_is_unavailable_not_in_the_nav() -> None:
    manifest = PluginManifest.model_validate({"name": "notes", "webui": {"screens": [NOTES]}})
    nav = web_nav(PluginRegistry(), config_dir=CONFIG, installed=[("notes", manifest)])
    assert "/notes" not in _available(nav)
    assert nav["unavailable"] == [
        {
            "id": "notes",
            "label": "Notes",
            "route": "/notes",
            "plugin": "notes",
            "reason": "not in the profile",
        }
    ]


def test_a_degraded_plugin_keeps_its_screens_and_says_so() -> None:
    record = _plugin("notes", NOTES, status=PluginStatus.DEGRADED)
    nav = web_nav(_registry(record), config_dir=CONFIG, installed=[])
    item = next(i for g in nav["groups"] for i in g["items"] if i["route"] == "/notes")
    assert item["status"] == "degraded"


def test_a_plugin_screen_slots_between_core_entries_by_order() -> None:
    inbox = {"id": "inbox", "label": "Inbox", "route": "/inbox", "group": "operations"}
    nav = web_nav(
        _registry(_plugin("mail", {**inbox, "order": 30})), config_dir=CONFIG, installed=[]
    )
    ops = next(g for g in nav["groups"] if g["id"] == "operations")
    assert [i["route"] for i in ops["items"]] == [
        "/actions",
        "/digest",
        "/inbox",
        "/activity",
        "/sessions",
        "/calltrace",
    ]


def test_an_off_nav_screen_is_owned_but_not_listed() -> None:
    sheet = {"id": "sheet", "label": "Sheet", "route": "/sheet", "nav": False}
    nav = web_nav(_registry(_plugin("notes", sheet)), config_dir=CONFIG, installed=[])
    assert "/sheet" not in _nav_routes(nav)
    assert "/sheet" in {s["route"] for s in nav["off_nav"]}


def test_a_plugin_cannot_take_a_core_route_or_another_plugins() -> None:
    registry = _registry(
        _plugin("rogue", {"id": "rogue", "label": "Rogue", "route": "/chat"}),
        _plugin("first", NOTES),
        _plugin("second", {"id": "notes2", "label": "N2", "route": "/notes"}),
    )
    nav = web_nav(registry, config_dir=CONFIG, installed=[])
    chat = nav["groups"][0]["items"]
    assert [(i["route"], i["plugin"]) for i in chat] == [("/chat", None)]
    notes = [i for g in nav["groups"] for i in g["items"] if i["route"] == "/notes"]
    assert [i["plugin"] for i in notes] == ["first"]
    assert len(nav["problems"]) == 2
    assert any("rogue" in p and "core" in p for p in nav["problems"])


def test_an_unknown_group_lands_in_the_default_group() -> None:
    registry = _registry(_plugin("notes", {**NOTES, "group": "nope"}))
    nav = web_nav(registry, config_dir=CONFIG, installed=[])
    apps = next(g for g in nav["groups"] if g["id"] == "apps")
    assert [i["route"] for i in apps["items"]] == ["/notes"]
    assert nav["problems"] and "nope" in nav["problems"][0]


# ── what a manifest may declare ────────────────────────────────────────────


def test_screen_defaults() -> None:
    screen = WebScreen(id="notes", label="Notes", route="/notes")
    assert (screen.title, screen.icon, screen.group, screen.order, screen.nav) == (
        None,
        "puzzle",
        "apps",
        100,
        True,
    )


@pytest.mark.parametrize(
    "screen",
    [
        {"id": "x", "label": "X", "route": "/a/b"},  # one segment only
        {"id": "x", "label": "X", "route": "no-slash"},
        {"id": "X", "label": "X", "route": "/x"},  # id shape
        {"id": "x", "label": "", "route": "/x"},
        {"id": "x", "label": "X", "route": "/x", "icon": "ChartLine"},
        {"id": "x", "label": "X", "route": "/x", "order": -1},
        # The phone's bar is the core's: a plugin cannot pin itself to it.
        {"id": "x", "label": "X", "route": "/x", "mobile_tab": {"label": "X", "order": 1}},
        {"id": "x", "label": "X", "route": "/x", "component": "Portfolio"},
    ],
)
def test_manifest_rejects_a_bad_screen(screen: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        PluginManifest.model_validate({"name": "p", "webui": {"screens": [screen]}})


def test_manifest_rejects_two_screens_on_one_route_or_id() -> None:
    a = {"id": "a", "label": "A", "route": "/x"}
    with pytest.raises(ValueError, match="route"):
        PluginWebUI.model_validate({"screens": [a, {"id": "b", "label": "B", "route": "/x"}]})
    with pytest.raises(ValueError, match="id"):
        PluginWebUI.model_validate({"screens": [a, {"id": "a", "label": "B", "route": "/y"}]})
    with pytest.raises(ValueError):
        PluginManifest.model_validate({"name": "p", "webui": {"pages": []}})


def test_a_manifest_without_webui_owns_no_screens() -> None:
    assert PluginManifest(name="p").webui.screens == ()


# ── the core's own YAML ────────────────────────────────────────────────────


def _core_raw() -> dict[str, Any]:
    raw = yaml.safe_load((CONFIG / "webui" / "nav.yaml").read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_core_nav_rejects_a_screen_in_a_missing_group() -> None:
    raw = _core_raw()
    raw["screens"][0]["group"] = "nowhere"
    with pytest.raises(ValueError, match="unknown group"):
        CoreNav.model_validate(raw)


def test_core_nav_rejects_two_pins_in_one_slot() -> None:
    raw = _core_raw()
    pinned = [s for s in raw["screens"] if s.get("mobile_tab")]
    pinned[1]["mobile_tab"]["order"] = pinned[0]["mobile_tab"]["order"]
    with pytest.raises(ValueError, match="mobile_tab"):
        CoreNav.model_validate(raw)


def test_core_nav_rejects_a_default_group_it_does_not_have() -> None:
    raw = _core_raw()
    raw["default_group"] = "nowhere"
    with pytest.raises(ValueError, match="default_group"):
        CoreNav.model_validate(raw)
