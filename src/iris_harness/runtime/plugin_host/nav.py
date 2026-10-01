"""The web console's navigation: the core's screens plus every mounted plugin's (OSS plan R17).

The console renders what this returns and nothing else, so a screen is in the nav
exactly when its owner is: the core's screens always (``config/webui/nav.yaml``), a
plugin's (its manifest's ``webui.screens``) while the plugin is mounted. A plugin that
is installed but not mounted -- failed, disabled, or not in the profile -- has its
screens listed under ``unavailable`` with the reason, so a direct link can say so
instead of drawing a screen with no API behind it.

Read-only: ``GET /api/v1/webui/nav`` is a thin wrapper over :func:`web_nav`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from iris_harness.foundation.paths import config_dir as resolve_config_dir
from iris_harness.foundation.paths import default_config_dir
from iris_harness.foundation.plugin_dirs import MANIFEST_FILENAME, installed_plugin_dirs

from .manifest import PluginManifest, WebScreen, load_manifest
from .registry import MOUNTED, PluginRegistry, PluginStatus

logger = logging.getLogger(__name__)

NAV_CONFIG = ("webui", "nav.yaml")


class NavGroup(BaseModel):
    """A heading in the sidebar. ``label: null`` is the pinned top group (Chat)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., pattern=r"^[a-z][a-z0-9_-]*$")
    label: str | None = Field(default=None, min_length=1, max_length=40)
    order: int = Field(default=100, ge=0, le=10_000)


class MobileTab(BaseModel):
    """A pin on the phone's bottom bar: its own word for the screen, and its slot."""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., min_length=1, max_length=12)
    order: int = Field(..., ge=1, le=9)


class CoreScreen(WebScreen):
    """A screen the core owns. Only the core pins screens to the phone's bar: its slots
    are fixed, and a plugin that came and went would move the owner's thumbs."""

    mobile_tab: MobileTab | None = None


class CoreNav(BaseModel):
    """``config/webui/nav.yaml``: the groups every screen lands in, and the core's screens."""

    model_config = ConfigDict(extra="forbid")

    # Where a plugin screen goes when it names a group the core does not have.
    default_group: str
    groups: tuple[NavGroup, ...]
    screens: tuple[CoreScreen, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _consistent(self) -> CoreNav:
        group_ids = [g.id for g in self.groups]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("nav: two groups share an id")
        if self.default_group not in group_ids:
            raise ValueError(f"nav: default_group {self.default_group!r} is not a group")
        for attr in ("id", "route"):
            values = [getattr(s, attr) for s in self.screens]
            if len(set(values)) != len(values):
                raise ValueError(f"nav: two core screens share a {attr}")
        for screen in self.screens:
            if screen.group not in group_ids:
                raise ValueError(f"nav: screen {screen.id!r} names unknown group {screen.group!r}")
            if screen.mobile_tab is not None and not screen.nav:
                raise ValueError(f"nav: screen {screen.id!r} is pinned but has no nav entry")
        slots = [s.mobile_tab.order for s in self.screens if s.mobile_tab is not None]
        if len(set(slots)) != len(slots):
            raise ValueError("nav: two screens pin the same mobile_tab order")
        return self


def load_core_nav(config_dir: Path | None = None) -> CoreNav:
    """The core's nav config. An override config dir without the file reads the shipped one."""
    base = config_dir if config_dir is not None else resolve_config_dir()
    path = base.joinpath(*NAV_CONFIG)
    if not path.is_file():
        path = default_config_dir().joinpath(*NAV_CONFIG)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return CoreNav.model_validate(raw)
    except Exception as exc:  # pydantic ValidationError -- keep the path in the message
        raise ValueError(f"web nav config invalid: {path}: {exc}") from exc


def installed_manifests() -> list[tuple[str, PluginManifest]]:
    """``(name, manifest)`` of every installed plugin whose manifest reads. No imports."""
    found: list[tuple[str, PluginManifest]] = []
    for name, directory in installed_plugin_dirs():
        try:
            found.append((name, load_manifest(directory / MANIFEST_FILENAME)))
        except ValueError as exc:
            logger.warning("web nav: skipping plugin %r (%s)", name, exc)
    return found


def _screen_view(
    screen: WebScreen, *, plugin: str | None, status: str | None, group: str
) -> dict[str, Any]:
    view: dict[str, Any] = {
        "id": screen.id,
        "label": screen.label,
        "route": screen.route,
        "title": screen.title or screen.label,
        "subtitle": screen.subtitle,
        "icon": screen.icon,
        "group": group,
        "order": screen.order,
        "nav": screen.nav,
        # None = the core's own screen.
        "plugin": plugin,
        "status": status,
    }
    if isinstance(screen, CoreScreen) and screen.mobile_tab is not None:
        view["mobile_tab"] = screen.mobile_tab.model_dump()
    return view


def _unmounted_reason(status: PluginStatus | None, error: str | None) -> str:
    if status is None:
        return "not in the profile"
    if status is PluginStatus.DISABLED:
        return "disabled in the profile"
    if error:
        return f"{status.value}: {error}"
    return status.value


def web_nav(
    registry: PluginRegistry | None,
    *,
    config_dir: Path | None = None,
    installed: Iterable[tuple[str, PluginManifest]] | None = None,
) -> dict[str, Any]:
    """The console's navigation for the running profile.

    ``groups``: sidebar groups in order, each with its nav entries in order (empty
    groups dropped). ``off_nav``: available screens with no menu entry. ``mobile_tabs``:
    the phone bar's pins in slot order. ``unavailable``: screens of installed plugins
    that are not mounted, with why. ``problems``: declarations the nav could not honour
    (a route another owner already has, a group the core lacks).
    """
    core = load_core_nav(config_dir)
    group_ids = {g.id for g in core.groups}
    problems: list[str] = []
    screens: list[dict[str, Any]] = [
        _screen_view(s, plugin=None, status=None, group=s.group) for s in core.screens
    ]
    taken_ids = {s.id: "core" for s in core.screens}
    taken_routes = {s.route: "core" for s in core.screens}

    records = registry.plugins() if registry is not None else []
    mounted_names: set[str] = set()
    for rec in records:
        if rec.status not in MOUNTED or rec.manifest is None:
            continue
        mounted_names.add(rec.name)
        for screen in rec.manifest.webui.screens:
            clash = taken_routes.get(screen.route) or taken_ids.get(screen.id)
            if clash is not None:
                problems.append(
                    f"{rec.name}: screen {screen.id!r} ({screen.route}) is already {clash}'s"
                )
                continue
            group = screen.group
            if group not in group_ids:
                problems.append(
                    f"{rec.name}: screen {screen.id!r} names unknown group {group!r}; "
                    f"shown under {core.default_group!r}"
                )
                group = core.default_group
            taken_routes[screen.route] = rec.name
            taken_ids[screen.id] = rec.name
            screens.append(
                _screen_view(screen, plugin=rec.name, status=rec.status.value, group=group)
            )

    by_group: dict[str, list[dict[str, Any]]] = {g.id: [] for g in core.groups}
    for view in screens:
        if view["nav"]:
            by_group[view["group"]].append(view)
    groups = [
        {
            "id": g.id,
            "label": g.label,
            # Stable sort: equal orders keep core-first, then the profile's mount order.
            "items": sorted(by_group[g.id], key=lambda v: v["order"]),
        }
        for g in sorted(core.groups, key=lambda g: g.order)
        if by_group[g.id]
    ]
    mobile_tabs = sorted(
        (v for v in screens if "mobile_tab" in v), key=lambda v: v["mobile_tab"]["order"]
    )

    records_by_name = {rec.name: rec for rec in records}
    unavailable: list[dict[str, Any]] = []
    for name, manifest in installed if installed is not None else installed_manifests():
        if name in mounted_names:
            continue
        record = records_by_name.get(name)
        reason = _unmounted_reason(
            record.status if record is not None else None,
            (record.load_error or record.last_error) if record is not None else None,
        )
        for screen in manifest.webui.screens:
            if screen.route in taken_routes:
                continue
            unavailable.append(
                {
                    "id": screen.id,
                    "label": screen.label,
                    "route": screen.route,
                    "plugin": name,
                    "reason": reason,
                }
            )

    return {
        "groups": groups,
        "off_nav": [v for v in screens if not v["nav"]],
        "mobile_tabs": [
            {"route": v["route"], "label": v["mobile_tab"]["label"]} for v in mobile_tabs
        ],
        "unavailable": unavailable,
        "problems": problems,
    }


__all__ = [
    "CoreNav",
    "CoreScreen",
    "MobileTab",
    "NAV_CONFIG",
    "NavGroup",
    "installed_manifests",
    "load_core_nav",
    "web_nav",
]
