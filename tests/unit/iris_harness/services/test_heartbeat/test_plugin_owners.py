"""Every ``plugin:`` in the shipped config/heartbeats.yaml names a real owner (issue #110).

``plugin:`` is a free string, and a typo in it reads as "plugin not installed": quiet
forever. The shipped file therefore declares every owner it may name, in two lists --
``plugins_in_tree`` (checked here against the manifests this tree ships) and
``plugins_external`` (owners shipped elsewhere, which this tree cannot check).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from iris_harness.services.heartbeat.config import (
    HeartbeatConfigError,
    load_heartbeats,
    load_plugin_owners,
)

ROOT = Path(__file__).resolve().parents[5]
HEARTBEATS = ROOT / "config" / "heartbeats.yaml"
_MANIFEST_ROOTS = (
    ROOT / "src" / "iris_harness" / "plugins_builtin",
    ROOT / "src" / "iris_personal" / "plugins",
)


def _in_tree_plugins() -> set[str]:
    names: set[str] = set()
    for base in _MANIFEST_ROOTS:
        for manifest in base.glob("*/manifest.yaml"):
            names.add(yaml.safe_load(manifest.read_text(encoding="utf-8"))["name"])
    return names


def _declared(key: str) -> set[str]:
    raw = yaml.safe_load(HEARTBEATS.read_text(encoding="utf-8"))
    return set(raw[key])


def test_the_manifest_scan_is_not_vacuous() -> None:
    assert {"system", "email_workflows"} <= _in_tree_plugins()


def test_every_plugin_value_is_a_declared_owner() -> None:
    owners = {d.plugin for d in load_heartbeats(HEARTBEATS) if d.plugin}
    declared = load_plugin_owners(HEARTBEATS)
    assert owners  # the shipped file does name owners
    assert owners - declared == set(), "plugin: value in neither declared list (typo?)"


def test_in_tree_list_is_exactly_real_manifests() -> None:
    """A typo cannot hide in the list itself: its entries must be shipped manifests."""
    assert _declared("plugins_in_tree") - _in_tree_plugins() == set()


def test_external_list_names_nothing_that_ships_in_tree() -> None:
    """A plugin that moves into this tree must move lists, so it gets the manifest check."""
    assert _declared("plugins_external") & _in_tree_plugins() == set()


def test_every_declared_owner_is_used() -> None:
    owners = {d.plugin for d in load_heartbeats(HEARTBEATS) if d.plugin}
    assert load_plugin_owners(HEARTBEATS) - owners == set(), "declared but no heartbeat names it"


def test_a_typo_would_be_caught(tmp_path: Path) -> None:
    path = tmp_path / "heartbeats.yaml"
    raw = yaml.safe_load(HEARTBEATS.read_text(encoding="utf-8"))
    raw["heartbeats"][0]["plugin"] = "emial_workflows"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    owners = {d.plugin for d in load_heartbeats(path) if d.plugin}
    assert owners - load_plugin_owners(path) == {"emial_workflows"}


def test_load_plugin_owners_unions_both_lists_and_tolerates_absence(tmp_path: Path) -> None:
    path = tmp_path / "h.yaml"
    path.write_text("plugins_in_tree: [a]\nplugins_external: [b, c]\n", encoding="utf-8")
    assert load_plugin_owners(path) == {"a", "b", "c"}
    path.write_text("heartbeats: []\n", encoding="utf-8")
    assert load_plugin_owners(path) == frozenset()
    assert load_plugin_owners(tmp_path / "missing.yaml") == frozenset()


def test_load_plugin_owners_rejects_a_malformed_list(tmp_path: Path) -> None:
    path = tmp_path / "h.yaml"
    path.write_text("plugins_external: calendar\n", encoding="utf-8")
    with pytest.raises(HeartbeatConfigError, match="plugins_external"):
        load_plugin_owners(path)
