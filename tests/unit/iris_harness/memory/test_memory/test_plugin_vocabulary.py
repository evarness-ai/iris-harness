"""Plugins bring their own memory vocabulary (memris PR 10; ADR-0115).

The core ontology knows people, places and organisations; a plugin brings its own terms
(the finance plugin: ``fin:`` banks, cards, insurers; here, the test vocabulary's ``tv:``
and throwaway ``$IRIS_HOME`` plugins). Every INSTALLED plugin's fragment is loaded,
mounted or not — the ontology is grammar, and a bank fact stored yesterday must still
read when the plugin that brought its term is switched off today.

What the real finance plugin's fragment does is asserted beside that plugin:
tests/unit/iris_personal/plugins/test_finance_workflows/test_finance_vocabulary.py.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_harness.foundation.plugin_dirs import installed_plugin_dirs
from iris_harness.memory.ontology import (
    memory_config_dir,
    memory_ontology,
    reset_cache,
    vocabulary_fragments,
)
from iris_harness.memory.store import MemoryStore, UserFact
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from memris.graph import MemoryGraph
from memris.model import OWNER_ID
from memris.store import SQLiteGraphStore

TEST_VOCABULARY = Path(__file__).resolve().parents[5] / "tests" / "fixtures" / "test_vocabulary"


@pytest.fixture(autouse=True)
def _fresh() -> None:
    reset_cache()


def _home_plugin(home: Path, name: str, prefix: str, attribute: str) -> None:
    """A throwaway ``$IRIS_HOME`` plugin that ships one attribute under ``prefix``."""
    plugin = home / "plugins" / name
    (plugin / "vocab").mkdir(parents=True)
    (plugin / "manifest.yaml").write_text(f"name: {name}\nontology: vocab\n", encoding="utf-8")
    (plugin / "vocab" / "ontology.yaml").write_text(
        f'prefixes: {{ {prefix}: "urn:test:{name}#" }}\n'
        f"attributes:\n  {prefix}:{attribute}: {{ domain: Animal, datatype: string }}\n",
        encoding="utf-8",
    )


def test_the_core_config_names_no_plugin_s_vocabulary() -> None:
    """The boundary, kept: core YAML may not use a prefix a plugin fragment declares --
    every installed plugin's, and the test vocabulary's, so the check never runs empty."""
    import yaml

    plugin_prefixes = set()
    for fragment in [*vocabulary_fragments(), TEST_VOCABULARY / "ontology"]:
        raw = yaml.safe_load((fragment / "ontology.yaml").read_text(encoding="utf-8")) or {}
        plugin_prefixes |= set((raw.get("prefixes") or {}).keys())
    assert "tv" in plugin_prefixes  # the test vocabulary at least
    core = memory_config_dir()
    for name in ("ontology.yaml", "shapes.yaml", "mappings.yaml"):
        text = (core / name).read_text(encoding="utf-8")
        for prefix in plugin_prefixes:
            code = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
            used = [ln for ln in code if re.search(rf"(?<![\w]){re.escape(prefix)}:\w", ln)]
            assert not used, f"{name} uses the plugin prefix {prefix}: {used[:2]}"


@pytest.mark.usefixtures("test_vocabulary")
def test_a_stored_plugin_fact_reads_with_the_plugin_installed_and_not_without(
    tmp_path: Path,
) -> None:
    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.ensure_schema()
    now = datetime.now(UTC)
    store.upsert_user_fact(UserFact("bank", "Example Bank", 0.9, "test", now, now, 1, True))

    installed = MemoryGraph(memory_ontology(), SQLiteGraphStore(store.db_path))
    missing = MemoryGraph(memory_ontology(fragments=False), SQLiteGraphStore(store.db_path))

    assert [s.predicate for s in installed.current(OWNER_ID)] == ["tv:banks_with"]
    assert installed.check_usage() == []
    assert any("tv:banks_with" in str(i) for i in missing.check_usage())


def test_any_installed_plugin_can_bring_a_fragment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "iris-home"
    _home_plugin(home, "pets", "pet", "favourite_treat")
    _home_plugin(home, "garden", "gdn", "favourite_plant")
    silent = home / "plugins" / "quiet"
    silent.mkdir(parents=True)
    (silent / "manifest.yaml").write_text("name: quiet\n", encoding="utf-8")
    monkeypatch.setenv("IRIS_HOME", str(home))

    names = [name for name, _dir in installed_plugin_dirs()]
    onto = memory_ontology()

    assert {"pets", "garden", "quiet"} <= set(names)
    assert "pet:favourite_treat" in onto.attributes
    assert "gdn:favourite_plant" in onto.attributes  # the others still load
    assert all(p.parent.name != "quiet" for p in vocabulary_fragments())


def test_a_fragment_that_is_declared_but_missing_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = tmp_path / "home" / "plugins" / "ghost"
    plugin.mkdir(parents=True)
    (plugin / "manifest.yaml").write_text("name: ghost\nontology: nowhere\n", encoding="utf-8")
    _home_plugin(tmp_path / "home", "pets", "pet", "favourite_treat")
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "home"))

    assert all(p.parent.name != "ghost" for p in vocabulary_fragments())
    assert "pet:favourite_treat" in memory_ontology().attributes  # the rest still load


def test_the_manifest_declares_it() -> None:
    assert PluginManifest(name="x", ontology="ontology").ontology == "ontology"
    for bad in ("/etc", "../up"):
        with pytest.raises(ValueError):
            PluginManifest(name="x", ontology=bad)


def test_a_fragment_outside_its_plugin_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "ontology.yaml").write_text('prefixes: { x: "urn:x#" }\n', encoding="utf-8")
    plugin = tmp_path / "home" / "plugins" / "sneaky"
    plugin.mkdir(parents=True)
    (plugin / "manifest.yaml").write_text(
        "name: sneaky\nontology: a/../../../../elsewhere\n", encoding="utf-8"
    )
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "home"))

    assert outside.resolve() not in vocabulary_fragments()


def test_plugins_are_found_where_the_loader_looks_first_place_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.foundation.plugin_dirs import BUILTIN_PACKAGE, _package_dir

    builtin_root = _package_dir(BUILTIN_PACKAGE)
    assert builtin_root is not None
    builtin = next(
        c.name for c in sorted(builtin_root.iterdir()) if (c / "manifest.yaml").is_file()
    )
    shadow = tmp_path / "home" / "plugins" / builtin
    shadow.mkdir(parents=True)
    (shadow / "manifest.yaml").write_text(f"name: {builtin}\n", encoding="utf-8")
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "home"))

    found = dict(installed_plugin_dirs())

    assert found[builtin] == builtin_root / builtin  # builtin, not the home copy
