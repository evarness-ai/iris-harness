"""config/memory's ontology is the one vocabulary memory has (memris plan PR 0, PR 3).

Since PR 3 retired fact_keys.yaml, the fact-key allowlist is derived from
mappings.yaml: a mapping's first key is canonical, its others are aliases. These tests
hold that there is no second list to drift from, and that the derivation is exact.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from memris.ontology import check_directory, load_or_raise

REPO = Path(__file__).resolve().parents[5]
MEMORY_CONFIG = REPO / "config" / "memory"


def _yaml(name: str) -> dict:
    return yaml.safe_load((MEMORY_CONFIG / name).read_text(encoding="utf-8"))


def _fact_mappings() -> dict[str, str]:
    """fact key → predicate, from mappings.yaml."""
    onto = load_or_raise(MEMORY_CONFIG)
    return {key: m.predicate for m in onto.mappings if m.source_type == "fact" for key in m.keys}


def test_the_memory_ontology_compiles_without_errors_or_warnings() -> None:
    result, issues = check_directory(MEMORY_CONFIG)
    assert result is not None
    assert [str(i) for i in issues] == []


def test_there_is_no_second_allowlist() -> None:
    assert not (MEMORY_CONFIG / "fact_keys.yaml").exists()


def test_the_allowlist_is_exactly_the_fact_mappings() -> None:
    from iris_harness.memory import fact_keys

    fact_keys.reset_cache()
    # The core's mappings plus every installed plugin's fragment (memris PR 10: the
    # finance keys live in the finance plugin now).
    from iris_harness.memory.ontology import vocabulary_fragments

    onto = load_or_raise(MEMORY_CONFIG, vocabulary_fragments())
    rules = [m for m in onto.mappings if m.source_type == "fact"]
    assert fact_keys.allowed_keys() == {m.keys[0] for m in rules}
    for rule in rules:
        for key in rule.keys:  # every key a mapping lists folds to the mapping's first key
            assert fact_keys.canonical_key(key) == rule.keys[0], key
    assert fact_keys.canonical_key("topic") is None  # and nothing else is a fact key


def test_residence_and_origin_never_share_a_property() -> None:
    """ "From India, living in the UK" is two facts; neither may overwrite the other."""
    mapped = _fact_mappings()
    residence = {mapped["city"], mapped["country"], mapped["location"]}
    origin = {mapped["hometown"], mapped["nationality"]}
    assert residence.isdisjoint(origin)


def test_every_map_group_names_a_declared_class() -> None:
    """entity_aliases.yaml groups classes for the Map; a typo would silently drop nodes."""
    onto = load_or_raise(MEMORY_CONFIG)
    for class_name in _yaml("entity_aliases.yaml")["groups"]:
        assert onto.qualify(class_name) in onto.classes, class_name


def test_every_class_a_relation_fact_points_at_has_a_map_group() -> None:
    """Otherwise the fact's entity has no node and its edge vanishes from the Map."""
    onto = load_or_raise(MEMORY_CONFIG)
    grouped = {onto.qualify(c) for c in _yaml("entity_aliases.yaml")["groups"]}
    for rule in onto.mappings:
        if rule.source_type == "fact" and rule.object_class:
            ancestors = set(onto.ancestors(rule.object_class))
            assert ancestors & grouped, f"{rule.id}: {rule.object_class} has no Map group"
