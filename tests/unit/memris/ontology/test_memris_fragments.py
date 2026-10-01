"""Vocabulary fragments: a base ontology plus directories that only add to it.

ADR-0115's vocabulary is ``core YAML ∪ plugin YAML ∪ learned rows``; this is the middle
term. The names here are test data; memris knows none of them.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from memris.ontology import OntologyError, check_directory, load_or_raise

BASE = """
ontology: { id: "urn:test", version: "1.0.0", default_prefix: t }
prefixes: { t: "urn:test#" }
classes:
  Thing:  { abstract: true }
  Person: { subclass_of: Thing }
  Org:    { subclass_of: Thing }
relations:
  works_at: { domain: Person, range: Org }
"""
BASE_SHAPES = """
Person:
  properties:
    works_at: { max_count: 1 }
"""
BASE_MAPPINGS = """
mappings:
  - id: fact_employer
    when: { source_type: fact, key: employer }
    emit: { subject: $owner, predicate: works_at, object: { from: $value, class: Org } }
"""
MONEY = """
fragment: { id: "urn:test:money", version: "0.1.0" }
prefixes: { m: "urn:test:money#" }
classes:
  m:Bank: { subclass_of: Org }
relations:
  m:banks_with: { domain: Person, range: m:Bank }
"""
MONEY_SHAPES = """
Person:
  properties:
    m:banks_with: { max_count: 1 }
"""
MONEY_MAPPINGS = """
mappings:
  - id: fact_bank
    when: { source_type: fact, key: bank }
    emit: { subject: $owner, predicate: m:banks_with, object: { from: $value, class: m:Bank } }
"""


def _dir(root: Path, **files: str) -> Path:
    root.mkdir(parents=True)
    for name, text in files.items():
        (root / f"{name}.yaml").write_text(dedent(text), encoding="utf-8")
    return root


@pytest.fixture
def base(tmp_path: Path) -> Path:
    return _dir(tmp_path / "core", ontology=BASE, shapes=BASE_SHAPES, mappings=BASE_MAPPINGS)


@pytest.fixture
def money(tmp_path: Path) -> Path:
    return _dir(tmp_path / "money", ontology=MONEY, shapes=MONEY_SHAPES, mappings=MONEY_MAPPINGS)


def test_a_fragment_adds_terms_shapes_and_mappings(base: Path, money: Path) -> None:
    onto = load_or_raise(base, [money])

    assert "m:Bank" in onto.classes and "m:banks_with" in onto.relations
    assert onto.prefixes["m"] == "urn:test:money#"
    assert onto.shapes["t:Person"]["m:banks_with"].max_count == 1
    assert onto.shapes["t:Person"]["t:works_at"].max_count == 1  # the base's stays
    assert {r.id for r in onto.mappings} == {"fact_employer", "fact_bank"}
    assert onto.id == "urn:test"  # the base owns identity


def test_without_the_fragment_its_terms_do_not_exist(base: Path, money: Path) -> None:
    assert "m:banks_with" not in load_or_raise(base).relations


def test_a_fragment_may_be_only_an_ontology(base: Path, tmp_path: Path) -> None:
    bare = _dir(tmp_path / "bare", ontology="classes: { Pet: { subclass_of: Thing } }")
    assert "t:Pet" in load_or_raise(base, [bare]).classes


@pytest.mark.parametrize(
    ("files", "where"),
    [
        ({"ontology": "relations: { works_at: { domain: Person, range: Org } }"}, "ontology.yaml"),
        ({"ontology": "", "shapes": BASE_SHAPES}, "shapes.yaml"),
        ({"ontology": "", "mappings": BASE_MAPPINGS}, "mappings.yaml"),
    ],
)
def test_a_fragment_only_adds(
    base: Path, tmp_path: Path, files: dict[str, str], where: str
) -> None:
    clash = _dir(tmp_path / "clash", **files)

    _result, issues = check_directory(base, [clash])

    conflicts = [i for i in issues if i.code == "fragment-conflict"]
    assert conflicts and conflicts[0].where == f"clash/{where}"
    with pytest.raises(OntologyError):
        load_or_raise(base, [clash])


def test_two_fragments_cannot_claim_one_name(base: Path, money: Path, tmp_path: Path) -> None:
    again = _dir(tmp_path / "again", ontology="classes: { m:Bank: { subclass_of: Org } }")
    with pytest.raises(OntologyError, match="already defined"):
        load_or_raise(base, [money, again])


def test_a_fragment_does_not_redefine_the_header(base: Path, tmp_path: Path) -> None:
    rogue = _dir(tmp_path / "rogue", ontology=BASE)
    _result, issues = check_directory(base, [rogue])
    assert any(i.code == "fragment-header" for i in issues)


def test_an_unknown_section_is_reported(base: Path, tmp_path: Path) -> None:
    odd = _dir(tmp_path / "odd", ontology="widgets: {}")
    _result, issues = check_directory(base, [odd])
    assert any("widgets" in i.message for i in issues)


def test_a_fragment_term_is_checked_like_any_other(base: Path, tmp_path: Path) -> None:
    broken = _dir(tmp_path / "broken", ontology="relations: { m:x: { domain: Nope, range: Org } }")
    with pytest.raises(OntologyError):
        load_or_raise(base, [broken])
