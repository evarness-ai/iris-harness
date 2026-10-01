"""The ontology compiler: each rule is shown failing on a small, deliberately broken file.

The vocabulary here (Person, Org, works_at …) is test data. memris itself knows none of
it; that is what test_memris_no_vocabulary.py holds true.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from memris.ontology import (
    OntologyError,
    check_directory,
    check_usage,
    load_or_raise,
)
from memris.ontology.__main__ import main

BASE = """
ontology: { id: "urn:test", version: "0.1.0", default_prefix: t }
prefixes: { t: "urn:test#", ext: "urn:ext#" }
classes:
  Thing:  { abstract: true }
  Person: { subclass_of: Thing }
  Org:    { subclass_of: Thing }
  Team:   { subclass_of: Org }
  Place:  { subclass_of: Thing }
  City:   { subclass_of: Place }
relations:
  works_at: { label: "works at", domain: Person, range: Org, inverse: employs, inverse_label: "employs" }
  lives_in: { domain: Person, range: Place }
  resides_in_city: { domain: Person, range: City, subproperty_of: lives_in }
  knows:    { domain: Person, range: Person, symmetric: true }
attributes:
  name:  { domain: Thing }
  email: { domain: Person }
"""


def _write(
    tmp_path: Path, ontology: str = BASE, shapes: str | None = None, mappings: str | None = None
) -> Path:
    (tmp_path / "ontology.yaml").write_text(dedent(ontology), encoding="utf-8")
    if shapes is not None:
        (tmp_path / "shapes.yaml").write_text(dedent(shapes), encoding="utf-8")
    if mappings is not None:
        (tmp_path / "mappings.yaml").write_text(dedent(mappings), encoding="utf-8")
    return tmp_path


def _codes(tmp_path: Path) -> list[str]:
    _, issues = check_directory(tmp_path)
    return [i.code for i in issues if i.severity == "error"]


def _with(**sections: str) -> str:
    """BASE with extra lines appended under the named sections."""
    text = BASE
    for section, extra in sections.items():
        text = text.replace(f"\n{section}:\n", f"\n{section}:\n{extra}", 1)
    return text


# --- a good file ------------------------------------------------------------------


def test_a_clean_ontology_compiles_with_qualified_names(tmp_path: Path) -> None:
    onto = load_or_raise(_write(tmp_path))
    assert "t:Person" in onto.classes
    assert onto.relations["t:works_at"].domain == "t:Person"
    assert onto.relations["t:works_at"].range == "t:Org"
    assert onto.expand("Person") == "urn:test#Person"
    assert onto.attributes["t:name"].datatype.endswith("#string")


def test_an_undeclared_inverse_is_synthesized_with_domain_and_range_swapped(tmp_path: Path) -> None:
    onto = load_or_raise(_write(tmp_path))
    employs = onto.relations["t:employs"]
    assert (employs.domain, employs.range, employs.inverse) == ("t:Org", "t:Person", "t:works_at")
    assert employs.label == "employs" and employs.synthesized


def test_subclass_queries_follow_the_chain(tmp_path: Path) -> None:
    onto = load_or_raise(_write(tmp_path))
    assert onto.ancestors("t:Team") == ["t:Team", "t:Org", "t:Thing"]
    assert onto.is_subclass("t:City", "t:Place")
    assert not onto.is_subclass("t:Place", "t:City")


# --- the file itself --------------------------------------------------------------


def test_an_unknown_dsl_key_is_refused_not_ignored(tmp_path: Path) -> None:
    _write(tmp_path, BASE.replace("Team:   { subclass_of: Org }", "Team:   { subclas_of: Org }"))
    result, issues = check_directory(tmp_path)
    assert result is None
    assert any(i.code == "schema" and "subclas_of" in i.where for i in issues)


def test_invalid_yaml_is_an_issue_not_a_crash(tmp_path: Path) -> None:
    _write(tmp_path, "ontology: [unclosed")
    result, issues = check_directory(tmp_path)
    assert result is None and issues


# --- names and references ---------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "extra", "code"),
    [
        ("classes", "  Robot: { subclass_of: Machine }\n", "unknown-class"),
        ("classes", "  Robot: { subclass_of: nope:Machine }\n", "unknown-prefix"),
        ("classes", "  Robot: { maps_to: Machine }\n", "unknown-prefix"),
        ("relations", "  owns: { domain: Person, range: Vehicle }\n", "unknown-class"),
        ("relations", "  owns: { domain: Person, range: email }\n", "wrong-kind"),
        ("attributes", "  age: { domain: Person, datatype: number }\n", "unknown-datatype"),
        ("attributes", "  Person: { domain: Person }\n", "duplicate-term"),
    ],
)
def test_bad_references_are_reported_by_code(
    tmp_path: Path, section: str, extra: str, code: str
) -> None:
    _write(tmp_path, _with(**{section: extra}))
    assert code in _codes(tmp_path)


def test_owner_class_must_be_a_concrete_declared_class(tmp_path: Path) -> None:
    header = "default_prefix: t }"
    _write(tmp_path, BASE.replace(header, "default_prefix: t, owner_class: Person }"))
    assert load_or_raise(tmp_path).owner_class == "t:Person"
    _write(tmp_path, BASE.replace(header, "default_prefix: t, owner_class: Thing }"))
    assert "owner-class" in _codes(tmp_path)
    _write(tmp_path, BASE.replace(header, "default_prefix: t, owner_class: Nobody }"))
    assert "unknown-class" in _codes(tmp_path)


def test_a_subclass_cycle_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path,
        BASE.replace("Thing:  { abstract: true }", "Thing:  { abstract: true, subclass_of: City }"),
    )
    assert "class-cycle" in _codes(tmp_path)


def test_every_problem_is_reported_in_one_run(tmp_path: Path) -> None:
    _write(
        tmp_path,
        _with(
            classes="  A: { subclass_of: Nope }\n",
            relations="  r: { domain: Nope, range: Person }\n",
        ),
    )
    assert _codes(tmp_path).count("unknown-class") == 2


# --- relation semantics -----------------------------------------------------------


def test_a_declared_inverse_must_point_back(tmp_path: Path) -> None:
    _write(tmp_path, _with(relations="  employs: { domain: Org, range: Person }\n"))
    assert "inverse-mismatch" in _codes(tmp_path)


def test_a_declared_inverse_that_points_back_is_accepted(tmp_path: Path) -> None:
    _write(
        tmp_path, _with(relations="  employs: { domain: Org, range: Person, inverse: works_at }\n")
    )
    assert _codes(tmp_path) == []


def test_symmetric_needs_one_class_on_both_ends(tmp_path: Path) -> None:
    _write(
        tmp_path,
        BASE.replace(
            "knows:    { domain: Person, range: Person, symmetric: true }",
            "knows:    { domain: Person, range: Org, symmetric: true }",
        ),
    )
    assert "symmetric-domain-range" in _codes(tmp_path)


def test_a_subproperty_must_narrow_its_parent(tmp_path: Path) -> None:
    _write(
        tmp_path,
        _with(relations="  bad: { domain: Person, range: Org, subproperty_of: lives_in }\n"),
    )
    assert "subproperty-range" in _codes(tmp_path)


def test_a_relation_cannot_be_a_subproperty_of_an_attribute(tmp_path: Path) -> None:
    _write(
        tmp_path, _with(relations="  bad: { domain: Person, range: Org, subproperty_of: email }\n")
    )
    assert "wrong-kind" in _codes(tmp_path)


# --- deprecation (decision 8) -----------------------------------------------------


def test_replaced_by_must_name_a_term_of_the_same_kind(tmp_path: Path) -> None:
    _write(
        tmp_path,
        _with(
            relations="  employed_by: { domain: Person, range: Org, deprecated: true, replaced_by: email }\n"
        ),
    )
    assert "wrong-kind" in _codes(tmp_path)


def test_replaced_by_on_a_live_term_is_an_error(tmp_path: Path) -> None:
    _write(
        tmp_path,
        _with(relations="  employed_by: { domain: Person, range: Org, replaced_by: works_at }\n"),
    )
    assert "replaced-not-deprecated" in _codes(tmp_path)


def test_a_replacement_cycle_is_reported(tmp_path: Path) -> None:
    _write(
        tmp_path,
        _with(
            relations=(
                "  a: { domain: Person, range: Org, deprecated: true, replaced_by: b }\n"
                "  b: { domain: Person, range: Org, deprecated: true, replaced_by: a }\n"
            )
        ),
    )
    assert "replacement-cycle" in _codes(tmp_path)


def test_a_split_lists_several_replacements(tmp_path: Path) -> None:
    _write(
        tmp_path,
        _with(
            classes="  Region: { subclass_of: Thing, deprecated: true, replaced_by: [Place, City] }\n"
        ),
    )
    onto = load_or_raise(tmp_path)
    assert onto.classes["t:Region"].replaced_by == ("t:Place", "t:City")


def test_usage_check_passes_a_replaced_term_and_fails_a_vanished_one(tmp_path: Path) -> None:
    _write(
        tmp_path,
        _with(
            relations=(
                "  employed_by: { domain: Person, range: Org, deprecated: true, replaced_by: works_at }\n"
                "  dead_end: { domain: Person, range: Org, deprecated: true }\n"
            )
        ),
    )
    onto = load_or_raise(tmp_path)
    assert check_usage(onto, ["works_at", "t:employed_by"]) == []
    vanished = {i.message.split("'")[1] for i in check_usage(onto, ["gone", "dead_end"])}
    assert vanished == {"t:gone", "t:dead_end"}


# --- shapes -----------------------------------------------------------------------


def test_shapes_compile_to_constraints(tmp_path: Path) -> None:
    _write(tmp_path, shapes="Person:\n  properties:\n    works_at: { max_count: 1, class: Team }\n")
    onto = load_or_raise(tmp_path)
    constraint = onto.shapes["t:Person"]["t:works_at"]
    assert (constraint.max_count, constraint.object_class) == (1, "t:Team")


@pytest.mark.parametrize(
    ("prop", "code"),
    [
        ("nope: { max_count: 1 }", "unknown-property"),
        ("works_at: { min_count: 2, max_count: 1 }", "shape-counts"),
        ("works_at: { class: Place }", "shape-class"),
        ("email: { class: Person }", "wrong-kind"),
    ],
)
def test_bad_shapes_are_reported(tmp_path: Path, prop: str, code: str) -> None:
    _write(tmp_path, shapes=f"Person:\n  properties:\n    {prop}\n")
    assert code in _codes(tmp_path)


def test_a_shape_cannot_constrain_a_property_outside_its_domain(tmp_path: Path) -> None:
    _write(tmp_path, shapes="Org:\n  properties:\n    email: { max_count: 1 }\n")
    assert "shape-not-applicable" in _codes(tmp_path)


# --- mappings ---------------------------------------------------------------------

GOOD_MAPPING = """
mappings:
  - id: employer
    when: { source_type: fact, key: [employer, org] }
    emit: { subject: $owner, predicate: works_at, object: { from: $value, class: Team } }
  - id: email
    when: { source_type: fact, key: email }
    emit: { subject: $owner, predicate: email, value: $value }
"""


def test_mappings_compile_with_keys_and_qualified_predicates(tmp_path: Path) -> None:
    onto = load_or_raise(_write(tmp_path, mappings=GOOD_MAPPING))
    employer = onto.mappings[0]
    assert employer.keys == ("employer", "org")
    assert (employer.predicate, employer.object_class) == ("t:works_at", "t:Team")


@pytest.mark.parametrize(
    ("emit", "code"),
    [
        ("{ subject: $owner, predicate: nope, value: $value }", "unknown-property"),
        ("{ subject: $owner, predicate: works_at, value: $value }", "mapping-kind"),
        (
            "{ subject: $owner, predicate: email, object: { from: $value, class: Person } }",
            "mapping-kind",
        ),
        (
            "{ subject: $owner, predicate: works_at, object: { from: $value, class: Place } }",
            "mapping-range",
        ),
        (
            "{ subject: $owner, predicate: lives_in, object: { from: $value, class: Thing } }",
            "abstract-class",
        ),
    ],
)
def test_bad_mappings_are_reported(tmp_path: Path, emit: str, code: str) -> None:
    _write(
        tmp_path,
        mappings=f"mappings:\n  - id: m\n    when: {{ source_type: fact, key: k }}\n    emit: {emit}\n",
    )
    assert code in _codes(tmp_path)


def test_one_source_key_may_feed_only_one_mapping(tmp_path: Path) -> None:
    overlap = GOOD_MAPPING + (
        "  - id: second\n    when: { source_type: fact, key: org }\n"
        "    emit: { subject: $owner, predicate: works_at, object: { from: $value, class: Org } }\n"
    )
    _write(tmp_path, mappings=overlap)
    assert "mapping-overlap" in _codes(tmp_path)


def test_using_a_deprecated_predicate_warns_but_does_not_fail(tmp_path: Path) -> None:
    _write(
        tmp_path,
        _with(
            relations="  employed_by: { domain: Person, range: Org, deprecated: true, replaced_by: works_at }\n"
        ),
        mappings="mappings:\n  - id: m\n    when: { source_type: fact, key: k }\n"
        "    emit: { subject: $owner, predicate: employed_by, object: { from: $value, class: Org } }\n",
    )
    _, issues = check_directory(tmp_path)
    assert [i.code for i in issues] == ["deprecated-in-use"]
    assert issues[0].severity == "warning"


# --- entry points -----------------------------------------------------------------


def test_load_or_raise_lists_every_error(tmp_path: Path) -> None:
    _write(tmp_path, _with(classes="  A: { subclass_of: Nope }\n  B: { subclass_of: Gone }\n"))
    with pytest.raises(OntologyError) as caught:
        load_or_raise(tmp_path)
    assert len(caught.value.issues) == 2


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "good"
    good.mkdir()
    _write(good)
    assert main([str(good)]) == 0
    assert "0 error(s)" in capsys.readouterr().out
    bad = tmp_path / "bad"
    bad.mkdir()
    _write(bad, _with(classes="  A: { subclass_of: Nope }\n"))
    assert main([str(bad)]) == 1
    assert main([]) == 2
