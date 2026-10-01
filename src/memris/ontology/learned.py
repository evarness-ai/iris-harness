"""The vocabulary in use: declared YAML ∪ learned rows (ADR-0115 decision 7).

``with_learned(ontology, terms)`` returns a copy of a compiled ontology that also holds
every learned term that ever became a term of its own (``active``, or rejected after it
was — its statements must still resolve). Candidates and aliases add nothing: a
candidate is only being counted, and an alias is a spelling of a term that exists.

A learned term that a declared term now matches — same local name or label, same domain,
same range — is linked to it: kept as ``deprecated`` with ``replaced_by`` the declared
term, so what was said with the learned word reads through to the promoted one
(decision 8) and nothing stored is rewritten.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace

from memris.model import LearnedTerm
from memris.ontology.compiler import AttributeTerm, Ontology, RelationTerm


def _local(name: str) -> str:
    return name.partition(":")[2] or name


def learned_prefix_iri(ontology: Ontology, prefix: str) -> str:
    """A stable IRI for a learned namespace that has no declared one."""
    return f"{ontology.id.rstrip('#/')}/{prefix}#"


def _same(ontology: Ontology, a: str, b: str) -> bool:
    """Two class or datatype names are the same once prefixes are expanded."""
    return a == b or (ontology.expand(a) or a) == (ontology.expand(b) or b)


def _declared_match(ontology: Ontology, term: LearnedTerm) -> str | None:
    """The declared term a learned one has since become, if any."""
    local, label = _local(term.name).lower(), term.label.strip().lower()
    table = ontology.attributes if term.kind == "attribute" else ontology.relations
    for name, declared in table.items():
        if declared.deprecated or name == term.name:
            continue
        same_name = _local(name).lower() == local or declared.label.strip().lower() == label
        if not same_name or declared.domain != term.domain:
            continue
        if isinstance(declared, AttributeTerm) and _same(ontology, declared.datatype, term.range):
            return name
        if isinstance(declared, RelationTerm) and _same(ontology, declared.range, term.range):
            return name
    return None


def with_learned(ontology: Ontology, terms: Iterable[LearnedTerm]) -> Ontology:
    """The ontology plus the learned terms that are terms (see the module docstring)."""
    relations = dict(ontology.relations)
    attributes = dict(ontology.attributes)
    prefixes = dict(ontology.prefixes)
    for term in terms:
        if term.status != "active" and term.activated_at is None:
            continue
        if term.name in ontology.relations or term.name in ontology.attributes:
            continue  # a declared term always wins its own name
        prefix = term.name.partition(":")[0]
        if prefix and prefix not in prefixes:
            prefixes[prefix] = learned_prefix_iri(ontology, prefix)
        promoted = _declared_match(ontology, term)
        replaced_by = (promoted,) if promoted else ()
        if term.kind == "attribute":
            attributes[term.name] = AttributeTerm(
                name=term.name,
                label=term.label,
                domain=term.domain,
                datatype=term.range,
                parent=None,
                maps_to=None,
                deprecated=promoted is not None,
                replaced_by=replaced_by,
            )
        else:
            relations[term.name] = RelationTerm(
                name=term.name,
                label=term.label,
                domain=term.domain,
                range=term.range,
                inverse=None,
                symmetric=False,
                parent=None,
                maps_to=None,
                deprecated=promoted is not None,
                replaced_by=replaced_by,
            )
    return replace(ontology, relations=relations, attributes=attributes, prefixes=prefixes)


__all__ = ["learned_prefix_iri", "with_learned"]
