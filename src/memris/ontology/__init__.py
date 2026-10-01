"""The ontology: YAML in, a checked :class:`Ontology` out.

``check_directory(path)`` compiles a directory of ``ontology.yaml`` / ``shapes.yaml`` /
``mappings.yaml`` and reports every issue; ``load_or_raise(path)`` is the strict form.
``check_usage(ontology, terms)`` is the data check for terms stored statements use.
From a shell: ``python -m memris.ontology <dir>``.
"""

from __future__ import annotations

from memris.ontology.compiler import (
    AttributeTerm,
    ClassTerm,
    CompileResult,
    Constraint,
    Issue,
    MappingRule,
    Ontology,
    RelationTerm,
    check_usage,
    compile_mappings,
    compile_ontology,
)
from memris.ontology.learned import learned_prefix_iri, with_learned
from memris.ontology.loader import (
    OntologyError,
    check_directory,
    load_mappings,
    load_ontology,
    load_or_raise,
)

__all__ = [
    "learned_prefix_iri",
    "with_learned",
    "AttributeTerm",
    "ClassTerm",
    "CompileResult",
    "Constraint",
    "Issue",
    "MappingRule",
    "Ontology",
    "OntologyError",
    "RelationTerm",
    "check_directory",
    "check_usage",
    "compile_mappings",
    "compile_ontology",
    "load_mappings",
    "load_ontology",
    "load_or_raise",
]
