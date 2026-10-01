"""memris — a memory graph of claims, described by a YAML ontology.

memris knows no vocabulary. Every class, relation and attribute comes from the YAML it
is given; the engine only interprets it (ADR-0115, decision 2). It imports nothing from
the IRIS harness and calls no model, so another project can use it on its own
(decision 10).

    from memris import MemoryGraph, SQLiteGraphStore, load_or_raise

    graph = MemoryGraph(load_or_raise("config/memory"), SQLiteGraphStore("memory.db"))
    graph.ensure_owner("Me")  # class from ontology.owner_class

Pieces: ``memris.ontology`` (compile and check the YAML), ``memris.model`` (entities and
bitemporal statements), ``memris.store`` (persistence: SQLite or in memory) and
``memris.graph`` (supersede / end / retract and time-aware reads). Entity resolution,
the learned vocabulary and interchange follow in later PRs
(docs/architecture/memris-plan.md).
"""

from __future__ import annotations

from memris.graph import MemoryGraph, StatementError
from memris.model import OWNER_ID, Entity, Statement
from memris.ontology import OntologyError, check_directory, load_or_raise
from memris.store import GraphStore, InMemoryGraphStore, SQLiteGraphStore

__version__ = "0.2.0"

__all__ = [
    "OWNER_ID",
    "Entity",
    "GraphStore",
    "InMemoryGraphStore",
    "MemoryGraph",
    "OntologyError",
    "SQLiteGraphStore",
    "Statement",
    "StatementError",
    "__version__",
    "check_directory",
    "load_or_raise",
]
