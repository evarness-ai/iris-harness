"""memris knows no vocabulary (ADR-0115 decisions 2 and 13).

Every class, relation and attribute name in IRIS's ontology is collected from the YAML,
and no string literal in ``src/memris`` may equal one of them. Code that needs to know a
name is code that stops working the day the YAML changes, which is what the owner rule
(vocabulary in config, not code) exists to prevent.
"""

from __future__ import annotations

import ast
from pathlib import Path

from memris.ontology import load_or_raise

REPO = Path(__file__).resolve().parents[4]
ONTOLOGY_DIR = REPO / "config" / "memory"
MEMRIS_SRC = REPO / "src" / "memris"
# IRIS code that draws or queries memory by the ontology must be as vocabulary-free as
# the engine (memris plan PR 5): every word it shows comes from the YAML.
VOCABULARY_FREE = [
    *sorted(MEMRIS_SRC.rglob("*.py")),
    REPO / "src" / "iris_harness" / "memory" / "graph.py",
    # Closed extraction (memris PR 3): the allowlist and the prompt come from the YAML.
    REPO / "src" / "iris_harness" / "memory" / "fact_keys.py",
    REPO / "src" / "iris_harness" / "memory" / "fact_extractor.py",
    # The agent's view of the graph (memris PR 6): linking and the memory_graph tool.
    REPO / "src" / "iris_harness" / "memory" / "graph_context.py",
]


def _vocabulary() -> set[str]:
    # The core YAML plus every installed plugin's fragment (memris PR 10): a plugin's
    # names (fin:banks_with) are no more welcome in vocabulary-free code than the core's.
    from iris_harness.memory.ontology import vocabulary_fragments

    onto = load_or_raise(ONTOLOGY_DIR, vocabulary_fragments())
    names: set[str] = set()
    for table in (onto.classes, onto.relations, onto.attributes):
        for qualified, term in table.items():
            names.add(qualified)
            names.add(qualified.partition(":")[2])
            names.add(term.label)
    return names


def _string_literals(path: Path) -> list[tuple[int, str]]:
    """Every string constant, except the names in ``__all__``.

    ``__all__`` lists Python symbols (``Entity`` is memris's record type); a symbol that
    happens to share a name with an ontology class is not a reference to it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    exported = {
        id(elt)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
        and isinstance(node.value, ast.List | ast.Tuple)
        for elt in node.value.elts
    }
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in exported
    ]


def test_the_vocabulary_is_non_trivial() -> None:
    # Guards the guard: an empty vocabulary would make the test below pass vacuously.
    assert len(_vocabulary()) > 50


def test_no_string_literal_in_vocabulary_free_code_names_an_ontology_term() -> None:
    vocabulary = _vocabulary()
    hits = [
        f"{path.relative_to(REPO)}:{line}: {value!r}"
        for path in VOCABULARY_FREE
        for line, value in _string_literals(path)
        if value in vocabulary
    ]
    assert hits == [], "memris must not name ontology terms:\n" + "\n".join(hits)
