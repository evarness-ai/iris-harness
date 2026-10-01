"""IRIS's learned vocabulary: memris's Learner with IRIS's names and thresholds.

memris decides *policy* (ADR-0115 decision 7): a property the ontology lacks is an alias
of a near-synonym, a term of its own once it recurs, or a counted candidate that expires.
IRIS supplies what memris leaves to its caller, all from ``learning.yaml`` →
``learned_terms``: the prefix learned terms live under, the thresholds, how alike two
names look (the same cheap, local comparison entity resolution uses), the fact keys that
already name each term (``job`` is ``profession``), and the keys never worth learning.

Learned terms are rows in the memory database, never YAML: the deployed config is
read-only, and nothing restarts when a word is learned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from iris_harness.memory.ontology import normalise_fact_key
from iris_harness.memory.resolution import name_similarity
from memris.graph import MemoryGraph
from memris.learn import Learner, Observation, term_local_name
from memris.model import LearnedTerm

logger = logging.getLogger(__name__)


def settings() -> dict[str, Any]:
    from iris_harness.memory.fact_keys import learning_config

    raw = learning_config().get("learned_terms") or {}
    return raw if isinstance(raw, dict) else {}


def learned_prefix() -> str:
    return str(settings().get("prefix") or "learned")


def never_learned() -> frozenset[str]:
    return frozenset(term_local_name(str(k)) for k in settings().get("never_learn") or [])


def _fact_keys_by_predicate(graph: MemoryGraph) -> dict[str, list[str]]:
    keys: dict[str, list[str]] = {}
    for rule in graph.declared.mappings:
        if rule.source_type == "fact":
            keys.setdefault(rule.predicate, []).extend(
                normalise_fact_key(k).replace("_", " ") for k in rule.keys
            )
    return keys


def learner_for(graph: MemoryGraph) -> Learner:
    config = settings()
    keys = _fact_keys_by_predicate(graph)

    def as_int(name: str, default: int) -> int:
        try:
            return int(config.get(name, default))
        except (TypeError, ValueError):
            return default

    try:
        alias_similarity = float(config.get("alias_similarity", 0.85))
    except (TypeError, ValueError):
        alias_similarity = 0.85
    return Learner(
        graph,
        prefix=learned_prefix(),
        similarity=name_similarity,
        names_for=lambda term: keys.get(term, []),
        alias_similarity=alias_similarity,
        min_observations=as_int("min_observations", 5),
        min_episodes=as_int("min_episodes", 3),
        expire_days=as_int("expire_days", 90),
        max_examples=as_int("max_examples", 3),
    )


@dataclass(frozen=True)
class Learned:
    """What capture does with a key the ontology lacks."""

    observation: Observation | None
    fact_key: str | None  # the key to write the fact with now, or None: only counted


class Vocabulary:
    """The learned layer over one memory graph — observe, list, decide."""

    def __init__(self, graph: MemoryGraph, fact_key_for: dict[str, str]) -> None:
        self.graph = graph
        self.learner = learner_for(graph)
        self._fact_key_for = fact_key_for

    def observe(
        self, raw_key: str, *, domain: str, datatype: str, example: str, episode: str | None
    ) -> Learned:
        """Count ``raw_key``; say which fact key (if any) the fact may be written with."""
        local = term_local_name(raw_key)
        if not local or local in never_learned():
            return Learned(None, None)
        seen = self.learner.observe(
            local, domain=domain, range_=datatype, example=example[:120], episode=episode
        )
        if seen is None:
            return Learned(None, None)
        usable = seen.usable_as
        if usable is None:
            return Learned(seen, None)
        if seen.outcome == "alias":
            return Learned(seen, self._fact_key_for.get(usable))
        return Learned(seen, usable.partition(":")[2] or usable)

    def terms(self, status: str | None = None) -> list[LearnedTerm]:
        return self.graph.store.terms(status)

    def reject(self, name: str, *, decided_by: str = "owner") -> LearnedTerm | None:
        """No: never learned again, and what was said with it leaves memory (kept in history)."""
        rejected = self.learner.reject(self.qualified(name), decided_by=decided_by)
        if rejected is not None and rejected.activated_at is not None:
            for s in self.graph.current(None, rejected.name, include_proposed=True):
                self.graph.retract(s.id, reason="rejected")
        return rejected

    def activate(self, name: str, *, decided_by: str = "owner") -> LearnedTerm | None:
        return self.learner.activate(self.qualified(name), decided_by=decided_by)

    def expire(self) -> list[str]:
        return self.learner.expire()

    def qualified(self, name: str) -> str:
        return name if ":" in name else self.learner.name_for(name)


def _short(ontology: Any, name: str) -> str:
    """``mem:Person`` → ``Person`` (the YAML's own spelling); other prefixes stay."""
    prefix = f"{ontology.default_prefix}:"
    return name[len(prefix) :] if name.startswith(prefix) else name


def _datatype(term: LearnedTerm) -> str:
    return term.range.rsplit("#", 1)[-1] if "#" in term.range else term.range


def term_yaml(term: LearnedTerm, ontology: Any) -> str:
    """The ontology.yaml line that declares ``term`` — for a human to review and commit."""
    local = term.name.partition(":")[2] or term.name
    domain = _short(ontology, term.domain)
    if term.kind == "attribute":
        body = f'domain: {domain}, datatype: {_datatype(term)}, label: "{term.label}"'
        return f"  {local}: {{ {body} }}"
    body = f'domain: {domain}, range: {_short(ontology, term.range)}, label: "{term.label}"'
    return f"  {local}: {{ {body} }}"


def promotion(term: LearnedTerm, ontology: Any) -> str:
    """What promoting ``term`` into the YAML means — a snippet, never a write.

    ``iris ontology promote`` is optional curation (decision 7): once a human commits
    this, the declared term takes over and the learned one reads through to it.
    """
    local = term.name.partition(":")[2] or term.name
    table = "attributes" if term.kind == "attribute" else "relations"
    emit = (
        f"{{ subject: $owner, predicate: {local}, value: $value }}"
        if term.kind == "attribute"
        else f"{{ subject: $owner, predicate: {local}, object: {{ from: $value, class: "
        f"{_short(ontology, term.range)} }} }}"
    )
    seen = ", ".join(repr(e) for e in term.examples) or "none kept"
    return (
        f"# Promote {term.name} (seen {term.observations}x in {len(term.episodes)} "
        f"conversation(s); examples: {seen}).\n"
        f"# The learned term then reads through to the declared one; nothing is rewritten.\n"
        f"\n# config/memory/ontology.yaml → {table}:\n{term_yaml(term, ontology)}\n"
        f"\n# config/memory/mappings.yaml → mappings:\n"
        f"  - id: fact_{local}\n"
        f"    when: {{ source_type: fact, key: {local} }}\n"
        f"    emit: {emit}\n"
    )


def export_yaml(terms: list[LearnedTerm], ontology: Any) -> str:
    """Learned terms as an ontology fragment (status and counts as comments)."""
    lines = ["# Learned vocabulary (memory database; ADR-0115 decision 7). Not loaded from here."]
    for kind, table in (("attribute", "attributes"), ("relation", "relations")):
        chosen = [t for t in terms if t.kind == kind]
        if not chosen:
            continue
        lines.append(f"{table}:")
        for term in chosen:
            note = f"  # {term.status}, seen {term.observations}x in {len(term.episodes)}"
            if term.alias_of:
                note += f", alias of {term.alias_of}"
            lines.append(term_yaml(term, ontology) + note)
    return "\n".join(lines) + "\n"


__all__ = [
    "Learned",
    "Vocabulary",
    "export_yaml",
    "learned_prefix",
    "learner_for",
    "never_learned",
    "promotion",
    "settings",
    "term_yaml",
]
