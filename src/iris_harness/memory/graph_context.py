"""The memory graph, as the agent sees it (memris PR 6; ADR-0115 decision 9).

Two ways in, one rendering:

- **Automatic linking.** Every turn, the entities the message names (label or alias,
  folded the way the Map folds them) bring their current one-hop statements into the
  prompt, within ``learning.yaml`` → ``linking.max_tokens``. What did not fit becomes a
  one-line pointer to the tool. The Context tab shows the block with its token cost.
- **The ``memory_graph`` tool**, for deliberate traversal and history: an entity, an
  optional relation, a direction, up to ``graph_tool.max_hops`` hops, and an optional
  ``as_of`` date. Its description lists the relations the ontology declares, so a term
  added to the YAML is offered with no code change. No query language is exposed.

Only confirmed statements are shown — a proposal is a question, not something memory
knows. Every line carries its provenance ("told 2026-09-16, confirmed"). Nothing here
names a class or a relation: labels come from the ontology.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from iris_harness.llm.budget import estimate_tokens
from memris.model import OWNER_ID, Entity, Statement, name_key

if TYPE_CHECKING:
    from iris_harness.memory.store import MemoryStore
    from memris.graph import MemoryGraph

logger = logging.getLogger(__name__)

# How the owner is named in what the model reads, and the tool argument that means them.
OWNER_IN_PROMPT = "the user"
OWNER_ARG = "user"

LINKED_HEADER = "## What you remember about names in this message"


def _learning(section: str) -> dict[str, Any]:
    from iris_harness.memory.fact_keys import learning_config

    raw = learning_config().get(section) or {}
    return raw if isinstance(raw, dict) else {}


def linking_max_tokens() -> int:
    """``linking.max_tokens`` — the linked block's share of the prompt (0 turns it off)."""
    try:
        return max(0, int(_learning("linking").get("max_tokens", 400)))
    except (TypeError, ValueError):
        return 400


def graph_tool_max_hops() -> int:
    """``graph_tool.max_hops`` — how far one ``memory_graph`` call may reach."""
    try:
        return max(1, int(_learning("graph_tool").get("max_hops", 2)))
    except (TypeError, ValueError):
        return 2


@dataclass(frozen=True)
class Linked:
    """The block linking adds to one turn's prompt."""

    text: str  # "" when nothing was named
    entities: tuple[str, ...]  # labels, in the order the message names them
    shown: int  # statements in the block
    left_out: int  # statements that did not fit the budget

    @property
    def pointer(self) -> str | None:
        if not self.left_out:
            return None
        return (
            f"{self.left_out} more remembered statement(s) about "
            f"{', '.join(self.entities)} did not fit here — memory_graph reads them."
        )


class GraphContext:
    """Renders the memory graph for the model — linking and the ``memory_graph`` tool."""

    def __init__(self, graph: MemoryGraph) -> None:
        self.graph = graph
        self.ontology = graph.ontology

    # -- naming ------------------------------------------------------------------

    def label(self, entity_id: str) -> str:
        if self.graph.canonical_id(entity_id) == OWNER_ID:
            return OWNER_IN_PROMPT
        entity = self.graph.get_entity(self.graph.canonical_id(entity_id))
        return entity.label if entity is not None else entity_id

    def predicate_label(self, predicate: str) -> str:
        term = self.ontology.relations.get(predicate) or self.ontology.attributes.get(predicate)
        return term.label if term is not None else predicate

    def sentence(self, s: Statement) -> str:
        """ "Petra: works at Infosys" — subject first, whichever way it was reached."""
        subject = self.label(s.subject_id)
        if s.object_id is not None:
            return f"{subject}: {self.predicate_label(s.predicate)} {self.label(s.object_id)}"
        return f"{subject}: {self.predicate_label(s.predicate)} = {s.literal}"

    @staticmethod
    def provenance(s: Statement) -> str:
        told = f"told {s.recorded_at.astimezone(UTC).date().isoformat()}"
        state = "confirmed" if s.status == "confirmed" else s.status
        parts = [told, state]
        if s.valid_to is not None:
            parts.append(f"until {s.valid_to.astimezone(UTC).date().isoformat()}")
        return ", ".join(parts)

    def line(self, s: Statement) -> str:
        return f"- {self.sentence(s)} ({self.provenance(s)})"

    # -- which entities a message names ----------------------------------------------

    def _names(self) -> dict[str, list[str]]:
        """canonical entity id → every name it goes by (label, aliases, merged names,
        and the Map's configured aliases that fold to it)."""
        from iris_harness.memory.graph import canonical_entity, graph_config

        system = {n for n, c in self.ontology.classes.items() if c.system}
        names: dict[str, list[str]] = {}
        by_key: dict[str, str] = {}
        for entity in self.graph.store.find_entities():
            target = self.graph.canonical_id(entity.id)
            if target == OWNER_ID or entity.class_ in system:
                continue
            for name in (entity.label, *entity.aliases):
                names.setdefault(target, []).append(name)
                by_key.setdefault(name_key(canonical_entity(name)), target)
        configured = graph_config().get("aliases") or {}
        for alias, folded in configured.items():
            owner_of = by_key.get(name_key(canonical_entity(str(folded))))
            if owner_of is not None:
                names[owner_of].append(str(alias))
        return names

    def named_in(self, message: str) -> list[str]:
        """Entities the message names, as canonical ids, in the order they appear."""
        text = message or ""
        if not text.strip():
            return []
        first_seen: dict[str, int] = {}
        for target, names in self._names().items():
            for name in names:
                if len(name_key(name)) < 2:
                    continue
                pattern = r"(?<!\w)" + re.escape(" ".join(name.split())) + r"(?!\w)"
                match = re.search(pattern, text, re.IGNORECASE)
                if match is not None:
                    first_seen[target] = min(first_seen.get(target, match.start()), match.start())
        return sorted(first_seen, key=lambda t: (first_seen[t], t))

    # -- automatic linking ---------------------------------------------------------

    def linked(self, message: str, *, max_tokens: int) -> Linked:
        """The current one-hop statements of every entity the message names, in budget."""
        empty = Linked("", (), 0, 0)
        if max_tokens <= 0:
            return empty
        named = self.named_in(message)
        statements: list[Statement] = []
        seen: set[str] = set()
        mentioned: list[str] = []
        for entity_id in named:
            around = [s for _hop, s in self.graph.neighbourhood(entity_id, hops=1)]
            if not around:
                continue
            mentioned.append(self.label(entity_id))
            for s in around:
                if s.id not in seen:
                    seen.add(s.id)
                    statements.append(s)
        if not statements:
            return empty
        lines: list[str] = []
        used = estimate_tokens(LINKED_HEADER)
        for s in statements:
            line = self.line(s)
            cost = estimate_tokens(line)
            if used + cost > max_tokens:
                break
            lines.append(line)
            used += cost
        if not lines:
            return Linked("", tuple(mentioned), 0, len(statements))
        text = LINKED_HEADER + "\n" + "\n".join(lines)
        return Linked(text, tuple(mentioned), len(lines), len(statements) - len(lines))

    # -- the memory_graph tool -----------------------------------------------------

    def relation_names(self) -> list[str]:
        """Relations and attributes the agent may ask about (current, not retired)."""
        # Not the bookkeeping relations whose ends are system classes (conversations,
        # lessons): memory keeps those for the Map, not as things to ask about.
        system = {n for n, c in self.ontology.classes.items() if c.system}

        def about_the_world(term: Any) -> bool:
            ends = {term.domain, getattr(term, "range", None)}
            return not any(
                self.ontology.is_subclass(end, cls) for end in ends if end for cls in system
            )

        names = [
            n
            for table in (self.ontology.relations, self.ontology.attributes)
            for n, term in table.items()
            if not term.deprecated
            and not getattr(term, "synthesized", False)
            and about_the_world(term)
        ]
        return sorted(self._short(n) for n in names)

    def _short(self, name: str) -> str:
        prefix = f"{self.ontology.default_prefix}:"
        return name[len(prefix) :] if name.startswith(prefix) else name

    def _predicate(self, raw: str) -> str | None:
        """A relation named by its term (``works_at``, ``fin:banks_with``), its label, or a
        fact key that maps onto it (``bank``, ``city`` — the names memory_correct uses)."""
        from iris_harness.memory.ontology import fact_mappings, normalise_fact_key

        rule = fact_mappings(self.ontology).get(normalise_fact_key(raw))
        if rule is not None:
            return self.ontology.qualify(rule.predicate)
        wanted = " ".join(raw.replace("_", " ").split()).lower()
        for table in (self.ontology.relations, self.ontology.attributes):
            for name, term in table.items():
                if self.ontology.qualify(raw) == name or self._short(name) == raw:
                    return name
                if " ".join(self._short(name).replace("_", " ").split()).lower() == wanted:
                    return name
                if term.label.lower() == wanted:
                    return name
        return None

    def _entity(self, raw: str) -> Entity | None:
        """The entity a tool argument names — the owner, or any name folded like the Map's."""
        from iris_harness.memory.graph import canonical_entity

        if name_key(raw) in {name_key(OWNER_ARG), name_key(OWNER_IN_PROMPT)}:
            return self.graph.get_entity(OWNER_ID)
        wanted = {name_key(raw), name_key(canonical_entity(raw))} - {""}
        for target, names in self._names().items():
            if any({name_key(n), name_key(canonical_entity(n))} & wanted for n in names):
                return self.graph.get_entity(target)
        return None

    def query(
        self,
        entity: str,
        *,
        relation: str | None = None,
        direction: str = "both",
        hops: int = 1,
        as_of: str | None = None,
    ) -> str:
        """What ``memory_graph`` answers: lines with provenance, or why there are none."""
        name = " ".join((entity or "").split())
        if not name:
            return f'Error: memory_graph needs an "entity" (a name, or "{OWNER_ARG}").'
        found = self._entity(name)
        if found is None:
            return f"Nothing remembered about '{name}'."
        predicate = None
        if relation:
            predicate = self._predicate(relation)
            if predicate is None:
                return (
                    f"Error: '{relation}' is not a relation memory keeps. "
                    f"Known: {', '.join(self.relation_names())}."
                )
        if direction not in ("out", "in", "both"):
            return 'Error: direction is "out", "in" or "both".'
        limit = graph_tool_max_hops()
        hops = max(1, min(int(hops), limit))
        when = None
        if as_of:
            try:
                when = datetime.fromisoformat(as_of)
            except ValueError:
                return "Error: as_of is a date like 2026-09-01."
            if when.tzinfo is None:
                when = when.replace(tzinfo=UTC)
        found_statements = self.graph.neighbourhood(
            found.id,
            predicate=predicate,
            direction=direction,  # type: ignore[arg-type]
            hops=hops,
            as_of=when,
        )
        if not found_statements:
            scope = f" ({self.predicate_label(predicate)})" if predicate else ""
            return f"Nothing confirmed about {self.label(found.id)}{scope}."
        lines = [
            f"{self.line(s)}{'' if hop == 1 else f' [hop {hop}]'}" for hop, s in found_statements
        ]
        return "\n".join(lines)

    def tool_description(self) -> str:
        return (
            "Read what memory knows about a person, place or organisation, and how they "
            "connect — with when you were told and whether it is confirmed. Use it for "
            "questions about someone or something the user has mentioned, their "
            "relationships, or what was true at an earlier date. "
            f'Args: {{"entity": str (a name, or "{OWNER_ARG}"), "relation": str (optional; '
            f"one of: {', '.join(self.relation_names())}), "
            '"direction": "out" | "in" | "both" (default both), '
            f'"hops": int (1-{graph_tool_max_hops()}, default 1), '
            '"as_of": "YYYY-MM-DD" (optional)}.'
        )


def graph_context(store: MemoryStore) -> GraphContext:
    return GraphContext(store.memory_graph())


def memory_graph_tool(store: MemoryStore | None, args: dict[str, Any]) -> str:
    """The ``memory_graph`` tool body, shared by the ReAct loop and native tool calling."""
    if store is None:
        return "memory_graph unavailable (memory store not configured)."
    try:
        hops = int(args.get("hops") or 1)
    except (TypeError, ValueError):
        hops = 1
    try:
        return graph_context(store).query(
            str(args.get("entity") or args.get("input") or ""),
            relation=str(args["relation"]) if args.get("relation") else None,
            direction=str(args.get("direction") or "both").strip().lower(),
            hops=hops,
            as_of=str(args["as_of"]) if args.get("as_of") else None,
        )
    except Exception as exc:
        logger.exception("memory_graph failed")
        return f"memory_graph failed: {exc}"


def memory_graph_description(store: MemoryStore | None) -> str:
    """The tool's description, generated from the ontology (a fixed one without a store)."""
    if store is not None:
        try:
            return graph_context(store).tool_description()
        except Exception:
            logger.exception("memory_graph description unavailable")
    return (
        "Read what memory knows about a person, place or organisation and how they "
        'connect. Args: {"entity": str, "relation": str, "direction": str, '
        '"hops": int, "as_of": "YYYY-MM-DD"}.'
    )


__all__ = [
    "LINKED_HEADER",
    "GraphContext",
    "Linked",
    "graph_context",
    "graph_tool_max_hops",
    "linking_max_tokens",
    "memory_graph_description",
    "memory_graph_tool",
]
