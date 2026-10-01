"""The memory graph — a VIEW over what is already stored, never a store of its own.

The wiki was the last attempt at this: 2,505 pages built by a regex extractor from
every chat turn and every classified email, 2,175 of them typed "person" (including
"Account Ending" and "Actually Use"), read back zero times in four months, and drawn
as one unpaginated React Flow layout that struggled in the browser.

So this is deliberately not that:

- **Nothing is extracted or stored.** Nodes are computed on request from the memris
  statements about the owner, session summaries, approved behaviors and episodic
  patterns. Forget a fact and its node is gone on the next read.
- **It knows no vocabulary** (ADR-0115 decisions 2 and 13). An edge's label is its
  property's label in ``config/memory/ontology.yaml``; which records become which edges
  is ``config/memory/mappings.yaml``; a node's kind (the Map's colour group) is its
  class's group in ``config/memory/entity_aliases.yaml``. This module only knows how
  to *read* each kind of record and how to draw.
- **It opens small.** "You" plus what touches you; a click loads one node's neighbours.
  There is a hard server-side cap, and what it leaves out is a "+N more" node rather
  than a silent truncation.
- **Unconfirmed things are drawn dimmed**, never counted as knowledge.
- **It has a time axis.** ``as_of`` draws what memory held true at that moment
  (the statements' valid time); records without a time axis are drawn as they are.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from iris_harness.foundation.process_state import track_globals
from iris_harness.memory import removal
from iris_harness.memory.ontology import memory_ontology
from memris.model import OWNER_ID, Statement
from memris.ontology import MappingRule, Ontology

logger = logging.getLogger(__name__)

_CONFIG_CACHE: dict[str, Any] | None = None

DEFAULT_NODE_CAP = 150
MORE = "more"
_DEFAULT_OWNER = {"id": "you", "group": "you", "label": "You"}


def graph_config() -> dict[str, Any]:
    """Read ``config/memory/entity_aliases.yaml`` (cached)."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    from iris_harness.foundation.paths import config_path, default_config_dir

    path = config_path("memory", "entity_aliases.yaml")
    if not path.exists():
        path = default_config_dir() / "memory" / "entity_aliases.yaml"
    config: dict[str, Any] = {
        "aliases": {},
        "suffixes": [],
        "groups": {},
        "attribute_group": None,
        "owner": dict(_DEFAULT_OWNER),
        "empty_items": [],
        "drawn_mentions": {},
    }
    if path.exists():
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config.update({k: v for k, v in loaded.items() if v is not None})
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read %s: %s — graph falls back to raw names", path, exc)
    _CONFIG_CACHE = config
    return config


def reset_config_cache() -> None:
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def canonical_entity(name: str) -> str:
    """Fold a name to the one the graph counts: alias first, then light normalisation."""
    raw = " ".join((name or "").split())
    if not raw:
        return ""
    config = graph_config()
    aliases = {str(k).strip().lower(): str(v) for k, v in (config.get("aliases") or {}).items()}
    if raw.lower() in aliases:
        return aliases[raw.lower()]
    suffixes = {str(s).strip().lower() for s in (config.get("suffixes") or [])}
    words = raw.split()
    while words and words[-1].lower().strip(".,") in suffixes:
        words.pop()
    return " ".join(words) or raw


def _mention_rules() -> tuple[int, list[re.Pattern[str]]]:
    """``drawn_mentions`` in entity_aliases.yaml: how many sessions a name needs, and the shapes
    that are never a name. A bad pattern is skipped with a warning, not fatal."""
    rules = graph_config().get("drawn_mentions") or {}
    try:
        min_sessions = max(1, int(rules.get("min_sessions", 1)))
    except (TypeError, ValueError):
        min_sessions = 1
    shapes: list[re.Pattern[str]] = []
    for raw in rules.get("never_draw") or []:
        try:
            shapes.append(re.compile(str(raw)))
        except re.error as exc:
            logger.warning("graph: bad never_draw pattern %r: %s", raw, exc)
    return min_sessions, shapes


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment is not None else None


@dataclass
class GraphNode:
    id: str
    kind: str
    label: str
    confirmed: bool = True
    meta: dict[str, Any] = field(default_factory=dict)
    # What the Remove action passes back (ADR-0119): {kind, id}, or None when the node
    # cannot be removed from the Map (you, lessons, patterns, "+N more").
    ref: dict[str, str] | None = None
    # A name the owner removed that a confirmed fact has since brought back.
    previously_removed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "confirmed": self.confirmed,
            "meta": self.meta,
            "ref": self.ref,
            "previously_removed": self.previously_removed,
        }


@dataclass
class GraphEdge:
    source: str
    target: str
    label: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.source}->{self.target}:{self.label}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class Record:
    """One non-fact record a reader yields, and the variables a mapping can bind."""

    episode: str | None = None
    item: str | None = None
    value: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- readers
#
# A reader knows how to FIND one kind of IRIS record. What the record becomes — which
# node, which edge, under which label — is the mapping's and the ontology's business.


def _removed_sessions(store: Any) -> set[str] | None:
    """Sessions the owner removed (ADR-0119), or ``None`` when the ledger is unreadable.

    ``None`` fails closed, like the retriever: without the ledger the Map cannot tell
    which sessions the owner removed, so it draws none rather than risk drawing one.
    """
    try:
        return set(store.removed_session_ids())
    except AttributeError:  # a store without the ledger removes nothing
        return set()
    except Exception as exc:  # the Map still draws, without sessions
        logger.warning(
            "graph: removed-session ledger unreadable (%s); drawing no sessions",
            type(exc).__name__,
            exc_info=True,
        )
        return None


def _sessions_with_summaries(store: Any) -> Iterator[tuple[str, Any, int, str]]:
    from iris_harness.memory.retention import is_ephemeral_session

    try:
        sessions = list(store.session_activity())
    except Exception as exc:  # the Map still draws, without sessions
        logger.warning(
            "graph: session activity unreadable (%s); drawing no sessions",
            type(exc).__name__,
            exc_info=True,
        )
        return
    removed = _removed_sessions(store)
    if removed is None:  # fail closed: a removed session must not reappear
        return
    for session_id, last_ts, turns in sessions:
        # A playground or test run, or a session the owner removed, is not memory.
        if is_ephemeral_session(session_id) or session_id in removed:
            continue
        try:
            summary = store.load_conversation_summary(session_id) or ""
        except Exception as exc:  # skip this session, draw the rest
            logger.warning(
                "graph: summary of session %s unreadable (%s); leaving it off the Map",
                session_id,
                type(exc).__name__,
                exc_info=True,
            )
            summary = ""
        if summary:
            yield session_id, last_ts, turns, summary


def _read_sessions(store: Any, _rule: MappingRule) -> Iterator[Record]:
    for session_id, last_ts, turns, summary in _sessions_with_summaries(store):
        yield Record(
            episode=session_id,
            meta={"turns": turns, "last_activity": last_ts, "summary": summary[:300]},
        )


def _section_items(summary: str, key: str) -> list[str]:
    """The comma-separated items of one summary section, found by its configured label."""
    from iris_harness.memory.compactor import summary_config

    labels = [
        str(s.get("label", "")).strip()
        for s in (summary_config().get("sections") or [])
        if isinstance(s, dict) and str(s.get("key", "")).strip() == key
    ]
    empty = {str(e).strip().lower() for e in (graph_config().get("empty_items") or [])}
    items: list[str] = []
    for label in filter(None, labels):
        pattern = re.compile(rf"^\s*{re.escape(label)}\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
        for match in pattern.finditer(summary):
            for raw in match.group(1).split(","):
                item = " ".join(raw.split())
                if len(item) >= 2 and item.lower() not in empty:
                    items.append(item)
    return items


def _read_summary_sections(store: Any, rule: MappingRule) -> Iterator[Record]:
    """Each listed name with the number of sessions that list it (``meta["sessions"]``).

    A name shaped like an ID, an address or a file (``drawn_mentions.never_draw``) is never
    yielded. How many sessions a name needs is the builder's call: a name that is a
    memris entity needs none.
    """
    _min, shapes = _mention_rules()
    found: list[tuple[str, str]] = []
    sessions_of: dict[str, set[str]] = {}
    for session_id, _last, _turns, summary in _sessions_with_summaries(store):
        for key in rule.keys:
            for item in _section_items(summary, key):
                if any(shape.search(item) for shape in shapes):
                    continue
                found.append((session_id, item))
                sessions_of.setdefault(canonical_entity(item).lower(), set()).add(session_id)
    for session_id, item in found:
        count = len(sessions_of[canonical_entity(item).lower()])
        yield Record(episode=session_id, item=item, meta={"sessions": count})


def _read_behaviors(_store: Any, _rule: MappingRule) -> Iterator[Record]:
    try:
        from iris_harness.memory.identity import list_behaviors

        for behavior in list_behaviors():
            yield Record(
                value=behavior.name,
                # Where to edit it: lessons have no Remove on the Map (ADR-0119).
                meta={
                    "summary": behavior.headline,
                    "intents": list(behavior.match_intents),
                    "file": str(behavior.path),
                },
            )
    except Exception:  # the Map still draws, without lessons
        logger.warning("graph: behaviors unavailable", exc_info=True)


def _read_patterns(_store: Any, _rule: MappingRule) -> Iterator[Record]:
    try:
        from iris_harness.memory.identity import list_episodic_patterns, loader

        path = str(loader.episodic_md_path())
        for pattern in list_episodic_patterns()[:20]:
            text = str(getattr(pattern, "text", pattern))
            yield Record(value=text[:80], meta={"file": path})
    except Exception:  # the Map still draws, without patterns
        logger.warning("graph: patterns unavailable", exc_info=True)


# The mapping's source_type names which reader supplies its records. Facts are not here:
# they are memris statements already, drawn directly.
READERS: dict[str, Callable[[Any, MappingRule], Iterator[Record]]] = {
    "session": _read_sessions,
    "summary_section": _read_summary_sections,
    "behavior": _read_behaviors,
    "pattern": _read_patterns,
}


# --------------------------------------------------------------------------- builder


class _Builder:
    def __init__(self, store: Any, ontology: Ontology, *, confirmed_only: bool) -> None:
        self.store = store
        self.onto = ontology
        self.config = graph_config()
        self.confirmed_only = confirmed_only
        self.min_sessions, _shapes = _mention_rules()
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []
        owner = {**_DEFAULT_OWNER, **(self.config.get("owner") or {})}
        self.owner_id = str(owner["id"])
        self.add_node(GraphNode(self.owner_id, str(owner["group"]), str(owner["label"])))
        self.groups = {
            self.onto.qualify(str(k)): str(v) for k, v in (self.config.get("groups") or {}).items()
        }
        # The class each mapping variable carries ($episode is a Conversation …), learned
        # from the mappings that emit it as an object.
        self.var_class: dict[str, str] = {}
        for rule in self.onto.mappings:
            if rule.object_from and rule.object_class:
                self.var_class.setdefault(rule.object_from, rule.object_class)
        self._excluded: frozenset[str] | None = None
        self._graph = None
        try:
            # memory_graph() runs the schema step (the one-time fact migration) first.
            self._graph = store.memory_graph()  # read-only: the statements themselves
        except Exception:  # the Map still draws the other records
            logger.warning("graph: statements unavailable", exc_info=True)
        self.suppressed = self._suppressed()

    def _suppressed(self) -> set[str]:
        """Names the owner removed (ADR-0119): a summary mention of one is not drawn."""
        try:
            return removal.suppressed_keys(self.store)
        except AttributeError:  # a store without the ledger suppresses nothing
            return set()
        except Exception as exc:  # the Map still draws
            logger.warning(
                "graph: removal ledger unreadable (%s); suppressing no names",
                type(exc).__name__,
                exc_info=True,
            )
            return set()

    def _is_suppressed(self, name: str) -> bool:
        return bool(self.suppressed) and removal.fold(name) in self.suppressed

    def _names_removed(self, entity_id: str) -> bool:
        if self._graph is None:
            return False
        entity = self._graph.get_entity(self._graph.canonical_id(entity_id))
        return entity is not None and self._is_suppressed(entity.label)

    def _holds_confirmed(self, entity_id: str) -> bool:
        """A confirmed fact names the entity now — which wins over a removed name."""
        if self._graph is None:
            return False
        return any(s.status == "confirmed" for s in self._graph.current(None, object_id=entity_id))

    # -- helpers ---------------------------------------------------------------

    def add_node(self, node: GraphNode) -> GraphNode:
        existing = self.nodes.get(node.id)
        if existing is None:
            self.nodes[node.id] = node
            return node
        existing.confirmed = existing.confirmed or node.confirmed
        return existing

    def group_of(self, class_name: str | None) -> str | None:
        if class_name is None:
            return None
        for cls in self.onto.ancestors(self.onto.qualify(class_name)):
            if cls in self.groups:
                return self.groups[cls]
        return None

    def label_of(self, predicate: str) -> str:
        name = self.onto.qualify(predicate)
        term = self.onto.relations.get(name) or self.onto.attributes.get(name)
        return term.label if term is not None else name.partition(":")[2]

    def _entity_node(self, entity_id: str, confirmed: bool) -> str | None:
        if self._graph is None:
            return None
        # A merged name draws as what it stands for now (ADR-0115 decision 4).
        entity = self._graph.get_entity(self._graph.canonical_id(entity_id))
        if entity is None or entity.removed:
            return None
        group = self.group_of(entity.class_)
        if group is None:
            return None
        node = self.add_node(
            GraphNode(
                f"{group}:{entity.id}",
                group,
                entity.label,
                confirmed=confirmed,
                meta={"class": entity.class_, "entity_id": entity.id},
                ref={"kind": removal.ENTITY, "id": entity.id},
                previously_removed=self._is_suppressed(entity.label),
            )
        )
        return node.id

    def _named_node(
        self,
        name: str,
        class_name: str | None,
        meta: dict[str, Any],
        *,
        fold: bool,
        ref_kind: str | None = None,
    ) -> str | None:
        """A record's node. A name read out of text (``fold``) is folded and matched to the
        memris entity it names, so "Northwind Bank Ltd" in a summary is the fact's Northwind Bank.

        A name that matches no entity is drawn only when enough sessions list it
        (``drawn_mentions.min_sessions``): a one-off mention is summary text, not memory.
        A removed name is not drawn from text — unless a confirmed fact names it again,
        which wins (ADR-0119)."""
        group = self.group_of(class_name)
        if group is None:
            return None
        folded = canonical_entity(name) if fold else name
        suppressed = fold and self._is_suppressed(name)
        if fold and self._graph is not None and class_name is not None:
            # The same resolution facts use — read-only here: the Map never creates.
            from memris.resolve import Resolver

            found = Resolver(self._graph, normalise=canonical_entity).find(name, class_name)
            if found is not None:
                if suppressed and not self._holds_confirmed(found[0].id):
                    return None
                return self._entity_node(found[0].id, confirmed=True)
        if suppressed:
            return None
        if fold and int(meta.get("sessions", 0)) < self.min_sessions:
            return None
        node = self.add_node(
            GraphNode(
                f"{group}:{_slug(folded) if fold else name}",
                group,
                folded,
                meta={"class": self.onto.qualify(class_name) if class_name else None, **meta},
                ref={"kind": ref_kind, "id": folded} if ref_kind else None,
            )
        )
        return node.id

    def _plugin_excluded(self, mention: str) -> bool:
        """A plugin said never to draw this name (ADR-0119). Asked once per draw.

        Only a summary mention is dropped: an entity a confirmed fact points at is drawn
        by the statements pass, whatever a plugin says.
        """
        if self._excluded is None:
            from iris_harness.memory.map_exclusions import excluded_names

            self._excluded = excluded_names(canonical_entity)
        return canonical_entity(mention).strip().lower() in self._excluded

    def _resolve(self, var: str, record: Record, class_name: str | None) -> str | None:
        if var == "$owner":
            return self.owner_id
        chosen = class_name or self.var_class.get(var)
        if var == "$episode" and record.episode:
            return self._named_node(
                record.episode, chosen, record.meta, fold=False, ref_kind=removal.SESSION
            )
        if var == "$item" and record.item:
            if self._plugin_excluded(record.item):
                return None
            return self._named_node(
                record.item, chosen, dict(record.meta), fold=True, ref_kind=removal.NAME
            )
        if var == "$value" and record.value:
            return self._named_node(record.value, chosen, record.meta, fold=False)
        return None

    # -- passes ----------------------------------------------------------------

    def statements(self, as_of: datetime | None) -> None:
        """Every current statement about the owner — the facts — drawn as it holds."""
        if self._graph is None:
            return
        found: list[Statement] = self._graph.current(
            OWNER_ID, include_proposed=not self.confirmed_only, as_of=as_of
        )
        attribute_group = self.config.get("attribute_group")
        for s in found:
            confirmed = s.status == "confirmed"
            label = self.label_of(s.predicate)
            edge_meta = {
                "predicate": s.predicate,
                "statement_id": s.id,
                "status": s.status,
                "confidence": s.confidence,
                "recorded_at": _iso(s.recorded_at),
                "valid_from": _iso(s.valid_from),
                "valid_to": _iso(s.valid_to),
            }
            if s.object_id is not None:
                # Only a CONFIRMED fact brings a removed name back.
                target = (
                    None
                    if not confirmed and self._names_removed(s.object_id)
                    else self._entity_node(s.object_id, confirmed)
                )
            elif attribute_group:
                node = self.add_node(
                    GraphNode(
                        f"{attribute_group}:{s.id}",
                        str(attribute_group),
                        f"{label}: {s.literal}",
                        confirmed=confirmed,
                        meta={"predicate": s.predicate, "datatype": s.datatype},
                        ref={"kind": removal.FACT, "id": s.id},
                    )
                )
                target = node.id
            else:
                target = None
            if target is not None:
                self.edges.append(GraphEdge(self.owner_id, target, label, edge_meta))

    def records(self) -> None:
        """Everything a non-fact mapping projects: sessions, mentions, lessons, patterns."""
        for rule in self.onto.mappings:
            reader = READERS.get(rule.source_type)
            if reader is None:
                continue
            label = self.label_of(rule.predicate)
            for record in reader(self.store, rule):
                subject = self._resolve(rule.subject, record, None)
                target = (
                    self._resolve(rule.object_from, record, rule.object_class)
                    if rule.object_from
                    else None
                )
                if subject is not None and target is not None:
                    self.edges.append(GraphEdge(subject, target, label, {"mapping": rule.id}))


def build_memory_graph(
    store: Any,
    *,
    focus: str | None = None,
    depth: int = 1,
    kinds: set[str] | None = None,
    confirmed_only: bool = False,
    node_cap: int = DEFAULT_NODE_CAP,
    as_of: datetime | None = None,
    ontology: Ontology | None = None,
) -> dict[str, Any]:
    """Compute the graph. ``focus`` is a node id; omitted, it centres on the owner."""
    builder = _Builder(store, ontology or memory_ontology(), confirmed_only=confirmed_only)
    builder.statements(as_of)
    builder.records()
    nodes, edges, owner = builder.nodes, builder.edges, builder.owner_id

    if kinds:
        keep = {i for i, n in nodes.items() if n.kind in kinds or i == owner}
        nodes = {i: n for i, n in nodes.items() if i in keep}
        edges = [e for e in edges if e.source in nodes and e.target in nodes]

    if focus:
        nodes, edges = _ego(nodes, edges, focus=focus, depth=max(1, depth))

    total = len(nodes)
    if total > node_cap:
        nodes, edges = _cap(nodes, edges, node_cap, owner)
        nodes[MORE] = GraphNode(
            id=MORE,
            kind=MORE,
            label=f"+{total - node_cap} {MORE}",
            meta={"hidden": total - node_cap},
        )
        edges.append(GraphEdge(owner, MORE, "…"))

    counts: dict[str, int] = {}
    for node in nodes.values():
        counts[node.kind] = counts.get(node.kind, 0) + 1
    return {
        "nodes": [n.as_dict() for n in nodes.values()],
        "edges": [e.as_dict() for e in edges],
        "stats": {"total_nodes": total, "shown": len(nodes), "by_kind": counts},
        "focus": focus or owner,
        "as_of": _iso(as_of),
    }


def _ego(
    nodes: dict[str, GraphNode], edges: list[GraphEdge], *, focus: str, depth: int
) -> tuple[dict[str, GraphNode], list[GraphEdge]]:
    """Keep ``focus`` and everything within ``depth`` hops of it."""
    if focus not in nodes:
        return nodes, edges
    reachable = {focus}
    frontier = {focus}
    for _ in range(depth):
        nxt: set[str] = set()
        for edge in edges:
            if edge.source in frontier and edge.target not in reachable:
                nxt.add(edge.target)
            if edge.target in frontier and edge.source not in reachable:
                nxt.add(edge.source)
        reachable |= nxt
        frontier = nxt
        if not frontier:
            break
    kept = {i: n for i, n in nodes.items() if i in reachable}
    return kept, [e for e in edges if e.source in kept and e.target in kept]


def _cap(
    nodes: dict[str, GraphNode], edges: list[GraphEdge], cap: int, owner: str
) -> tuple[dict[str, GraphNode], list[GraphEdge]]:
    """Keep the most connected nodes, always keeping the owner."""
    degree: dict[str, int] = {}
    for edge in edges:
        degree[edge.source] = degree.get(edge.source, 0) + 1
        degree[edge.target] = degree.get(edge.target, 0) + 1
    ordered = sorted(nodes.values(), key=lambda n: (n.id != owner, -degree.get(n.id, 0), n.label))
    kept = {n.id: n for n in ordered[:cap]}
    return kept, [e for e in edges if e.source in kept and e.target in kept]


__all__ = [
    "DEFAULT_NODE_CAP",
    "READERS",
    "GraphEdge",
    "GraphNode",
    "Record",
    "build_memory_graph",
    "canonical_entity",
    "graph_config",
    "reset_config_cache",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_CONFIG_CACHE")
