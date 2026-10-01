"""The records memris keeps: entities and the statements made about them.

A **statement** is one claim — ``(subject, predicate, object | literal)`` — with the two
time axes of ADR-0115 decision 5:

- *valid time* (``valid_from`` / ``valid_to``): when the claim was true in the world.
  ``None`` means unknown, not "forever".
- *record time* (``recorded_at`` / ``retracted_at``): when memris learned it, and when
  it withdrew it.

Records are immutable values; :class:`memris.graph.MemoryGraph` produces new versions
of them and a :class:`memris.store.GraphStore` persists them. Nothing here knows a
class or predicate name — those are qualified strings from the ontology.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Literal

Status = Literal["proposed", "confirmed", "retracted"]
STATUSES: tuple[Status, ...] = ("proposed", "confirmed", "retracted")

# The one entity every install has: the person the memory belongs to. Its class comes
# from configuration (MemoryGraph.ensure_owner), never from here.
OWNER_ID = "owner"


def new_id(prefix: str) -> str:
    """A sortable, collision-safe id: ``<prefix>_<ms since epoch, hex><80 random bits>``.

    ULID-shaped without the dependency: ids sort by creation time, which keeps a table
    scan in insertion order and makes ids readable in logs.
    """
    return f"{prefix}_{int(time.time() * 1000):012x}{secrets.token_hex(10)}"


def name_key(name: str) -> str:
    """How two entity names are compared: case and runs of whitespace do not count."""
    return " ".join(name.split()).casefold()


def utc(moment: datetime) -> datetime:
    """Refuse naive datetimes: a moment without a zone cannot be ordered safely."""
    if moment.tzinfo is None:
        raise ValueError(f"naive datetime {moment.isoformat()} — pass an aware (UTC) datetime")
    return moment.astimezone(UTC)


@dataclass(frozen=True)
class Entity:
    id: str
    class_: str
    label: str
    aliases: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    merged_into: str | None = None
    # Removed by its owner (MemoryGraph.remove_entity): out of every read until restored.
    # ``removed_statements`` is what the removal withdrew — (statement id, its status
    # and reason before) — so a restore puts back exactly those and nothing else.
    removed_at: datetime | None = None
    removed_reason: str | None = None
    removed_statements: tuple[tuple[str, str, str | None], ...] = ()

    @property
    def removed(self) -> bool:
        return self.removed_at is not None

    def evolve(self, **changes: object) -> Entity:
        return replace(self, **changes)  # type: ignore[arg-type]


Decision = Literal["candidate", "same", "distinct", "undone"]


@dataclass(frozen=True)
class EntityDecision:
    """Whether two entities are one (ADR-0115 decision 4) — and the evidence for it.

    ``candidate``: they look alike; evidence (episodes both names turned up in) is
    gathering. ``same``: ``b`` was merged into ``a``. ``distinct``: someone said no;
    never asked again. ``undone``: a merge that was reversed. Nothing here is deleted.
    """

    id: str
    a: str
    b: str
    decision: Decision
    decided_at: datetime
    decided_by: str | None = None
    score: float | None = None
    evidence: tuple[str, ...] = ()
    added_aliases: tuple[str, ...] = ()
    # When the owner was asked about this pair (ADR-0115 decision 4: asked once). Kept
    # through later evidence, so an unanswered question is never asked again.
    asked_at: datetime | None = None


@dataclass(frozen=True)
class Statement:
    id: str
    subject_id: str
    predicate: str
    recorded_at: datetime
    object_id: str | None = None
    literal: str | None = None
    datatype: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    retracted_at: datetime | None = None
    status: Status = "proposed"
    confidence: float | None = None
    source_episode: str | None = None
    source_turn: str | None = None
    extractor: str | None = None
    supersedes: str | None = None
    ontology_version: str | None = None
    # How often the same claim was made again, and when last — repetition is evidence,
    # and a claim said ten times is not the same as one said once.
    reinforced: int = 1
    last_reinforced_at: datetime | None = None
    # What supports the claim (the words it came from), why it is in its current state
    # ("approved", "rejected", "refused", "forgot" …), the statement it contradicted,
    # and when an owner last reviewed it.
    evidence: str | None = None
    reason: str | None = None
    contradicts: str | None = None
    reviewed_at: datetime | None = None

    @property
    def is_literal(self) -> bool:
        return self.object_id is None

    @property
    def effective_from(self) -> datetime:
        """When the claim counts as true for an ``as_of`` query.

        An unknown start is read as "since memris learned it". Reading it as "always"
        would put a value someone moved to *today* into every past year, and "as of
        last year, where did I live?" would return both the old and the new city.
        """
        return self.valid_from or self.recorded_at

    def valid_at(self, moment: datetime) -> bool:
        return self.effective_from <= moment and (self.valid_to is None or moment < self.valid_to)

    def known_at(self, moment: datetime) -> bool:
        return self.recorded_at <= moment and (
            self.retracted_at is None or moment < self.retracted_at
        )

    def evolve(self, **changes: object) -> Statement:
        return replace(self, **changes)  # type: ignore[arg-type]


TermStatus = Literal["candidate", "alias", "active", "rejected"]
TermKind = Literal["relation", "attribute"]


@dataclass(frozen=True)
class LearnedTerm:
    """A word memory picked up on its own (ADR-0115 decision 7) — data, never YAML.

    ``candidate``: seen, counted, a few examples kept; forgotten when it stops turning
    up. ``alias``: a near-synonym of an existing term (``alias_of``); what is said with it
    is written against that term. ``active``: seen often enough, in enough episodes, to
    be a term of its own — ``name`` then reads like any declared property. ``rejected``:
    the owner said no; never learned again (a term rejected after it was active stays
    readable, so the statements made with it still resolve — ``activated_at`` says so).
    """

    name: str  # qualified, e.g. "learned:mentors"
    kind: TermKind
    label: str
    domain: str  # qualified class
    range: str  # a class for a relation, a datatype for an attribute
    status: TermStatus
    first_seen: datetime
    last_seen: datetime
    alias_of: str | None = None
    observations: int = 1
    episodes: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()
    activated_at: datetime | None = None
    decided_by: str | None = None

    def evolve(self, **changes: object) -> LearnedTerm:
        return replace(self, **changes)  # type: ignore[arg-type]


__all__ = [
    "OWNER_ID",
    "STATUSES",
    "Decision",
    "Entity",
    "EntityDecision",
    "LearnedTerm",
    "Statement",
    "Status",
    "TermKind",
    "TermStatus",
    "name_key",
    "new_id",
    "utc",
]
