"""Removing things from memory, reversibly (ADR-0119).

What the owner removes leaves the Map, recall and the prompt together; the Map never
shows a cleaner memory than the one the agent uses. Four kinds of thing can go:

- **entity**: a memris entity. Every claim that still holds about it is withdrawn with
  it (``MemoryGraph.remove_entity``), and its name stays suppressed so a summary
  mention cannot draw it back.
- **name**: something a summary mentions that is not an entity. Its folded name is
  suppressed in every summary, past and future. Summaries are never edited.
- **session**: a conversation. It leaves the Map, cross-session recall and the chat
  list; its turns and summary stay until deleted for good.
- **fact**: one fact value, through the same forget the About-you tab uses (retracted,
  reason ``forgot``). Every forgotten fact is listed here, so each is restorable.

Restore undoes a removal exactly. **Delete permanently** works only on removed things
and means gone and staying gone: a deleted entity's name stays suppressed, and a
suppressed name has no permanent delete of its own — restoring it is how it is lifted.

The ledger lives in ``memory_removals`` (``MemoryStore``); an entity's own mark lives in
memris.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from iris_harness.foundation.clock import utc_now_iso
from iris_harness.memory.store import MemoryStore
from memris.graph import MemoryGraph, StatementError
from memris.model import OWNER_ID, Statement, name_key, new_id

logger = logging.getLogger(__name__)

Kind = Literal["entity", "name", "session", "fact"]
KINDS: tuple[Kind, ...] = ("entity", "name", "session", "fact")
# The same kinds, by role — the Map tags each removable node with one (graph.py may
# not spell them: "name" is also an ontology term, and it knows no vocabulary).
ENTITY, NAME, SESSION, FACT = KINDS
# The reason a forget records on a fact (FactStatements.forget) — what lists it here.
FORGOT = "forgot"
DELETE_WORD = "delete"


class RemovalError(ValueError):
    """A removal, restore or delete that cannot be done as asked."""


class NotFoundError(RemovalError):
    """Nothing by that id or name."""


@dataclass(frozen=True)
class Target:
    kind: Kind
    id: str

    @classmethod
    def of(cls, raw: dict[str, Any]) -> Target:
        kind = str(raw.get("kind") or "")
        target_id = str(raw.get("id") or "").strip()
        if kind not in KINDS:
            raise RemovalError(f"kind is one of {', '.join(KINDS)}, not {kind!r}")
        if not target_id:
            raise RemovalError("a target needs an id")
        return cls(kind, target_id)

    def as_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "id": self.id}


def fold(name: str) -> str:
    """How a suppressed name is compared: the Map's folding, then case and spaces."""
    from iris_harness.memory.graph import canonical_entity  # graph imports us

    return name_key(canonical_entity(name))


class MemoryRemoval:
    """The Removed list and every change to it.

    ``rederive`` rebuilds a fact key's derived homes (recall index, USER.md) after its
    statements changed — ``FactCoordinator.rederive``. ``purge_session`` deletes a
    session's turns, their vectors and its summary for good; without one, turns and
    summary are deleted and the vectors are left to ``iris memory doctor``.
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        rederive: Callable[[str], None] | None = None,
        purge_session: Callable[[str], None] | None = None,
    ) -> None:
        self.store = store
        self._rederive = rederive
        self._purge_session = purge_session

    @property
    def graph(self) -> MemoryGraph:
        return self.store.memory_graph()

    # -- describing ------------------------------------------------------------

    def _text(self, s: Statement) -> str:
        facts = self.store.fact_statements()
        key = facts.key_of(s) or s.predicate.partition(":")[2] or s.predicate
        line = f"{key}: {facts.value_of(s)}"
        if s.subject_id == OWNER_ID:
            return line
        subject = self.graph.get_entity(s.subject_id)
        return f"{subject.label if subject else s.subject_id} · {line}"

    def _cascade(self, statement_ids: Iterable[str]) -> list[dict[str, str]]:
        out = []
        for statement_id in statement_ids:
            s = self.graph.store.get_statement(statement_id)
            if s is not None:
                out.append({"statement_id": s.id, "text": self._text(s)})
        return out

    def _live_about(self, entity_id: str) -> list[Statement]:
        graph = self.graph
        now = datetime.now(UTC)
        found: dict[str, Statement] = {}
        for member in graph.members(entity_id):
            for s in (
                *graph.store.statements(subject_id=member),
                *graph.store.statements(object_id=member),
            ):
                if s.status == "proposed" or (s.status == "confirmed" and s.valid_at(now)):
                    found[s.id] = s
        return sorted(found.values(), key=lambda s: (s.recorded_at, s.id))

    def _item(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "kind": row["kind"],
            "label": row["label"],
            "removed_at": row["removed_at"],
            "cascade": row["cascade"],
            "permanent": row["permanent"],
        }

    def _forgotten(self) -> list[Statement]:
        """Forgotten owner facts whose claim is not held again by a later statement."""
        graph = self.graph
        held = {
            (s.predicate, s.object_id, s.literal)
            for s in graph.current(OWNER_ID, include_proposed=True)
        }
        facts = self.store.fact_statements()
        return [
            s
            for s in graph.store.statements(subject_id=OWNER_ID)
            if s.status == "retracted"
            and s.reason == FORGOT
            and facts.key_of(s) is not None
            and (s.predicate, s.object_id, s.literal) not in held
        ]

    def _fact_item(self, s: Statement) -> dict[str, Any]:
        text = self._text(s)
        moment = s.retracted_at or s.recorded_at
        return {
            "id": s.id,
            "kind": "fact",
            "label": text,
            "removed_at": moment.isoformat(),
            "cascade": [{"statement_id": s.id, "text": text}],
            "permanent": False,
        }

    # -- reads -------------------------------------------------------------------

    def items(self) -> list[dict[str, Any]]:
        """The Removed list, newest first: every removal, and every forgotten fact."""
        found = [self._item(r) for r in self.store.removals()]
        found += [self._fact_item(s) for s in self._forgotten()]
        return sorted(found, key=lambda i: (str(i["removed_at"]), str(i["id"])), reverse=True)

    def _row_for(self, kind: str, target_id: str) -> dict[str, Any] | None:
        return next(
            (r for r in self.store.removals() if r["kind"] == kind and r["target_id"] == target_id),
            None,
        )

    # -- preview -----------------------------------------------------------------

    def preview(self, targets: Iterable[Target]) -> list[dict[str, Any]]:
        """What removing each target would do, without doing it."""
        return [self._effect(t) for t in targets]

    def _entity(self, target_id: str) -> Any:
        graph = self.graph
        entity = graph.get_entity(graph.canonical_id(target_id))
        if entity is None or entity.id == OWNER_ID:
            raise NotFoundError(f"no entity '{target_id}'")
        return entity

    def _statement(self, target_id: str) -> Statement:
        s = self.graph.store.get_statement(target_id)
        if s is None or self.store.fact_statements().key_of(s) is None:
            raise NotFoundError(f"no fact '{target_id}'")
        return s

    def _session_exists(self, session_id: str) -> bool:
        return bool(
            self.store.load_conversation_summary(session_id)
            or self.store.fetch_turn_ids(session_id)
        )

    def _effect(self, target: Target) -> dict[str, Any]:
        lines: list[str]
        if target.kind == "entity":
            entity = self._entity(target.id)
            live = self._live_about(entity.id)
            texts = [self._text(s) for s in live]
            lines = (
                [f"forgets {len(texts)} fact{'s' if len(texts) != 1 else ''}: {'; '.join(texts)}"]
                if texts
                else ["no live facts"]
            )
            lines.append(f"hides “{entity.label}” in every summary")
            label = entity.label
        elif target.kind == "name":
            label = target.id
            lines = [f"hides “{fold(target.id)}” in every summary, past and future"]
        elif target.kind == "session":
            if not self._session_exists(target.id):
                raise NotFoundError(f"no session '{target.id}'")
            label = target.id
            lines = ["leaves the Map, recall and the chat list"]
        else:
            s = self._statement(target.id)
            label = self._text(s)
            lines = ["forgets this fact (the same as Forget on About you)"]
        return {"target": target.as_dict(), "label": label, "lines": lines}

    # -- remove ------------------------------------------------------------------

    def remove(self, targets: Iterable[Target]) -> list[dict[str, Any]]:
        """Remove each target; checked first, so an unknown one removes nothing."""
        wanted = list(targets)
        for target in wanted:
            self._effect(target)
        return [self._remove(t) for t in wanted]

    def _ledger(
        self, kind: str, target_id: str, label: str, key: str | None, cascade: list[dict[str, str]]
    ) -> dict[str, Any]:
        existing = self._row_for(kind, target_id)
        if existing is not None:
            return self._item(existing)
        row = {
            "id": new_id("rm"),
            "kind": kind,
            "target_id": target_id,
            "label": label,
            "name_key": key,
            "removed_at": utc_now_iso(),
            "cascade": cascade,
            "permanent": False,
        }
        self.store.add_removal(row)
        return self._item(row)

    def _remove(self, target: Target) -> dict[str, Any]:
        if target.kind == "entity":
            entity = self._entity(target.id)
            existing = self._row_for("entity", entity.id)
            if existing is not None:
                return self._item(existing)
            removed = self.graph.remove_entity(entity.id)
            withdrawn = [r[0] for r in removed.removed_statements]
            self._rederive_statements(withdrawn)
            return self._ledger(
                "entity", entity.id, entity.label, fold(entity.label), self._cascade(withdrawn)
            )
        if target.kind == "name":
            key = fold(target.id)
            existing = self._row_for("name", key)
            if existing is not None:
                return self._item(existing)
            return self._ledger("name", key, target.id, key, [])
        if target.kind == "session":
            return self._ledger("session", target.id, target.id, None, [])
        s = self._statement(target.id)
        fact_key = self.store.fact_statements().key_of(s)
        if s.status != "retracted":
            if s.subject_id != OWNER_ID or fact_key is None:
                raise RemovalError("only a fact about you is removed this way")
            self.store.delete_user_fact(fact_key, statement_id=s.id)
            if self._rederive is not None:
                self._rederive(fact_key)
        after = self.graph.store.get_statement(s.id) or s
        return self._fact_item(after)

    def _rederive_statements(self, statement_ids: Iterable[str]) -> None:
        if self._rederive is None:
            return
        facts = self.store.fact_statements()
        keys = set()
        for statement_id in statement_ids:
            s = self.graph.store.get_statement(statement_id)
            if s is not None and s.subject_id == OWNER_ID and (key := facts.key_of(s)):
                keys.add(key)
        for key in sorted(keys):
            self._rederive(key)

    # -- restore -----------------------------------------------------------------

    def restore(self, removal_id: str) -> dict[str, Any]:
        """Undo one removal exactly. A permanent row's restore lifts its suppression."""
        row = self.store.get_removal(removal_id)
        if row is not None:
            item = self._item(row)
            if row["kind"] == "entity" and not row["permanent"]:
                self.graph.restore_entity(row["target_id"])
                self._rederive_statements(c["statement_id"] for c in row["cascade"])
            self.store.delete_removal(removal_id)
            return item
        s = self.graph.store.get_statement(removal_id)
        if s is None or s not in self._forgotten():
            raise NotFoundError(f"'{removal_id}' is not in the Removed list")
        item = self._fact_item(s)
        try:
            self.graph.reinstate(s.id, status="confirmed", reason="restored")
        except StatementError as exc:
            raise RemovalError(str(exc)) from exc
        key = self.store.fact_statements().key_of(s)
        if key is not None and self._rederive is not None:
            self._rederive(key)
        return item

    # -- delete for good ---------------------------------------------------------

    def delete(self, ids: Iterable[str], *, confirm: str) -> dict[str, Any]:
        """Delete removed things for good. ``confirm`` must be the word ``delete``."""
        if confirm.strip().lower() != DELETE_WORD:
            raise RemovalError(f"type '{DELETE_WORD}' to confirm a permanent delete")
        deleted: list[str] = []
        refused: list[dict[str, str]] = []
        for removal_id in dict.fromkeys(ids):
            why = self._delete_one(removal_id)
            if why is None:
                deleted.append(removal_id)
            else:
                refused.append({"id": removal_id, "reason": why})
        return {"deleted": deleted, "refused": refused}

    def _delete_one(self, removal_id: str) -> str | None:
        row = self.store.get_removal(removal_id)
        if row is None:
            s = self.graph.store.get_statement(removal_id)
            if s is None or s not in self._forgotten():
                return "not in the Removed list"
            try:
                self.graph.purge([s.id])
            except StatementError as exc:
                return str(exc)
            return None
        if row["permanent"]:
            return "already deleted"
        if row["kind"] == "name":
            return "a hidden name has no permanent delete; restore it to show it again"
        if row["kind"] == "entity":
            try:
                self.graph.delete_entity(row["target_id"])
            except StatementError as exc:
                return str(exc)
        elif row["kind"] == "session":
            if self._purge_session is not None:
                self._purge_session(row["target_id"])
            else:
                self.store.delete_session_turns(row["target_id"])
                self.store.delete_conversation_summary(row["target_id"])
        self.store.mark_removal_permanent(removal_id)
        return None


def suppressed_keys(store: MemoryStore) -> set[str]:
    """Every folded name the Map must not draw from a summary: suppressed names, and the
    names and aliases of removed entities (a deleted one keeps its ledger row)."""
    keys = set(store.suppressed_name_keys())
    try:
        for entity in store.memory_graph().store.find_entities():
            if entity.removed:
                keys |= {fold(n) for n in (entity.label, *entity.aliases)}
    except Exception:  # a Map without statements still draws
        logger.warning(
            "removal: removed entities unreadable; their names are not suppressed",
            exc_info=True,
        )
    return keys


__all__ = [
    "DELETE_WORD",
    "ENTITY",
    "FACT",
    "KINDS",
    "NAME",
    "SESSION",
    "MemoryRemoval",
    "NotFoundError",
    "RemovalError",
    "Target",
    "fold",
    "suppressed_keys",
]
