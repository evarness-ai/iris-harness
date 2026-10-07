"""A SQLite GraphStore — one file, standard library only.

Tables are prefixed ``memris_`` so the store can live inside a database an application
already has (IRIS will put it in its memory database) without colliding with it.
Datetimes are stored as ISO-8601 text in UTC, which sorts and compares correctly as
text.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Collection, Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from memris.model import Entity, EntityDecision, LearnedTerm, Statement, name_key, utc

SCHEMA_VERSION = 6

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memris_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memris_entities (
    id          TEXT PRIMARY KEY,
    class       TEXT NOT NULL,
    label       TEXT NOT NULL,
    aliases     TEXT NOT NULL DEFAULT '[]',
    created_at  TEXT NOT NULL,
    merged_into TEXT,
    removed_at  TEXT,
    removed_reason TEXT,
    removed_statements TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS memris_entity_names (
    entity_id TEXT NOT NULL REFERENCES memris_entities(id) ON DELETE CASCADE,
    name_key  TEXT NOT NULL,
    PRIMARY KEY (entity_id, name_key)
);
CREATE INDEX IF NOT EXISTS memris_entity_names_key ON memris_entity_names(name_key);
CREATE TABLE IF NOT EXISTS memris_statements (
    id               TEXT PRIMARY KEY,
    subject_id       TEXT NOT NULL,
    predicate        TEXT NOT NULL,
    object_id        TEXT,
    literal          TEXT,
    datatype         TEXT,
    valid_from       TEXT,
    valid_to         TEXT,
    recorded_at      TEXT NOT NULL,
    retracted_at     TEXT,
    status           TEXT NOT NULL,
    confidence       REAL,
    source_episode   TEXT,
    source_turn      TEXT,
    extractor        TEXT,
    supersedes       TEXT,
    ontology_version TEXT,
    reinforced       INTEGER NOT NULL DEFAULT 1,
    last_reinforced_at TEXT,
    evidence         TEXT,
    reason           TEXT,
    contradicts      TEXT,
    reviewed_at      TEXT,
    CHECK ((object_id IS NULL) <> (literal IS NULL))
);
CREATE TABLE IF NOT EXISTS memris_entity_decisions (
    id            TEXT PRIMARY KEY,
    a             TEXT NOT NULL,
    b             TEXT NOT NULL,
    decision      TEXT NOT NULL,
    decided_at    TEXT NOT NULL,
    decided_by    TEXT,
    score         REAL,
    evidence      TEXT NOT NULL DEFAULT '[]',
    added_aliases TEXT NOT NULL DEFAULT '[]',
    asked_at      TEXT
);
CREATE TABLE IF NOT EXISTS memris_ontology_terms (
    name          TEXT PRIMARY KEY,
    kind          TEXT NOT NULL,
    label         TEXT NOT NULL,
    domain        TEXT NOT NULL,
    range         TEXT NOT NULL,
    status        TEXT NOT NULL,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    alias_of      TEXT,
    observations  INTEGER NOT NULL DEFAULT 1,
    episodes      TEXT NOT NULL DEFAULT '[]',
    examples      TEXT NOT NULL DEFAULT '[]',
    activated_at  TEXT,
    decided_by    TEXT
);
CREATE INDEX IF NOT EXISTS memris_entity_decisions_pair ON memris_entity_decisions(a, b);
CREATE INDEX IF NOT EXISTS memris_statements_subject ON memris_statements(subject_id, predicate);
CREATE INDEX IF NOT EXISTS memris_statements_object ON memris_statements(object_id);
CREATE INDEX IF NOT EXISTS memris_statements_predicate ON memris_statements(predicate);
CREATE INDEX IF NOT EXISTS memris_entities_merged ON memris_entities(merged_into);
CREATE INDEX IF NOT EXISTS memris_entities_class ON memris_entities(class);
CREATE INDEX IF NOT EXISTS memris_entity_decisions_b ON memris_entity_decisions(b);
"""

# Indexes only speed reads up: they change no row and no query's answer, so adding one
# needs no schema-version bump. Every open runs _SCHEMA, whose ``IF NOT EXISTS`` builds
# a missing index in place on an existing database (a one-time cost on first open),
# and a database opened by an older memris is still readable by it.

_ENTITY_COLUMNS = (
    "id, class, label, aliases, created_at, merged_into, removed_at, removed_reason, "
    "removed_statements"
)

_STATEMENT_COLUMNS = (
    "id, subject_id, predicate, object_id, literal, datatype, valid_from, valid_to, "
    "recorded_at, retracted_at, status, confidence, source_episode, source_turn, "
    "extractor, supersedes, ontology_version, reinforced, last_reinforced_at, evidence, "
    "reason, contradicts, reviewed_at"
)

# Upgrades from each older schema version, applied in order. Additive only: a column
# added with a default, so a reader of the older version keeps working.
_UPGRADES: dict[int, tuple[str, ...]] = {
    1: (
        "ALTER TABLE memris_statements ADD COLUMN reinforced INTEGER NOT NULL DEFAULT 1",
        "ALTER TABLE memris_statements ADD COLUMN last_reinforced_at TEXT",
    ),
    2: (
        "ALTER TABLE memris_statements ADD COLUMN evidence TEXT",
        "ALTER TABLE memris_statements ADD COLUMN reason TEXT",
        "ALTER TABLE memris_statements ADD COLUMN contradicts TEXT",
        "ALTER TABLE memris_statements ADD COLUMN reviewed_at TEXT",
    ),
    3: ("ALTER TABLE memris_entity_decisions ADD COLUMN asked_at TEXT",),
    # v5 adds memris_ontology_terms (learned vocabulary) — a new table, made by _SCHEMA.
    4: (),
    5: (
        "ALTER TABLE memris_entities ADD COLUMN removed_at TEXT",
        "ALTER TABLE memris_entities ADD COLUMN removed_reason TEXT",
        "ALTER TABLE memris_entities ADD COLUMN removed_statements TEXT NOT NULL DEFAULT '[]'",
    ),
}


# How long a process waits for another to finish initialising the database: the same 30 s a
# connection waits for any other lock (``_open``).
_INIT_WAIT_SECONDS = 30.0

# Ids per IN (...) query: well under SQLite's bound-parameter limit on any build.
_CHUNK = 500


def _ts(moment: datetime | None) -> str | None:
    return None if moment is None else utc(moment).isoformat()


def _dt(text: str | None) -> datetime | None:
    return None if text is None else datetime.fromisoformat(text)


class SQLiteGraphStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._connect() as conn:
            # WAL is a property of the database file: set once here, not per connection.
            self._enable_wal(conn)
            conn.executescript(_SCHEMA)
            # One initialiser at a time. Reading the version and then writing it (or upgrading
            # the tables) in a deferred transaction let two processes both read "no version" and
            # both insert it, or both upgrade; ``BEGIN IMMEDIATE`` takes the write lock first, so
            # the second process waits and then finds the work done.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT value FROM memris_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO memris_meta(key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            elif int(row[0]) > SCHEMA_VERSION:
                raise RuntimeError(
                    f"{self.path} has memris schema {row[0]}; this memris reads {SCHEMA_VERSION}"
                )
            elif int(row[0]) < SCHEMA_VERSION:
                for version in range(int(row[0]), SCHEMA_VERSION):
                    for sql in _UPGRADES[version]:
                        try:
                            conn.execute(sql)
                        except sqlite3.OperationalError as exc:
                            # A table newer than this database's version was just made
                            # by _SCHEMA with the column already in it.
                            if "duplicate column" not in str(exc):
                                raise
                conn.execute(
                    "UPDATE memris_meta SET value = ? WHERE key = 'schema_version'",
                    (str(SCHEMA_VERSION),),
                )

    @staticmethod
    def _enable_wal(conn: sqlite3.Connection) -> None:
        """``PRAGMA journal_mode = WAL``, waiting out another process doing the same.

        Switching a database's journal mode needs a lock the busy timeout does not wait for:
        a second process opening a brand-new file at the same instant gets "database is
        locked" at once. Retry for as long as the connections wait for any other lock.
        """
        deadline = time.monotonic() + _INIT_WAIT_SECONDS
        delay = 0.005
        while True:
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 0.25)

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA foreign_keys = ON")  # per connection, so on every open
        return conn

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        shared: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if shared is not None:
            # Inside session(): the same connection, still one transaction per call.
            with shared:
                yield shared
            return
        conn = self._open()
        try:
            with conn:  # one transaction: commit on success, roll back on any error
                yield conn
        finally:
            conn.close()

    @contextmanager
    def session(self) -> Iterator[None]:
        """Reuse one connection for every call this thread makes inside the block.

        A read such as ``MemoryGraph.neighbourhood`` makes dozens of store calls; without
        this each one opens (and closes) its own connection. Nothing else changes: every
        call is still its own transaction, committed when it returns, so no lock is held
        between calls. Nested sessions reuse the outer one.
        """
        if getattr(self._local, "conn", None) is not None:
            yield
            return
        conn = self._open()
        self._local.conn = conn
        try:
            yield
        finally:
            self._local.conn = None
            conn.close()

    # -- writes ------------------------------------------------------------------

    def save(
        self, *, entities: Iterable[Entity] = (), statements: Iterable[Statement] = ()
    ) -> None:
        new_entities = list(entities)
        new_statements = list(statements)
        with self._connect() as conn:
            for e in new_entities:
                conn.execute(
                    f"INSERT OR REPLACE INTO memris_entities({_ENTITY_COLUMNS}) "  # noqa: S608
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        e.id,
                        e.class_,
                        e.label,
                        json.dumps(list(e.aliases)),
                        _ts(e.created_at),
                        e.merged_into,
                        _ts(e.removed_at),
                        e.removed_reason,
                        json.dumps([list(r) for r in e.removed_statements]),
                    ),
                )
                conn.execute("DELETE FROM memris_entity_names WHERE entity_id = ?", (e.id,))
                conn.executemany(
                    "INSERT OR IGNORE INTO memris_entity_names(entity_id, name_key) VALUES (?, ?)",
                    [(e.id, name_key(n)) for n in (e.label, *e.aliases)],
                )
            conn.executemany(
                f"INSERT OR REPLACE INTO memris_statements({_STATEMENT_COLUMNS}) "  # noqa: S608
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        s.id,
                        s.subject_id,
                        s.predicate,
                        s.object_id,
                        s.literal,
                        s.datatype,
                        _ts(s.valid_from),
                        _ts(s.valid_to),
                        _ts(s.recorded_at),
                        _ts(s.retracted_at),
                        s.status,
                        s.confidence,
                        s.source_episode,
                        s.source_turn,
                        s.extractor,
                        s.supersedes,
                        s.ontology_version,
                        s.reinforced,
                        _ts(s.last_reinforced_at),
                        s.evidence,
                        s.reason,
                        s.contradicts,
                        _ts(s.reviewed_at),
                    )
                    for s in new_statements
                ],
            )

    # -- reads -------------------------------------------------------------------

    @staticmethod
    def _entity(row: Any) -> Entity:
        created = _dt(row[4])
        assert created is not None  # NOT NULL column
        return Entity(
            row[0],
            row[1],
            row[2],
            () if row[3] == "[]" else tuple(json.loads(row[3])),  # most have none: skip parse
            created,
            row[5],
            removed_at=_dt(row[6]),
            removed_reason=row[7],
            removed_statements=(
                ()
                if row[8] == "[]"
                else tuple(
                    (str(r[0]), str(r[1]), None if r[2] is None else str(r[2]))
                    for r in json.loads(row[8])
                )
            ),
        )

    @staticmethod
    def _statement(row: Any) -> Statement:
        recorded = _dt(row[8])
        assert recorded is not None  # NOT NULL column
        return Statement(
            id=str(row[0]),
            subject_id=str(row[1]),
            predicate=str(row[2]),
            object_id=row[3],
            literal=row[4],
            datatype=row[5],
            valid_from=_dt(row[6]),
            valid_to=_dt(row[7]),
            recorded_at=recorded,
            retracted_at=_dt(row[9]),
            status=row[10],
            confidence=row[11],
            source_episode=row[12],
            source_turn=row[13],
            extractor=row[14],
            supersedes=row[15],
            ontology_version=row[16],
            reinforced=int(row[17]),
            last_reinforced_at=_dt(row[18]),
            evidence=row[19],
            reason=row[20],
            contradicts=row[21],
            reviewed_at=_dt(row[22]),
        )

    def get_entity(self, entity_id: str) -> Entity | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_ENTITY_COLUMNS} FROM memris_entities WHERE id = ?",  # noqa: S608
                (entity_id,),
            ).fetchone()
        return None if row is None else self._entity(row)

    def find_entities(self, *, label: str | None = None, class_: str | None = None) -> list[Entity]:
        columns = ", ".join(f"e.{c.strip()}" for c in _ENTITY_COLUMNS.split(","))
        sql = f"SELECT DISTINCT {columns} FROM memris_entities e"  # noqa: S608
        where: list[str] = []
        args: list[str] = []
        if label is not None:
            sql += " JOIN memris_entity_names n ON n.entity_id = e.id"
            where.append("n.name_key = ?")
            args.append(name_key(label))
        if class_ is not None:
            where.append("e.class = ?")
            args.append(class_)
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY e.created_at, e.id", args).fetchall()
        return [self._entity(r) for r in rows]

    def merged_members(self, targets: Collection[str]) -> dict[str, list[str]]:
        """For each target id, the entities merged directly into it, oldest first."""
        wanted = list(dict.fromkeys(targets))
        found: dict[str, list[str]] = {}
        with self._connect() as conn:
            for start in range(0, len(wanted), _CHUNK):
                chunk = wanted[start : start + _CHUNK]
                rows = conn.execute(
                    "SELECT id, merged_into FROM memris_entities "  # noqa: S608
                    f"WHERE merged_into IN ({','.join('?' for _ in chunk)}) "
                    "ORDER BY created_at, id",
                    chunk,
                ).fetchall()
                for member, target in rows:
                    found.setdefault(target, []).append(member)
        return found

    def live_entities(self, classes: Collection[str] | None = None) -> list[Entity]:
        """Entities neither merged nor removed — of ``classes``, when given — oldest first."""
        sql = (
            f"SELECT {_ENTITY_COLUMNS} FROM memris_entities "  # noqa: S608
            "WHERE merged_into IS NULL AND removed_at IS NULL"
        )
        args: list[str] = []
        if classes is not None:
            args = list(dict.fromkeys(classes))
            if not args:
                return []
            sql += f" AND class IN ({','.join('?' for _ in args)})"
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY created_at, id", args).fetchall()
        return [self._entity(r) for r in rows]

    def get_statement(self, statement_id: str) -> Statement | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {_STATEMENT_COLUMNS} FROM memris_statements WHERE id = ?",  # noqa: S608
                (statement_id,),
            ).fetchone()
        return None if row is None else self._statement(row)

    def statements(
        self,
        *,
        subject_id: str | None = None,
        predicates: Collection[str] | None = None,
        object_id: str | None = None,
    ) -> list[Statement]:
        where: list[str] = []
        args: list[str] = []
        if subject_id is not None:
            where.append("subject_id = ?")
            args.append(subject_id)
        if predicates is not None:
            wanted = list(predicates)
            if not wanted:
                return []
            where.append(f"predicate IN ({','.join('?' for _ in wanted)})")
            args.extend(wanted)
        if object_id is not None:
            where.append("object_id = ?")
            args.append(object_id)
        sql = f"SELECT {_STATEMENT_COLUMNS} FROM memris_statements"  # noqa: S608
        if where:
            sql += " WHERE " + " AND ".join(where)
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY recorded_at, id", args).fetchall()
        return [self._statement(r) for r in rows]

    def delete_statements(self, ids: Collection[str]) -> int:
        wanted = list(ids)
        if not wanted:
            return 0
        with self._connect() as conn:
            cursor = conn.execute(
                f"DELETE FROM memris_statements WHERE id IN ({','.join('?' for _ in wanted)})",  # noqa: S608
                wanted,
            )
            return cursor.rowcount

    def delete_entity(self, entity_id: str) -> bool:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM memris_entity_decisions WHERE a = ? OR b = ?", (entity_id, entity_id)
            )
            cursor = conn.execute("DELETE FROM memris_entities WHERE id = ?", (entity_id,))
            return cursor.rowcount == 1

    def save_decision(self, decision: EntityDecision) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memris_entity_decisions(id, a, b, decision, decided_at, "
                "decided_by, score, evidence, added_aliases, asked_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    decision.id,
                    decision.a,
                    decision.b,
                    decision.decision,
                    _ts(decision.decided_at),
                    decision.decided_by,
                    decision.score,
                    json.dumps(list(decision.evidence)),
                    json.dumps(list(decision.added_aliases)),
                    _ts(decision.asked_at),
                ),
            )

    def decisions(self, entity_id: str | None = None) -> list[EntityDecision]:
        sql = (
            "SELECT id, a, b, decision, decided_at, decided_by, score, evidence, added_aliases, "
            "asked_at "
            "FROM memris_entity_decisions"
        )
        args: tuple[str, ...] = ()
        if entity_id is not None:
            sql += " WHERE a = ? OR b = ?"
            args = (entity_id, entity_id)
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY decided_at, id", args).fetchall()
        return [
            EntityDecision(
                r[0],
                r[1],
                r[2],
                r[3],
                _dt(r[4]) or datetime.min,
                r[5],
                r[6],
                tuple(json.loads(r[7])),
                tuple(json.loads(r[8])),
                _dt(r[9]),
            )
            for r in rows
        ]

    # -- learned vocabulary (decision 7) ---------------------------------------------

    _TERM_COLUMNS = (
        "name, kind, label, domain, range, status, first_seen, last_seen, alias_of, "
        "observations, episodes, examples, activated_at, decided_by"
    )

    def save_term(self, term: LearnedTerm) -> None:
        with self._connect() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO memris_ontology_terms({self._TERM_COLUMNS}) "  # noqa: S608
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    term.name,
                    term.kind,
                    term.label,
                    term.domain,
                    term.range,
                    term.status,
                    _ts(term.first_seen),
                    _ts(term.last_seen),
                    term.alias_of,
                    term.observations,
                    json.dumps(list(term.episodes)),
                    json.dumps(list(term.examples)),
                    _ts(term.activated_at),
                    term.decided_by,
                ),
            )

    @staticmethod
    def _term(r: Any) -> LearnedTerm:
        return LearnedTerm(
            name=r[0],
            kind=r[1],
            label=r[2],
            domain=r[3],
            range=r[4],
            status=r[5],
            first_seen=_dt(r[6]) or datetime.min,
            last_seen=_dt(r[7]) or datetime.min,
            alias_of=r[8],
            observations=int(r[9]),
            episodes=tuple(json.loads(r[10])),
            examples=tuple(json.loads(r[11])),
            activated_at=_dt(r[12]),
            decided_by=r[13],
        )

    def get_term(self, name: str) -> LearnedTerm | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {self._TERM_COLUMNS} FROM memris_ontology_terms WHERE name = ?",  # noqa: S608
                (name,),
            ).fetchone()
        return None if row is None else self._term(row)

    def terms(self, status: str | None = None) -> list[LearnedTerm]:
        sql = f"SELECT {self._TERM_COLUMNS} FROM memris_ontology_terms"  # noqa: S608
        args: tuple[str, ...] = ()
        if status is not None:
            sql += " WHERE status = ?"
            args = (status,)
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY first_seen, name", args).fetchall()
        return [self._term(r) for r in rows]

    def delete_term(self, name: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memris_ontology_terms WHERE name = ?", (name,))
            return cursor.rowcount == 1

    def get_meta(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM memris_meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def claim_meta(self, key: str, value: str) -> bool:
        """Set ``key`` only if nobody has: an atomic claim two processes cannot both win."""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO memris_meta(key, value) VALUES (?, ?)", (key, value)
            )
            return cursor.rowcount == 1

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO memris_meta(key, value) VALUES (?, ?)", (key, value)
            )

    def used_terms(self) -> set[str]:
        with self._connect() as conn:
            predicates = {
                r[0] for r in conn.execute("SELECT DISTINCT predicate FROM memris_statements")
            }
            classes = {r[0] for r in conn.execute("SELECT DISTINCT class FROM memris_entities")}
        return predicates | classes


__all__ = ["SCHEMA_VERSION", "SQLiteGraphStore"]
