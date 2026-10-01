"""Plan the move of user facts onto memris statements (memris plan PR 2a; ADR-0115).

This module only PLANS and REPORTS. It opens the legacy database read-only and writes
nothing to it; applying the plan is PR 2b. The owner reads the report on their real
data first — the retention work showed why (an age-only rule would have deleted 864
turns).

What becomes of each ``user_facts`` row (owner decisions, 2026-09-18):

- **map**: its key has a mapping in ``config/memory/mappings.yaml`` → one current
  statement under that property.
- **keep-as**: off the allowlist, but the owner rescued it in the triage file by
  naming an allowed key → mapped under that key's property.
- **drop**: off the allowlist and not rescued — the default. No statement is made. The
  legacy row is never deleted (PR 2b keeps the old tables as an archive), so a drop is
  undone by editing the triage file and running the migration again.
- **collision**: two keys map onto one single-valued property (``employer`` and
  ``organization`` both mean ``works_at``). The most recently confirmed value stays current; the other is
  closed at migration time, visible in the report.

A key's history becomes its timeline: each earlier value is a closed statement
(superseded when the value changed, retracted when it was forgotten). Keys that live
only in history — forgotten facts — become retracted statements when they map, and
stay in the archive when they do not. Every row is accounted for, and the report
says so, or says what does not add up.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml

from iris_harness.memory.ontology import fact_mappings, normalise_fact_key
from memris.ontology import Ontology

Outcome = Literal["map", "keep-as", "drop", "collision", "history-only", "archive"]
DROP = "drop"
# A key YAML reads back unchanged without quotes; anything else is written quoted.
_PLAIN_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
# ...and not a word YAML 1.1 reads as a boolean or null.
_YAML_WORDS = frozenset({"y", "n", "yes", "no", "on", "off", "true", "false", "null", "none", "~"})


# --------------------------------------------------------------------------- inputs


@dataclass(frozen=True)
class LegacyFact:
    key: str
    value: str
    confidence: float
    source: str
    first_seen: datetime
    last_confirmed: datetime
    times_confirmed: int
    confirmed: bool


@dataclass(frozen=True)
class LegacyHistory:
    id: int
    key: str
    old_value: str | None
    new_value: str | None
    new_confidence: float | None
    source: str
    reason: str
    changed_at: datetime


@dataclass(frozen=True)
class LegacyProposal:
    key: str
    value: str
    confidence: float
    source: str
    evidence: str
    created_at: datetime
    seen_count: int


@dataclass(frozen=True)
class LegacyContradiction:
    key: str
    stored_value: str
    incoming_value: str
    incoming_confidence: float | None
    resolution: str  # superseded | blocked
    source: str
    detected_at: datetime
    acknowledged: bool
    seen_count: int


@dataclass(frozen=True)
class LegacyData:
    facts: list[LegacyFact]
    history: list[LegacyHistory]
    contradictions: int
    pending_proposals: int
    proposals: list[LegacyProposal] = field(default_factory=list)
    conflicts: list[LegacyContradiction] = field(default_factory=list)


def _dt(text: str) -> datetime:
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def open_read_only(db_path: Path) -> sqlite3.Connection:
    """A connection SQLite itself refuses to write through — not just one we don't write to."""
    return sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)


def read_legacy(db_path: Path) -> LegacyData:
    """Read the legacy fact tables through a read-only connection."""
    conn = open_read_only(db_path)
    try:
        facts = [
            LegacyFact(r[0], r[1], float(r[2]), r[3], _dt(r[4]), _dt(r[5]), int(r[6]), bool(r[7]))
            for r in conn.execute(
                "SELECT key, value, confidence, source, first_seen, last_confirmed, "
                "times_confirmed, confirmed FROM user_facts ORDER BY key"
            )
        ]
        history = [
            LegacyHistory(r[0], r[1], r[2], r[3], r[4], r[5], r[6], _dt(r[7]))
            for r in conn.execute(
                "SELECT id, key, old_value, new_value, new_confidence, source, reason, "
                "changed_at FROM user_fact_history ORDER BY id"
            )
        ]
        contradictions = conn.execute("SELECT COUNT(*) FROM user_fact_contradictions").fetchone()[0]
        proposals = [
            LegacyProposal(r[0], r[1], float(r[2]), r[3], r[4] or "", _dt(r[5]), int(r[6]))
            for r in conn.execute(
                "SELECT key, value, confidence, source, evidence, created_at, seen_count "
                "FROM fact_proposals WHERE status = 'pending' ORDER BY id"
            )
        ]
        conflicts = [
            LegacyContradiction(
                r[0], r[1], r[2], r[3], r[4], r[5], _dt(r[6]), bool(r[7]), int(r[8])
            )
            for r in conn.execute(
                "SELECT key, stored_value, incoming_value, incoming_confidence, resolution, "
                "source, detected_at, acknowledged, seen_count FROM user_fact_contradictions "
                "ORDER BY id"
            )
        ]
    finally:
        conn.close()
    return LegacyData(facts, history, int(contradictions), len(proposals), proposals, conflicts)


# --------------------------------------------------------------------------- triage


class TriageError(ValueError):
    """The triage file says something the migration cannot do."""


def load_triage(path: Path, mappable: Iterable[str]) -> dict[str, str]:
    """``{key: "drop" | <allowed key>}`` from the owner's triage file (missing = empty).

    Each entry is ``drop`` or ``{keep: <key>}``; the kept key must be one the mappings
    know, or the rescue would have nowhere to go.
    """
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("facts") or {}
    if not isinstance(entries, dict):
        raise TriageError(f"{path}: 'facts' must be a mapping of key → drop | {{keep: <key>}}")
    known = set(mappable)
    decisions: dict[str, str] = {}
    for key, decision in entries.items():
        if decision == DROP:
            decisions[str(key)] = DROP
            continue
        target = decision.get("keep") if isinstance(decision, dict) else None
        if not isinstance(target, str):
            raise TriageError(
                f"{path}: '{key}' must be 'drop' or {{keep: <key>}}, got {decision!r}"
            )
        if normalise_fact_key(target) not in known:
            raise TriageError(f"{path}: '{key}' keeps as '{target}', which no mapping knows")
        decisions[str(key)] = normalise_fact_key(target)
    return decisions


def render_triage(dropped: list[LegacyFact], existing: Mapping[str, str]) -> str:
    """The triage file: every off-allowlist fact, ``drop`` unless the owner chose."""
    lines = [
        "# Facts on keys outside the allowlist (memris plan PR 2; ADR-0115).",
        "#",
        "# Each is dropped by default: no statement is made, the old row stays in the",
        "# archive, and nothing reaches a prompt. To keep one, replace `drop` with",
        "#   {keep: <allowed key>}      e.g.  to-do: {keep: project}",
        "# then run `iris memory migrate-statements` again. Your edits are never overwritten.",
        "",
        "facts:",
    ]
    for fact in dropped:
        chosen = existing.get(fact.key, DROP)
        decision = DROP if chosen == DROP else f"{{keep: {chosen}}}"
        value = " ".join(fact.value.split())
        shown = value if len(value) <= 60 else value[:57] + "…"
        plain = _PLAIN_KEY.match(fact.key) and fact.key.lower() not in _YAML_WORDS
        key = fact.key if plain else json.dumps(fact.key)
        lines.append(f"  {key}: {decision}   # {shown!r}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- plan


@dataclass(frozen=True)
class PlannedStatement:
    predicate: str
    value: str
    object_class: str | None  # None → a literal
    status: Literal["confirmed", "proposed", "retracted"]
    recorded_at: datetime
    valid_to: datetime | None = None
    retracted_at: datetime | None = None
    confidence: float | None = None
    source: str | None = None
    times_confirmed: int = 1
    last_confirmed: datetime | None = None
    reason: str | None = None  # forgot / corrected / restored — read back by the history view

    @property
    def current(self) -> bool:
        return self.status != "retracted" and self.valid_to is None


@dataclass
class KeyPlan:
    key: str
    outcome: Outcome
    predicate: str | None = None
    kept_as: str | None = None
    current_value: str | None = None
    statements: list[PlannedStatement] = field(default_factory=list)
    history_rows: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class MigrationPlan:
    keys: list[KeyPlan]
    fact_rows: int
    history_rows: int
    contradictions: int
    pending_proposals: int
    migrated_at: datetime
    issues: list[str] = field(default_factory=list)

    def by_outcome(self, outcome: Outcome) -> list[KeyPlan]:
        return [k for k in self.keys if k.outcome == outcome]

    @property
    def statements(self) -> list[PlannedStatement]:
        return [s for k in self.keys for s in k.statements]

    def accounting(self) -> list[str]:
        """Why the numbers do not add up — empty when every row has one outcome."""
        problems = []
        row_outcomes = {"map", "keep-as", "drop", "collision"}
        with_row = [k for k in self.keys if k.outcome in row_outcomes]
        if len(with_row) != self.fact_rows:
            problems.append(f"{self.fact_rows} fact rows but {len(with_row)} planned")
        counted = sum(k.history_rows for k in self.keys)
        if counted != self.history_rows:
            problems.append(f"{self.history_rows} history rows but {counted} planned")
        return problems


def _timeline(
    row: LegacyFact | None, events: list[LegacyHistory], notes: list[str]
) -> list[dict[str, Any]]:
    """The values a key held, in order, each with how it began and ended."""
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for event in events:
        if current is None and event.old_value is not None:
            # history starts mid-life: the value before the first recorded change
            current = {
                "value": event.old_value,
                "start": event.changed_at,
                "confidence": None,
                "source": None,
            }
            notes.append("history begins mid-life; the first value's start is its first change")
        if event.new_value is None:  # forget / junk removal
            if current is not None:
                segments.append({**current, "end": event.changed_at, "ended_by": "retracted"})
            current = None
            continue
        if current is not None and current["value"] == event.new_value:
            continue  # a re-confirmation, not a change
        if current is not None:
            segments.append({**current, "end": event.changed_at, "ended_by": "superseded"})
        current = {
            "value": event.new_value,
            "start": event.changed_at,
            "confidence": event.new_confidence,
            "source": event.source,
            "began": event.reason,
        }
    if row is not None:
        if current is None:
            current = {
                "value": row.value,
                "start": row.first_seen,
                "confidence": row.confidence,
                "source": row.source,
            }
        elif current["value"] != row.value:
            notes.append(
                f"history ends at {current['value']!r} but the stored value is {row.value!r}; the stored value wins"
            )
            segments.append({**current, "end": row.last_confirmed, "ended_by": "superseded"})
            current = {
                "value": row.value,
                "start": row.last_confirmed,
                "confidence": row.confidence,
                "source": row.source,
            }
        segments.append({**current, "end": None, "ended_by": None})
    elif current is not None:
        notes.append(
            f"history leaves {current['value']!r} current but no row holds it; kept as retracted"
        )
        segments.append({**current, "end": current["start"], "ended_by": "retracted"})
    return segments


def _statements(
    predicate: str, object_class: str | None, row: LegacyFact | None, segments: list[dict[str, Any]]
) -> list[PlannedStatement]:
    out = []
    for seg in segments:
        live = seg["ended_by"] is None
        if seg["ended_by"] == "retracted":
            status: Literal["confirmed", "proposed", "retracted"] = "retracted"
        elif live and row is not None and not row.confirmed:
            status = "proposed"
        else:
            status = "confirmed"  # a superseded value was the store's belief while it held
        out.append(
            PlannedStatement(
                predicate=predicate,
                value=seg["value"],
                object_class=object_class,
                status=status,
                recorded_at=seg["start"],
                valid_to=seg["end"] if seg["ended_by"] == "superseded" else None,
                retracted_at=seg["end"] if seg["ended_by"] == "retracted" else None,
                confidence=(row.confidence if live and row else seg["confidence"]),
                source=(row.source if live and row else seg["source"]),
                times_confirmed=(row.times_confirmed if live and row else 1),
                last_confirmed=(row.last_confirmed if live and row else None),
                reason=(
                    "forgot"
                    if seg["ended_by"] == "retracted"
                    else _BEGAN_AS.get(seg.get("began") or "")
                ),
            )
        )
    return out


# How a legacy history row that started a value maps onto the statement's reason.
_BEGAN_AS = {"correct": "corrected", "restore": "restored"}


def plan_migration(
    data: LegacyData,
    ontology: Ontology,
    triage: Mapping[str, str],
    *,
    now: datetime | None = None,
) -> MigrationPlan:
    migrated_at = now or datetime.now(UTC)
    mappings = fact_mappings(ontology)
    rows = {f.key: f for f in data.facts}
    events: dict[str, list[LegacyHistory]] = {}
    for event in data.history:
        events.setdefault(event.key, []).append(event)

    plans: list[KeyPlan] = []
    for key in sorted(set(rows) | set(events)):
        row = rows.get(key)
        key_events = events.get(key, [])
        norm = normalise_fact_key(key)
        plan = KeyPlan(
            key=key,
            outcome="map",
            history_rows=len(key_events),
            current_value=row.value if row else None,
        )
        target = norm if norm in mappings else None
        if target is None and row is not None:
            decision = triage.get(key, DROP)
            if decision == DROP:
                plan.outcome = "drop"
                plans.append(plan)
                continue
            target, plan.outcome, plan.kept_as = decision, "keep-as", decision
        if target is None:  # history only, and off the allowlist
            plan.outcome = "archive"
            plans.append(plan)
            continue
        if row is None:
            plan.outcome = "history-only"
        rule = mappings[target]
        plan.predicate = rule.predicate
        segments = _timeline(row, key_events, plan.notes)
        plan.statements = _statements(rule.predicate, rule.object_class, row, segments)
        plans.append(plan)

    issues = _resolve_collisions(plans, ontology, rows, migrated_at)
    result = MigrationPlan(
        keys=plans,
        fact_rows=len(data.facts),
        history_rows=len(data.history),
        contradictions=data.contradictions,
        pending_proposals=data.pending_proposals,
        migrated_at=migrated_at,
        issues=issues,
    )
    result.issues.extend(result.accounting())
    return result


def _single_valued(ontology: Ontology, predicate: str) -> bool:
    owner = ontology.owner_class
    if owner is None:
        return False
    for cls in ontology.ancestors(owner):
        constraint = ontology.shapes.get(cls, {}).get(predicate)
        if constraint is not None:
            return constraint.max_count == 1
    return False


def _resolve_collisions(
    plans: list[KeyPlan], ontology: Ontology, rows: Mapping[str, LegacyFact], migrated_at: datetime
) -> list[str]:
    issues: list[str] = []
    by_predicate: dict[str, list[KeyPlan]] = {}
    for plan in plans:
        if plan.predicate and plan.key in rows and any(s.current for s in plan.statements):
            by_predicate.setdefault(plan.predicate, []).append(plan)
    for predicate, group in by_predicate.items():
        values = {rows[p.key].value for p in group}
        if len(group) < 2 or len(values) < 2 or not _single_valued(ontology, predicate):
            continue
        group.sort(key=lambda p: rows[p.key].last_confirmed)
        winner = group[-1]
        for loser in group[:-1]:
            loser.outcome = "collision"
            loser.notes.append(
                f"shares single-valued {predicate} with '{winner.key}'; the newer "
                f"'{winner.key}' stays current and this value closes at migration"
            )
            loser.statements = [
                replace(s, valid_to=migrated_at) if s.current else s for s in loser.statements
            ]
        issues.append(
            f"collision on {predicate}: {', '.join(p.key for p in group)} — '{winner.key}' kept current"
        )
    return issues


# --------------------------------------------------------------------------- report


def _cell(text: str | None, width: int = 48) -> str:
    if text is None:
        return "—"
    flat = " ".join(str(text).split()).replace("|", "\\|")
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def render_report(
    plan: MigrationPlan,
    *,
    db_path: Path,
    triage_path: Path,
    backup: Path | None = None,
    written: int | None = None,
) -> str:
    """The migration report — a dry run's, or, given ``backup``, the applied migration's.

    An applied report says what was written and where the backup is; it once carried the
    dry run's "opened read-only; nothing was written" line, the opposite of the truth.
    """
    statements = plan.statements
    current = [s for s in statements if s.current]
    applied = backup is not None
    when = plan.migrated_at.isoformat(timespec="seconds")
    if applied:
        head = [
            "# Fact → statement migration: applied",
            "",
            f"Database: `{db_path}` — migrated; {written} statements written.  ",
            f"Backup taken first: `{backup}`  ",
            f"Migrated at: {when}  ",
        ]
    else:
        head = [
            "# Fact → statement migration: dry run",
            "",
            f"Database: `{db_path}` (opened read-only; nothing was written).  ",
            f"Planned at: {when}  ",
        ]
    lines = [
        *head,
        f"Triage file: `{triage_path}`",
        "",
        "## Summary",
        "",
        "| | Count |",
        "|---|---|",
        f"| fact rows | {plan.fact_rows} |",
        f"| → mapped | {len(plan.by_outcome('map'))} |",
        f"| → kept by your triage | {len(plan.by_outcome('keep-as'))} |",
        f"| → dropped (triage default; row stays in the archive) | {len(plan.by_outcome('drop'))} |",
        f"| → collision (closed at migration) | {len(plan.by_outcome('collision'))} |",
        f"| history-only keys that map (become retracted statements) | {len(plan.by_outcome('history-only'))} |",
        f"| history-only keys off the allowlist (archive only) | {len(plan.by_outcome('archive'))} |",
        f"| history rows | {plan.history_rows} |",
        f"| statements planned | {len(statements)} ({len(current)} current) |",
        f"| contradictions (kept as the legacy log, not migrated) | {plan.contradictions} |",
        f"| pending proposals | {plan.pending_proposals} |",
        *(
            [f"| statements written (with proposals and conflicts) | {written} |"]
            if applied
            else []
        ),
        "",
    ]
    lines += ["## Checks", ""]
    if plan.issues:
        lines += [f"- ⚠ {issue}" for issue in plan.issues]
    else:
        lines.append("- ✓ every fact row and every history row has exactly one outcome")
    lines.append("")

    def table(title: str, keys: list[KeyPlan], note: str = "") -> None:
        if not keys:
            return
        lines.extend([f"## {title} ({len(keys)})", ""])
        if note:
            lines.extend([note, ""])
        lines.extend(
            ["| key | value now | becomes | earlier values | notes |", "|---|---|---|---|---|"]
        )
        for k in keys:
            earlier = [s for s in k.statements if not s.current]
            shape = (
                f"`{k.predicate}`" + (f" (as `{k.kept_as}`)" if k.kept_as else "")
                if k.predicate
                else "—"
            )
            past = (
                f"{sum(s.status != 'retracted' for s in earlier)} closed, "
                f"{sum(s.status == 'retracted' for s in earlier)} retracted"
                if earlier
                else "—"
            )
            lines.append(
                f"| `{k.key}` | {_cell(k.current_value)} | {shape} | {past} | {_cell('; '.join(k.notes), 60)} |"
            )
        lines.append("")

    table("Mapped", plan.by_outcome("map"))
    table("Kept by your triage", plan.by_outcome("keep-as"))
    table("Collisions", plan.by_outcome("collision"))
    dropped_note = (
        "Off the allowlist. No statement was made and nothing reaches a prompt; the row stays "
        "in the archive (the legacy `user_facts` table). The migration has run, so editing "
        f"`{triage_path.name}` no longer changes it: to redo it, restore the backup above."
        if applied
        else "Off the allowlist. No statement is made and nothing reaches a prompt; the row "
        f"stays in the archive. To keep one, edit `{triage_path.name}` and run again."
    )
    table("Dropped by default", plan.by_outcome("drop"), dropped_note)
    table("Forgotten facts that map (become retracted statements)", plan.by_outcome("history-only"))
    table("Forgotten facts off the allowlist (archive only)", plan.by_outcome("archive"))
    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------- apply

MIGRATED_KEY = "facts_migrated_at"
CLAIM_KEY = "facts_migration_claim"


@dataclass(frozen=True)
class ApplyResult:
    applied: bool
    reason: str
    statements: int = 0
    backup: Path | None = None
    report: Path | None = None


def _statement_records(plan: MigrationPlan, ontology: Ontology) -> tuple[list[Any], list[Any]]:
    """Entities and statements for every planned statement, deduplicating entities by name."""
    from memris.model import OWNER_ID, Entity, Statement, name_key, new_id

    entities: dict[tuple[str, str], Entity] = {}
    statements: list[Statement] = []
    for key_plan in plan.keys:
        previous: tuple[PlannedStatement, str] | None = None
        for ps in key_plan.statements:
            object_id = literal = datatype = None
            if ps.object_class is not None:
                slot = (ps.object_class, name_key(ps.value))
                if slot not in entities:
                    label = " ".join(ps.value.split())
                    entities[slot] = Entity(
                        new_id("ent"), ps.object_class, label, (), ps.recorded_at
                    )
                object_id = entities[slot].id
            else:
                literal = ps.value
                datatype = ontology.attributes[ps.predicate].datatype
            statements.append(
                Statement(
                    id=new_id("st"),
                    subject_id=OWNER_ID,
                    predicate=ps.predicate,
                    recorded_at=ps.recorded_at,
                    object_id=object_id,
                    literal=literal,
                    datatype=datatype,
                    valid_to=ps.valid_to,
                    retracted_at=ps.retracted_at,
                    status=ps.status,
                    confidence=ps.confidence,
                    extractor=ps.source,
                    ontology_version=ontology.version,
                    reinforced=max(1, ps.times_confirmed),
                    last_reinforced_at=ps.last_confirmed,
                    reason=ps.reason,
                    # the value this one replaced, so history and review read old → new
                    supersedes=(
                        previous[1]
                        if previous is not None
                        and previous[0].status != "retracted"
                        and previous[0].valid_to == ps.recorded_at
                        else None
                    ),
                )
            )
            previous = (ps, statements[-1].id)
    return list(entities.values()), statements


def _pending_proposal_records(
    proposals: list[LegacyProposal], ontology: Ontology, entities: list[Any], statements: list[Any]
) -> None:
    """Pending review proposals become proposed statements (memris plan PR 2c).

    Only keys the ontology has a property for; the rest stay in the archive table, as
    off-list facts do. Resolved proposals are history the archive keeps as it is.
    """
    from memris.model import OWNER_ID, Entity, Statement, name_key, new_id

    mappings = fact_mappings(ontology)
    by_name = {(e.class_, name_key(e.label)): e for e in entities}
    for p in proposals:
        rule = mappings.get(normalise_fact_key(p.key))
        if rule is None:
            continue
        object_id = literal = datatype = None
        if rule.object_class is not None:
            slot = (rule.object_class, name_key(p.value))
            if slot not in by_name:
                by_name[slot] = Entity(
                    new_id("ent"), rule.object_class, " ".join(p.value.split()), (), p.created_at
                )
                entities.append(by_name[slot])
            object_id = by_name[slot].id
        else:
            literal, datatype = p.value, ontology.attributes[rule.predicate].datatype
        statements.append(
            Statement(
                id=new_id("st"),
                subject_id=OWNER_ID,
                predicate=rule.predicate,
                recorded_at=p.created_at,
                object_id=object_id,
                literal=literal,
                datatype=datatype,
                status="proposed",
                confidence=p.confidence,
                extractor=p.source,
                ontology_version=ontology.version,
                reinforced=max(1, p.seen_count),
                evidence=p.evidence[:500] or None,
                reason="proposed",
            )
        )


def _conflict_records(
    conflicts: list[LegacyContradiction],
    ontology: Ontology,
    entities: list[Any],
    statements: list[Any],
) -> None:
    """Legacy contradictions onto the chain (memris plan PR 2c-ii).

    A refused ("blocked") value becomes a refused statement pointing at the value it
    lost against; an acknowledged conflict carries its review onto the statement.
    Superseded conflicts are already in the chain (each value links to the one it
    replaced), so only their acknowledgement moves. Conflicts on keys with no property,
    or whose values are not in the chain, stay in the archive table.
    """
    from memris.model import OWNER_ID, Entity, Statement, name_key, new_id

    mappings = fact_mappings(ontology)
    labels = {e.id: e.label for e in entities}
    by_name = {(e.class_, name_key(e.label)): e for e in entities}

    def value(st: Statement) -> str:
        if st.object_id is None:
            return st.literal or ""
        return str(labels.get(st.object_id, ""))

    for c in conflicts:
        rule = mappings.get(normalise_fact_key(c.key))
        if rule is None:
            continue
        chain = [
            st for st in statements if st.predicate == rule.predicate and st.reason != "refused"
        ]
        if c.resolution == "superseded":
            if not c.acknowledged:
                continue
            by_id = {st.id: st for st in chain}
            for i, st in enumerate(statements):
                before = by_id.get(st.supersedes or "")
                if (
                    st.predicate == rule.predicate
                    and value(st) == c.incoming_value
                    and before is not None
                    and value(before) == c.stored_value
                ):
                    statements[i] = replace(st, reviewed_at=c.detected_at)
                    break
            continue
        lost_to = [
            st for st in chain if value(st) == c.stored_value and st.recorded_at <= c.detected_at
        ]
        if not lost_to:
            continue
        object_id = literal = datatype = None
        if rule.object_class is not None:
            slot = (rule.object_class, name_key(c.incoming_value))
            if slot not in by_name:
                by_name[slot] = Entity(
                    new_id("ent"),
                    rule.object_class,
                    " ".join(c.incoming_value.split()),
                    (),
                    c.detected_at,
                )
                entities.append(by_name[slot])
                labels[by_name[slot].id] = by_name[slot].label
            object_id = by_name[slot].id
        else:
            literal, datatype = c.incoming_value, ontology.attributes[rule.predicate].datatype
        statements.append(
            Statement(
                id=new_id("st"),
                subject_id=OWNER_ID,
                predicate=rule.predicate,
                recorded_at=c.detected_at,
                object_id=object_id,
                literal=literal,
                datatype=datatype,
                retracted_at=c.detected_at,
                status="retracted",
                confidence=c.incoming_confidence,
                extractor=c.source,
                ontology_version=ontology.version,
                reinforced=max(1, c.seen_count),
                reason="refused",
                contradicts=lost_to[-1].id,
                reviewed_at=c.detected_at if c.acknowledged else None,
            )
        )


def _backup(db_path: Path, backup_dir: Path, stamp: str) -> Path:
    """An online, consistent copy (SQLite's backup API), taken before anything moves."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"memory-before-memris-{stamp}.db"
    source = open_read_only(db_path)
    dest = sqlite3.connect(target)
    try:
        source.backup(dest)
    finally:
        dest.close()
        source.close()
    return target


def apply_migration(
    db_path: Path,
    ontology: Ontology,
    *,
    out_dir: Path,
    backup_dir: Path,
    now: datetime | None = None,
    wait_seconds: float = 30.0,
) -> ApplyResult:
    """Move user facts onto statements once per database (owner decision: automatic).

    Safe to call on every open: a done marker makes it a no-op, a claim marker keeps
    two processes from both migrating, the database is backed up first, and the
    statements land in one atomic save. A fresh database is simply marked done.
    """
    import os
    import time

    from memris.graph import MemoryGraph
    from memris.store import SQLiteGraphStore

    store = SQLiteGraphStore(db_path)
    if store.get_meta(MIGRATED_KEY):
        return ApplyResult(False, "already migrated")
    moment = now or datetime.now(UTC)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    if not store.claim_meta(CLAIM_KEY, f"{os.getpid()}@{moment.isoformat()}"):
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if store.get_meta(MIGRATED_KEY):
                return ApplyResult(False, "migrated by another process")
            time.sleep(0.1)
        raise RuntimeError(
            f"{db_path}: another process claimed the fact migration and has not finished "
            f"({store.get_meta(CLAIM_KEY)}); delete the '{CLAIM_KEY}' row in memris_meta "
            "if that process is gone"
        )

    data = read_legacy(db_path)
    if not data.facts and not data.history and not data.proposals:
        store.set_meta(MIGRATED_KEY, moment.isoformat())
        return ApplyResult(False, "nothing to migrate")

    triage_path = out_dir / "triage.yaml"
    plan = plan_migration(
        data, ontology, load_triage(triage_path, fact_mappings(ontology)), now=moment
    )
    problems = plan.accounting()
    if problems:  # a planner bug, never a data condition — refuse rather than lose facts
        raise RuntimeError(f"fact migration plan does not add up: {'; '.join(problems)}")

    backup = _backup(db_path, backup_dir, stamp)
    entities, statements = _statement_records(plan, ontology)
    _pending_proposal_records(data.proposals, ontology, entities, statements)
    _conflict_records(data.conflicts, ontology, entities, statements)
    graph = MemoryGraph(ontology, store)
    graph.ensure_owner("Owner")
    graph.import_(statements, entities)
    store.set_meta(MIGRATED_KEY, moment.isoformat())

    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"report-applied-{stamp}.md"
    report.write_text(
        render_report(
            plan,
            db_path=db_path,
            triage_path=triage_path,
            backup=backup,
            written=len(statements),
        ),
        encoding="utf-8",
    )
    return ApplyResult(True, "migrated", len(statements), backup, report)


__all__ = [
    "ApplyResult",
    "apply_migration",
    "KeyPlan",
    "LegacyContradiction",
    "LegacyData",
    "LegacyFact",
    "LegacyHistory",
    "LegacyProposal",
    "MigrationPlan",
    "PlannedStatement",
    "TriageError",
    "load_triage",
    "open_read_only",
    "plan_migration",
    "read_legacy",
    "render_report",
    "render_triage",
]
