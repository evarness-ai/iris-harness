"""The fact → statement migration plan (memris plan PR 2a).

The legacy database is built with MemoryStore's own writers, so the history rows are
the ones production writes, not hand-made imitations.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.cli.memory import memory_app
from iris_harness.memory.ontology import fact_mappings, memory_ontology
from iris_harness.memory.statement_migration import (
    LegacyData,
    LegacyFact,
    LegacyHistory,
    TriageError,
    load_triage,
    open_read_only,
    plan_migration,
    read_legacy,
    render_report,
    render_triage,
)
from iris_harness.memory.store import MemoryStore

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = pytest.mark.usefixtures("test_vocabulary")

T0 = datetime(2026, 6, 1, tzinfo=UTC)


class _Legacy:
    """A database in the shape the pre-2b store left it: user_facts plus its history.

    Rows are written the way the old writers wrote them (capture / supersede / forget
    history rows), straight into the legacy tables, without ever running the migration.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        MemoryStore(db_path=path)._ensure_tables()  # the tables only — no migration

    def _sql(self, sql: str, args: tuple) -> None:  # type: ignore[type-arg]
        with sqlite3.connect(self.path) as conn:
            conn.execute(sql, args)

    def _history(
        self, key: str, old: str | None, new: str | None, reason: str, at: datetime
    ) -> None:
        self._sql(
            "INSERT INTO user_fact_history(key, old_value, old_confidence, new_value, "
            "new_confidence, source, reason, changed_at) VALUES (?, ?, 0.9, ?, 0.9, ?, ?, ?)",
            (key, old, new, "conversation:llm", reason, at.isoformat()),
        )

    def capture(
        self, key: str, value: str, *, at: datetime = T0, confirmed: bool = True, conf: float = 0.9
    ) -> None:
        with sqlite3.connect(self.path) as conn:
            prior = conn.execute("SELECT value FROM user_facts WHERE key = ?", (key,)).fetchone()
        self._sql(
            "INSERT OR REPLACE INTO user_facts(key, value, confidence, source, first_seen, "
            "last_confirmed, times_confirmed, confirmed) VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
            (key, value, conf, "conversation:llm", at.isoformat(), at.isoformat(), int(confirmed)),
        )
        if prior is None:
            self._history(key, None, value, "capture", at)
        elif prior[0] != value:
            self._history(key, prior[0], value, "supersede", at)

    def forget(self, key: str, at: datetime = T0) -> None:
        with sqlite3.connect(self.path) as conn:
            (value,) = conn.execute("SELECT value FROM user_facts WHERE key = ?", (key,)).fetchone()
            conn.execute("DELETE FROM user_facts WHERE key = ?", (key,))
        self._history(key, value, None, "forget", at)


@pytest.fixture
def legacy_db(tmp_path: Path) -> Path:
    db = _Legacy(tmp_path / "memory.db")
    db.capture("employer", "Barclays")
    db.capture("employer", "Litware", at=T0 + timedelta(days=30))
    db.capture("email", "old@example.org")
    db.forget("email")  # forgotten: lives only in history now
    db.capture("to-do", "buy milk")  # off the allowlist
    db.capture("error", "it shows error")  # off the allowlist
    db.capture("topic", "murder plot")
    db.forget("topic")  # forgotten AND off the allowlist
    db.capture("card", "Visa", at=T0)
    db.capture("credit_card", "Proseware", at=T0 + timedelta(days=5))
    db.capture("organization", "Acme", at=T0 + timedelta(days=40))
    db.capture("blog", "web3notes.example", confirmed=False)
    return db.path


def _plan(db: Path, triage: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    return plan_migration(
        read_legacy(db), memory_ontology(), triage or {}, now=T0 + timedelta(days=90)
    )


def _key(plan, key: str):  # type: ignore[no-untyped-def]
    return next(k for k in plan.keys if k.key == key)


# --- outcomes ---------------------------------------------------------------------


def test_every_row_has_exactly_one_outcome(legacy_db: Path) -> None:
    plan = _plan(legacy_db)
    assert plan.accounting() == []
    assert plan.fact_rows == 7  # employer, organization, to-do, error, card, credit_card, blog
    outcomes = {k.key: k.outcome for k in plan.keys}
    assert outcomes == {
        "employer": "collision",
        "organization": "map",
        "email": "history-only",
        "to-do": "drop",
        "error": "drop",
        "topic": "archive",
        "card": "map",
        "credit_card": "map",
        "blog": "map",
    }


def test_a_changed_value_becomes_a_closed_statement_then_the_next(legacy_db: Path) -> None:
    employer = _key(_plan(legacy_db), "employer")
    assert employer.predicate == "mem:works_at"
    first, second = employer.statements
    # (Litware is then closed by the organization collision — see the collision test)
    assert (first.value, first.status, second.value, second.status) == (
        "Barclays",
        "confirmed",
        "Litware",
        "confirmed",
    )
    assert first.valid_to == second.recorded_at  # closed when the new value arrived
    assert second.object_class == "mem:Organization"


def test_a_forgotten_fact_that_maps_becomes_a_retracted_statement(legacy_db: Path) -> None:
    [email] = _key(_plan(legacy_db), "email").statements
    assert (email.predicate, email.value, email.status) == (
        "mem:email",
        "old@example.org",
        "retracted",
    )
    assert email.retracted_at is not None and email.object_class is None


def test_unconfirmed_facts_stay_proposals(legacy_db: Path) -> None:
    [blog] = _key(_plan(legacy_db), "blog").statements
    assert blog.status == "proposed"


def test_off_allowlist_facts_are_dropped_unless_the_owner_keeps_them(legacy_db: Path) -> None:
    plan = _plan(legacy_db, {"to-do": "project"})
    todo = _key(plan, "to-do")
    assert (todo.outcome, todo.kept_as, todo.predicate) == ("keep-as", "project", "mem:works_on")
    assert _key(plan, "error").statements == []
    assert plan.accounting() == []


def test_a_collision_keeps_the_newer_value_current_and_says_so(legacy_db: Path) -> None:
    """employer and organization both mean works_at, which holds one value."""
    plan = _plan(legacy_db)
    org, employer = _key(plan, "organization"), _key(plan, "employer")
    # organization (Acme) was confirmed last, so it stays; employer's Litware closes
    assert [s.current for s in org.statements] == [True]
    assert not any(s.current for s in employer.statements)
    assert org.outcome == "map" and employer.outcome == "collision"
    assert employer.statements[-1].valid_to == plan.migrated_at
    assert any("collision on mem:works_at" in i for i in plan.issues)


def test_a_property_without_a_limit_keeps_every_value(legacy_db: Path) -> None:
    """Two cards are two cards: holds_card has no max_count, so nothing collides."""
    plan = _plan(legacy_db)
    for key in ("card", "credit_card"):
        k = _key(plan, key)
        assert k.outcome == "map" and [s.current for s in k.statements] == [True]
    assert not any("holds_card" in i for i in plan.issues)


def test_a_history_that_disagrees_with_the_row_is_reported_and_the_row_wins() -> None:
    row = LegacyFact("city", "Leeds", 0.9, "s", T0, T0 + timedelta(days=2), 1, True)
    events = [LegacyHistory(1, "city", None, "London", 0.9, "s", "capture", T0)]
    plan = plan_migration(LegacyData([row], events, 0, 0), memory_ontology(), {})
    city = _key(plan, "city")
    assert [s.value for s in city.statements] == ["London", "Leeds"]
    assert any("stored value wins" in n for n in city.notes)


def test_accounting_catches_a_row_nobody_planned(legacy_db: Path) -> None:
    plan = _plan(legacy_db)
    plan.keys.pop(0)
    assert plan.accounting()


# --- triage file ------------------------------------------------------------------


def test_the_triage_file_round_trips_and_keeps_owner_choices(
    legacy_db: Path, tmp_path: Path
) -> None:
    data = read_legacy(legacy_db)
    off = [f for f in data.facts if f.key in {"to-do", "error"}]
    path = tmp_path / "triage.yaml"
    path.write_text(render_triage(off, {"to-do": "project"}), encoding="utf-8")
    assert load_triage(path, fact_mappings(memory_ontology())) == {
        "to-do": "project",
        "error": "drop",
    }


@pytest.mark.parametrize(
    "key", ["to-do", "has space", "yes", "No", "null", "2fa: key", "ünï", "a#b"]
)
def test_any_legacy_key_survives_the_triage_round_trip(tmp_path: Path, key: str) -> None:
    fact = LegacyFact(key, "v # 'quoted'", 1.0, "s", T0, T0, 1, True)
    path = tmp_path / "triage.yaml"
    path.write_text(render_triage([fact], {}), encoding="utf-8")
    assert load_triage(path, fact_mappings(memory_ontology())) == {key: "drop"}


@pytest.mark.parametrize(
    ("body", "why"),
    [
        ("facts:\n  to-do: {keep: nonsense}\n", "no mapping knows"),
        ("facts:\n  to-do: maybe\n", "must be 'drop'"),
        ("facts: [a, b]\n", "must be a mapping"),
    ],
)
def test_a_bad_triage_file_is_refused(tmp_path: Path, body: str, why: str) -> None:
    path = tmp_path / "triage.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(TriageError, match=why):
        load_triage(path, fact_mappings(memory_ontology()))


def test_the_report_states_the_counts_and_the_checks(legacy_db: Path, tmp_path: Path) -> None:
    report = render_report(
        _plan(legacy_db), db_path=legacy_db, triage_path=tmp_path / "triage.yaml"
    )
    assert "| fact rows | 7 |" in report
    assert "| → dropped (triage default; row stays in the archive) | 2 |" in report
    assert "collision on mem:works_at" in report
    assert "## Dropped by default (2)" in report
    assert "dry run" in report and "nothing was written" in report  # it is one


def test_an_applied_report_says_it_was_applied(legacy_db: Path, tmp_path: Path) -> None:
    """It once said "opened read-only; nothing was written" about a migrated database."""
    backup = tmp_path / "backups" / "memory-before-memris-x.db"
    report = render_report(
        _plan(legacy_db),
        db_path=legacy_db,
        triage_path=tmp_path / "triage.yaml",
        backup=backup,
        written=12,
    )
    assert report.startswith("# Fact → statement migration: applied")
    assert "dry run" not in report and "nothing was written" not in report
    assert "12 statements written" in report and str(backup) in report
    assert "restore the backup" in report and "run again" not in report


# --- the command ------------------------------------------------------------------


def test_the_legacy_connection_cannot_write(legacy_db: Path) -> None:
    conn = open_read_only(legacy_db)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM user_facts")
    finally:
        conn.close()


def test_a_path_with_spaces_opens(tmp_path: Path) -> None:
    """The owner's repo lives under 'My Workspace' — a URI must be escaped, not glued."""
    spaced = tmp_path / "My Workspace" / "memory.db"
    spaced.parent.mkdir()
    MemoryStore(db_path=spaced).ensure_schema()
    assert read_legacy(spaced).facts == []


def _digest(path: Path) -> str:
    """What the database holds — not its bytes.

    The file is in WAL mode: committed rows sit in the -wal file until any closing
    connection checkpoints them into the main file, so hashing the main file measures
    checkpoint timing, not writes. A logical dump is what "nothing was written" means.
    """
    conn = sqlite3.connect(path)
    try:
        return hashlib.sha256("\n".join(conn.iterdump()).encode()).hexdigest()
    finally:
        conn.close()


def test_the_command_is_a_dry_run_that_never_writes_the_database(
    legacy_db: Path, tmp_path: Path
) -> None:
    before = _digest(legacy_db)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        memory_app, ["migrate-statements", "--db-path", str(legacy_db), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert _digest(legacy_db) == before
    assert (out / "report.md").exists() and "dry run" in result.output

    # the owner rescues one fact; a second run honours it and keeps the edit
    triage = out / "triage.yaml"
    triage.write_text(
        triage.read_text(encoding="utf-8").replace("to-do: drop", "to-do: {keep: project}"),
        encoding="utf-8",
    )
    again = CliRunner().invoke(
        memory_app, ["migrate-statements", "--db-path", str(legacy_db), "--out", str(out)]
    )
    assert again.exit_code == 0, again.output
    assert "kept 1" in again.output
    assert "to-do: {keep: project}" in triage.read_text(encoding="utf-8")
    assert _digest(legacy_db) == before


def test_the_command_refuses_a_broken_triage_file(legacy_db: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "triage.yaml").write_text("facts:\n  to-do: {keep: nonsense}\n", encoding="utf-8")
    result = CliRunner().invoke(
        memory_app, ["migrate-statements", "--db-path", str(legacy_db), "--out", str(out)]
    )
    assert result.exit_code == 1
