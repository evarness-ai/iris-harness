"""Tests for ``iris email recategorize`` (Track 1J / ADR-0024)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.main import app
from iris_personal.email.category_store import Category, CategoryStore
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore

ACCOUNT = "gmail:user@gmail.com"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _seed_category(db_path: Path, path: str = "email/finance/banking/northwind-savings") -> None:
    store = CategoryStore(db_path=db_path)
    store.ensure_schema()
    type_, root, branch, leaf = path.split("/")
    store.upsert_if_new(
        Category(path=path, type=type_, root=root, branch=branch, leaf=leaf, account_id=ACCOUNT)
    )


def _seed_email(
    email_db_path: Path, *, message_id: str = "m-1", classified: str | None = None
) -> None:
    store = EmailStore(db_path=email_db_path)
    store.ensure_schema()
    msg = EmailMessage(
        id=message_id,
        provider="gmail",
        account_id=ACCOUNT,
        from_address="bob@example.com",
        from_domain="example.com",
        subject="hi",
        snippet="hi",
        received_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    store.upsert(msg)
    if classified is not None:
        store.mark_classified(msg.id, category=classified, confidence=0.6)


def _invoke(
    runner: CliRunner,
    *,
    account: str = ACCOUNT,
    message_id: str = "m-1",
    to: str,
    db: Path,
    email_db: Path,
):  # type: ignore[no-untyped-def]
    return runner.invoke(
        app,
        [
            "email",
            "recategorize",
            "--account",
            account,
            "--message-id",
            message_id,
            "--to",
            to,
            "--db-path",
            str(db),
            "--email-db",
            str(email_db),
        ],
    )


def test_recategorize_writes_history_and_updates_row(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "iris.db"
    email_db = tmp_path / "email.db"
    _seed_category(db, "email/finance/banking/northwind-savings")
    _seed_email(email_db, message_id="m-1", classified="email/shopping/apparel/outlet-brand")

    result = _invoke(
        runner,
        message_id="m-1",
        to="email/finance/banking/northwind-savings",
        db=db,
        email_db=email_db,
    )
    assert result.exit_code == 0, result.output
    assert "recategorized" in result.output
    assert "shopping/apparel/outlet-brand" in result.output
    assert "finance/banking/northwind-savings" in result.output

    # Email row updated
    import sqlite3

    with sqlite3.connect(email_db) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT classified_category, classified_confidence, triage_state FROM emails WHERE id = ?",
            ("m-1",),
        ).fetchone()
    assert row["classified_category"] == "email/finance/banking/northwind-savings"
    assert row["classified_confidence"] == 1.0
    assert row["triage_state"] == "classified"

    # History row written
    cat_store = CategoryStore(db_path=db)
    rows = cat_store.list_corrections(account_id=ACCOUNT)
    assert len(rows) == 1
    payload = rows[0]["payload"]
    assert payload["message_id"] == "m-1"
    assert payload["old_path"] == "email/shopping/apparel/outlet-brand"
    assert payload["new_path"] == "email/finance/banking/northwind-savings"


def test_recategorize_with_reason_is_persisted(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "iris.db"
    email_db = tmp_path / "email.db"
    _seed_category(db)
    _seed_email(email_db, message_id="m-1")

    result = runner.invoke(
        app,
        [
            "email",
            "recategorize",
            "--account",
            ACCOUNT,
            "--message-id",
            "m-1",
            "--to",
            "email/finance/banking/northwind-savings",
            "--reason",
            "this is my bank",
            "--db-path",
            str(db),
            "--email-db",
            str(email_db),
        ],
    )
    assert result.exit_code == 0
    rows = CategoryStore(db_path=db).list_corrections(account_id=ACCOUNT)
    assert rows[0]["payload"]["reason"] == "this is my bank"


def test_recategorize_exits_2_when_message_missing(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "iris.db"
    email_db = tmp_path / "email.db"
    _seed_category(db)
    # No email seeded
    EmailStore(db_path=email_db).ensure_schema()

    result = _invoke(
        runner,
        message_id="ghost",
        to="email/finance/banking/northwind-savings",
        db=db,
        email_db=email_db,
    )
    assert result.exit_code == 2
    assert "no email with id" in result.output


def test_recategorize_exits_3_when_target_category_missing(
    runner: CliRunner, tmp_path: Path
) -> None:
    db = tmp_path / "iris.db"
    email_db = tmp_path / "email.db"
    CategoryStore(db_path=db).ensure_schema()  # empty
    _seed_email(email_db, message_id="m-1")

    result = _invoke(
        runner,
        message_id="m-1",
        to="email/finance/banking/does-not-exist",
        db=db,
        email_db=email_db,
    )
    assert result.exit_code == 3
    assert "not found or inactive" in result.output


def test_recategorize_unclassified_message_logs_old_path_as_none(
    runner: CliRunner, tmp_path: Path
) -> None:
    db = tmp_path / "iris.db"
    email_db = tmp_path / "email.db"
    _seed_category(db)
    _seed_email(email_db, message_id="m-1")  # no prior classification

    result = _invoke(
        runner,
        message_id="m-1",
        to="email/finance/banking/northwind-savings",
        db=db,
        email_db=email_db,
    )
    assert result.exit_code == 0
    rows = CategoryStore(db_path=db).list_corrections()
    assert rows[0]["old_path"] is None
    assert rows[0]["payload"]["old_path"] is None


def test_recategorize_emits_email_classified(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR-0025 §6: recategorize re-fires email.classified so the wiki
    subscriber follows user corrections."""
    from iris_harness.foundation.eventbus import EventBus
    from iris_personal.email.events import EMAIL_CLASSIFIED, EmailClassifiedPayload

    db = tmp_path / "iris.db"
    email_db = tmp_path / "email.db"
    _seed_category(db, "email/finance/banking/northwind-savings")
    _seed_email(email_db, message_id="m-1")

    # Stand up an isolated bus + capture every email.classified
    bus = EventBus()
    captured: list = []
    bus.on(EMAIL_CLASSIFIED, captured.append)

    import iris_harness.main as main_mod

    # The CLI imports get_default_bus from iris_harness.sdk.events; patch the
    # function reference inside main so it returns our test bus.
    monkeypatch.setattr("iris_harness.sdk.events.get_default_bus", lambda: bus)
    # Re-import the symbol that main.py grabbed
    monkeypatch.setattr(main_mod, "_main_test_unused", None, raising=False)

    result = _invoke(
        runner,
        message_id="m-1",
        to="email/finance/banking/northwind-savings",
        db=db,
        email_db=email_db,
    )
    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    payload = captured[0]
    assert isinstance(payload, EmailClassifiedPayload)
    assert payload.id == "m-1"
    assert payload.category_path == "email/finance/banking/northwind-savings"
    assert payload.confidence == 1.0
    assert payload.classifier == "user-classification-correction"
