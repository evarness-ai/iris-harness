"""Tests for ``iris email reingest-wiki`` (Track 1K / ADR-0025 §7)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.foundation.eventbus import EventBus
from iris_harness.main import app
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.events import EMAIL_CLASSIFIED
from iris_personal.email.store import EmailStore

ACCOUNT = "gmail:user@gmail.com"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _seed_email(
    email_db: Path,
    *,
    id: str,
    classified: str | None = "email/shopping/apparel/gap",
    confidence: float = 0.8,
) -> None:
    store = EmailStore(db_path=email_db)
    store.ensure_schema()
    msg = EmailMessage(
        id=id,
        provider="gmail",
        account_id=ACCOUNT,
        from_address="x@gap.com",
        from_domain="gap.com",
        subject="x",
        snippet="y",
        received_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    store.upsert(msg)
    if classified is not None:
        store.mark_classified(id, category=classified, confidence=confidence)


def test_reingest_exits_3_when_no_classified_emails(runner: CliRunner, tmp_path: Path) -> None:
    email_db = tmp_path / "email.db"
    EmailStore(db_path=email_db).ensure_schema()  # empty

    result = runner.invoke(
        app,
        ["email", "reingest-wiki", "--account", ACCOUNT, "--email-db", str(email_db)],
    )
    assert result.exit_code == 3
    assert "no classified emails" in result.output


def test_reingest_exits_2_when_email_db_missing(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "email",
            "reingest-wiki",
            "--account",
            ACCOUNT,
            "--email-db",
            str(tmp_path / "nope.db"),
        ],
    )
    assert result.exit_code == 2
    assert "email.db not found" in result.output


def test_reingest_emits_one_event_per_classified_email(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three classified emails → three email.classified events on the bus."""
    email_db = tmp_path / "email.db"
    _seed_email(email_db, id="m-1", classified="email/shopping/apparel/gap")
    _seed_email(email_db, id="m-2", classified="email/finance/banking/northwind")
    _seed_email(email_db, id="m-3", classified="email/social/facebook/updates")

    bus = EventBus()
    captured: list = []
    bus.on(EMAIL_CLASSIFIED, captured.append)
    monkeypatch.setattr("iris_harness.sdk.events.get_default_bus", lambda: bus)

    result = runner.invoke(
        app,
        [
            "email",
            "reingest-wiki",
            "--account",
            ACCOUNT,
            "--email-db",
            str(email_db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "re-emitted" in result.output
    assert "3" in result.output

    assert len(captured) == 3
    ids = {p.id for p in captured}
    assert ids == {"m-1", "m-2", "m-3"}
    # Backfill payloads are tagged so wiki subscribers / future consumers
    # can distinguish a fresh classification from a backfill re-emit.
    assert all(p.classifier == "reingest-wiki" for p in captured)


def test_reingest_respects_limit(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    email_db = tmp_path / "email.db"
    for i in range(5):
        _seed_email(email_db, id=f"m-{i}")

    bus = EventBus()
    captured: list = []
    bus.on(EMAIL_CLASSIFIED, captured.append)
    monkeypatch.setattr("iris_harness.sdk.events.get_default_bus", lambda: bus)

    result = runner.invoke(
        app,
        [
            "email",
            "reingest-wiki",
            "--account",
            ACCOUNT,
            "--email-db",
            str(email_db),
            "--limit",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert len(captured) == 2


def test_reingest_skips_unclassified_emails(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only emails with classified_category IS NOT NULL get re-emitted."""
    email_db = tmp_path / "email.db"
    _seed_email(email_db, id="classified-1", classified="email/shopping/apparel/gap")
    _seed_email(email_db, id="unclassified-1", classified=None)

    bus = EventBus()
    captured: list = []
    bus.on(EMAIL_CLASSIFIED, captured.append)
    monkeypatch.setattr("iris_harness.sdk.events.get_default_bus", lambda: bus)

    runner.invoke(
        app,
        ["email", "reingest-wiki", "--account", ACCOUNT, "--email-db", str(email_db)],
    )
    assert len(captured) == 1
    assert captured[0].id == "classified-1"


def test_reingest_skips_vendor_classified_rows(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mailbox tab (source='vendor') is not an IRIS verdict: never re-emitted."""
    email_db = tmp_path / "email.db"
    _seed_email(email_db, id="iris-row", classified="email/shopping/apparel/gap")
    store = EmailStore(db_path=email_db)
    store.upsert(
        EmailMessage(
            id="vendor-row",
            provider="gmail",
            account_id=ACCOUNT,
            from_address="x@shop.com",
            received_at=datetime(2026, 5, 2, tzinfo=UTC),
            vendor_category="email/promotions",
        )
    )

    bus = EventBus()
    captured: list = []
    bus.on(EMAIL_CLASSIFIED, captured.append)
    monkeypatch.setattr("iris_harness.sdk.events.get_default_bus", lambda: bus)

    result = runner.invoke(
        app, ["email", "reingest-wiki", "--account", ACCOUNT, "--email-db", str(email_db)]
    )
    assert result.exit_code == 0, result.output
    assert [p.id for p in captured] == ["iris-row"]


def test_reingest_skips_mail_the_judge_has_not_released(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A waiting (unjudged) email is invisible: no email.classified for it (PR 5)."""
    from iris_personal.plugins.email_workflows.judgments import JudgmentStore

    email_db = tmp_path / "email.db"
    for mid in ("m-judged", "m-waiting", "m-promo"):
        _seed_email(email_db, id=mid)
    judgments = JudgmentStore(db_path=email_db)
    judgments.ensure_schema()
    judgments.mark_waiting(ACCOUNT, ["m-judged", "m-waiting"])
    judgments.record("m-judged", ACCOUNT, bucket="fyi", confidence=0.9)

    bus = EventBus()
    captured: list = []
    bus.on(EMAIL_CLASSIFIED, captured.append)
    monkeypatch.setattr("iris_harness.sdk.events.get_default_bus", lambda: bus)

    result = runner.invoke(
        app, ["email", "reingest-wiki", "--account", ACCOUNT, "--email-db", str(email_db)]
    )
    assert result.exit_code == 0, result.output
    assert {p.id for p in captured} == {"m-judged", "m-promo"}
