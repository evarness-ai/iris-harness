"""Tests for ``iris email search`` (Track 1M / ADR-0026)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.main import app
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore

ACCOUNT = "gmail:user@gmail.com"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _seed(email_db: Path) -> None:
    store = EmailStore(db_path=email_db)
    store.ensure_schema()
    store.upsert(
        EmailMessage(
            id="northwind-1",
            provider="gmail",
            account_id=ACCOUNT,
            from_address="alerts@northwindbank.test",
            from_domain="northwindbank.test",
            subject="Northwind: Your May statement is ready",
            snippet="Account balance details inside",
            received_at=datetime(2026, 5, 1, tzinfo=UTC),
        )
    )
    store.upsert(
        EmailMessage(
            id="gap-1",
            provider="gmail",
            account_id=ACCOUNT,
            from_address="acme-apparel@email.acme-apparel.com",
            from_domain="email.acme-apparel.com",
            subject="Acme Apparel: 60% off summer sale",
            snippet="Shop summer styles today",
            received_at=datetime(2026, 5, 10, tzinfo=UTC),
        )
    )
    store.mark_classified(
        "northwind-1", category="email/finance/banking/northwind-savings", confidence=0.9
    )
    store.mark_classified("gap-1", category="email/shopping/apparel/acme-apparel", confidence=0.8)


def test_search_finds_matching_email(runner: CliRunner, tmp_path: Path) -> None:
    email_db = tmp_path / "email.db"
    _seed(email_db)

    result = runner.invoke(
        app,
        ["email", "search", "northwind", "--email-db", str(email_db)],
    )
    assert result.exit_code == 0, result.output
    assert "1 match" in result.output
    # The sender shows up in the rendered table
    assert "northwindbank.test" in result.output


def test_search_renders_no_matches_friendly(runner: CliRunner, tmp_path: Path) -> None:
    email_db = tmp_path / "email.db"
    _seed(email_db)

    result = runner.invoke(
        app,
        ["email", "search", "zzznomatch", "--email-db", str(email_db)],
    )
    assert result.exit_code == 0
    assert "no matches" in result.output


def test_search_filters_by_category_prefix(runner: CliRunner, tmp_path: Path) -> None:
    """--category restricts to a prefix of classified_category."""
    email_db = tmp_path / "email.db"
    _seed(email_db)

    # "summer OR statement" matches both rows; the category filter narrows
    # to the shopping leaf only.
    result = runner.invoke(
        app,
        [
            "email",
            "search",
            "summer OR statement",
            "--category",
            "email/shopping/",
            "--email-db",
            str(email_db),
        ],
    )
    assert result.exit_code == 0
    assert "1 match" in result.output
    # Rich may truncate cells; look for substrings that survive truncation
    assert "summer" in result.output  # the matched + highlighted term
    # northwindbank.test (the other email's sender) should NOT appear
    assert "northwindbank" not in result.output


def test_search_supports_phrase_query(runner: CliRunner, tmp_path: Path) -> None:
    email_db = tmp_path / "email.db"
    _seed(email_db)
    result = runner.invoke(
        app,
        ["email", "search", '"acme apparel"', "--email-db", str(email_db)],
    )
    assert result.exit_code == 0
    assert "1 match" in result.output
    assert "acme-apparel" in result.output or "acme-apparel" in result.output


def test_search_exit_2_on_malformed_query(runner: CliRunner, tmp_path: Path) -> None:
    email_db = tmp_path / "email.db"
    _seed(email_db)
    # Unterminated quote — FTS5 raises OperationalError; the CLI maps to exit 2
    result = runner.invoke(
        app,
        ["email", "search", '"unterminated', "--email-db", str(email_db)],
    )
    assert result.exit_code == 2
    assert "FTS5 query error" in result.output
