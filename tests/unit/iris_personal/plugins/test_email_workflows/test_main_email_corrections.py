"""Tests for ``iris email corrections list`` (Track 1J / ADR-0024)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.main import app
from iris_personal.email.category_store import Category, CategoryStore

ACCOUNT = "gmail:user@gmail.com"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _seed_category_and_corrections(db_path: Path, corrections: list[dict]) -> None:
    store = CategoryStore(db_path=db_path)
    store.ensure_schema()
    store.upsert_if_new(
        Category(
            path="email/finance/banking/northwind-savings",
            type="email",
            root="finance",
            branch="banking",
            leaf="northwind-savings",
            account_id=ACCOUNT,
        )
    )
    for c in corrections:
        store.record_correction(
            message_id=c["message_id"],
            account_id=c.get("account_id", ACCOUNT),
            old_path=c.get("old_path"),
            new_path=c.get("new_path", "email/finance/banking/northwind-savings"),
            previous_classifier=c.get("previous_classifier"),
            reason=c.get("reason"),
        )


def test_corrections_list_exits_3_when_empty(runner: CliRunner, tmp_path: Path) -> None:
    """No corrections for the account → exit 3 with friendly message."""
    db = tmp_path / "iris.db"
    CategoryStore(db_path=db).ensure_schema()  # empty

    result = runner.invoke(
        app,
        ["email", "corrections", "list", "--account", ACCOUNT, "--db-path", str(db)],
    )
    assert result.exit_code == 3
    assert "no corrections" in result.output


def test_corrections_list_renders_table(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "iris.db"
    _seed_category_and_corrections(
        db,
        [
            {
                "message_id": "m-1",
                "old_path": "email/shopping/apparel/gap",
                "reason": "this is my bank not retail",
            },
            {
                "message_id": "m-2",
                "old_path": "email/news/india/etimes",
            },
        ],
    )

    result = runner.invoke(
        app,
        ["email", "corrections", "list", "--account", ACCOUNT, "--db-path", str(db)],
    )
    assert result.exit_code == 0, result.output
    assert "2 correction" in result.output
    assert "m-1" in result.output or "m-2" in result.output
    # The reason on m-1 surfaces
    assert "bank" in result.output


def test_corrections_list_filters_by_account(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "iris.db"
    _seed_category_and_corrections(
        db,
        [
            {"message_id": "m-my", "account_id": ACCOUNT},
            {"message_id": "m-other", "account_id": "gmail:other@x.com"},
        ],
    )
    result = runner.invoke(
        app,
        ["email", "corrections", "list", "--account", ACCOUNT, "--db-path", str(db)],
    )
    assert result.exit_code == 0
    assert "1 correction" in result.output
    assert "m-my" in result.output
    assert "m-other" not in result.output


def test_corrections_list_respects_limit(runner: CliRunner, tmp_path: Path) -> None:
    db = tmp_path / "iris.db"
    _seed_category_and_corrections(
        db,
        [{"message_id": f"m-{i}"} for i in range(5)],
    )
    result = runner.invoke(
        app,
        [
            "email",
            "corrections",
            "list",
            "--account",
            ACCOUNT,
            "--db-path",
            str(db),
            "--limit",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "2 correction" in result.output
