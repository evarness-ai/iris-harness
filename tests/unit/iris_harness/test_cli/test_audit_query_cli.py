"""`iris audit query` reports a DuckDB error as one line, not a traceback."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from iris_harness.main import app


def test_unknown_table_is_a_one_line_error(tmp_path: Path) -> None:
    """A query against the SQLite table name (``audit_log``) instead of the view
    (``audit_archive``) used to escape as a raw ``CatalogException`` traceback."""
    result = CliRunner().invoke(
        app,
        [
            "audit",
            "query",
            "select ts from audit_log limit 1",
            "--db",
            str(tmp_path / "audit.db"),
            "--archive-root",
            str(tmp_path / "archive"),
        ],
    )
    assert result.exit_code == 2
    assert "audit_log" in result.output
    assert "Traceback" not in result.output


def test_view_query_on_empty_store_succeeds(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "audit",
            "query",
            "select ts from audit_archive limit 1",
            "--db",
            str(tmp_path / "audit.db"),
            "--archive-root",
            str(tmp_path / "archive"),
        ],
    )
    assert result.exit_code == 0, result.output
