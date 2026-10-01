from __future__ import annotations

import importlib.util
from pathlib import Path

from typer.testing import CliRunner

from iris_harness.main import app


def _has_dep(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def test_audit_query_rejects_non_select_sql() -> None:
    result = CliRunner().invoke(app, ["audit", "query", "DELETE FROM audit_archive"])
    if _has_dep("duckdb") and _has_dep("pyarrow"):
        assert result.exit_code == 2
        assert "only SELECT/CTE queries" in result.output
    else:
        assert result.exit_code == 2
        assert "optional dependency" in result.output


def test_audit_export_invalid_since_returns_usage_error() -> None:
    result = CliRunner().invoke(app, ["audit", "export", "--since", "yesterday"])
    if _has_dep("duckdb") and _has_dep("pyarrow"):
        assert result.exit_code == 2
        assert "--since must be ISO date/datetime" in result.output
    else:
        assert result.exit_code == 2
        assert "optional dependency" in result.output


def test_audit_export_invalid_format() -> None:
    result = CliRunner().invoke(
        app,
        ["audit", "export", "--since", "2026-01-01", "--format", "xml"],
    )
    if _has_dep("duckdb") and _has_dep("pyarrow"):
        assert result.exit_code == 2
        assert "unknown --format" in result.output
    else:
        assert result.exit_code == 2
        assert "optional dependency" in result.output


def test_audit_export_creates_file_when_deps_available(tmp_path: Path) -> None:
    if not (_has_dep("duckdb") and _has_dep("pyarrow")):
        return

    output = tmp_path / "audit.jsonl"
    result = CliRunner().invoke(
        app,
        [
            "audit",
            "export",
            "--since",
            "2026-01-01",
            "--out",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
