"""Tests for ``iris email knn-gate`` (Track 1L / ADR-0023)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.main import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_knn_gate_exits_2_when_no_labels(runner: CliRunner, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "email",
            "knn-gate",
            "--account",
            "gmail:nobody@gmail.com",
            "--workspace-dir",
            str(tmp_path / "ws"),
            "--db-path",
            str(tmp_path / "iris.db"),
        ],
    )
    assert result.exit_code == 2
    assert "no labels" in result.output


def test_knn_gate_renders_report_with_stubbed_runner(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stub the runner to return a synthetic report; assert the rendered
    output contains the expected sections."""
    from iris_personal.plugins.email_workflows import knn_gate as knn_gate_module
    from iris_personal.plugins.email_workflows.knn_gate import MeasurementReport, SweepCell

    fake_report = MeasurementReport(
        account_id="gmail:user@gmail.com",
        measured_at=datetime(2026, 5, 25, tzinfo=UTC),
        total_labels=10,
        label_distribution={"shopping": 4, "finance": 3, "social": 3},
        predictions=[],
        sweep=[
            SweepCell(
                cos_min=0.70,
                margin_min=0.05,
                total=10,
                gated_count=7,
                gated_correct=6,
                gated_accuracy=6 / 7,
                queue_rate=3 / 10,
            ),
            SweepCell(
                cos_min=0.80,
                margin_min=0.05,
                total=10,
                gated_count=4,
                gated_correct=4,
                gated_accuracy=1.0,
                queue_rate=6 / 10,
            ),
        ],
        recommended_cos_min=0.70,
        recommended_margin_min=0.05,
        recommended_accuracy=6 / 7,
        recommended_gated_count=7,
        confusion={"shopping": {"shopping": 4}, "finance": {"finance": 3}},
    )

    def fake_measure(self, account_id, *, labels_path=None, include_corrections=False):  # type: ignore[no-untyped-def]
        return fake_report

    monkeypatch.setattr(knn_gate_module.KnnGateRunner, "measure", fake_measure)

    writeup_path = tmp_path / "writeup.md"
    result = runner.invoke(
        app,
        [
            "email",
            "knn-gate",
            "--account",
            "gmail:user@gmail.com",
            "--workspace-dir",
            str(tmp_path / "ws"),
            "--db-path",
            str(tmp_path / "iris.db"),
            "--writeup-to",
            str(writeup_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Holdout distribution" in result.output
    assert "Threshold sweep" in result.output
    assert "Recommended thresholds" in result.output
    # Both grid cells appear; the recommended one is highlighted
    assert "0.70" in result.output
    assert "0.80" in result.output

    # Writeup was written
    assert writeup_path.exists()
    md = writeup_path.read_text()
    assert "kNN-gate measurement" in md
    assert "## Recommendation" in md
    assert "0.70" in md
