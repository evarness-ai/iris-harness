"""Tests for ``iris email triage`` (Track 1G manual CLI)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iris_harness.main import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _invoke(
    runner: CliRunner,
    *,
    account: str = "gmail:user@gmail.com",
    limit: int = 20,
    workspace: Path | None = None,
    db: Path | None = None,
    email_db: Path | None = None,
) -> Any:
    args = ["email", "triage", "--account", account, "--limit", str(limit)]
    if workspace is not None:
        args += ["--workspace-dir", str(workspace)]
    if db is not None:
        args += ["--db-path", str(db)]
    if email_db is not None:
        args += ["--email-db", str(email_db)]
    return runner.invoke(app, args)


def test_triage_exits_3_when_no_unclassified(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No unclassified emails → exit code 3 with friendly message."""
    from iris_personal.plugins.email_workflows import triage as triage_module

    def fake_classify_unclassified(self, account_id, *, limit=20, store=None, use_llm=False):  # type: ignore[no-untyped-def]
        return []

    monkeypatch.setattr(
        triage_module.EmailTriageClassifier,
        "classify_unclassified",
        fake_classify_unclassified,
    )

    result = _invoke(
        runner,
        workspace=tmp_path / "ws",
        db=tmp_path / "iris.db",
        email_db=tmp_path / "email.db",
    )
    assert result.exit_code == 3
    assert "no unclassified emails" in result.output


def test_triage_exits_2_when_categories_missing(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FileNotFoundError from the classifier → exit 2 with the error message."""
    from iris_personal.plugins.email_workflows import triage as triage_module

    def fake_classify_unclassified(self, account_id, *, limit=20, store=None, use_llm=False):  # type: ignore[no-untyped-def]
        raise FileNotFoundError("proposals.jsonl missing for gmail:user@gmail.com")

    monkeypatch.setattr(
        triage_module.EmailTriageClassifier,
        "classify_unclassified",
        fake_classify_unclassified,
    )

    result = _invoke(
        runner,
        workspace=tmp_path / "ws",
        db=tmp_path / "iris.db",
        email_db=tmp_path / "email.db",
    )
    assert result.exit_code == 2
    assert "proposals.jsonl missing" in result.output


def test_triage_renders_summary_table(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two classifications + one soft-fail render correctly."""
    from iris_personal.plugins.email_workflows import triage as triage_module
    from iris_personal.plugins.email_workflows.triage import TriageResult

    def fake_classify_unclassified(self, account_id, *, limit=20, store=None, use_llm=False):  # type: ignore[no-untyped-def]
        return [
            TriageResult("m-1", "email/shopping/apparel/gap", 0.85),
            TriageResult("m-2", "email/social/facebook/updates", 0.63),
            TriageResult("m-3", None, None, error="llama-server unreachable"),
        ]

    monkeypatch.setattr(
        triage_module.EmailTriageClassifier,
        "classify_unclassified",
        fake_classify_unclassified,
    )

    result = _invoke(
        runner,
        workspace=tmp_path / "ws",
        db=tmp_path / "iris.db",
        email_db=tmp_path / "email.db",
    )
    assert result.exit_code == 0, result.output
    # New ADR-0022 title format includes "queued" when there are queued items.
    assert "2 classified" in result.output
    # Rich may truncate cells; look for substrings that survive.
    assert "shopping" in result.output or "apparel" in result.output
    assert "0.85" in result.output
    assert "llama" in result.output  # error column carries the soft-fail note


# ─── ADR-0022 — queued result rendering + triage-batch CLI ──────────────────


def test_triage_renders_queued_results_with_hint(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per ADR-0022: queued rows show as state='queued' and the CLI
    hints the user to run triage-batch."""
    from iris_personal.plugins.email_workflows import triage as triage_module
    from iris_personal.plugins.email_workflows.triage import TriageResult

    def fake_classify_unclassified(self, account_id, *, limit=20, store=None, use_llm=False):  # type: ignore[no-untyped-def]
        return [
            TriageResult("m-1", "email/shopping/apparel/gap", 0.85, classifier="pure-knn"),
            TriageResult(
                "m-2",
                None,
                None,
                error="queued: cos1=0.55 margin=0.02",
                queued=True,
                classifier="pure-knn",
            ),
            TriageResult(
                "m-3",
                None,
                None,
                error="queued: cos1=0.60 margin=0.03",
                queued=True,
                classifier="pure-knn",
            ),
        ]

    monkeypatch.setattr(
        triage_module.EmailTriageClassifier,
        "classify_unclassified",
        fake_classify_unclassified,
    )

    result = _invoke(
        runner,
        workspace=tmp_path / "ws",
        db=tmp_path / "iris.db",
        email_db=tmp_path / "email.db",
    )
    assert result.exit_code == 0, result.output
    assert "1 classified" in result.output
    assert "2 queued" in result.output
    # The CLI surfaces the triage-batch hint when there's a queue
    assert "triage-batch" in result.output


def test_triage_batch_exits_3_when_no_pending(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_personal.plugins.email_workflows import triage as triage_module

    def fake_batch(self, account_id, *, limit=20, store=None):  # type: ignore[no-untyped-def]
        return []

    monkeypatch.setattr(
        triage_module.EmailTriageClassifier,
        "classify_pending_review_batch",
        fake_batch,
    )

    result = runner.invoke(
        app,
        [
            "email",
            "triage-batch",
            "--account",
            "gmail:user@gmail.com",
            "--workspace-dir",
            str(tmp_path / "ws"),
            "--db-path",
            str(tmp_path / "iris.db"),
            "--email-db",
            str(tmp_path / "email.db"),
        ],
    )
    assert result.exit_code == 3
    assert "no pending-review emails" in result.output


def test_triage_batch_renders_summary_when_drained(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """triage-batch returns the LLM-resolved classifications."""
    from iris_personal.plugins.email_workflows import triage as triage_module
    from iris_personal.plugins.email_workflows.triage import TriageResult

    def fake_batch(self, account_id, *, limit=20, store=None):  # type: ignore[no-untyped-def]
        return [
            TriageResult(
                "m-1",
                "email/shopping/apparel/gap",
                0.77,
                classifier="tier3-local-knn",
            ),
            TriageResult(
                "m-2",
                "email/shopping/apparel/shopmart",
                0.70,
                classifier="tier3-local-knn",
            ),
        ]

    monkeypatch.setattr(
        triage_module.EmailTriageClassifier,
        "classify_pending_review_batch",
        fake_batch,
    )

    result = runner.invoke(
        app,
        [
            "email",
            "triage-batch",
            "--account",
            "gmail:user@gmail.com",
            "--workspace-dir",
            str(tmp_path / "ws"),
            "--db-path",
            str(tmp_path / "iris.db"),
            "--email-db",
            str(tmp_path / "email.db"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "2 drained from queue" in result.output
    assert "0.77" in result.output
