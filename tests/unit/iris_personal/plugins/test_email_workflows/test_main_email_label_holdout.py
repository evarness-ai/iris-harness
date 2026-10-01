"""Tests for ``iris email label-holdout`` (Track 1L / ADR-0023).

The interactive labeling flow is exercised via the spike-import path
only; the interactive prompt loop is verified by the holdout module's
unit tests (append_label semantics) plus the e2e validation in
Track 1L Commit 6.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.main import app
from iris_personal.plugins.email_workflows.holdout import load_holdout


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_spike(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_import_spike_writes_holdout_jsonl(runner: CliRunner, tmp_path: Path) -> None:
    """--import-spike round-trips the flat spike schema into HoldoutLabel."""
    spike = tmp_path / "spike.jsonl"
    _write_spike(
        spike,
        [
            {
                "uid": "10",
                "user_category": "finance",
                "from_address": "bank@x.com",
                "from_domain": "x.com",
                "subject": "Statement",
                "snippet": "Your statement",
                "received_at": "2026-05-01T00:00:00+00:00",
            },
            {
                "uid": "11",
                "user_category": "marketing",
                "from_address": "shop@y.com",
                "from_domain": "y.com",
                "subject": "Sale",
                "snippet": "Big sale",
                "received_at": "2026-05-02T00:00:00+00:00",
            },
        ],
    )

    workspace = tmp_path / "ws"
    result = runner.invoke(
        app,
        [
            "email",
            "label-holdout",
            "--account",
            "gmail:user@gmail.com",
            "--import-spike",
            str(spike),
            "--workspace-dir",
            str(workspace),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "imported" in result.output
    assert "2" in result.output

    target = workspace / "email" / "gmail-user-at-gmail.com" / "holdout-labels.jsonl"
    assert target.exists()
    labels = load_holdout(target)
    assert {lbl.true_root for lbl in labels} == {"finance", "shopping"}


def test_import_spike_exits_2_when_source_missing(runner: CliRunner, tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    result = runner.invoke(
        app,
        [
            "email",
            "label-holdout",
            "--account",
            "gmail:user@gmail.com",
            "--import-spike",
            str(tmp_path / "nope.jsonl"),
            "--workspace-dir",
            str(workspace),
        ],
    )
    assert result.exit_code == 2
    assert "spike file not found" in result.output


def test_interactive_exits_3_when_no_unlabeled_in_email_db(
    runner: CliRunner, tmp_path: Path
) -> None:
    """No emails in email.db for the account → exit 3 with friendly message."""
    from iris_personal.email.store import EmailStore

    email_db = tmp_path / "email.db"
    store = EmailStore(db_path=email_db)
    store.ensure_schema()
    # Don't insert any emails

    result = runner.invoke(
        app,
        [
            "email",
            "label-holdout",
            "--account",
            "gmail:user@gmail.com",
            "--workspace-dir",
            str(tmp_path / "ws"),
            "--email-db",
            str(email_db),
        ],
    )
    assert result.exit_code == 3
    assert "no new emails to label" in result.output
