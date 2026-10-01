"""Tests for ``iris email bootstrap-categories`` (Track 1E.2).

Exercises the Typer surface — the orchestrator itself is unit-tested
elsewhere; here we verify exit codes, output paths, and the slug
translation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iris_harness.main import app
from iris_personal.plugins.email_workflows.cli import _account_slug_for_path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_slug_translation() -> None:
    assert _account_slug_for_path("gmail:user@gmail.com") == "gmail-user-at-gmail.com"
    assert _account_slug_for_path("outlook:foo@bar.org") == "outlook-foo-at-bar.org"


def test_bootstrap_categories_exits_2_when_no_corpus(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No emails in email.db + --no-fetch-if-low → exit code 2."""
    from iris_personal.plugins.email_workflows import discovery

    # Force the orchestrator to look at a clean tmp DB by patching the
    # default Path("data/email.db") used inside bootstrap_categories.
    monkeypatch.setattr(discovery, "load_corpus", lambda db_path, account_id, **kw: [])

    result = runner.invoke(
        app,
        [
            "email",
            "bootstrap-categories",
            "--account",
            "gmail:nobody@gmail.com",
            "--no-fetch-if-low",
            "--skip-naming",
            "--workspace-dir",
            str(tmp_path / "ws"),
        ],
    )
    assert result.exit_code == 2, result.output


def test_bootstrap_categories_writes_jsonl_and_prints_table(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end happy path: stub the orchestrator to return 2
    proposals; verify the CLI writes them to the workspace JSONL and
    prints a table."""
    from iris_personal.email.contracts import CategoryProposal
    from iris_personal.plugins.email_workflows import discovery

    def fake_orchestrator(account_id, **kwargs):  # type: ignore[no-untyped-def]
        return [
            CategoryProposal(
                cluster_id=0,
                size=15,
                cohesion=0.92,
                top_domains=(("gap.com", 15),),
                proposed_root="shopping",
                proposed_branch="apparel",
                proposed_leaf="outlet-brand",
            ),
            CategoryProposal(
                cluster_id=1,
                size=8,
                cohesion=0.61,
                top_domains=(("facebook.com", 8),),
                proposed_root="social",
                proposed_branch="facebook",
                proposed_leaf="updates",
            ),
        ]

    monkeypatch.setattr(discovery, "bootstrap_categories", fake_orchestrator)

    ws = tmp_path / "ws"
    result = runner.invoke(
        app,
        [
            "email",
            "bootstrap-categories",
            "--account",
            "gmail:user@gmail.com",
            "--skip-naming",  # bypass any LLM call
            "--no-fetch-if-low",
            "--workspace-dir",
            str(ws),
        ],
    )
    assert result.exit_code == 0, result.output

    out_path = ws / "email" / "gmail-user-at-gmail.com" / "proposals.jsonl"
    assert out_path.exists(), result.output
    lines = out_path.read_text().splitlines()
    assert len(lines) == 2
    payloads = [json.loads(line) for line in lines]
    assert payloads[0]["proposed_root"] == "shopping"
    assert payloads[0]["cohesion"] == 0.92
    assert payloads[1]["proposed_root"] == "social"

    # Output table mentions counts + cohesion + at least one root.
    assert "2 category candidates" in result.output
    assert "shopping" in result.output


def test_bootstrap_categories_uses_default_workspace_when_omitted(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--workspace-dir omitted → uses $IRIS_HOME/workspace (never Path.home() directly,
    so a relocated home, like this test's, is honoured)."""
    from iris_personal.email.contracts import CategoryProposal
    from iris_personal.plugins.email_workflows import discovery

    def fake_orchestrator(account_id, **kwargs):  # type: ignore[no-untyped-def]
        return [
            CategoryProposal(cluster_id=0, size=5, cohesion=0.8),
        ]

    monkeypatch.setattr(discovery, "bootstrap_categories", fake_orchestrator)
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "fake-home"))

    result = runner.invoke(
        app,
        [
            "email",
            "bootstrap-categories",
            "--account",
            "gmail:a@b.com",
            "--skip-naming",
            "--no-fetch-if-low",
        ],
    )
    assert result.exit_code == 0, result.output

    out_path = (
        tmp_path / "fake-home" / "workspace" / "email" / "gmail-a-at-b.com" / "proposals.jsonl"
    )
    assert out_path.exists()
