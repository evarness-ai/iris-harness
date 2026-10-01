"""Tests for the agent-callable propose_skill_from_sandbox tool."""

from __future__ import annotations

from pathlib import Path

from iris_harness.tools.propose_skill_from_sandbox import NARRATIVE_FILE, propose_skill_from_sandbox
from iris_harness.tools.skills.loader import (
    DEFAULT_SANDBOX_SCRIPT_NAME,
    PROPOSAL_FILE,
    RUNS_FILE,
    load_skill_proposal,
)

_SCRIPT = "import httpx\nprint(httpx.get('https://example.com').status_code)\n"


def test_propose_writes_quarantined_proposal(tmp_path: Path) -> None:
    result = propose_skill_from_sandbox(
        tmp_path,
        script=_SCRIPT,
        intent="Fetch top GitHub repos",
        narrative="Reusable for daily morning briefs.",
    )

    assert result["ok"] is True
    assert result["slug"] == "fetch-top-github-repos"
    assert result["status"] == "proposed"

    proposal_dir = tmp_path / str(result["proposal_dir"])
    assert (proposal_dir / DEFAULT_SANDBOX_SCRIPT_NAME).read_text(encoding="utf-8") == _SCRIPT
    assert (proposal_dir / RUNS_FILE).exists()
    assert (
        (proposal_dir / NARRATIVE_FILE)
        .read_text(encoding="utf-8")
        .startswith("Reusable for daily morning briefs.")
    )

    proposal = load_skill_proposal(tmp_path, "fetch-top-github-repos")
    assert proposal.source_kind == "sandbox"
    assert proposal.sandbox_script_path == DEFAULT_SANDBOX_SCRIPT_NAME


def test_propose_uses_explicit_slug_when_provided(tmp_path: Path) -> None:
    result = propose_skill_from_sandbox(
        tmp_path,
        script=_SCRIPT,
        intent="anything goes",
        narrative="reason",
        slug="github-trending-daily",
    )

    assert result["ok"] is True
    assert result["slug"] == "github-trending-daily"
    assert (tmp_path / "config/skills/auto/github-trending-daily" / PROPOSAL_FILE).exists()


def test_propose_updates_existing_slug(tmp_path: Path) -> None:
    first = propose_skill_from_sandbox(tmp_path, script=_SCRIPT, intent="dup", narrative="first")
    second = propose_skill_from_sandbox(
        tmp_path, script="print('updated')\n", intent="dup", narrative="second"
    )

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["updated_existing"] is True
    assert second["slug"] == "dup"
    assert second["run_count"] == 2
    proposal_dir = tmp_path / str(second["proposal_dir"])
    assert (proposal_dir / DEFAULT_SANDBOX_SCRIPT_NAME).read_text(
        encoding="utf-8"
    ) == "print('updated')\n"
    assert (proposal_dir / NARRATIVE_FILE).read_text(encoding="utf-8") == "second\n"


def test_propose_rejects_empty_inputs(tmp_path: Path) -> None:
    assert (
        propose_skill_from_sandbox(tmp_path, script="   ", intent="x", narrative="y")["ok"] is False
    )
    assert (
        propose_skill_from_sandbox(tmp_path, script=_SCRIPT, intent="", narrative="y")["ok"]
        is False
    )
    assert (
        propose_skill_from_sandbox(tmp_path, script=_SCRIPT, intent="x", narrative="")["ok"]
        is False
    )
