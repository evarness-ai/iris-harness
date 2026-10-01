"""Schema and lifecycle tests for sandbox-origin :class:`SkillProposal` flow."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from iris_harness.tools.skills.loader import (
    PROPOSAL_FILE,
    RUNS_FILE,
    TOOLS_FILE,
    load_skill_proposal,
    record_sandbox_run,
    scaffold_skill_proposal,
)
from iris_harness.tools.skills.models import SkillProposal


def _base_kwargs() -> dict[str, object]:
    return {
        "proposal_id": "prop-abc",
        "task_id": "task-1",
        "skill_name": "demo skill",
        "skill_slug": "demo-skill",
        "scope": "learning",
        "source_description": "demo",
        "proposal_dir": "config/skills/auto/demo-skill",
        "manifest_path": "config/skills/auto/demo-skill/manifest.yaml",
    }


def test_proposal_defaults_preserve_crystallized_origin() -> None:
    proposal = SkillProposal(**_base_kwargs())

    assert proposal.source_kind == "crystallized"
    assert proposal.sandbox_script_path is None
    assert proposal.run_count == 0
    assert proposal.last_run_at is None
    assert proposal.wiki_page_id is None
    assert proposal.promotion_threshold == 3


def test_proposal_accepts_sandbox_origin_fields() -> None:
    last_run = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)
    proposal = SkillProposal(
        **_base_kwargs(),
        source_kind="sandbox",
        sandbox_script_path="sandbox_script.py",
        run_count=2,
        last_run_at=last_run,
        wiki_page_id="wiki-42",
        promotion_threshold=5,
    )

    assert proposal.source_kind == "sandbox"
    assert proposal.sandbox_script_path == "sandbox_script.py"
    assert proposal.run_count == 2
    assert proposal.last_run_at == last_run
    assert proposal.wiki_page_id == "wiki-42"
    assert proposal.promotion_threshold == 5


def test_proposal_rejects_unknown_source_kind() -> None:
    with pytest.raises(ValidationError):
        SkillProposal(**_base_kwargs(), source_kind="bogus")  # type: ignore[arg-type]


def test_proposal_rejects_negative_run_count() -> None:
    with pytest.raises(ValidationError):
        SkillProposal(**_base_kwargs(), run_count=-1)


def test_proposal_rejects_zero_promotion_threshold() -> None:
    with pytest.raises(ValidationError):
        SkillProposal(**_base_kwargs(), promotion_threshold=0)


# ── Scaffold + run-recording lifecycle ────────────────────────────────────────


def _sandbox_proposal(slug: str = "demo-skill") -> SkillProposal:
    return SkillProposal(
        **{
            **_base_kwargs(),
            "skill_slug": slug,
            "proposal_dir": f"config/skills/auto/{slug}",
            "manifest_path": f"config/skills/auto/{slug}/manifest.yaml",
        },
        source_kind="sandbox",
        sandbox_script_path="sandbox_script.py",
        promotion_threshold=3,
    )


def test_scaffold_sandbox_writes_script_and_skips_tools(tmp_path: Path) -> None:
    proposal = _sandbox_proposal()
    proposal_dir, created = scaffold_skill_proposal(
        tmp_path, proposal, sandbox_script="print('hello')\n"
    )

    assert created is True
    assert (proposal_dir / "sandbox_script.py").read_text(encoding="utf-8") == "print('hello')\n"
    assert (proposal_dir / RUNS_FILE).exists()
    assert (proposal_dir / RUNS_FILE).read_text(encoding="utf-8") == ""
    assert (proposal_dir / PROPOSAL_FILE).exists()
    # Sandbox proposals don't get the placeholder tools.py / __init__.py.
    assert not (proposal_dir / TOOLS_FILE).exists()
    assert not (proposal_dir / "__init__.py").exists()


def test_scaffold_crystallized_keeps_existing_artifacts(tmp_path: Path) -> None:
    proposal = SkillProposal(**_base_kwargs())  # default source_kind='crystallized'
    proposal_dir, _ = scaffold_skill_proposal(tmp_path, proposal)

    assert (proposal_dir / TOOLS_FILE).exists()
    assert (proposal_dir / "__init__.py").exists()
    # proposal.yaml is now persisted for both origins so /queue can read it.
    assert (proposal_dir / PROPOSAL_FILE).exists()


def test_scaffold_sandbox_requires_script_content(tmp_path: Path) -> None:
    proposal = _sandbox_proposal()
    with pytest.raises(ValueError, match="sandbox_script is required"):
        scaffold_skill_proposal(tmp_path, proposal)


def test_record_sandbox_run_appends_to_jsonl_and_bumps_count(tmp_path: Path) -> None:
    proposal = _sandbox_proposal()
    scaffold_skill_proposal(tmp_path, proposal, sandbox_script="x = 1\n")

    updated = record_sandbox_run(
        tmp_path, proposal.skill_slug, args={"q": "stars"}, exit_code=0, duration_ms=120
    )

    assert updated.run_count == 1
    assert updated.last_run_at is not None
    assert updated.status == "proposed"  # still below threshold of 3

    runs_path = tmp_path / proposal.proposal_dir / RUNS_FILE
    lines = runs_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["args"] == {"q": "stars"}
    assert entry["exit_code"] == 0
    assert entry["duration_ms"] == 120


def test_record_sandbox_run_flips_to_ready_at_threshold(tmp_path: Path) -> None:
    proposal = _sandbox_proposal()
    scaffold_skill_proposal(tmp_path, proposal, sandbox_script="x = 1\n")

    record_sandbox_run(tmp_path, proposal.skill_slug)
    record_sandbox_run(tmp_path, proposal.skill_slug)
    after_third = record_sandbox_run(tmp_path, proposal.skill_slug)

    assert after_third.run_count == 3
    assert after_third.status == "ready"

    reloaded = load_skill_proposal(tmp_path, proposal.skill_slug)
    assert reloaded.status == "ready"
    assert reloaded.run_count == 3


def test_record_sandbox_run_failed_run_does_not_count(tmp_path: Path) -> None:
    proposal = _sandbox_proposal()
    scaffold_skill_proposal(tmp_path, proposal, sandbox_script="x = 1\n")

    updated = record_sandbox_run(tmp_path, proposal.skill_slug, exit_code=1)

    assert updated.run_count == 0
    assert updated.status == "proposed"
    # Failed run is still logged for telemetry.
    runs_path = tmp_path / proposal.proposal_dir / RUNS_FILE
    assert len(runs_path.read_text(encoding="utf-8").splitlines()) == 1


def test_record_sandbox_run_rejects_non_sandbox_proposal(tmp_path: Path) -> None:
    proposal = SkillProposal(**_base_kwargs())  # crystallized
    scaffold_skill_proposal(tmp_path, proposal)

    with pytest.raises(ValueError, match="source_kind='sandbox'"):
        record_sandbox_run(tmp_path, proposal.skill_slug)
