"""Tests for the pure functions backing the /queue slash command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from iris_harness.cli.queue import (
    DEFAULT_VISIBLE_STATUSES,
    dismiss_proposal,
    list_proposals,
    promote_proposal,
    queue_stats,
    show_proposal,
)
from iris_harness.tools.skills.loader import (
    DEFAULT_SANDBOX_SCRIPT_NAME,
    record_sandbox_run,
    scaffold_skill_proposal,
)
from iris_harness.tools.skills.models import SkillProposal


def _make_sandbox_proposal(slug: str, *, threshold: int = 3) -> SkillProposal:
    return SkillProposal(
        proposal_id=f"prop-{slug}",
        task_id=f"sandbox:{slug}",
        skill_name=slug.replace("-", " ").title(),
        skill_slug=slug,
        scope="sandbox",
        source_description=f"demo {slug}",
        proposal_dir=f"config/skills/auto/{slug}",
        manifest_path=f"config/skills/auto/{slug}/manifest.yaml",
        source_kind="sandbox",
        sandbox_script_path=DEFAULT_SANDBOX_SCRIPT_NAME,
        promotion_threshold=threshold,
    )


def _scaffold(repo_root: Path, slug: str, *, threshold: int = 3) -> SkillProposal:
    proposal = _make_sandbox_proposal(slug, threshold=threshold)
    scaffold_skill_proposal(repo_root, proposal, sandbox_script="x = 1\n")
    return proposal


def test_list_proposals_filters_by_status_and_sorts_by_run_count(tmp_path: Path) -> None:
    _scaffold(tmp_path, "alpha")
    _scaffold(tmp_path, "beta")
    record_sandbox_run(tmp_path, "alpha")
    record_sandbox_run(tmp_path, "alpha")

    result = list_proposals(tmp_path)
    slugs = [s.slug for s in result]
    assert slugs == ["alpha", "beta"]  # alpha has higher run_count
    assert all(s.status in DEFAULT_VISIBLE_STATUSES for s in result)


def test_list_proposals_excludes_dismissed_by_default(tmp_path: Path) -> None:
    _scaffold(tmp_path, "alpha")
    _scaffold(tmp_path, "beta")
    dismiss_proposal(tmp_path, "beta")

    visible = list_proposals(tmp_path)
    assert [s.slug for s in visible] == ["alpha"]

    everything = list_proposals(tmp_path, status_filter=None)
    assert sorted(s.slug for s in everything) == ["alpha", "beta"]


def test_show_proposal_returns_narrative_and_recent_runs(tmp_path: Path) -> None:
    _scaffold(tmp_path, "alpha")
    narrative_path = tmp_path / "config/skills/auto/alpha/NARRATIVE.md"
    narrative_path.write_text("Useful for daily briefs.\n", encoding="utf-8")
    record_sandbox_run(tmp_path, "alpha", args={"q": "stars"}, exit_code=0, duration_ms=42)

    detail = show_proposal(tmp_path, "alpha")
    assert detail.proposal.skill_slug == "alpha"
    assert detail.narrative is not None
    assert "Useful for daily briefs." in detail.narrative
    assert len(detail.recent_runs) == 1
    assert detail.recent_runs[0]["exit_code"] == 0
    assert detail.recent_runs[0]["duration_ms"] == 42


def test_show_proposal_raises_for_unknown_slug(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        show_proposal(tmp_path, "nope")


def test_dismiss_proposal_flips_status_and_persists(tmp_path: Path) -> None:
    _scaffold(tmp_path, "alpha")

    updated = dismiss_proposal(tmp_path, "alpha")
    assert updated.status == "dismissed"

    # Idempotent: dismissing again returns the same status.
    again = dismiss_proposal(tmp_path, "alpha")
    assert again.status == "dismissed"


def test_promote_proposal_flips_to_promoting(tmp_path: Path) -> None:
    _scaffold(tmp_path, "alpha")

    updated = promote_proposal(tmp_path, "alpha")
    assert updated.status == "promoting"


def test_promote_proposal_invokes_runner_with_proposal_dir(tmp_path: Path) -> None:
    _scaffold(tmp_path, "alpha")
    runner = MagicMock()

    updated = promote_proposal(tmp_path, "alpha", runner=runner)

    assert updated.status == "promoting"
    runner.assert_called_once()
    proposal_dir_arg, proposal_arg = runner.call_args.args
    assert proposal_dir_arg == tmp_path / "config/skills/auto/alpha"
    assert proposal_arg.skill_slug == "alpha"
    assert proposal_arg.status == "promoting"


def test_promote_proposal_rejects_non_promotable_status(tmp_path: Path) -> None:
    _scaffold(tmp_path, "alpha")
    dismiss_proposal(tmp_path, "alpha")

    with pytest.raises(ValueError, match="cannot promote proposal in status 'dismissed'"):
        promote_proposal(tmp_path, "alpha")


def test_queue_stats_groups_by_status(tmp_path: Path) -> None:
    _scaffold(tmp_path, "alpha")
    _scaffold(tmp_path, "beta")
    _scaffold(tmp_path, "gamma")
    dismiss_proposal(tmp_path, "gamma")

    counts = queue_stats(tmp_path)
    assert counts["proposed"] == 2
    assert counts["dismissed"] == 1
    assert counts["total"] == 3


def test_queue_stats_empty_when_no_proposals(tmp_path: Path) -> None:
    counts = queue_stats(tmp_path)
    assert counts == {"total": 0}
