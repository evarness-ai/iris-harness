"""Pure functions backing the ``/queue`` slash command.

Reads sandbox skill proposals from ``config/skills/auto/<slug>/proposal.yaml``
and supports inspecting, dismissing, and promoting them. Promotion flips
status to ``"promoting"`` and (optionally) invokes a runner — step 5 will
wire that runner to the ``iris-code`` pipeline; this module stays pure so
it's easy to unit-test.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from iris_harness.tools.skills.loader import (
    PROPOSAL_FILE,
    RUNS_FILE,
    load_skill_proposal,
    resolve_auto_skills_root,
    save_skill_proposal,
)
from iris_harness.tools.skills.models import SkillProposal

DEFAULT_VISIBLE_STATUSES: tuple[str, ...] = ("ready", "proposed")


@dataclass(frozen=True)
class ProposalSummary:
    """One-line view of a proposal for the queue listing."""

    slug: str
    skill_name: str
    status: str
    source_kind: str
    run_count: int
    promotion_threshold: int
    last_run_at: datetime | None


@dataclass(frozen=True)
class ProposalDetail:
    """Full view of a proposal for ``/queue show``."""

    proposal: SkillProposal
    narrative: str | None
    recent_runs: tuple[dict[str, object], ...]


def _iter_proposal_slugs(repo_root: Path) -> Iterable[str]:
    auto_root = resolve_auto_skills_root(repo_root)
    if not auto_root.exists():
        return ()
    return tuple(sorted(p.name for p in auto_root.iterdir() if (p / PROPOSAL_FILE).exists()))


def _summary(proposal: SkillProposal) -> ProposalSummary:
    return ProposalSummary(
        slug=proposal.skill_slug,
        skill_name=proposal.skill_name,
        status=proposal.status,
        source_kind=proposal.source_kind,
        run_count=proposal.run_count,
        promotion_threshold=proposal.promotion_threshold,
        last_run_at=proposal.last_run_at,
    )


def list_proposals(
    repo_root: Path,
    *,
    status_filter: tuple[str, ...] | None = DEFAULT_VISIBLE_STATUSES,
) -> tuple[ProposalSummary, ...]:
    """Return summaries for proposals whose status matches ``status_filter``.

    Pass ``status_filter=None`` to include every status (e.g., ``--all`` flag).
    Sorted by ``run_count`` descending so the most-used drafts surface first.
    """
    summaries: list[ProposalSummary] = []
    for slug in _iter_proposal_slugs(repo_root):
        try:
            proposal = load_skill_proposal(repo_root, slug)
        except (FileNotFoundError, ValueError):
            continue
        if status_filter is not None and proposal.status not in status_filter:
            continue
        summaries.append(_summary(proposal))
    summaries.sort(key=lambda s: (-s.run_count, s.slug))
    return tuple(summaries)


def _read_recent_runs(
    repo_root: Path, proposal: SkillProposal, *, limit: int = 5
) -> tuple[dict[str, object], ...]:
    runs_path = repo_root / proposal.proposal_dir / RUNS_FILE
    if not runs_path.exists():
        return ()
    lines = runs_path.read_text(encoding="utf-8").splitlines()
    parsed: list[dict[str, object]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return tuple(parsed)


def show_proposal(repo_root: Path, slug: str) -> ProposalDetail:
    """Return proposal metadata, narrative text, and recent run entries."""
    proposal = load_skill_proposal(repo_root, slug)
    proposal_dir = repo_root / proposal.proposal_dir
    narrative_path = proposal_dir / "NARRATIVE.md"
    narrative = narrative_path.read_text(encoding="utf-8") if narrative_path.exists() else None
    return ProposalDetail(
        proposal=proposal,
        narrative=narrative,
        recent_runs=_read_recent_runs(repo_root, proposal),
    )


def dismiss_proposal(repo_root: Path, slug: str) -> SkillProposal:
    """Flip a proposal's status to ``dismissed`` and persist the change."""
    proposal = load_skill_proposal(repo_root, slug)
    if proposal.status == "dismissed":
        return proposal
    updated = proposal.model_copy(update={"status": "dismissed"})
    save_skill_proposal(repo_root, updated)
    return updated


def promote_proposal(
    repo_root: Path,
    slug: str,
    *,
    runner: Callable[[Path, SkillProposal], None] | None = None,
    preflight: Callable[[SkillProposal], Any] | None = None,
) -> SkillProposal:
    """Flip a proposal to ``promoting`` and (optionally) hand it to a runner.

    The runner is the integration seam for the ``iris-code`` coding-agent
    pipeline; step 5 wires the real implementation. Tests pass a mock here.
    Promotion is rejected if the proposal isn't ``proposed``/``ready``/``promoting``.
    ``promoting`` is treated as idempotent so a stale or interrupted coding-task
    handoff can be staged again without forcing the user to mutate proposal state.

    ``preflight`` (ADR-0070): an optional replay-eval gate run BEFORE flipping
    status. If it returns a failing verdict the proposal is left untouched and
    ``PreflightError`` is raised — so a stale/regressed intent is never handed to
    the coding agent.
    """
    proposal = load_skill_proposal(repo_root, slug)
    if proposal.status not in ("proposed", "ready", "promoting"):
        raise ValueError(
            f"cannot promote proposal in status '{proposal.status}'; "
            "expected 'proposed', 'ready', or 'promoting'"
        )
    if preflight is not None:
        from iris_harness.services.learning.preflight import PreflightError

        verdict = preflight(proposal)
        if not verdict.passed:
            raise PreflightError(verdict)
    updated = (
        proposal
        if proposal.status == "promoting"
        else proposal.model_copy(update={"status": "promoting"})
    )
    save_skill_proposal(repo_root, updated)
    if runner is not None:
        runner(repo_root / updated.proposal_dir, updated)
    return updated


def queue_stats(repo_root: Path) -> dict[str, int]:
    """Return proposal counts grouped by status (and a ``total`` key)."""
    counts: dict[str, int] = {}
    total = 0
    for slug in _iter_proposal_slugs(repo_root):
        try:
            proposal = load_skill_proposal(repo_root, slug)
        except (FileNotFoundError, ValueError):
            continue
        counts[proposal.status] = counts.get(proposal.status, 0) + 1
        total += 1
    counts["total"] = total
    return counts
