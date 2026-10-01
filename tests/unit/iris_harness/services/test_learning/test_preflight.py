"""ADR-0070 replay-eval pre-flight for crystallized proposals — pure scoring +
the promote_proposal gate (both injectable, no model needed)."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from iris_harness.cli.queue import promote_proposal
from iris_harness.services.learning.eval_harness import EvalQuery, RunOutcome
from iris_harness.services.learning.preflight import (
    PreflightError,
    proposal_intent,
    replay_preflight,
)
from iris_harness.tools.skills.loader import scaffold_skill_proposal
from iris_harness.tools.skills.models import SkillProposal


def _workload(n: int) -> list[EvalQuery]:
    return [EvalQuery(query=f"q{i}", intent="calendar") for i in range(n)]


def _run_query(completions: list[bool]):
    it: Iterator[bool] = iter(completions)

    def rq(_query: EvalQuery, _config: object) -> RunOutcome:
        return RunOutcome(completed=next(it))

    return rq


# ── pure scoring ─────────────────────────────────────────────────────────────


def test_replay_preflight_passes_on_clean_completion() -> None:
    v = replay_preflight(_workload(3), run_query=_run_query([True, True, True]))
    assert v.passed is True
    assert v.completion_rate == 1.0
    assert v.queries == 3


def test_replay_preflight_fails_below_threshold() -> None:
    # 2/5 complete = 40% < 80% target
    v = replay_preflight(_workload(5), run_query=_run_query([True, True, False, False, False]))
    assert v.passed is False
    assert v.completion_rate == pytest.approx(0.4)
    assert "<" in v.reason


def test_replay_preflight_insufficient_workload() -> None:
    v = replay_preflight(_workload(1), run_query=_run_query([True]))
    assert v.passed is False
    assert "not enough replay queries" in v.reason


def test_proposal_intent_from_signal_task_id() -> None:
    p = SkillProposal(
        proposal_id="p1",
        task_id="signal:calendar",
        skill_name="cal",
        skill_slug="calendar-calendar",
        scope="learning",
        source_description="x",
        proposal_dir="config/skills/auto/calendar-calendar",
        manifest_path="config/skills/auto/calendar-calendar/manifest.yaml",
        source_kind="crystallized",
    )
    assert proposal_intent(p) == "calendar"


# ── promote_proposal gate ────────────────────────────────────────────────────


def _crystallized(repo_root: Path, slug: str = "calendar-calendar") -> SkillProposal:
    proposal = SkillProposal(
        proposal_id=f"prop-{slug}",
        task_id="signal:calendar",
        skill_name="calendar-management",
        skill_slug=slug,
        scope="learning",
        source_description="recurring SUCCESS: intent 'calendar'",
        proposal_dir=f"config/skills/auto/{slug}",
        manifest_path=f"config/skills/auto/{slug}/manifest.yaml",
        source_kind="crystallized",
    )
    scaffold_skill_proposal(repo_root, proposal)
    return proposal


def test_promote_blocked_when_preflight_fails(tmp_path: Path) -> None:
    from iris_harness.cli.queue import load_skill_proposal
    from iris_harness.services.learning.preflight import PreflightVerdict

    _crystallized(tmp_path)
    fail = PreflightVerdict(False, 0.4, 5, 5, "replay completion 40% < 80%")

    with pytest.raises(PreflightError):
        promote_proposal(tmp_path, "calendar-calendar", preflight=lambda _p: fail)

    # left untouched — NOT flipped to promoting
    assert load_skill_proposal(tmp_path, "calendar-calendar").status == "proposed"


def test_promote_proceeds_when_preflight_passes(tmp_path: Path) -> None:
    from iris_harness.services.learning.preflight import PreflightVerdict

    _crystallized(tmp_path)
    ok = PreflightVerdict(True, 1.0, 3, 3, "replay completion 100% >= 80%")

    updated = promote_proposal(tmp_path, "calendar-calendar", preflight=lambda _p: ok)
    assert updated.status == "promoting"


def test_build_proposal_preflight_fails_without_workload(tmp_path: Path) -> None:
    # No model needed: an empty workload loader → fail-closed before any runtime build.
    from iris_harness.services.learning.preflight import build_proposal_preflight

    proposal = _crystallized(tmp_path)
    preflight = build_proposal_preflight(tmp_path, workload_loader=lambda _intent, _n: [])
    verdict = preflight(proposal)
    assert verdict.passed is False
    assert "no replay workload" in verdict.reason


# ── sandbox modes (ADR-0070; default-on effect containment) ──────────────────


def test_resolve_eval_sandbox_mode_defaults_on(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.services.learning.preflight import resolve_eval_sandbox_mode

    monkeypatch.delenv("IRIS_EVAL_SANDBOX", raising=False)
    assert resolve_eval_sandbox_mode() == "effect"  # default ON
    monkeypatch.setenv("IRIS_EVAL_SANDBOX", "off")
    assert resolve_eval_sandbox_mode() == "off"
    monkeypatch.setenv("IRIS_EVAL_SANDBOX", "docker")
    assert resolve_eval_sandbox_mode() == "docker"
    assert resolve_eval_sandbox_mode("off") == "off"  # explicit arg wins


def test_effect_mode_disables_external_writes_and_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.services.learning.preflight import build_proposal_preflight

    monkeypatch.delenv("IRIS_DISABLE_EXTERNAL_WRITES", raising=False)
    seen: dict[str, str | None] = {}

    def _fake_runtime() -> object:
        seen["during"] = os.environ.get("IRIS_DISABLE_EXTERNAL_WRITES")
        return object()

    # 3 queries so we pass the min_queries guard and reach the runtime build.
    preflight = build_proposal_preflight(
        tmp_path,
        sandbox="effect",
        runtime_factory=_fake_runtime,
        workload_loader=lambda _i, _n: [
            EvalQuery(query=f"q{i}", intent="calendar") for i in range(3)
        ],
    )
    # make_runtime_run_query will wrap our dummy runtime; the dummy has no .chat,
    # so score_arm scores each run as a non-completion — fine, we only assert env.
    verdict = preflight(_crystallized_intent_proposal())
    assert seen["during"] == "1"  # external writes disabled DURING the eval
    assert os.environ.get("IRIS_DISABLE_EXTERNAL_WRITES") is None  # restored after
    assert verdict.queries == 3


def test_docker_mode_refuses_when_unprovisioned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.services.learning import eval_docker
    from iris_harness.services.learning.preflight import (
        SandboxUnavailable,
        build_proposal_preflight,
    )

    # Force the "docker not on PATH" path so this is deterministic regardless of
    # whether the host actually has Docker + the iris-eval image provisioned.
    monkeypatch.setattr(eval_docker.shutil, "which", lambda _name: None)

    preflight = build_proposal_preflight(
        tmp_path,
        sandbox="docker",
        workload_loader=lambda _i, _n: [
            EvalQuery(query=f"q{i}", intent="calendar") for i in range(3)
        ],
    )
    with pytest.raises(SandboxUnavailable, match="docker"):
        preflight(_crystallized_intent_proposal())


def _crystallized_intent_proposal() -> SkillProposal:
    return SkillProposal(
        proposal_id="p1",
        task_id="signal:calendar",
        skill_name="cal",
        skill_slug="calendar-calendar",
        scope="learning",
        source_description="x",
        proposal_dir="config/skills/auto/calendar-calendar",
        manifest_path="config/skills/auto/calendar-calendar/manifest.yaml",
        source_kind="crystallized",
    )
