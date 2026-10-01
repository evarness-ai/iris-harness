"""Replay-eval pre-flight for crystallized skill proposals (ADR-0070 + ADR-0068).

A crystallized proposal is mined from *historical* successes. Before we hand it
to the coding agent (``/queue promote``), this re-checks that the intent still
completes cleanly **now**: replay a frozen workload of the intent's recent
queries through the live agent and require a minimum completion rate.

The scoring is the single-arm case of the ADR-0070 harness (:func:`score_arm`),
so it's pure given an injected ``run_query`` — unit-testable without a model. The
heavy glue (isolated runtime + workload sourcing) lives in :func:`build_proposal_preflight`.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iris_harness.services.learning.eval_harness import (
    EvalQuery,
    RunQuery,
    VariantConfig,
    score_arm,
)
from iris_harness.tools.skills.models import SkillProposal

logger = logging.getLogger(__name__)

DEFAULT_MIN_QUERIES = 3
DEFAULT_MIN_COMPLETION_RATE = 0.8

# Eval sandbox modes (IRIS_EVAL_SANDBOX), default "effect":
#   effect — in-process isolated runtime + external side-effects neutralized (default)
#   docker — full gVisor/Docker container (opt-in escalation; needs the image + Ollama egress)
#   off    — in-process, no effect containment (debugging only; can touch the real world)
SANDBOX_EFFECT = "effect"
SANDBOX_DOCKER = "docker"
SANDBOX_OFF = "off"
_VALID_SANDBOX = {SANDBOX_EFFECT, SANDBOX_DOCKER, SANDBOX_OFF}


class SandboxUnavailable(RuntimeError):
    """The requested eval sandbox mode can't run in this environment."""


def resolve_eval_sandbox_mode(explicit: str | None = None) -> str:
    """Eval sandbox mode: explicit arg > IRIS_EVAL_SANDBOX > 'effect' (default on)."""
    raw = (explicit or os.getenv("IRIS_EVAL_SANDBOX") or SANDBOX_EFFECT).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return SANDBOX_EFFECT
    if raw in {"0", "false", "no"}:
        return SANDBOX_OFF
    return raw if raw in _VALID_SANDBOX else SANDBOX_EFFECT


@contextmanager
def _effect_containment() -> Iterator[None]:
    """Scope external-side-effect lockdown to the eval, then restore the parent env.

    Sets ``IRIS_DISABLE_EXTERNAL_WRITES=1`` (+ calendar apple-write off) so a replayed
    write can only touch the eval's scratch stores, never the real Apple/Google
    calendar or other external systems. Restored on exit so the parent is untouched.
    """
    overrides = {"IRIS_DISABLE_EXTERNAL_WRITES": "1", "IRIS_CALENDAR_APPLE_WRITE": "0"}
    prior = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass(frozen=True)
class PreflightVerdict:
    """Outcome of a crystallized proposal's replay pre-flight."""

    passed: bool
    completion_rate: float
    runs: int
    queries: int
    reason: str


class PreflightError(RuntimeError):
    """Raised to block promotion when the pre-flight fails. Carries the verdict."""

    def __init__(self, verdict: PreflightVerdict) -> None:
        super().__init__(verdict.reason)
        self.verdict = verdict


def proposal_intent(proposal: SkillProposal) -> str:
    """The intent a crystallized proposal serves (``task_id`` is ``signal:<intent>``)."""
    task_id = proposal.task_id or ""
    if ":" in task_id:
        return task_id.split(":", 1)[1].strip()
    return task_id.strip()


def replay_preflight(
    workload: Sequence[EvalQuery],
    *,
    run_query: RunQuery,
    repeats: int = 1,
    min_queries: int = DEFAULT_MIN_QUERIES,
    min_completion_rate: float = DEFAULT_MIN_COMPLETION_RATE,
) -> PreflightVerdict:
    """Replay ``workload`` once-per-query (×``repeats``) and gate on completion rate."""
    n = len(workload)
    if n < min_queries:
        return PreflightVerdict(
            passed=False,
            completion_rate=0.0,
            runs=0,
            queries=n,
            reason=f"not enough replay queries ({n} < {min_queries}) — insufficient evidence",
        )
    score = score_arm(
        workload, run_query=run_query, config=VariantConfig(label="preflight"), repeats=repeats
    )
    passed = score.completion_rate >= min_completion_rate
    comparator = ">=" if passed else "<"
    return PreflightVerdict(
        passed=passed,
        completion_rate=score.completion_rate,
        runs=score.runs,
        queries=n,
        reason=(
            f"replay completion {score.completion_rate:.0%} {comparator} "
            f"{min_completion_rate:.0%} target over {n} queries ({score.runs} runs)"
        ),
    )


def build_proposal_preflight(
    repo_root: Path,
    *,
    repeats: int = 1,
    max_items: int = 8,
    min_queries: int = DEFAULT_MIN_QUERIES,
    min_completion_rate: float = DEFAULT_MIN_COMPLETION_RATE,
    runtime_factory: Callable[[], Any] | None = None,
    workload_loader: Callable[[str, int], Sequence[EvalQuery]] | None = None,
    sandbox: str | None = None,
) -> Callable[[SkillProposal], PreflightVerdict]:
    """Live pre-flight: replays the proposal's intent through an ISOLATED eval
    runtime (throwaway temp data dir, so signals never reach the real store).

    Sandbox modes (``IRIS_EVAL_SANDBOX``, default ``effect``): ``effect`` neutralizes
    external side effects (default on); ``docker`` is the full-container escalation;
    ``off`` disables containment. ``runtime_factory`` (no-arg) / ``workload_loader``
    are injectable for tests; the defaults need a model backend. ``repo_root`` is
    reserved for future repo-scoped workloads.
    """
    _ = repo_root  # reserved; default workload reads the shared trace store
    mode = resolve_eval_sandbox_mode(sandbox)

    def _default_workload(intent: str, limit: int) -> Sequence[EvalQuery]:
        from iris_harness.services.learning.eval_workload import (
            build_workload_from_traces,
        )

        return build_workload_from_traces(intent=intent, max_items=limit)

    load = workload_loader or _default_workload

    def _run_replay(workload: Sequence[EvalQuery]) -> PreflightVerdict:
        from iris_harness.services.learning.eval_adapter import (
            make_runtime_run_query,
        )

        scratch: Path | None = None
        if runtime_factory is not None:
            runtime = runtime_factory()
        else:
            from iris_harness.services.learning.eval_runtime import (
                build_eval_runtime,
            )

            runtime, scratch = build_eval_runtime()
        try:
            return replay_preflight(
                workload,
                run_query=make_runtime_run_query(runtime),
                repeats=repeats,
                min_queries=min_queries,
                min_completion_rate=min_completion_rate,
            )
        finally:
            if scratch is not None:
                import shutil

                shutil.rmtree(scratch, ignore_errors=True)

    def _preflight(proposal: SkillProposal) -> PreflightVerdict:
        intent = proposal_intent(proposal)
        workload = list(load(intent, max_items))
        if len(workload) < min_queries:
            return PreflightVerdict(
                passed=False,
                completion_rate=0.0,
                runs=0,
                queries=len(workload),
                reason=(
                    f"no replay workload for intent '{intent}' "
                    f"({len(workload)} < {min_queries} queries)"
                ),
            )

        if mode == SANDBOX_DOCKER:
            # Full gVisor/Docker isolation: run the entrypoint in the iris-eval image
            # (no host creds mounted). Raises SandboxUnavailable if docker/image absent.
            from iris_harness.services.learning.eval_docker import (
                run_eval_in_docker,
            )

            return run_eval_in_docker(
                proposal_intent(proposal),
                workload,
                repeats=repeats,
                max_items=max_items,
                min_queries=min_queries,
                min_rate=min_completion_rate,
            )
        if mode == SANDBOX_OFF:
            return _run_replay(workload)  # no containment (debugging)
        # effect (default): neutralize external side effects for the eval's lifetime
        with _effect_containment():
            return _run_replay(workload)

    return _preflight


__all__ = [
    "PreflightVerdict",
    "PreflightError",
    "SandboxUnavailable",
    "resolve_eval_sandbox_mode",
    "replay_preflight",
    "build_proposal_preflight",
    "proposal_intent",
    "DEFAULT_MIN_QUERIES",
    "DEFAULT_MIN_COMPLETION_RATE",
]
