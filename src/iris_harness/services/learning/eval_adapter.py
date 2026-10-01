"""Live execution adapter for the eval harness (ADR-0070, slice 1).

The harness core (:mod:`iris_harness.services.learning.eval_harness`) is pure — it needs a
``RunQuery`` that actually runs a query under a config and reports the outcome.
This module provides that, against a real ``IrisRuntime``, plus the isolation that
makes it safe.

Two safety properties matter, and both come from running against a *dedicated*
eval runtime, never production:

1. **No signal pollution.** Eval turns emit the same ``task_completed`` /
   ``turn_tokens`` learning signals as real turns. If those landed in the real
   ``learning.db`` they would corrupt the very measurements slice 1 reads. So the
   eval runtime is built with a **throwaway temp ``data_dir``** — its signals go to
   a scratch DB that is discarded.
2. **No production config leak.** The variant is applied by mutating the eval
   runtime's tier-router priors (`set_intent_tier_priors`); a separate runtime
   object means production routing is never touched.

Governance still applies: the eval runtime has its own governance kernel, so the
egress gate runs on eval turns exactly as in production (the sandbox isolates, it
does not bypass).

Note: the runtime build + the actual chat turns require a live model backend
(Ollama) and so are validated on a real host, not in unit tests. The pure mapping
(:func:`chat_result_to_outcome`) and the adapter wiring
(:func:`make_runtime_run_query`) are unit-tested with fakes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from iris_harness.services.learning.eval_harness import (
    COMPLETION_RATE,
    EvalComparison,
    EvalQuery,
    Metric,
    RunOutcome,
    RunQuery,
    VariantConfig,
    evaluate,
)

# A fixed session id for eval turns so they never mingle with real conversations.
_EVAL_SESSION_ID = "__eval_replay__"


def chat_result_to_outcome(result: Any) -> RunOutcome:
    """Map a ``ChatResult`` (duck-typed) to a :class:`RunOutcome`.

    ``completed`` is the absence of errors. ``tools_called`` is read from
    ``metadata`` when the runtime recorded it (keys ``tools`` or ``tools_called``);
    if it didn't, tool-correctness simply stays unmeasurable for that run rather
    than guessing.
    """
    completed = not bool(getattr(result, "has_errors", False))
    meta = getattr(result, "metadata", None) or {}
    raw_tools = meta.get("tools") or meta.get("tools_called") or ()
    if isinstance(raw_tools, str):
        raw_tools = [raw_tools]
    tools = tuple(str(t) for t in raw_tools if t)
    return RunOutcome(completed=completed, tools_called=tools)


def make_runtime_run_query(runtime: Any, *, session_id: str = _EVAL_SESSION_ID) -> RunQuery:
    """Build a ``RunQuery`` over an **isolated eval runtime**.

    ``runtime`` MUST be an eval runtime (throwaway ``data_dir``, scheduler off) — it
    mutates the runtime's intent-tier priors per arm and runs real turns whose
    signals land in that runtime's scratch ``learning.db``. Never pass a production
    runtime (see :func:`iris_harness.services.learning.eval_runtime.build_eval_runtime`).
    """

    def run(item: EvalQuery, config: VariantConfig) -> RunOutcome:
        # Apply the arm's config: variant priors, or {} for the baseline arm.
        runtime.tier_router.set_intent_tier_priors(dict(config.intent_tier_priors))
        result = runtime.chat(item.query, session_id=session_id)
        return chat_result_to_outcome(result)

    return run


def run_preflight(
    runtime: Any,
    workload: Sequence[EvalQuery],
    variant: VariantConfig,
    *,
    repeats: int = 1,
    metric: Metric = COMPLETION_RATE,
    min_improvement_pct: float = 5.0,
    min_runs: int = 1,
) -> EvalComparison:
    """Replay ``workload`` through an isolated eval ``runtime`` under baseline vs
    ``variant`` and return the comparison.

    The one-call pre-flight: a losing verdict can spare the user from ever applying
    a bad change; a winning one is a fast, clean signal to apply it and let slice 3
    confirm on live traffic. ``runtime`` must be an isolated eval runtime
    (:func:`iris_harness.services.learning.eval_runtime.build_eval_runtime`).
    """
    return evaluate(
        workload,
        run_query=make_runtime_run_query(runtime),
        variant=variant,
        repeats=repeats,
        metric=metric,
        min_improvement_pct=min_improvement_pct,
        min_runs=min_runs,
    )


__all__ = [
    "chat_result_to_outcome",
    "make_runtime_run_query",
    "run_preflight",
]
