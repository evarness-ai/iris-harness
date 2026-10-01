"""ADR-0072 slice 3 — the judge-gated clarify-nudge decision (`_clarify_warranted`).

Pure decision logic: a strong signal (judge `warn`/`retry`) or a weak one (very low
router confidence) warrants a clarifying nudge, but errored / already-clarified /
clarify-agent turns and confident clean answers never do.
"""

from __future__ import annotations

from iris_harness.runtime.facade import _clarify_warranted


def _bundle(*verdicts: str) -> dict:
    return {"signals": [{"name": f"s{i}", "verdict": v} for i, v in enumerate(verdicts)]}


def test_judge_warn_warrants_clarify() -> None:
    assert _clarify_warranted(
        intent_confidence=0.9,
        judge_bundle=_bundle("pass", "warn"),
        has_errors=False,
        already_clarified=False,
        agent_type="email",
    )


def test_judge_retry_warrants_clarify() -> None:
    assert _clarify_warranted(
        intent_confidence=0.9,
        judge_bundle=_bundle("retry"),
        has_errors=False,
        already_clarified=False,
        agent_type="general",
    )


def test_low_router_confidence_warrants_clarify() -> None:
    assert _clarify_warranted(
        intent_confidence=0.2,
        judge_bundle=_bundle("pass"),
        has_errors=False,
        already_clarified=False,
        agent_type="general",
    )


def test_confident_clean_answer_is_not_nagged() -> None:
    assert not _clarify_warranted(
        intent_confidence=0.95,
        judge_bundle=_bundle("pass", "skipped"),
        has_errors=False,
        already_clarified=False,
        agent_type="email",
    )


def test_no_double_ask_or_error_or_clarify_agent() -> None:
    strong = _bundle("warn")
    # errored turn → the error is already surfaced; don't pile on.
    assert not _clarify_warranted(
        intent_confidence=0.1,
        judge_bundle=strong,
        has_errors=True,
        already_clarified=False,
        agent_type="email",
    )
    # escalation already asked a clarifying question this turn.
    assert not _clarify_warranted(
        intent_confidence=0.1,
        judge_bundle=strong,
        has_errors=False,
        already_clarified=True,
        agent_type="email",
    )
    # the clarify agent's own turn is already a question.
    assert not _clarify_warranted(
        intent_confidence=0.1,
        judge_bundle=strong,
        has_errors=False,
        already_clarified=False,
        agent_type="clarify",
    )


def test_missing_bundle_and_confidence_is_safe() -> None:
    assert not _clarify_warranted(
        intent_confidence=None,
        judge_bundle=None,
        has_errors=False,
        already_clarified=False,
        agent_type="general",
    )
