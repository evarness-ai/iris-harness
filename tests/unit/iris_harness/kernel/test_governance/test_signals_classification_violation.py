from __future__ import annotations

import pytest

from iris_harness.kernel.governance.evaluator import StepRecord
from iris_harness.kernel.governance.evaluator.signals import ClassificationViolationSignal


def _step(
    *,
    tier: str | None = None,
    classification: str | None = None,
) -> StepRecord:
    return StepRecord(
        run_id="r",
        step_id=1,
        agent_type="chat",
        tier=tier,  # type: ignore[arg-type]
        classification=classification,  # type: ignore[arg-type]
    )


def test_local_tiers_never_halt_regardless_of_class() -> None:
    signal = ClassificationViolationSignal()
    for tier in ("tier_1", "tier_2"):
        for classification in ("public", "internal", "personal", "secret"):
            result = signal(_step(tier=tier, classification=classification), state={})
            assert result.verdict == "ok", (tier, classification)


def test_cloud_with_public_or_internal_is_ok() -> None:
    signal = ClassificationViolationSignal()
    assert signal(_step(tier="tier_3", classification="public"), state={}).verdict == "ok"
    assert signal(_step(tier="tier_3", classification="internal"), state={}).verdict == "ok"


@pytest.mark.parametrize("classification", ["personal", "secret"])
def test_cloud_with_personal_or_secret_halts_critical(classification: str) -> None:
    signal = ClassificationViolationSignal()
    result = signal(
        _step(tier="tier_3", classification=classification),
        state={},
    )
    assert result.verdict == "halt"
    assert result.severity == "critical"
    assert result.audit_metadata["classification"] == classification
    assert result.audit_metadata["tier"] == "tier_3"


def test_unclassified_step_is_ok() -> None:
    signal = ClassificationViolationSignal()
    # No tier yet → not the signal's job to halt (egress_gate decides on
    # the in-flight call); the signal only catches confirmed violations.
    assert signal(_step(), state={}).verdict == "ok"
