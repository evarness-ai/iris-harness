"""The opt-in input safety screen: Llama Guard on the user turn at PRE_TURN.

Step b2 of docs/architecture/deterministic-path-parity.md. Off by default; follows the
threat config's mode; a slow or missing model lets the turn through.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from iris_harness.kernel.governance import HookContext, HookPoint, kernel_from_env
from iris_harness.kernel.governance.plugins.input_safety import InputSafetyHook
from iris_harness.kernel.governance.threat import ThreatDetectionConfig
from iris_harness.kernel.governance.threat.types import ThreatVerdict
from iris_harness.kernel.governance.wiring import _input_safety_from_env


class _Stub:
    """A classifier that returns one fixed verdict."""

    name = "stub"

    def __init__(self, verdict: ThreatVerdict) -> None:
        self._verdict = verdict

    async def score(self, *, text: str, surface: str) -> ThreatVerdict:
        return self._verdict


def _unsafe(*categories: str) -> _Stub:
    return _Stub(
        ThreatVerdict(
            label="unsafe", score=1.0, surface="inbound", backend="stub", categories=categories
        )
    )


def _hook(stub: _Stub, *, shadow: bool) -> InputSafetyHook:
    return InputSafetyHook(
        classifier=stub,  # type: ignore[arg-type]
        enforce=frozenset({"self_harm", "violent_crimes"}),
        log_only=frozenset({"hate"}),
        shadow=shadow,
    )


def _ctx(message: str = "a user message") -> HookContext:
    return HookContext(
        hook_point=HookPoint.PRE_TURN, run_id="t", agent_type="turn", payload={"message": message}
    )


# -- the hook -------------------------------------------------------------------------


async def test_enforce_refuses_an_enforced_category() -> None:
    decision = await _hook(_unsafe("self_harm"), shadow=False)(_ctx())
    assert decision.outcome == "deny" and decision.audit_metadata["enforced"] == ["self_harm"]


async def test_shadow_audits_but_allows() -> None:
    decision = await _hook(_unsafe("self_harm"), shadow=True)(_ctx())
    assert decision.outcome == "allow" and decision.severity == "critical"
    assert decision.audit_metadata["shadow"] is True


async def test_a_log_only_category_is_audited_and_allowed() -> None:
    decision = await _hook(_unsafe("hate"), shadow=False)(_ctx())
    assert decision.outcome == "allow" and decision.severity == "warn"


async def test_an_uncategorized_unsafe_verdict_fails_safe() -> None:
    assert (await _hook(_unsafe(), shadow=False)(_ctx())).outcome == "deny"


async def test_a_missing_model_lets_the_turn_through() -> None:
    down = _Stub(ThreatVerdict.failure(surface="inbound", backend="stub", detail="timeout"))
    decision = await _hook(down, shadow=False)(_ctx())
    assert decision.outcome == "allow" and decision.severity == "warn"


async def test_safe_and_empty_messages_pass() -> None:
    safe = _Stub(ThreatVerdict.benign(surface="inbound", backend="stub"))
    assert (await _hook(safe, shadow=False)(_ctx())).outcome == "allow"
    assert (await _hook(_unsafe("self_harm"), shadow=False)(_ctx("   "))).outcome == "allow"


# -- the config -----------------------------------------------------------------------


def test_categories_default_to_the_output_guards() -> None:
    config = ThreatDetectionConfig()
    enforce, log_only = config.input_safety_categories()
    assert enforce == frozenset(config.output.enforce) and "self_harm" in enforce
    assert log_only == frozenset(config.output.log_only)


def test_categories_can_be_set_for_the_input_screen_alone() -> None:
    config = ThreatDetectionConfig.model_validate(
        {"input_safety": {"enforce": ["self_harm"], "log_only": []}}
    )
    assert config.input_safety_categories() == (frozenset({"self_harm"}), frozenset())


def test_the_shipped_config_enables_the_section() -> None:
    config_dir = Path(__file__).resolve().parents[5] / "config"
    config = ThreatDetectionConfig.from_yaml(config_dir / "governance" / "threat-detection.yaml")
    assert config.input_safety.enabled is True


# -- off by default -------------------------------------------------------------------


def test_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_GOVERNANCE_INPUT_SAFETY", raising=False)
    assert _input_safety_from_env() is None
    kernel = kernel_from_env()
    assert kernel is not None
    assert "input_safety" not in kernel.hook_names(HookPoint.PRE_TURN)


def test_the_flag_installs_it_at_pre_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_INPUT_SAFETY", "1")
    kernel = kernel_from_env()
    assert kernel is not None
    assert "input_safety" in kernel.hook_names(HookPoint.PRE_TURN)


def test_the_flag_with_the_section_disabled_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = tmp_path / "threat.yaml"
    config.write_text("version: 1\ninput_safety:\n  enabled: false\n")
    monkeypatch.setenv("IRIS_GOVERNANCE_INPUT_SAFETY", "1")
    monkeypatch.setenv("IRIS_THREAT_DETECTION_CONFIG", str(config))
    with caplog.at_level(logging.WARNING):
        assert _input_safety_from_env() is None
    assert "input safety screen is OFF" in caplog.text
