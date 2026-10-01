"""The 2026-07-06 red-team case, as a regression: a hazardous message is refused before a
deterministic handler can claim it — when the opt-in input safety screen is on.

The red-team found a self-harm message captured by a keyword handler and answered with no
guard at all. Every turn now passes the PRE_TURN screen first (step a); with the input
safety screen installed (step b2, ``IRIS_GOVERNANCE_INPUT_SAFETY``) the screen can
recognise the hazard. The classifier is a stub: the test pins the path, not the model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.response_curator import GOVERNANCE_BLOCKED_TEXT
from iris_harness.kernel.governance import GovernanceKernel
from iris_harness.kernel.governance.plugins.input_safety import InputSafetyHook
from iris_harness.kernel.governance.threat.types import ThreatVerdict
from iris_harness.runtime import build_runtime

# A message the system plugin's deterministic handler answers; the stub classifier calls
# it hazardous, so only the screen stands between the message and the handler.
CLAIMED_BY_A_HANDLER = "what time is it?"


class _FlagsSelfHarm:
    name = "stub"

    async def score(self, *, text: str, surface: str) -> ThreatVerdict:
        return ThreatVerdict(
            label="unsafe", score=1.0, surface="inbound", backend="stub", categories=("self_harm",)
        )


def _kernel(*, shadow: bool) -> GovernanceKernel:
    kernel = GovernanceKernel(audit_log=None)
    kernel.register(
        InputSafetyHook(
            classifier=_FlagsSelfHarm(),  # type: ignore[arg-type]
            enforce=frozenset({"self_harm"}),
            shadow=shadow,
        )
    )
    kernel.init_lock()
    return kernel


@pytest.fixture()
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    config_dir = Path(__file__).resolve().parents[5] / "config"
    return build_runtime(
        config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False
    )


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_enforced_a_hazard_is_refused_before_any_handler(runtime: Any, entry: str) -> None:
    runtime.governance_kernel = _kernel(shadow=False)
    if entry == "chat":
        result = runtime.chat(CLAIMED_BY_A_HANDLER, session_id=f"b2-{entry}")
    else:
        events = list(runtime.chat_stream(CLAIMED_BY_A_HANDLER, session_id=f"b2-{entry}"))
        result = events[-1].result
    assert result.response == GOVERNANCE_BLOCKED_TEXT
    assert result.metadata.get("governance_screen") == "deny"
    assert result.metadata.get("deterministic_time_date") is None


def test_in_shadow_the_hazard_is_audited_and_the_turn_answered(runtime: Any) -> None:
    runtime.governance_kernel = _kernel(shadow=True)
    result = runtime.chat(CLAIMED_BY_A_HANDLER, session_id="b2-shadow")
    assert result.metadata.get("deterministic_time_date") is True
