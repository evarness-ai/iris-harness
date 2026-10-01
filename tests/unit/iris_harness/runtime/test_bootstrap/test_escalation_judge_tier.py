"""The escalation judge runs on ``judge_tier``, never on a weaker tier (ADR-0068).

It was built from the ``general`` intent's tier — tier1, the same model that
answers — which graded its own correct "I won't delete your emails" reply as a
capability gap at confidence 0.9. The tier2 judge accepts that reply.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris_harness.llm.client import CodingLLMConfig
from iris_harness.runtime.judges import build_curator_escalation_client


class _TierRouter:
    def __init__(self, tiers: dict[str, str]) -> None:
        self._tiers = tiers
        self.tier_requests: list[str] = []
        self.intent_requests: list[str] = []

    def get_llm_config_for_tier(self, tier_name: str) -> CodingLLMConfig | None:
        self.tier_requests.append(tier_name)
        model = self._tiers.get(tier_name)
        if model is None:
            return None
        return CodingLLMConfig(provider="ollama", model=model, tier_name=tier_name)

    def get_llm_config(self, intent: str) -> Any:
        self.intent_requests.append(intent)
        raise AssertionError("the judge must not be built from an intent's tier")


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("IRIS_CURATOR_ESCALATION", raising=False)
    (tmp_path / "escalation.yaml").write_text(
        "enabled: true\nmode: shadow\njudge_tier: tier2\n", encoding="utf-8"
    )
    return tmp_path


def test_judge_is_built_from_judge_tier(config_dir: Path) -> None:
    router = _TierRouter({"tier1": "small", "tier2": "big"})

    judge, cfg = build_curator_escalation_client(
        tier_router=router,  # type: ignore[arg-type]
        llm_call=None,
        config_dir=config_dir,
    )

    assert judge is not None
    assert cfg.judge_tier == "tier2"
    assert router.tier_requests == ["tier2"]
    assert router.intent_requests == []


def test_unknown_judge_tier_disables_the_judge(config_dir: Path) -> None:
    router = _TierRouter({"tier1": "small"})

    judge, cfg = build_curator_escalation_client(
        tier_router=router,  # type: ignore[arg-type]
        llm_call=None,
        config_dir=config_dir,
    )

    assert judge is None  # no judge rather than a weaker one
    assert cfg.enabled is True
    assert router.intent_requests == []
