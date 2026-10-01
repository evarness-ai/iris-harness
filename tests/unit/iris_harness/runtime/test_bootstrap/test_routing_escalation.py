"""Multi-step turns escalate model selection to the stronger tier-2 intent."""

from __future__ import annotations

from iris_harness.runtime.client_config import resolve_routing_intent


def test_multi_step_escalates_to_tier2_intent() -> None:
    # task_planning maps to tier2 (qwen2.5:7b) in config/llm_tiers.yaml.
    assert resolve_routing_intent("system", True) == "task_planning"
    assert resolve_routing_intent("general", True) == "task_planning"


def test_single_step_keeps_its_intent() -> None:
    assert resolve_routing_intent("system", False) == "system"
    assert resolve_routing_intent("search", False) == "search"
