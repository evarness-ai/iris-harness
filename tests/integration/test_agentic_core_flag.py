"""Integration test verifying the IRIS_AGENTIC_CORE_ENABLED flag wires the ReAct handler.

The flag defaults to ON. Setting it to "0"/"false"/"no" restores the legacy
general handler.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.runtime import build_runtime

# No model server in tests (tests/conftest.py network guard): runtime turns here
# reach the LLM on their degrade paths, so the model is a stubbed dead server.
pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("offline_llm")]


def _build(tmp_path: Path):
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    return build_runtime(
        config_dir=config_dir,
        data_dir=data_dir,
        use_background_scheduler=False,
    )


def test_flag_unset_defaults_to_react_handler(tmp_path, monkeypatch):
    """With the flag unset (default), the ReAct handler is wired in."""
    from iris_harness.agent.agent_executor import AgentTask
    from iris_harness.llm.client import CodingLLMClient

    monkeypatch.delenv("IRIS_AGENTIC_CORE_ENABLED", raising=False)
    monkeypatch.setattr(
        CodingLLMClient,
        "invoke",
        lambda self, *, system_prompt, user_prompt, **kwargs: "Thought: simple.\nFinal Answer: ok",
    )

    runtime = _build(tmp_path)
    result = runtime.agent_executor.execute(AgentTask(query="hi", agent_type="system"))

    assert result.success is True
    assert result.metadata.get("agentic_core") is True
    assert result.output == "ok"


def test_flag_explicitly_off_uses_general_handler(tmp_path, monkeypatch):
    """Setting the flag to '0' restores the legacy general handler."""
    from iris_harness.agent.agent_executor import AgentTask

    monkeypatch.setenv("IRIS_AGENTIC_CORE_ENABLED", "0")
    runtime = _build(tmp_path)

    # The general handler does not emit agentic_core metadata.
    assert "system" in runtime.agent_executor.registered_agents()
    # Execute to confirm metadata.agentic_core is absent (legacy path).
    result = runtime.agent_executor.execute(AgentTask(query="hi", agent_type="system"))
    assert result.metadata.get("agentic_core") is not True


def test_flag_explicitly_on_uses_react_handler(tmp_path, monkeypatch):
    """Explicit '1' keeps the ReAct handler wired."""
    from iris_harness.agent.agent_executor import AgentTask
    from iris_harness.llm.client import CodingLLMClient

    monkeypatch.setenv("IRIS_AGENTIC_CORE_ENABLED", "1")
    monkeypatch.setattr(
        CodingLLMClient,
        "invoke",
        lambda self, *, system_prompt, user_prompt, **kwargs: "Thought: simple.\nFinal Answer: ok",
    )

    runtime = _build(tmp_path)
    result = runtime.agent_executor.execute(AgentTask(query="hi", agent_type="system"))

    assert result.success is True
    assert result.metadata.get("agentic_core") is True
    assert result.output == "ok"


def test_flag_shadow_preserves_legacy_output_and_records_comparison(tmp_path, monkeypatch):
    """Shadow mode keeps the legacy response path while running AgenticCore too."""
    from iris_harness.agent.agent_executor import AgentTask
    from iris_harness.llm.client import CodingLLMClient

    monkeypatch.setattr(
        CodingLLMClient,
        "invoke",
        lambda self, *, system_prompt, user_prompt, **kwargs: "Thought: simple.\nFinal Answer: ok",
    )

    monkeypatch.setenv("IRIS_AGENTIC_CORE_ENABLED", "shadow")
    shadow_runtime = _build(tmp_path / "shadow")
    shadow = shadow_runtime.agent_executor.execute(AgentTask(query="hi", agent_type="system"))

    assert shadow.success is True
    assert shadow.metadata.get("agentic_core") is not True
    assert shadow.metadata["agentic_core_shadow_mode"] == "shadow"
    assert shadow.metadata["agentic_core_shadow_response_source"] == "legacy"
    comparison = shadow.metadata["agentic_core_shadow_compare"]
    assert comparison["candidate_success"] is True
    assert "latency_delta_ms" in comparison


# --- ADR-0077 P4: calendar + planner convergence (opt-in) --------------------


def _stub_llm(monkeypatch) -> None:
    from iris_harness.llm.client import CodingLLMClient

    monkeypatch.setattr(
        CodingLLMClient,
        "invoke",
        lambda self, *, system_prompt, user_prompt, **kwargs: "Thought: simple.\nFinal Answer: ok",
    )


def _stub_llm_reading(monkeypatch) -> None:
    """A model that reads before it answers. Calendar and planner are read-first intents
    (their plugins' ``read_first_intents``), so an answer with no read is turned back;
    memory_search is on every menu (a core tool)."""
    from iris_harness.agent.agentic_core import _SCRATCHPAD_HEADER
    from iris_harness.llm.client import CodingLLMClient

    def invoke(self, *, system_prompt, user_prompt, **kwargs) -> str:
        if _SCRATCHPAD_HEADER in user_prompt:  # a tool result is in hand
            return "Thought: I have what I need.\nFinal Answer: ok"
        return 'Thought: look first.\nAction: memory_search\nAction Input: {"query": "today"}'

    monkeypatch.setattr(CodingLLMClient, "invoke", invoke)


def test_p4_off_keeps_planner_and_calendar_deterministic(tmp_path, monkeypatch):
    """Default (P4 unset): calendar + planner stay deterministic lanes, even though
    the base loop is on — flipping P4 is the only thing that converges them."""
    from iris_harness.agent.agent_executor import AgentTask

    monkeypatch.delenv("IRIS_AGENTIC_CORE_P4", raising=False)
    _stub_llm(monkeypatch)
    runtime = _build(tmp_path)

    for intent in ("planner", "calendar"):
        result = runtime.agent_executor.execute(
            AgentTask(query="what's on today", agent_type=intent, params={"intent": intent})
        )
        # Deterministic handlers don't emit agentic_core metadata.
        assert result.metadata.get("agentic_core") is not True, intent


def test_p4_on_routes_planner_and_calendar_through_the_loop(tmp_path, monkeypatch):
    """With IRIS_AGENTIC_CORE_P4=1 (and the base loop on), calendar + planner run the
    unified ReAct handler — asserted by the agentic_core metadata (routing only)."""
    from iris_harness.agent.agent_executor import AgentTask

    monkeypatch.setenv("IRIS_AGENTIC_CORE_P4", "1")
    _stub_llm_reading(monkeypatch)
    runtime = _build(tmp_path)

    for intent in ("planner", "calendar"):
        result = runtime.agent_executor.execute(
            AgentTask(query="how is my day looking", agent_type=intent, params={"intent": intent})
        )
        assert result.success is True, intent
        assert result.metadata.get("agentic_core") is True, intent


def test_p4_on_requires_base_loop(tmp_path, monkeypatch):
    """P4 needs the base loop: with the base rollout off, calendar/planner fall back
    to deterministic even when P4 is set (no react_handler to route to)."""
    from iris_harness.agent.agent_executor import AgentTask

    monkeypatch.setenv("IRIS_AGENTIC_CORE_ENABLED", "0")
    monkeypatch.setenv("IRIS_AGENTIC_CORE_P4", "1")
    runtime = _build(tmp_path)

    result = runtime.agent_executor.execute(
        AgentTask(query="what's on today", agent_type="planner", params={"intent": "planner"})
    )
    assert result.metadata.get("agentic_core") is not True
