"""The streaming path (``run_stream``) bypasses the ResponseCurator, so it must
screen its own final answer for internal-architecture disclosure. Without this a
"who are you?" SOUL-summary streams IRIS's pipeline / tier→model internals
straight to the user (and into the trace metadata) — exactly what happened live.
"""

from __future__ import annotations

import json

from iris_harness.agent.agent_executor import ActivityChunk, TraceChunk
from iris_harness.agent.agentic_core import _ARCH_DISCLOSURE_REFUSAL, AgenticCore

# 5 pipeline-stage class names + a tier→model line ⇒ architecture disclosure.
_DUMP = (
    "I am IRIS. My pipeline is IntentRouter, TaskPlanner, ReActLoop, "
    "AgentExecutor, ResponseCurator. Tier 1 uses llama3.2:3b for orchestration."
)


def _blob(items: list[object]) -> str:
    parts: list[str] = []
    for item in items:
        if isinstance(item, (TraceChunk, ActivityChunk)):
            parts.append(item.text)
        elif isinstance(item, str):
            parts.append(item)
        else:
            parts.append(json.dumps(item, default=str))
    return " ".join(parts)


def test_run_stream_blocks_architecture_disclosure_final_answer() -> None:
    core = AgenticCore(llm_call=lambda prompt: f"Final Answer: {_DUMP}")
    items = list(core.run_stream("who are you?"))

    strings = [i for i in items if isinstance(i, str)]
    # The user-facing answer is the refusal, never the dump.
    assert _ARCH_DISCLOSURE_REFUSAL in strings
    # Nothing yielded — answer, trace chunk, or final metadata — carries the
    # internal-architecture artifacts.
    blob = _blob(items)
    assert "IntentRouter" not in blob
    assert "llama3.2" not in blob


def test_run_stream_passes_benign_final_answer() -> None:
    benign = "I'm IRIS, your local assistant — I can help with email and calendar."
    core = AgenticCore(llm_call=lambda prompt: f"Final Answer: {benign}")
    items = list(core.run_stream("who are you?"))
    assert any(isinstance(i, str) and benign in i for i in items)
    assert _ARCH_DISCLOSURE_REFUSAL not in [i for i in items if isinstance(i, str)]
