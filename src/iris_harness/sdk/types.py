"""The types a plugin's handlers and tools are written against.

A plugin that registers an intent handler signs a contract with the agent
executor: it is handed an ``AgentTask`` and returns a ``HandlerResult``, or
yields ``StreamChunk``s. Those names are declared where the executor lives; they
are re-exported here so a plugin author imports one module, not four, and so the
SDK boundary can be a contract rather than a convention (OSS plan M6, decision 5).

The verbs come with them. ``build_task_spec`` and ``verify`` are how a handler
declares what it promised to produce and then checks it did -- a plugin that skips
them can claim success while returning nothing, which is the failure the TaskSpec
shape exists to catch.

A heartbeat handler is handed a ``HeartbeatDefinition`` and returns a ``HeartbeatRun``
carrying a ``HeartbeatStatus``; ``HeartbeatHandler`` is that callable's Protocol, for the
factory that builds one.

``Principal`` is who is calling a plugin's API route (the owner's bearer token, or a
paired device), so a route can refuse what a device may not do.

Re-exports, not copies: there is one definition of each, and it is the one the
harness itself runs on.
"""

from __future__ import annotations

from iris_harness.agent.agent_executor import (
    ActivityChunk,
    AgentResult,
    AgentTask,
    HandlerResult,
    StreamChunk,
    TraceChunk,
)
from iris_harness.agent.agentic_core import ToolDescription, ToolSpec
from iris_harness.agent.task_spec import (
    Verdict,
    build_missing_artifact_answer,
    build_task_spec,
    describe_expected_artifact,
    render_code_exec_system_prompt_for_small,
    user_visible_artifacts,
    verify,
)
from iris_harness.foundation.auth import Principal
from iris_harness.services.heartbeat.models import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatStatus,
)
from iris_harness.services.heartbeat.scheduler import HeartbeatHandler

__all__ = [
    "Principal",
    "ActivityChunk",
    "AgentResult",
    "AgentTask",
    "HandlerResult",
    "HeartbeatDefinition",
    "HeartbeatHandler",
    "HeartbeatRun",
    "HeartbeatStatus",
    "StreamChunk",
    "ToolDescription",
    "ToolSpec",
    "TraceChunk",
    "Verdict",
    "build_missing_artifact_answer",
    "build_task_spec",
    "describe_expected_artifact",
    "render_code_exec_system_prompt_for_small",
    "user_visible_artifacts",
    "verify",
]
