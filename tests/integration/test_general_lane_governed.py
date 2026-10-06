"""The general lane's tool calls are governed on both surfaces (R14: every call audited).

With the governed loop off (``IRIS_AGENTIC_CORE_ENABLED=0``) the ``system`` agent is the
general handler, and in shadow mode its answer is the one the user reads. Its plugin tools
used to run with no governance at all. Driven through a real runtime on ``chat`` and
``chat_stream`` (the REPL and web surface), with only the model and the search backend
faked: each turn's ``research`` call leaves PRE and POST rows in the governance ledger,
attributed to the model, under one run.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from iris_harness.kernel.governance import HookPoint
from iris_harness.kernel.governance.audit.log import AuditLog, AuditRow
from iris_harness.llm.client import LLMInvocationResponse, LLMMessage, LLMToolCall
from iris_harness.runtime import build_runtime
from iris_harness.runtime.types import ChatResult

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("offline_llm")]

_ANSWER = "Here is what the web says."
_FOUND = "snippet-7c1e from the web"


@pytest.fixture()
def world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[tuple[Any, AuditLog]]:
    from iris_harness.llm.client import CodingLLMClient

    def invoke_turn(
        self: Any, *, messages: Sequence[LLMMessage], bound_tools: Sequence[Any] = (), **_: Any
    ) -> LLMInvocationResponse:
        # First step of a turn: search. Once the observation is in: answer.
        if any(m.role == "tool" for m in messages):
            return LLMInvocationResponse(content=_ANSWER)
        return LLMInvocationResponse(
            tool_calls=(
                LLMToolCall(id="c1", name="research", arguments={"query": "rust 2.0 release"}),
            )
        )

    monkeypatch.setenv("IRIS_AGENTIC_CORE_ENABLED", "0")
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    audit_path = tmp_path / "governance-audit.db"
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(audit_path))
    monkeypatch.setattr(CodingLLMClient, "invoke_turn", invoke_turn)
    monkeypatch.setattr(
        "iris_harness.plugins_builtin.research.tool.run_research", lambda _args, **_: _FOUND
    )
    config_dir, data_dir = tmp_path / "config", tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    rt = build_runtime(config_dir=config_dir, data_dir=data_dir, use_background_scheduler=False)
    rt.startup()
    try:
        yield rt, AuditLog(audit_path)
    finally:
        rt.shutdown()


def _research_rows(audit: AuditLog, session_id: str) -> list[tuple[AuditRow, dict[str, Any]]]:
    rows = []
    for row in audit.query():
        payload = json.loads(row.payload_json)
        if payload.get("tool_name") == "research" and payload.get("session_id") == session_id:
            rows.append((row, payload))
    return rows


def test_the_general_lane_governs_its_research_call_on_both_surfaces(world: Any) -> None:
    rt, audit = world
    assert rt.governance_kernel is not None
    assert "research" in {tool.name for tool in rt.plugin_registry.tools()}
    # Tool-shaped ("latest"), so the general lane binds its tools instead of streaming.
    message = "what is the latest on the rust 2.0 release?"

    sync = rt.chat(message, session_id="s-sync")
    done = [e for e in rt.chat_stream(message, session_id="s-stream") if e.kind == "done"]
    assert done, "chat_stream ended without a result"
    results: dict[str, ChatResult] = {"s-sync": sync, "s-stream": done[-1].result}

    for session_id, result in results.items():
        # The general lane answered (the loop is off), from what the search returned.
        assert result.agent_type == "system", result
        assert result.metadata.get("agentic_core") is not True
        assert result.response.strip() == _ANSWER, result.response
        rows = _research_rows(audit, session_id)
        assert {row.hook_point for row, _ in rows} == {
            HookPoint.PRE_TOOL_USE.value,
            HookPoint.POST_TOOL_USE.value,
        }
        # One run for the turn's governed calls, attributed to the model, digested.
        assert len({row.run_id for row, _ in rows}) == 1
        assert {row.agent_type for row, _ in rows} == {"system"}
        callers = {row.reason for row, _ in rows if row.plugin == "caller_policy"}
        assert callers == {"caller_policy: model:system"}
        assert all(payload.get("digest_alg") for _, payload in rows)
        assert all(_FOUND not in row.payload_json for row, _ in rows)


def test_each_general_lane_plugin_call_carries_one_minted_call_id_on_both_surfaces(
    world: Any,
) -> None:
    """#134: the lane's plugin tool call takes the runner, so it gets the runner's minted
    ULID on its PRE and POST rows (the provider's own ``c1`` is never the audit id)."""
    from iris_harness.foundation.ids import is_ulid

    rt, audit = world
    message = "what is the latest on the rust 2.0 release?"
    rt.chat(message, session_id="s-sync")
    assert [e for e in rt.chat_stream(message, session_id="s-stream") if e.kind == "done"]

    seen: set[str] = set()
    for session_id in ("s-sync", "s-stream"):
        rows = _research_rows(audit, session_id)
        ids = {payload.get("call_id") for _, payload in rows}
        assert len(ids) == 1, ids  # PRE and POST rows of the one call share its id
        (call_id,) = ids
        assert is_ulid(call_id) and call_id != "c1"
        seen.add(call_id)
    assert len(seen) == 2  # two turns, two calls, two ids
