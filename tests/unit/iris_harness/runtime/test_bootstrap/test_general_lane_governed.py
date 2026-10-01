"""The general lane's plugin tools run through the governed runner (R14: every call audited).

The general handler (the ``system`` agent with the loop off, and the answering side in
shadow mode) used to call ``research`` and every other plugin tool directly: no
``PRE_TOOL_USE`` / ``POST_TOOL_USE``, no approval, no audit row, and no refusal without an
audit key. These drive the lane with a real kernel and check each of those, with only the
model faked. ``test_no_bypass`` pins that no direct ``.call`` comes back.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iris_harness.agent.agent_executor import AgentTask
from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.audit.log import AuditLog
from iris_harness.kernel.governance.turn_label import (
    current_turn_label,
    lift_turn_label,
    turn_label_scope,
)
from iris_harness.llm.client import (
    CodingLLMConfig,
    LLMInvocationResponse,
    LLMMessage,
    LLMToolCall,
)
from iris_harness.runtime.handlers.general import _make_general_handler
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus

_RAW = "raw-result-text-7f3a"


class _TierRouter:
    class _Tier:
        model = "fake-model"
        provider = "github"

    def get_llm_config(self, _intent: str) -> CodingLLMConfig:
        return CodingLLMConfig(
            provider="github", model="fake-model", base_url="https://example.test/v1"
        )

    def get_tier(self, _intent: str) -> _Tier:
        return self._Tier()

    def trace_metadata_for_intent(self, _intent: str) -> dict[str, object]:
        return {"model": "fake-model", "provider": "github"}


class _Hook:
    """Records what one tool hook point was shown, and answers with ``decision``."""

    priority = 1

    def __init__(self, point: HookPoint, decision: HookDecision | None = None) -> None:
        self.name = f"test_{point.value}"
        self.hook_point = point
        self.decision = decision or HookDecision(outcome="allow", reason="seen")
        self.seen: list[HookContext] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.seen.append(ctx)
        return self.decision


def _install_model(
    monkeypatch: pytest.MonkeyPatch, responses: Sequence[LLMInvocationResponse]
) -> list[tuple[LLMMessage, ...]]:
    """A model that answers from a script; returns every message list it was shown."""
    shown: list[tuple[LLMMessage, ...]] = []
    queue = list(responses)

    class _FakeClient:
        def __init__(self, config: CodingLLMConfig) -> None:
            self.config = config

        def get_usage_mark(self) -> int:
            return 0

        def get_token_usage_since(self, _mark: int) -> dict[str, int]:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def invoke_turn(
            self, *, messages: Sequence[LLMMessage], bound_tools: Sequence[Any] = ()
        ) -> LLMInvocationResponse:
            shown.append(tuple(messages))
            return queue.pop(0)

    monkeypatch.setattr("iris_harness.llm.client.CodingLLMClient", _FakeClient)
    return shown


def _calls(*names: str) -> list[LLMInvocationResponse]:
    """The model calls each tool (one per turn), then answers."""
    turns = [
        LLMInvocationResponse(
            tool_calls=(LLMToolCall(id=f"c{i}", name=name, arguments={"query": "q"}),)
        )
        for i, name in enumerate(names)
    ]
    return [*turns, LLMInvocationResponse(content="done")]


def _tool_messages(shown: list[tuple[LLMMessage, ...]]) -> list[str]:
    return [m.content for m in shown[-1] if m.role == "tool"]


class _World:
    def __init__(self, tmp_path: Path, *hooks: _Hook) -> None:
        self.invoked: list[dict[str, Any]] = []
        self.audit = AuditLog(tmp_path / "audit.db")
        self.kernel = GovernanceKernel(audit_log=self.audit)
        for hook in hooks:
            self.kernel.register(hook)
        self.kernel.init_lock()
        registry = PluginRegistry()
        registry.add_plugin(PluginRecord(name="t", source="test", status=PluginStatus.LOADED))
        for name in ("research", "look_up"):
            registry.add_tool("t", ToolSpec(name, f"{name} tool", self._call(name)))
        self.holder = [SimpleNamespace(plugin_registry=registry, governance_kernel=self.kernel)]

    def _call(self, name: str) -> Any:
        def call(args: dict[str, Any]) -> str:
            self.invoked.append({"tool": name, **args})
            return f"{_RAW} from {name}"

        return call

    def turn(self, query: str = "look something up") -> str:
        handler, _stream = _make_general_handler(_TierRouter(), runtime_holder=self.holder)
        text, _meta = handler(
            AgentTask(query=query, agent_type="system", session_id="s-gen", origin_channel="web")
        )
        return text


def test_research_and_a_plugin_tool_fire_pre_and_post_as_the_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pre, post = _Hook(HookPoint.PRE_TOOL_USE), _Hook(HookPoint.POST_TOOL_USE)
    world = _World(tmp_path, pre, post)
    shown = _install_model(monkeypatch, _calls("research", "look_up"))

    assert world.turn() == "done"

    assert [c.payload["tool_name"] for c in pre.seen] == ["research", "look_up"]
    assert [c.payload["tool_name"] for c in post.seen] == ["research", "look_up"]
    assert {c.metadata["caller"] for c in pre.seen + post.seen} == {"model:system"}
    assert {c.agent_type for c in pre.seen} == {"system"}
    # One run for the turn's tool loop, as the ReAct loop has one per run.
    assert len({c.run_id for c in pre.seen + post.seen}) == 1
    assert {c.metadata["session_id"] for c in pre.seen} == {"s-gen"}
    assert {c.metadata["origin_channel"] for c in pre.seen} == {"web"}
    # Research keeps this lane's snippet-mode arguments; the digest covers them.
    assert pre.seen[0].payload["args"] == {"query": "q", "fetch_content": False, "max_results": 5}
    assert [i["tool"] for i in world.invoked] == ["research", "look_up"]
    assert all(_RAW in m for m in _tool_messages(shown))


def test_each_call_writes_audit_rows_with_keyed_digests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    world = _World(tmp_path, _Hook(HookPoint.PRE_TOOL_USE), _Hook(HookPoint.POST_TOOL_USE))
    _install_model(monkeypatch, _calls("research", "look_up"))

    world.turn()

    rows = world.audit.query()
    by_point: dict[str, list[str]] = {}
    for row in rows:
        payload = json.loads(row.payload_json)
        by_point.setdefault(row.hook_point, []).append(payload["tool_name"])
        assert "digest_alg" in payload
        assert _RAW not in row.payload_json  # a digest, never the result
    assert by_point[HookPoint.PRE_TOOL_USE.value] == ["research", "look_up"]
    assert by_point[HookPoint.POST_TOOL_USE.value] == ["research", "look_up"]


@pytest.mark.usefixtures("no_vault_master_key")
def test_no_audit_key_refuses_before_the_tool_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pre = _Hook(HookPoint.PRE_TOOL_USE)
    world = _World(tmp_path, pre)
    shown = _install_model(monkeypatch, _calls("research", "look_up"))

    world.turn()

    assert world.invoked == []
    assert pre.seen == []
    assert not world.audit.query()
    messages = _tool_messages(shown)
    assert len(messages) == 2
    assert all("can't audit this call" in m and _RAW not in m for m in messages)


def test_a_pre_deny_holds_the_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    deny = HookDecision(outcome="deny", reason="not today")
    world = _World(tmp_path, _Hook(HookPoint.PRE_TOOL_USE, deny))
    shown = _install_model(monkeypatch, _calls("research", "look_up"))

    world.turn()

    assert world.invoked == []
    messages = _tool_messages(shown)
    assert len(messages) == 2
    assert all("blocked by governance: not today" in m for m in messages)


def test_a_post_deny_withholds_the_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    deny = HookDecision(outcome="deny", reason="leaky")
    world = _World(tmp_path, _Hook(HookPoint.POST_TOOL_USE, deny))
    shown = _install_model(monkeypatch, _calls("research", "look_up"))

    world.turn()

    assert [i["tool"] for i in world.invoked] == ["research", "look_up"]
    messages = _tool_messages(shown)
    assert len(messages) == 2
    assert all(_RAW not in m and "blocked by governance: leaky" in m for m in messages)


def test_a_post_transform_hands_on_the_rewrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rewrite = HookDecision(
        outcome="transform",
        reason="redacted",
        transformed_payload={"tool_name": "look_up", "result": "[redacted]"},
    )
    world = _World(tmp_path, _Hook(HookPoint.POST_TOOL_USE, rewrite))
    shown = _install_model(monkeypatch, _calls("look_up"))

    world.turn()

    assert _tool_messages(shown) == ["[redacted]"]


def test_the_call_is_governed_as_the_turn_and_lifts_its_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pre = _Hook(HookPoint.PRE_TOOL_USE)
    raise_it = HookDecision(outcome="allow", reason="secret inside", set_classification="secret")
    world = _World(tmp_path, pre, _Hook(HookPoint.POST_TOOL_USE, raise_it))
    _install_model(monkeypatch, _calls("look_up", "look_up"))

    with turn_label_scope():
        lift_turn_label("personal")
        world.turn()
        after = current_turn_label()

    # The first call carries the turn's label; the second the label the first earned.
    assert [c.classification for c in pre.seen] == ["personal", "secret"]
    assert after == "secret"
