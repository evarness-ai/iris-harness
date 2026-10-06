"""Audit rows name who acted (G10): the model of every model call, the owner of every tool.

Driven through the real governed harness, reading the real ledger: the ReAct step's
``pre_llm_call`` row names its model and provider as every other model call's does; a
tool's ``pre_tool_use`` / ``post_tool_use`` rows name the plugin that owns it (``system``
for a core tool); and the stable ``TurnAuditRow`` carries ``caller`` and ``tool_plugin``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.audit import AuditLog
from iris_harness.testing import TurnAuditRow, harness, plugin

_SCRIPT: dict[str, Any] = {
    "rules": [
        {
            "name": "answer from the tool",
            "match": {"user": r"(?s)Observation:.*echo: banana"},
            "reply": {"content": "Thought: Done.\nFinal Answer: The tool said banana."},
        },
        {
            "name": "call the tool",
            "match": {"user": r"User: Please echo the word banana"},
            "reply": {
                "content": "Thought: Use the tool.\nAction: echo_back\n"
                'Action Input: {"word": "banana"}'
            },
        },
    ]
}


def _echo(api: PluginAPI) -> None:
    api.register_tool(
        "echo_back",
        'Repeat the given word back. Args: {"word": str}.',
        lambda args: f"echo: {args.get('word', '')}",
    )


def _echo_plugin() -> Any:
    return plugin(
        _echo,
        manifest={"name": "echo", "provides": ["tool"], "tools": {"echo_back": {"effect": "read"}}},
    )


def _raw_payloads(h: Any, hook_point: str) -> list[dict[str, Any]]:
    return [
        json.loads(r.payload_json)
        for r in AuditLog(db_path=h.audit_db).query()
        if r.hook_point == hook_point
    ]


def _run(entry: str, h: Any) -> None:
    if entry == "chat":
        h.chat("Please echo the word banana")
    else:
        assert h.chat_stream("Please echo the word banana").answered


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_every_model_call_row_names_model_and_provider(entry: str) -> None:
    with harness(plugins=[_echo_plugin()], fake_model=_SCRIPT) as h:
        _run(entry, h)
        payloads = _raw_payloads(h, "pre_llm_call")
        loop_rows = [r for r in h.audit_rows(hook_point="pre_llm_call") if r.step_id is not None]
    # The loop's two steps (call the tool, answer) are among them.
    assert len(loop_rows) >= 2
    assert payloads
    for payload in payloads:
        assert payload.get("model"), payload
        assert payload.get("provider"), payload


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_plugin_tools_rows_name_the_plugin_and_the_caller(entry: str) -> None:
    with harness(plugins=[_echo_plugin()], fake_model=_SCRIPT) as h:
        _run(entry, h)
        pre = h.audit_rows(hook_point="pre_tool_use")
        post = h.audit_rows(hook_point="post_tool_use")
    for rows in (pre, post):
        mine = [r for r in rows if r.tool == "echo_back"]
        assert mine, rows
        assert all(isinstance(r, TurnAuditRow) for r in mine)
        assert {r.tool_plugin for r in mine} == {"echo"}
        assert {r.caller for r in mine} == {"model:system"}


def test_a_core_tool_names_system() -> None:
    from iris_harness.agent.agentic_core import ToolSpec

    assert ToolSpec("t", "d", lambda a: "").plugin == "system"


def test_a_row_the_identity_does_not_describe_has_none() -> None:
    with harness(fake_model={"default": {"content": "Scripted answer."}}) as h:
        h.chat("hello there")
        rows = h.audit_rows(hook_point="pre_turn")
    assert rows
    assert all(r.tool_plugin is None for r in rows)
