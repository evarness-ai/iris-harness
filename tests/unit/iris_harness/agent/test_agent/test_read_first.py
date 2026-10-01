"""Read-first turns: an answer about the user's own data must come from a read tool.

2026-09-29, live on the VM: "What is in my calendar this week?" was answered at the
first step with five invented events. The model called no tool ("I can provide this
information directly"), none of the events were in the prompt or in calendar.db, and
the curator accepted it. A plugin now lists such intents under ``read_first_intents``;
the loop turns the first unread answer back, and the second ends the run so the handler
answers from the intent's deterministic digest.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from iris_harness.agent.agent_executor import AgentTask
from iris_harness.agent.agentic_core import (
    UNGROUNDED_ANSWER,
    UNGROUNDED_REASON,
    AgenticCore,
    AgenticCoreConfig,
    ToolSpec,
)
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus

# The reply the VM's calendar turn gave, word for word in shape.
_INVENTED = (
    "Thought: The user wants to know about their calendar this week. I can provide this "
    "information directly without needing to call any tools.\n"
    "Final Answer: This week, you have the following events in your calendar:\n"
    "- Monday, September 25: Meeting with the AI team at 10:00 AM"
)
_INVENTED_PLAIN = "This week you have a meeting with the AI team on Monday at 10:00 AM."
_LOOKUP = 'Thought: check the calendar\nAction: calendar_lookup\nAction Input: {"range": "week"}'
_REAL = "Tuesday 5:00 pm: swim lessons. Wednesday 12:00: Shortest Path webinar."
_FROM_READ = f"Thought: I have the events\nFinal Answer: {_REAL}"


class _Script:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)


def _core(script: _Script, *, read_first: bool = True) -> AgenticCore:
    lookup = ToolSpec(
        name="calendar_lookup",
        description="Answer questions about the user's calendar.",
        call=lambda _args: _REAL,
    )
    return AgenticCore(
        AgenticCoreConfig(max_iterations=5, read_first=read_first), llm_call=script, tools=[lookup]
    )


def _run(core: AgenticCore, mode: str) -> tuple[str, bool, str | None]:
    """(answer text, success, stream reason) for either loop."""
    query = "What is in my calendar this week?"
    if mode == "sync":
        trace = core.run(query)
        return trace.final_answer, trace.success, "ungrounded" if trace.ungrounded else None
    chunks: Iterator[object] = core.run_stream(query)
    items = list(chunks)
    meta = next(c for c in items if isinstance(c, dict))
    return "".join(c for c in items if isinstance(c, str)), bool(meta["success"]), meta["reason"]


@pytest.mark.parametrize("mode", ["sync", "stream"])
@pytest.mark.parametrize("invented", [_INVENTED, _INVENTED_PLAIN])
def test_an_answer_given_twice_without_reading_never_reaches_the_user(
    mode: str, invented: str
) -> None:
    script = _Script(invented, invented)
    answer, success, reason = _run(_core(script), mode)
    assert "AI team" not in answer
    assert success is False
    if mode == "sync":
        assert answer == UNGROUNDED_ANSWER and reason == "ungrounded"
    else:
        assert answer == "" and reason == UNGROUNDED_REASON  # the handler answers instead
    # The turn-back named the read to call.
    assert "calendar_lookup" in script.prompts[1]


@pytest.mark.parametrize("mode", ["sync", "stream"])
def test_a_turned_back_answer_gets_the_read_and_answers_from_it(mode: str) -> None:
    script = _Script(_INVENTED, _LOOKUP, _FROM_READ)
    answer, success, _ = _run(_core(script), mode)
    assert success is True
    assert answer == _REAL


@pytest.mark.parametrize("mode", ["sync", "stream"])
def test_an_answer_after_a_read_is_accepted_first_time(mode: str) -> None:
    script = _Script(_LOOKUP, _FROM_READ)
    answer, success, _ = _run(_core(script), mode)
    assert (answer, success) == (_REAL, True)
    assert len(script.prompts) == 2  # no turn-back


@pytest.mark.parametrize("mode", ["sync", "stream"])
def test_turns_not_marked_read_first_are_unchanged(mode: str) -> None:
    script = _Script("Thought: small talk\nFinal Answer: Hello!")
    answer, success, _ = _run(_core(script, read_first=False), mode)
    assert (answer, success) == ("Hello!", True)


# --- manifests and registry -----------------------------------------------------------

_PLUGINS = Path(__file__).resolve().parents[5] / "src" / "iris_personal" / "plugins"


@pytest.mark.parametrize(
    ("plugin", "intents"),
    [
        ("calendar", {"calendar"}),
        ("finance_workflows", {"finance"}),
        ("planner", {"planner"}),
        ("email_workflows", {"email", "communication"}),
    ],
)
def test_own_data_plugins_declare_their_read_first_intents(plugin: str, intents: set[str]) -> None:
    # The public tree ships the email slice only (OSS plan R2): a domain plugin it does
    # not carry has no manifest to check there.
    if not (_PLUGINS / plugin).is_dir():
        pytest.skip(f"the {plugin} plugin is not in this tree")
    raw = yaml.safe_load((_PLUGINS / plugin / "manifest.yaml").read_text(encoding="utf-8"))
    assert set(PluginManifest.model_validate(raw).read_first_intents) == intents


def _record(name: str, intents: tuple[str, ...], status: PluginStatus) -> PluginRecord:
    manifest = PluginManifest(name=name, read_first_intents=intents)
    return PluginRecord(name=name, source="test", status=status, manifest=manifest)


def test_the_registry_collects_read_first_intents_from_mounted_plugins_only() -> None:
    reg = PluginRegistry()
    reg.add_plugin(_record("cal", ("calendar",), PluginStatus.LOADED))
    reg.add_plugin(_record("fin", ("finance",), PluginStatus.DEGRADED))
    reg.add_plugin(_record("off", ("planner",), PluginStatus.DISABLED))
    assert reg.read_first_intents() == frozenset({"calendar", "finance"})


# --- the handler answers from the digest ----------------------------------------------


class _StubTierRouter:
    def get_llm_config(self, intent: str) -> Any:
        from iris_harness.llm.client import CodingLLMConfig

        return CodingLLMConfig(
            provider="github",
            model="stub",
            base_url="http://localhost",
            api_key_env="STUB",
            temperature=0.0,
            max_tokens=64,
            timeout_seconds=10,
        )


class _StubSkills:
    def list_packages(self) -> list[Any]:
        return []


class _StubRuntime:
    def __init__(self) -> None:
        self.plugin_registry = PluginRegistry()
        self.plugin_registry.add_plugin(_record("calendar", ("calendar",), PluginStatus.LOADED))

        class _Learning:
            def self_management_enabled(self) -> bool:
                return False

        self.learning = _Learning()


def _handlers(monkeypatch, replies: list[str]) -> tuple[Any, Any, list[str]]:
    from iris_harness.llm.client import CodingLLMClient
    from iris_harness.runtime.handlers.react import _make_react_handler

    it = iter(replies)
    monkeypatch.setattr(CodingLLMClient, "invoke", lambda self, **_kw: next(it))
    seen: list[str] = []

    def digest(_task: AgentTask) -> tuple[str, dict[str, object]]:
        seen.append("calendar")
        return "CAL-DIGEST", {"deterministic": True}

    sync, stream = _make_react_handler(
        _StubTierRouter(),  # type: ignore[arg-type]
        _StubSkills(),  # type: ignore[arg-type]
        fallback_handlers={"calendar": digest},
        runtime_holder=[_StubRuntime()],
    )
    return sync, stream, seen


def _task() -> AgentTask:
    return AgentTask(
        query="What is in my calendar this week?",
        agent_type="calendar",
        params={"intent": "calendar"},
    )


def test_the_sync_handler_answers_an_unread_calendar_turn_from_the_digest(monkeypatch) -> None:
    sync, _, seen = _handlers(monkeypatch, [_INVENTED, _INVENTED])
    text, _ = sync(_task())
    assert text == "CAL-DIGEST" and seen == ["calendar"]


def test_the_stream_handler_answers_an_unread_calendar_turn_from_the_digest(monkeypatch) -> None:
    _, stream, seen = _handlers(monkeypatch, [_INVENTED, _INVENTED])
    items = list(stream(_task()))
    text = "".join(c for c in items if isinstance(c, str))
    meta = next(c for c in items if isinstance(c, dict) and "reason" in c)
    assert text == "CAL-DIGEST" and seen == ["calendar"]
    assert meta["reason"] == UNGROUNDED_REASON
