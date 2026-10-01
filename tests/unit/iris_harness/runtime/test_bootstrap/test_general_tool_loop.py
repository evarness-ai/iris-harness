"""Tests for direct tool access in the general chat handler."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agent_executor import ActivityChunk, AgentTask, TraceChunk
from iris_harness.agent.intent_router import IntentResult
from iris_harness.llm.client import (
    CodingLLMConfig,
    LLMInvocationResponse,
    LLMMessage,
    LLMToolCall,
)
from iris_harness.runtime.facade import IrisRuntime
from iris_harness.runtime.handlers.general import _make_general_handler
from iris_harness.tools.skills.registry import SkillRegistry


class _TierRouter:
    class _Tier:
        def __init__(self) -> None:
            self.model = "fake-model"
            self.provider = "github"

    def get_llm_config(self, _intent: str) -> CodingLLMConfig:
        return CodingLLMConfig(
            provider="github",
            model="fake-model",
            base_url="https://example.test/v1",
            api_key_env=None,
        )

    def get_tier(self, _intent: str) -> _Tier:
        return self._Tier()

    def trace_metadata_for_intent(self, _intent: str) -> dict[str, object]:
        return {
            "model": "fake-model",
            "provider": "github",
            "router_model": "fake-model",
            "router_provider": "github",
        }


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    responses: Sequence[LLMInvocationResponse],
    *,
    stream_chunks: Sequence[str] | None = None,
) -> list[Any]:
    instances: list[Any] = []

    class _FakeClient:
        def __init__(self, _config: CodingLLMConfig) -> None:
            self.config = _config
            self.responses = list(responses)
            self.stream_chunks = list(stream_chunks or [])
            self.bound_tools: list[tuple[dict[str, Any], ...]] = []
            self.messages: list[tuple[LLMMessage, ...]] = []
            self.stream_prompts: list[tuple[str, str]] = []
            instances.append(self)

        def get_usage_mark(self) -> int:
            return 0

        def get_token_usage_since(self, _mark: int) -> dict[str, int]:
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        def invoke_turn(
            self,
            *,
            messages: Sequence[LLMMessage],
            bound_tools: Sequence[dict[str, Any]] = (),
        ) -> LLMInvocationResponse:
            self.messages.append(tuple(messages))
            self.bound_tools.append(tuple(dict(tool) for tool in bound_tools))
            return self.responses.pop(0)

        def invoke_stream(
            self,
            *,
            system_prompt: str,
            user_prompt: str,
        ):
            self.stream_prompts.append((system_prompt, user_prompt))
            chunks = self.stream_chunks
            if not chunks and self.responses:
                chunks = [self.responses.pop(0).content]
            yield from chunks

    monkeypatch.setattr("iris_harness.llm.client.CodingLLMClient", _FakeClient)
    return instances


def _write_fetch_top_repos_skill(repo_root: Path) -> None:
    skill_dir = repo_root / "config" / "skills" / "fetch-top-repos"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "manifest.yaml").write_text(
        "name: fetch-top-repos\n"
        "version: 1.0.0\n"
        "description: Fetch top GitHub repositories\n"
        "author: iris-tests\n"
        "license: Apache-2.0\n"
        "tools:\n"
        "  - name: fetch_top_repos\n"
        "    description: Fetch top GitHub repositories for a recent period.\n"
        "    governor_route: system/read\n"
        "requires:\n"
        "  python: '>=3.12'\n"
        "  packages:\n"
        "    - pydantic>=2.0\n"
        "  env_vars: []\n"
        "  config_files: []\n"
        "  agents: []\n",
        encoding="utf-8",
    )
    (skill_dir / "tools.py").write_text(
        "from langchain_core.tools import BaseTool\n"
        "from pydantic import BaseModel, Field\n\n"
        "class FetchTopReposInput(BaseModel):\n"
        "    limit: int = Field(default=10, ge=1, le=50)\n"
        "    since: str = Field(default='daily')\n\n"
        "class FetchTopReposTool(BaseTool):\n"
        "    name: str = 'fetch_top_repos'\n"
        "    description: str = 'Fetch top GitHub repositories.'\n"
        "    args_schema: type[BaseModel] = FetchTopReposInput\n\n"
        "    def _run(self, limit: int = 10, since: str = 'daily') -> str:\n"
        "        return f'top repos limit={limit} since={since}'\n\n"
        "    async def _arun(self, limit: int = 10, since: str = 'daily') -> str:\n"
        "        return self._run(limit=limit, since=since)\n\n"
        "SKILL_TOOLS = [FetchTopReposTool]\n",
        encoding="utf-8",
    )


def _write_morning_brief_skill(repo_root: Path) -> None:
    skill_dir = repo_root / "config" / "skills" / "morning-briefing"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "manifest.yaml").write_text(
        "name: morning-briefing\n"
        "version: 1.0.0\n"
        "description: Daily morning briefing for the user.\n"
        "author: iris-tests\n"
        "license: Apache-2.0\n"
        "kind: brief\n"
        "brief:\n"
        '  subject: "IRIS Morning Briefing"\n'
        '  recipient: "user"\n'
        "  uses: []\n"
        "  layout: |\n"
        "    Good morning. Briefing for {{date}}.\n\n"
        "    - No reminders due.\n"
        "  slots:\n"
        "    date:\n"
        "      kind: literal\n"
        '      value: "{today:%A, %B %d, %Y}"\n'
        "requires:\n"
        "  python: '>=3.12'\n"
        "  packages: []\n"
        "  env_vars: []\n"
        "  config_files: []\n"
        "  agents: []\n",
        encoding="utf-8",
    )


def _write_broken_morning_brief_skill(repo_root: Path) -> None:
    skill_dir = repo_root / "config" / "skills" / "morning-briefing"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "manifest.yaml").write_text(
        "name: morning-briefing\n"
        "version: 1.0.0\n"
        "description: Daily morning briefing for the user.\n"
        "author: iris-tests\n"
        "license: Apache-2.0\n"
        "kind: brief\n"
        "brief:\n"
        '  subject: "IRIS Morning Briefing"\n'
        '  recipient: "user"\n'
        "  uses: [missing-skill]\n"
        "  layout: |\n"
        "    Good morning. Briefing for {{headline}}.\n"
        "  slots:\n"
        "    headline:\n"
        "      kind: tool\n"
        "      skill: missing-skill\n"
        "      tool: missing_tool\n"
        "requires:\n"
        "  python: '>=3.12'\n"
        "  packages: []\n"
        "  env_vars: []\n"
        "  config_files: []\n"
        "  agents: []\n",
        encoding="utf-8",
    )


def _holder_with_tools(*specs: object) -> list[object]:
    """A runtime holder whose plugin registry carries the given tools.

    The general lane resolves its tools through the registry as of M4.6/M4.7, so a
    test that wants `code_exec` or `research` on this lane mounts them the way a
    profile would.
    """
    from types import SimpleNamespace

    from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus

    registry = PluginRegistry()
    registry.add_plugin(PluginRecord(name="t", source="test", status=PluginStatus.LOADED))
    for spec in specs:
        registry.add_tool("t", spec)
    # No kernel bound (governance disabled): these tests are about binding and dispatch;
    # test_general_lane_governed.py drives the same lane with a kernel.
    return [SimpleNamespace(plugin_registry=registry, governance_kernel=None)]


def _code_exec_spec(call: object) -> object:
    from iris_harness.agent.agentic_core import ToolSpec

    return ToolSpec(
        name="code_exec",
        description="Execute a short coding task inside a sandboxed Docker container.",
        call=call,  # type: ignore[arg-type]
    )


def _holder_with_research_plugin() -> list[object]:
    """A runtime holder carrying a mounted ``research`` plugin.

    ``research`` is a reference plugin as of M4.7, so this lane reaches it through
    the plugin registry. That is not a detail: the lane used to call the engine
    directly, which skipped the tool's egress guards — the ADR-0102 refusal and
    the identifier stripping — on every turn it served.
    """
    from types import SimpleNamespace

    from iris_harness.plugins_builtin.research import tools as research_tools
    from iris_harness.runtime.plugin_host.api import HarnessServices, PluginAPI
    from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus

    registry = PluginRegistry()
    registry.add_plugin(PluginRecord(name="research", source="builtin", status=PluginStatus.LOADED))
    services = HarnessServices(
        config_dir=Path("/nonexistent"),
        data_dir=Path("/nonexistent"),
        tier_router=None,
        agent_executor=None,
        heartbeats=None,
        channels=None,
        deterministic_reply=lambda **kw: None,
    )
    research_tools.register(PluginAPI(plugin="research", services=services, registry=registry))
    # No kernel bound (governance disabled): these tests are about binding and dispatch;
    # test_general_lane_governed.py drives the same lane with a kernel.
    return [SimpleNamespace(plugin_registry=registry, governance_kernel=None)]


def test_general_handler_binds_and_executes_research(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        LLMInvocationResponse(
            tool_calls=(
                LLMToolCall(
                    id="call-1",
                    name="research",
                    arguments={"query": "GitHub Trending today"},
                ),
            )
        ),
        LLMInvocationResponse(content="Here are the current trending repositories."),
    ]
    instances = _install_fake_client(monkeypatch, responses)
    monkeypatch.setattr(
        "iris_harness.plugins_builtin.research.tool.run_research",
        lambda args, **_: f"live results for {args.get('query')}",
    )

    handler, _stream_handler = _make_general_handler(
        _TierRouter(), runtime_holder=_holder_with_research_plugin()
    )
    text, _metadata = handler(AgentTask(query="GitHub Trending today", agent_type="system"))

    assert text == "Here are the current trending repositories."
    assert instances
    first_tools = instances[0].bound_tools[0]
    assert any(tool["function"]["name"] == "research" for tool in first_tools)
    second_turn_messages = instances[0].messages[1]
    assert any(
        message.role == "tool" and "live results for GitHub Trending today" in message.content
        for message in second_turn_messages
    )


def test_general_handler_offers_no_research_when_no_plugin_is_mounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A profile with no web-access plugin must not advertise a tool it cannot run."""
    instances = _install_fake_client(monkeypatch, [LLMInvocationResponse(content="no web here")])

    handler, _stream_handler = _make_general_handler(_TierRouter())
    handler(AgentTask(query="what is trending", agent_type="system"))

    assert instances
    assert not any(tool["function"]["name"] == "research" for tool in instances[0].bound_tools[0])


def test_general_handler_research_honours_the_egress_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-0102 on THIS lane too — it used to call the engine straight through."""
    responses = [
        LLMInvocationResponse(
            tool_calls=(
                LLMToolCall(
                    id="call-1",
                    name="research",
                    arguments={"query": "what are my insurance dues"},
                ),
            )
        ),
        LLMInvocationResponse(content="Checking your local finance data instead."),
    ]
    instances = _install_fake_client(monkeypatch, responses)
    reached: list[dict[str, object]] = []
    monkeypatch.setattr(
        "iris_harness.plugins_builtin.research.tool.run_research",
        lambda args, **_: reached.append(args) or "web results",
    )

    handler, _stream_handler = _make_general_handler(
        _TierRouter(), runtime_holder=_holder_with_research_plugin()
    )
    handler(AgentTask(query="what are my insurance dues", agent_type="system"))

    assert reached == [], "a personal-finance query reached the web from the general lane"
    observation = next(m.content for m in instances[0].messages[1] if m.role == "tool")
    assert "won't web-search your personal finances" in observation


def test_general_handler_binds_and_executes_wiki_search(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        LLMInvocationResponse(
            tool_calls=(
                LLMToolCall(
                    id="call-1",
                    name="wiki_search",
                    arguments={"query": "IRIS memory architecture"},
                ),
            )
        ),
        LLMInvocationResponse(content="IRIS uses episodic.md as a compact wiki index."),
    ]
    instances = _install_fake_client(monkeypatch, responses)

    class _FakeWiki:
        def query(self, question: str) -> str:
            return f"**IRIS Memory Architecture**\n\nCompiled answer for {question}."

    handler, _stream_handler = _make_general_handler(_TierRouter(), wiki=_FakeWiki())  # type: ignore[arg-type]
    text, _metadata = handler(
        AgentTask(query="what did we decide about memory?", agent_type="system")
    )

    assert text == "IRIS uses episodic.md as a compact wiki index."
    assert instances
    first_tools = instances[0].bound_tools[0]
    assert any(tool["function"]["name"] == "wiki_search" for tool in first_tools)
    second_turn_messages = instances[0].messages[1]
    assert any(
        message.role == "tool" and "IRIS Memory Architecture" in message.content
        for message in second_turn_messages
    )


def test_general_handler_exposes_code_exec_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        LLMInvocationResponse(
            tool_calls=(
                LLMToolCall(
                    id="call-1",
                    name="code_exec",
                    arguments={"task": "scrape GitHub Trending and return the top 10"},
                ),
            )
        ),
        LLMInvocationResponse(content="The sandbox produced the top 10 list."),
    ]
    instances = _install_fake_client(monkeypatch, responses)
    executed: list[AgentTask] = []

    def code_exec_call(args: dict[str, object]) -> str:
        executed.append(
            AgentTask(
                query=str(args.get("task") or args.get("query") or ""), agent_type="code_exec"
            )
        )
        return "sandbox output with repository list"

    handler, _stream_handler = _make_general_handler(
        _TierRouter(),
        runtime_holder=_holder_with_tools(_code_exec_spec(code_exec_call)),
    )
    text, _metadata = handler(AgentTask(query="get the top repos today", agent_type="system"))

    assert text == "The sandbox produced the top 10 list."
    assert executed[0].query == "scrape GitHub Trending and return the top 10"
    first_tools = instances[0].bound_tools[0]
    assert any(tool["function"]["name"] == "code_exec" for tool in first_tools)
    second_turn_messages = instances[0].messages[1]
    assert any(
        message.role == "tool" and "sandbox output with repository list" in message.content
        for message in second_turn_messages
    )


def test_general_handler_binds_and_executes_local_skill(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_fetch_top_repos_skill(tmp_path)
    responses = [
        LLMInvocationResponse(
            tool_calls=(
                LLMToolCall(
                    id="call-1",
                    name="fetch_top_repos",
                    arguments={"limit": 3, "since": "daily"},
                ),
            )
        ),
        LLMInvocationResponse(content="Here are the top repos from the promoted skill."),
    ]
    instances = _install_fake_client(monkeypatch, responses)

    handler, _stream_handler = _make_general_handler(
        _TierRouter(),
        skill_registry=SkillRegistry(tmp_path),
    )
    text, _metadata = handler(AgentTask(query="use the local helper", agent_type="system"))

    assert text == "Here are the top repos from the promoted skill."
    first_tools = instances[0].bound_tools[0]
    assert any(tool["function"]["name"] == "fetch_top_repos" for tool in first_tools)
    second_turn_messages = instances[0].messages[1]
    assert any(
        message.role == "tool" and "top repos limit=3 since=daily" in message.content
        for message in second_turn_messages
    )


@pytest.mark.minilm
def test_general_handler_directly_prefers_matching_local_skill_over_code_exec(
    tmp_path: Path,
) -> None:
    _write_fetch_top_repos_skill(tmp_path)
    executed: list[AgentTask] = []

    def code_exec_call(args: dict[str, object]) -> str:
        executed.append(AgentTask(query=str(args.get("task") or ""), agent_type="code_exec"))
        return "sandbox should not run"

    handler, _stream_handler = _make_general_handler(
        _TierRouter(),
        runtime_holder=_holder_with_tools(_code_exec_spec(code_exec_call)),
        skill_registry=SkillRegistry(tmp_path),
    )

    text, metadata = handler(
        AgentTask(query="what are the top 10 github repo's today?", agent_type="system")
    )

    assert executed == []
    assert metadata["used_skill"] is True
    assert metadata["skill_tool"] == "fetch_top_repos"
    assert "Used skill `fetch_top_repos`" in text
    assert "top repos limit=10 since=daily" in text


@pytest.mark.minilm
def test_general_handler_directly_renders_matching_brief_skill_without_llm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_morning_brief_skill(tmp_path)
    instances = _install_fake_client(monkeypatch, [LLMInvocationResponse(content="should not run")])

    handler, _stream_handler = _make_general_handler(
        _TierRouter(),
        skill_registry=SkillRegistry(tmp_path),
    )

    text, metadata = handler(
        AgentTask(query="what's my morning briefing today?", agent_type="system")
    )

    assert instances == []
    assert metadata["used_skill"] is True
    assert metadata["skill_tool"] == "morning-briefing"
    assert metadata["skill_kind"] == "brief"
    assert text.startswith("Good morning. Briefing for ")
    assert "- No reminders due." in text


@pytest.mark.minilm
def test_general_handler_brief_failure_falls_back_with_recovery_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_broken_morning_brief_skill(tmp_path)
    instances = _install_fake_client(
        monkeypatch,
        [LLMInvocationResponse(content="Fallback answer with alternate tools.")],
    )

    handler, _stream_handler = _make_general_handler(
        _TierRouter(),
        skill_registry=SkillRegistry(tmp_path),
    )

    text, metadata = handler(
        AgentTask(query="what's my morning briefing today?", agent_type="system")
    )

    assert text == "Fallback answer with alternate tools."
    assert metadata.get("fallback_clarification") is False
    first_turn = instances[0].messages[0]
    user_message = next(message for message in first_turn if message.role == "user")
    assert "[Execution context]" in user_message.content
    assert "Direct capability `morning-briefing` failed before completing." in user_message.content


@pytest.mark.minilm
def test_general_handler_brief_failure_unresolved_fallback_becomes_structured_clarification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_broken_morning_brief_skill(tmp_path)
    _install_fake_client(
        monkeypatch,
        [
            LLMInvocationResponse(
                content="I was unable to find any specific information about that."
            )
        ],
    )

    handler, _stream_handler = _make_general_handler(
        _TierRouter(),
        skill_registry=SkillRegistry(tmp_path),
    )

    text, metadata = handler(
        AgentTask(query="what's my morning briefing today?", agent_type="system")
    )

    assert text.startswith("I couldn't complete that via `morning-briefing`")
    assert "Reply with one option:" in text
    assert "1. `retry now`" in text
    assert metadata.get("fallback_clarification") is True


@pytest.mark.minilm
def test_runtime_skill_match_overrides_code_exec_intent(tmp_path: Path) -> None:
    _write_fetch_top_repos_skill(tmp_path)
    runtime = object.__new__(IrisRuntime)
    runtime.skill_registry = SkillRegistry(tmp_path)

    resolved = runtime._resolve_skill_intent(
        "what are the top 10 github repo's today?",
        IntentResult(
            intent="code_exec",
            agent_type="code_exec",
            confidence=0.85,
            raw_query="what are the top 10 github repo's today?",
        ),
    )

    assert resolved.intent == "general"
    assert resolved.agent_type == "system"
    assert resolved.confidence == 0.92


def test_runtime_skill_match_does_not_override_protected_domain_intent(tmp_path: Path) -> None:
    """Issue 0003: a DOMAIN intent (email/finance/…) has a dedicated agent and
    must NOT be hijacked to a skill via the general handler, even when a skill
    matches the message — this was the 'any finance emails?' → general/system
    misroute that then stalled."""
    _write_fetch_top_repos_skill(tmp_path)  # a skill that DOES match the message
    runtime = object.__new__(IrisRuntime)
    runtime.skill_registry = SkillRegistry(tmp_path)

    resolved = runtime._resolve_skill_intent(
        "what are the top 10 github repo's today?",
        IntentResult(
            intent="communication",
            agent_type="email",
            confidence=0.85,
            raw_query="what are the top 10 github repo's today?",
        ),
    )

    assert resolved.intent == "communication"  # unchanged — dedicated agent wins
    assert resolved.agent_type == "email"


def test_general_handler_respects_configured_tool_budget_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_GENERAL_TOOL_MAX_TURNS", "1")
    monkeypatch.setenv("IRIS_GENERAL_TOOL_MAX_TURNS_CAP", "1")

    responses = [
        LLMInvocationResponse(
            tool_calls=(
                LLMToolCall(
                    id="call-1",
                    name="research",
                    arguments={"query": "latest ai news"},
                ),
            )
        )
    ]
    _install_fake_client(monkeypatch, responses)
    monkeypatch.setattr(
        "iris_harness.plugins_builtin.research.tool.run_research",
        lambda args, **_: f"ok {args.get('query')}",
    )

    handler, _stream_handler = _make_general_handler(_TierRouter())
    text, _metadata = handler(AgentTask(query="search once", agent_type="system"))

    assert "tool-call limit" in text


def test_general_handler_builtin_ollama_profile_uses_tier_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _install_fake_client(
        monkeypatch,
        [LLMInvocationResponse(content="ok")],
    )

    handler, _stream_handler = _make_general_handler(_TierRouter())
    text, _metadata = handler(
        AgentTask(
            query="hello",
            agent_type="system",
            params={"intent": "general", "provider_profile": "ollama"},
        )
    )

    assert text == "ok"
    assert instances[0].config.provider == "github"
    assert instances[0].config.model == "fake-model"


def test_general_handler_explicit_ollama_model_still_overrides_tier_router(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _install_fake_client(
        monkeypatch,
        [LLMInvocationResponse(content="ok")],
    )

    handler, _stream_handler = _make_general_handler(_TierRouter())
    text, _metadata = handler(
        AgentTask(
            query="hello",
            agent_type="system",
            params={
                "intent": "general",
                "provider_profile": "ollama",
                "preferred_model": "qwen3-coder:30b",
            },
        )
    )

    assert text == "ok"
    assert instances[0].config.provider == "ollama"
    assert instances[0].config.model == "qwen3-coder:30b"


def test_general_stream_handler_streams_simple_turn_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _install_fake_client(
        monkeypatch,
        [],
        stream_chunks=["hello", " there"],
    )

    _handler, stream_handler = _make_general_handler(_TierRouter())
    chunks = list(stream_handler(AgentTask(query="hello friend", agent_type="system")))

    assert [chunk for chunk in chunks if isinstance(chunk, str)] == ["hello", " there"]
    assert any(isinstance(chunk, ActivityChunk) for chunk in chunks)
    assert any(isinstance(chunk, TraceChunk) for chunk in chunks)
    assert any(isinstance(chunk, dict) and chunk.get("model") == "fake-model" for chunk in chunks)
    assert instances[0].stream_prompts
    assert instances[0].messages == []


def test_general_stream_handler_keeps_toolish_turn_on_tool_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances = _install_fake_client(
        monkeypatch,
        [LLMInvocationResponse(content="fresh result")],
        stream_chunks=["should not stream"],
    )

    _handler, stream_handler = _make_general_handler(_TierRouter())
    chunks = list(stream_handler(AgentTask(query="search latest ai news", agent_type="system")))

    assert [chunk for chunk in chunks if isinstance(chunk, str)] == ["fresh result"]
    assert instances[0].messages
    assert instances[0].stream_prompts == []
