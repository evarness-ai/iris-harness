"""Tests for the AgenticCore ReAct handler wired into the runtime.

The email agent's own loop is not here: it left for the email_workflows plugin at
M6.1b with the library it reads (OSS plan M6, decision 2), and its tests moved with
it to tests/unit/test_email_workflows/test_agent.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from iris_harness.agent.agent_executor import ActivityChunk, AgentTask, TraceChunk
from iris_harness.runtime.handlers.react import _make_react_handler, _skills_to_react_tools
from iris_harness.runtime.react_tools import builtin_react_tools
from iris_harness.tools.skills.models import (
    SkillManifest,
    SkillPackage,
    SkillRequirements,
    SkillToolManifest,
)
from iris_harness.tools.skills.registry import SkillRegistry


class _EchoArgs(BaseModel):
    text: str = ""


class _EchoTool(BaseTool):
    name: str = "echo_tool"
    description: str = "Echoes the input string."
    args_schema: type[BaseModel] = _EchoArgs

    def _run(self, text: str = "") -> str:
        return f"echoed:{text}"

    async def _arun(self, text: str = "") -> str:
        return self._run(text=text)


@pytest.fixture(autouse=True)
def _gmail_provider_mounted() -> None:
    """These tools reach the mailbox through the provider registry (M5.7 track A);
    the real Gmail provider is registered so the ``gf.*`` patches below are what runs."""
    from iris_personal.email.providers import clear_mail_providers, register_mail_provider
    from iris_personal.plugins.gmail.provider import GmailProvider

    clear_mail_providers()
    register_mail_provider(GmailProvider())
    yield  # type: ignore[misc]
    clear_mail_providers()


def _make_package() -> SkillPackage:
    manifest = SkillManifest(
        name="echo-skill",
        kind="tool",
        description="echo",
        version="0.1.0",
        author="test",
        license="MIT",
        requires=SkillRequirements(),
        tools=(
            SkillToolManifest(
                name="echo_tool",
                description="Echoes the input string.",
                governor_route="general",
            ),
        ),
    )
    return SkillPackage(
        manifest=manifest,
        source_dir=Path("/tmp/echo-skill"),
        skill_dir=Path("/tmp/echo-skill"),
        tools_module_path=Path("/tmp/echo-skill/tools.py"),
        tool_classes=(_EchoTool,),
        is_loadable=True,
    )


class _StubRegistry(SkillRegistry):
    def __init__(self) -> None:
        super().__init__(repo_root=Path("/tmp"))
        self._packages = (_make_package(),)

    def discover(self) -> tuple[SkillPackage, ...]:
        return self._packages


def test_skills_to_react_tools_wraps_each_tool_class() -> None:
    specs = _skills_to_react_tools(_StubRegistry())

    assert [s.name for s in specs] == ["echo_tool"]
    assert specs[0].call({"text": "hi"}) == "echoed:hi"


def test_skills_to_react_tools_dedupes_repeat_names() -> None:
    registry = _StubRegistry()
    # Duplicate the same package — second entry must be ignored.
    registry._packages = registry._packages + registry._packages
    specs = _skills_to_react_tools(registry)

    assert len(specs) == 1


def test_a_skill_tool_is_internal_unless_its_manifest_declares_it_external() -> None:
    """The injection guard scans ``content: external`` results; a skill tool reaches it
    only through the declaration, so the adapter must carry it onto the ToolSpec."""
    registry = _StubRegistry()
    assert [s.content for s in _skills_to_react_tools(registry)] == ["internal"]

    package = registry._packages[0]
    declared = package.manifest.tools[0].model_copy(update={"content": "external"})
    external = package.model_copy(
        update={"manifest": package.manifest.model_copy(update={"tools": (declared,)})}
    )
    registry._packages = (external,)
    assert [s.content for s in _skills_to_react_tools(registry)] == ["external"]


def test_the_shipped_web_fetch_skill_declares_its_output_external() -> None:
    """``fetch_web_content`` returns feed items, headlines and repo descriptions."""
    repo_root = Path(__file__).resolve().parents[5]
    registry = SkillRegistry(repo_root=repo_root)
    spec = next(s for s in _skills_to_react_tools(registry) if s.name == "fetch_web_content")
    assert spec.content == "external"


class _StubTierRouter:
    def get_llm_config(self, intent: str):
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


def _patch_llm_invoke(monkeypatch, responses: list[str]) -> None:
    iterator = iter(responses)

    def fake_invoke(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stop: object = None,  # added 2026-05-19 (commit 1aa9e47) — ignored by the stub
    ) -> str:
        return next(iterator)

    from iris_harness.llm.client import CodingLLMClient

    monkeypatch.setattr(CodingLLMClient, "invoke", fake_invoke)


def test_react_handler_sync_returns_text_and_metadata(monkeypatch) -> None:
    _patch_llm_invoke(
        monkeypatch,
        [
            'Thought: I will echo.\nAction: echo_tool\nAction Input: {"text": "pong"}',
            "Thought: Done.\nFinal Answer: pong-final",
        ],
    )
    handler, _ = _make_react_handler(_StubTierRouter(), _StubRegistry())  # type: ignore[arg-type]
    result = handler(AgentTask(query="ping", agent_type="system"))

    assert isinstance(result, tuple)
    text, meta = result
    assert text == "pong-final"
    assert meta["agentic_core"] is True
    assert meta["success"] is True
    assert meta["iterations"] == 2
    # ADR-0118 decision 5: the echo skill is a read, and this is a fresh run.
    assert meta["effects_executed"] == []
    assert meta["resumed"] is False


def test_react_handler_env_governance_blocks_secret_before_llm(monkeypatch) -> None:
    from iris_harness.llm.client import CodingLLMClient

    def fail_invoke(self, *, system_prompt: str, user_prompt: str) -> str:
        raise AssertionError("LLM should not be called when governance blocks")

    monkeypatch.setenv("IRIS_GOVERNANCE_ENABLED", "1")
    monkeypatch.setattr(CodingLLMClient, "invoke", fail_invoke)

    handler, _ = _make_react_handler(_StubTierRouter(), _StubRegistry())  # type: ignore[arg-type]
    text, meta = handler(
        AgentTask(query="here is my key sk-ABC1234567890abcdef1234", agent_type="system")
    )

    assert "blocked by governance" in text
    assert meta["agentic_core"] is True
    assert meta["success"] is False
    assert meta["iterations"] == 0


def test_react_handler_stream_yields_chunks_and_meta(monkeypatch) -> None:
    _patch_llm_invoke(
        monkeypatch,
        ["Thought: easy.\nFinal Answer: 42"],
    )
    _, stream = _make_react_handler(_StubTierRouter(), _StubRegistry())  # type: ignore[arg-type]
    items = list(stream(AgentTask(query="q", agent_type="system")))

    activities = [i for i in items if isinstance(i, ActivityChunk)]
    traces = [i for i in items if isinstance(i, TraceChunk)]
    texts = [i for i in items if isinstance(i, str)]
    metas = [i for i in items if isinstance(i, dict)]

    assert activities
    assert traces
    assert texts == ["42"]
    assert metas[-1]["agentic_core"] is True
    assert metas[-1]["success"] is True
    assert metas[-1]["effects_executed"] == []
    assert not any(m.get("resumed") for m in metas)


def test_builtin_react_tools_always_include_memory_wiki_propose() -> None:
    specs = builtin_react_tools(semantic_index=None, wiki=None, repo_root=Path("/tmp"))
    names = {s.name for s in specs}

    assert {"memory_search", "wiki_search", "propose_skill_from_sandbox"} <= names
    # `research` (M4.7) and `code_exec` (M4.6) are reference plugins — they join the
    # pool from the plugin registry, not from the builtin set. Their own tests live
    # in tests/unit/test_research and tests/unit/test_code_exec.
    assert "research" not in names
    assert "code_exec" not in names


def test_builtin_tools_return_guard_messages_when_deps_missing() -> None:
    specs = {s.name: s for s in builtin_react_tools(semantic_index=None, wiki=None, repo_root=None)}
    # memory_search degrades per scope now (facts and patterns need the index; sessions
    # need the store; behaviors need neither), so a missing dep means "nothing matched"
    # for the scopes that cannot answer, not a blanket "unavailable".
    assert (
        "no stored memory matched"
        in specs["memory_search"].call({"query": "x", "scope": "facts"}).lower()
    )
    assert "unavailable" in specs["recall_conversation"].call({"query": "x"}).lower()
    assert specs["wiki_search"].call({"query": "x"}) == "Wiki unavailable."
    assert (
        "unavailable"
        in specs["propose_skill_from_sandbox"]
        .call({"script": "s", "intent": "i", "narrative": "n"})
        .lower()
    )


def _memory_specs(memory_store=None):
    return {
        s.name: s
        for s in builtin_react_tools(
            semantic_index=None,
            wiki=None,
            repo_root=None,
            memory_store=memory_store,
        )
    }


def test_memory_curation_tools_registered() -> None:
    names = set(_memory_specs())
    assert {"memory_correct", "memory_forget", "memory_restore"} <= names


def test_memory_curation_unavailable_without_store() -> None:
    specs = _memory_specs(memory_store=None)
    assert "unavailable" in specs["memory_forget"].call({"key": "location"}).lower()
    assert "unavailable" in specs["memory_correct"].call({"key": "k", "value": "v"}).lower()
    assert "unavailable" in specs["memory_restore"].call({"key": "k"}).lower()


def test_memory_curation_guards_empty_args(tmp_path: Path) -> None:
    from iris_harness.memory.store import MemoryStore

    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.ensure_schema()
    specs = _memory_specs(memory_store=store)
    # Empty/hallucinated args must never act — a local model on a blank arg.
    assert "Error" in specs["memory_forget"].call({})
    assert "Error" in specs["memory_correct"].call({"key": "location"})  # missing value
    assert "Error" in specs["memory_restore"].call({})
    # Nothing was created by the guarded no-ops.
    assert store.fetch_all_user_facts() == []


def test_memory_curation_full_cycle(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from iris_harness.memory.store import MemoryStore, UserFact

    store = MemoryStore(db_path=tmp_path / "memory.db")
    store.ensure_schema()
    now = datetime.now(UTC)
    store.upsert_user_fact(
        UserFact(
            key="location",
            value="Berlin",
            confidence=0.9,
            source="test",
            first_seen=now,
            last_confirmed=now,
            times_confirmed=1,
        )
    )
    specs = _memory_specs(memory_store=store)

    # correct: Berlin -> New York
    assert "New York" in specs["memory_correct"].call({"key": "location", "value": "New York"})
    assert store.fetch_user_fact("location").value == "New York"

    # forget: reversible
    assert "Forgot" in specs["memory_forget"].call({"key": "location"})
    assert store.fetch_user_fact("location") is None

    # restore: brings back the pre-forget value (New York)
    assert "New York" in specs["memory_restore"].call({"key": "location"})
    assert store.fetch_user_fact("location").value == "New York"

    # forget a non-existent key reports cleanly (no crash)
    assert "No stored fact" in specs["memory_forget"].call({"key": "nope"})


def test_skills_adapter_skips_names_already_in_the_pool() -> None:
    """A skill that ships a tool named `research` cannot shadow the real one: the handler
    passes the pool's names as ``taken`` (ADR-0110 follow-up; no static reserved list)."""

    class _ReservedArgs(BaseModel):
        query: str = ""

    class _ReservedTool(BaseTool):
        name: str = "research"
        description: str = "shadowed"
        args_schema: type[BaseModel] = _ReservedArgs

        def _run(self, query: str = "") -> str:
            return query

        async def _arun(self, query: str = "") -> str:
            return query

    manifest = SkillManifest(
        name="shadow-skill",
        kind="tool",
        description="shadow",
        version="0.1.0",
        author="t",
        license="MIT",
        requires=SkillRequirements(),
        tools=(
            SkillToolManifest(name="research", description="shadowed", governor_route="general"),
        ),
    )
    pkg = SkillPackage(
        manifest=manifest,
        source_dir=Path("/tmp/shadow"),
        skill_dir=Path("/tmp/shadow"),
        tools_module_path=Path("/tmp/shadow/tools.py"),
        tool_classes=(_ReservedTool,),
        is_loadable=True,
    )

    class _Reg(SkillRegistry):
        def __init__(self) -> None:
            super().__init__(repo_root=Path("/tmp"))
            self._packages = (pkg,)

        def discover(self):
            return self._packages

    assert _skills_to_react_tools(_Reg(), taken=frozenset({"research"})) == []
    # With nothing taken (no research tool mounted) the skill's tool is offered.
    assert [t.name for t in _skills_to_react_tools(_Reg())] == ["research"]


def test_skills_adapter_filters_unrelated_when_query_given() -> None:
    """With a non-empty query that doesn't match the manifest, the skill is filtered out."""
    specs = _skills_to_react_tools(_StubRegistry(), query="what is the weather today")
    assert specs == []


@pytest.fixture()
def no_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """No MiniLM model on disk and no fetching one (the suite's rule, a fresh clone's
    state), with a router built fresh so an earlier test's loaded model is not reused."""
    from iris_harness.kernel.governance.evaluator import embeddings
    from iris_harness.runtime import routine_authoring

    monkeypatch.setenv("IRIS_TEST_NULL_EMBEDDINGS", "1")
    monkeypatch.setattr(embeddings, "default_model_on_disk", lambda: False)
    monkeypatch.setattr(routine_authoring, "_semantic_router_singleton", None)
    monkeypatch.setattr(routine_authoring, "_semantic_router_init_failed", False)


def test_a_router_that_cannot_embed_is_no_router(no_embedding_model: None) -> None:
    """Every caller's no-router path is its no-embeddings path. A router that scored
    every skill 0.0 instead filtered them all out of the pool."""
    from iris_harness.runtime.routine_authoring import _get_semantic_router

    assert _get_semantic_router() is None


def test_skills_adapter_includes_matching_skill(no_embedding_model: None) -> None:
    """With no embeddings, a query naming the skill includes it (the keyword scorer's
    manifest-name bonus)."""
    specs = _skills_to_react_tools(_StubRegistry(), query="run the echo-skill on my text")
    assert [s.name for s in specs] == ["echo_tool"]


def test_skills_adapter_without_embeddings_still_filters_unrelated(
    no_embedding_model: None,
) -> None:
    specs = _skills_to_react_tools(_StubRegistry(), query="what is the weather today")
    assert specs == []


# --- P4: intent-aware degrade fallback + opt-in flag (ADR-0077) ---------------


def test_p4_flag_defaults_off_and_honours_env(monkeypatch) -> None:
    from iris_harness.runtime.bootstrap import _p4_universal_surfacing_enabled

    monkeypatch.delenv("IRIS_AGENTIC_CORE_P4", raising=False)
    assert _p4_universal_surfacing_enabled() is False
    monkeypatch.setenv("IRIS_AGENTIC_CORE_P4", "1")
    assert _p4_universal_surfacing_enabled() is True
    monkeypatch.setenv("IRIS_AGENTIC_CORE_P4", "off")
    assert _p4_universal_surfacing_enabled() is False


def test_react_handler_fallback_dispatches_by_intent(monkeypatch) -> None:
    # When the loop fails, degrade to the deterministic handler for THAT intent
    # (ADR-0077 P4: calendar/planner, not just finance).
    from iris_harness.llm.client import CodingLLMClient

    def boom(self, **_kw) -> str:
        raise RuntimeError("loop down")

    monkeypatch.setattr(CodingLLMClient, "invoke", boom)

    seen: list[str] = []

    def _cal(_task) -> tuple[str, dict[str, object]]:
        seen.append("calendar")
        return "CAL-DIGEST", {"deterministic": True}

    def _plan(_task) -> tuple[str, dict[str, object]]:
        seen.append("planner")
        return "DAY-PLAN", {"deterministic": True}

    handler, _ = _make_react_handler(
        _StubTierRouter(),  # type: ignore[arg-type]
        _StubRegistry(),  # type: ignore[arg-type]
        fallback_handlers={"calendar": _cal, "planner": _plan},
    )

    text, _ = handler(
        AgentTask(query="any meetings today?", agent_type="calendar", params={"intent": "calendar"})
    )
    assert text == "CAL-DIGEST" and seen == ["calendar"]

    seen.clear()
    text2, _ = handler(
        AgentTask(query="plan my day", agent_type="planner", params={"intent": "planner"})
    )
    assert text2 == "DAY-PLAN" and seen == ["planner"]


def test_react_handler_fallback_is_intent_scoped(monkeypatch) -> None:
    # A failure on an intent that has no fallback (system) must NOT borrow another
    # domain's deterministic digest — the loop's own (degraded) result stands.
    from iris_harness.llm.client import CodingLLMClient

    def boom(self, **_kw) -> str:
        raise RuntimeError("loop down")

    monkeypatch.setattr(CodingLLMClient, "invoke", boom)

    used: list[str] = []
    handler, _ = _make_react_handler(
        _StubTierRouter(),  # type: ignore[arg-type]
        _StubRegistry(),  # type: ignore[arg-type]
        fallback_handlers={"calendar": lambda _t: (used.append("cal") or "CAL-DIGEST", {})},
    )
    text, _ = handler(AgentTask(query="hello", agent_type="system", params={"intent": "system"}))
    assert used == []  # calendar fallback never fires for a system turn
    assert text != "CAL-DIGEST"


def test_cloud_search_synthesis_client_carries_a_tier_label(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The Copilot synthesis model logged ``tier=None``; it now names its role."""
    import iris_harness.llm.client as llm_client
    from iris_harness.runtime.handlers import react

    monkeypatch.setenv("IRIS_SEARCH_SYNTHESIS_PROVIDER", "copilot")
    monkeypatch.setenv("IRIS_ENABLE_COPILOT_BACKEND", "1")
    monkeypatch.setattr(
        react, "config_from_profile", lambda _name: llm_client.PROVIDER_DEFAULTS["copilot"]
    )
    built: list[Any] = []
    monkeypatch.setattr(
        llm_client, "CodingLLMClient", lambda cfg, **_kw: built.append(cfg) or object()
    )

    assert react._build_search_synthesis_client() is not None
    assert built[0].tier_name == "search_synthesis"
    # Governance reads the call as cloud: the provider it dials is declared to run there.
    assert built[0].governance_tier is None
    assert llm_client.provider_locality(built[0].provider) == "cloud"


def _paused_run(run_id: str) -> None:
    """A real paused run in the store the handler reads (IRIS_HOME is a temp dir)."""
    from iris_harness.memory.state import CheckpointStore
    from iris_harness.memory.state.chat import ChatCheckpointPayload, CheckpointStep

    CheckpointStore().write(
        run_id=run_id,
        step_id=0,
        agent_type="chat",
        payload=ChatCheckpointPayload(
            query="pick a graph database",
            steps=(
                CheckpointStep(
                    thought="ask them",
                    action="ask_user",
                    action_input={"question": "Neo4j or Stardog?"},
                    observation="Asked the user: Neo4j or Stardog?",
                ),
            ),
            iteration=1,
        ).to_payload(),
        signal="awaiting_user_input",
        session_id="s1",
    )


def _resume_task(run_id: str) -> AgentTask:
    return AgentTask(
        query="use Stardog",
        agent_type="system",
        resume_run_id=run_id,
        resume_step_id=0,
        resume_reply="use Stardog",
    )


def test_react_handler_marks_a_resumed_run_on_both_paths(monkeypatch) -> None:
    """ADR-0118 decision 5: escalation never re-runs a turn that continued a paused run."""
    _paused_run("resume-sync")
    _patch_llm_invoke(monkeypatch, ["Thought: ok.\nFinal Answer: Stardog it is"])
    handler, stream = _make_react_handler(_StubTierRouter(), _StubRegistry())  # type: ignore[arg-type]

    _text, meta = handler(_resume_task("resume-sync"))
    assert meta["resumed"] is True

    _paused_run("resume-stream")
    _patch_llm_invoke(monkeypatch, ["Thought: ok.\nFinal Answer: Stardog it is"])
    metas = [i for i in stream(_resume_task("resume-stream")) if isinstance(i, dict)]
    assert any(m.get("resumed") is True for m in metas)
