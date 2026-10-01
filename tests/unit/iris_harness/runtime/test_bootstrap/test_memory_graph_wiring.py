"""The memory graph reaches the model on every path (memris PR 6; ADR-0115 decision 9).

A feature wired to nothing is this repo's most repeated failure, so each place the graph
has to arrive is pinned: the turn's context, both prompt builders, both tool paths, and
the Context tab.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iris_harness.agent.agent_executor import AgentTask
from iris_harness.foundation.auth import auth_headers
from iris_harness.llm.client import LLMToolCall
from iris_harness.memory.compactor import ConversationCompactor
from iris_harness.memory.graph_context import LINKED_HEADER
from iris_harness.memory.retriever import MemoryContext
from iris_harness.memory.store import MemoryStore, UserFact
from iris_harness.runtime.handlers.general_invoke import make_general_invoke
from iris_harness.runtime.handlers.general_tools import GeneralTools, make_general_tools
from iris_harness.runtime.handlers.local_skills import LocalSkills
from iris_harness.runtime.react_tools import builtin_react_tools
from iris_harness.runtime.session_memory import SessionMemory
from iris_harness.server.iris_api.main import create_app

# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = pytest.mark.usefixtures("test_vocabulary")

NOW = datetime(2026, 9, 16, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(db_path=tmp_path / "memory.db")
    s.ensure_schema()
    s.upsert_user_fact(UserFact("spouse", "Petra", 0.9, "test", NOW, NOW, 1, True))
    s.upsert_user_fact(UserFact("bank", "Northwind Bank", 0.9, "test", NOW, NOW, 1, True))
    return s


def _sessions(store: MemoryStore) -> SessionMemory:
    host = SimpleNamespace(
        memory_store=store,
        semantic_index=None,
        memory_retriever=SimpleNamespace(
            build_context=lambda **kw: MemoryContext(recent_turns=kw.get("recent_turns", ()))
        ),
        compactor=ConversationCompactor(compaction_threshold=50, keep_recent=10),
        learning=SimpleNamespace(behavior_miner=None),
    )
    return SessionMemory(host)  # type: ignore[arg-type]


class TestTheTurnsContext:
    def test_a_named_entity_brings_its_statements(self, store: MemoryStore) -> None:
        ctx = _sessions(store).build_memory_context("is Petra around?", session_id="s")

        assert ctx.linked is not None
        assert ctx.linked.startswith(LINKED_HEADER)
        assert "the user: spouse of Petra" in ctx.linked
        assert "Northwind" not in ctx.linked  # not named in this message

    def test_what_does_not_fit_becomes_a_pointer(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("iris_harness.memory.graph_context.linking_max_tokens", lambda: 12)

        ctx = _sessions(store).build_memory_context("Petra and Northwind Bank", session_id="s")

        assert any("memory_graph" in p for p in ctx.pointers)

    def test_a_broken_graph_costs_the_turn_nothing(
        self, store: MemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(_store: Any) -> Any:
            raise RuntimeError("graph unavailable")

        monkeypatch.setattr("iris_harness.memory.graph_context.graph_context", _boom)

        ctx = _sessions(store).build_memory_context("is Petra around?", session_id="s")

        assert ctx.linked is None


class TestTheGeneralHandlersPrompt:
    def test_the_linked_block_is_in_the_system_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Client:
            def __init__(self, _cfg: Any) -> None: ...

            def get_usage_mark(self) -> int:
                return 0

        monkeypatch.setattr("iris_harness.llm.client.CodingLLMClient", _Client)
        tier = SimpleNamespace(model="fake", provider="github", num_ctx=None)
        router = SimpleNamespace(
            get_tier=lambda _i: tier,
            get_llm_config=lambda _i: SimpleNamespace(
                provider="github",
                model="fake",
                base_url="https://example.test/v1",
                api_key_env=None,
            ),
        )
        invoke = make_general_invoke(
            tier_router=router,  # type: ignore[arg-type]
            general_tools=GeneralTools(bindings=lambda _q="": (), execute=None),
        )
        linked = f"{LINKED_HEADER}\n- Petra: works at Infosys (told 2026-09-16, confirmed)"
        task = AgentTask(
            query="is Petra around?",
            agent_type="general",
            memory_context=MemoryContext(linked=linked),
        )

        _client, system_prompt, _user, _mark, _strategy = invoke.build_prompts(task)

        assert linked in system_prompt


class TestTheTool:
    def test_the_react_loop_offers_it(self, store: MemoryStore) -> None:
        tools = {
            t.name: t
            for t in builtin_react_tools(
                semantic_index=None, wiki=None, repo_root=None, memory_store=store
            )
        }

        tool = tools["memory_graph"]
        assert "works_at" in tool.description  # generated from the ontology
        assert "the user: spouse of Petra" in tool.call({"entity": "Petra"})

    def test_native_tool_calling_binds_and_runs_it(self, store: MemoryStore) -> None:
        runtime = SimpleNamespace(
            memory_store=store, plugin_registry=SimpleNamespace(tools=lambda: [])
        )
        none = SimpleNamespace(
            tools=lambda: {}, relevant_tools=lambda _q: {}, direct_skill_response=None
        )
        general = make_general_tools(
            repo_root=None,
            wiki=None,
            semantic_index=None,
            runtime_holder=[runtime],
            local_skills=LocalSkills(
                tools=none.tools,
                relevant_tools=none.relevant_tools,
                direct_skill_response=None,
                direct_brief_response=None,
            ),
        )

        names = [b["function"]["name"] for b in general.bindings("who is Petra")]
        result = general.execute(
            AgentTask(query="who is Petra", agent_type="general"),
            LLMToolCall(id="1", name="memory_graph", arguments={"entity": "Petra"}),
            run_id="run-1",
        )

        assert "memory_graph" in names
        assert result["ok"] is True
        assert "spouse of Petra" in str(result["result"])


class TestTheContextTab:
    def test_it_shows_the_block_for_the_last_message(self, store: MemoryStore) -> None:
        sessions = _sessions(store)
        sessions.record_turn("s1", "is Petra around?", "I'll check.")
        runtime = SimpleNamespace(memory_store=store, semantic_index=None, sessions=sessions)

        with TestClient(
            create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
        ) as client:
            last = client.get("/memory/context/s1").json()
            asked = client.get("/memory/context/s1", params={"message": "Northwind Bank"}).json()

        graph_block = "memory graph (names in the message)"
        blocks = {b["block"]: b for b in last["blocks"]}
        assert last["message"] == "is Petra around?"
        assert blocks[graph_block]["present"] is True
        assert "spouse of Petra" in blocks[graph_block]["preview"]
        assert blocks[graph_block]["tokens"] > 0
        other = {b["block"]: b for b in asked["blocks"]}[graph_block]
        assert "Northwind Bank" in other["preview"] and "Petra" not in other["preview"]
