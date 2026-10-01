"""Acceptance: the memory chain, end to end, deterministically.

Nine PRs changed what IRIS remembers and what reaches a prompt. Each proved its own
piece. This proves the chain — one runtime, one conversation, asserting on the PROMPT
rather than on a model's words, so it can run in the gate instead of only by hand.

The failure mode it exists for: every block here has a producer that is well tested and
was, at some point, wired to nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agentic_core import _build_react_prompt
from iris_harness.memory.coordinator import FactCoordinator
from iris_harness.memory.lessons import SOURCE_CORRECTION, LessonCurator
from iris_harness.runtime.bootstrap import build_runtime

# No model server in tests (tests/conftest.py network guard): runtime turns here
# reach the LLM on their degrade paths, so the model is a stubbed dead server.
# Fact keys such as `bank` and `credit_card` come from the test vocabulary fragment
# (tests/fixtures/test_vocabulary, installed by the `test_vocabulary` fixture), not
# from whichever domain plugin the tree happens to carry.
pytestmark = [
    pytest.mark.integration,
    pytest.mark.usefixtures("offline_llm", "test_vocabulary"),
]

# Shaped like a real conversation id: an "accept…" name is a test run (retention.yaml's
# ephemeral prefixes), which cross-session recall leaves out (Map cleanup plan decision 7).
SID = "c0ffee000001"
BUDGET = 4300


@pytest.fixture
def runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real runtime on a throwaway store, with a deterministic summarizer."""
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("IRIS_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    # Small window so a six-exchange script actually rolls over (the real budget is
    # derived from the chat model's context and would need a far longer script).
    monkeypatch.setenv("IRIS_COMPACTION_TOKEN_BUDGET", "110")
    rt = build_runtime(use_background_scheduler=False)
    rt.compactor._llm = lambda prompt: (
        "Goal: track the Springfield flat.\n"
        "Decisions & outcomes: rent is due on the 5th; deposit was 180000.\n"
        "Open items: send the signed lease copy.\n"
        "Referenced: Petra Sutton, Remy, Northwind"
    )
    return rt


def _prompt(rt: Any, question: str) -> str:
    ctx = rt.sessions.build_memory_context(question, session_id=SID, intent="general")
    return _build_react_prompt(
        question, tools=[], history=[], memory_context=ctx, memory_token_budget=BUDGET
    )


def _talk(rt: Any, exchanges: list[tuple[str, str]]) -> None:
    for user, assistant in exchanges:
        rt.sessions.record_turn(SID, user, assistant)
    rt.sessions.wait_for_compaction(SID, timeout=30)


FLAT_CONVERSATION = [
    ("my landlord is Petra Sutton and rent is due on the 5th", "Noted — rent on the 5th."),
    ("the flat is in Springfield, Elmwood", "Got it."),
    ("the deposit was 180000 and Remy handles maintenance", "Understood."),
    ("i pay by UPI from Northwind", "Noted."),
    ("the lease renews in March", "Noted."),
    ("i still need to send the signed lease copy", "Noted."),
]


class TestTheConversationSurvives:
    def test_a_detail_from_the_first_exchange_is_still_in_the_prompt(self, runtime: Any) -> None:
        """The whole reason this work started: the model saw 3 lines of history."""
        _talk(runtime, FLAT_CONVERSATION)

        prompt = _prompt(runtime, "who is the landlord again?")

        assert "Petra Sutton" in prompt

    def test_more_than_three_lines_reach_the_model(self, runtime: Any) -> None:
        _talk(runtime, FLAT_CONVERSATION)

        prompt = _prompt(runtime, "what do we know so far?")
        lines = prompt.count("\n  user: ") + prompt.count("\n  assistant: ")

        assert lines > 3

    def test_the_summary_is_rendered_with_its_sections(self, runtime: Any) -> None:
        _talk(runtime, FLAT_CONVERSATION)

        prompt = _prompt(runtime, "what did we decide?")

        assert "## Earlier in this conversation" in prompt
        assert "Open items:" in prompt

    def test_a_pointer_names_how_to_read_the_rest(self, runtime: Any) -> None:
        _talk(runtime, FLAT_CONVERSATION)

        prompt = _prompt(runtime, "anything else?")

        assert "recall_conversation" in prompt

    def test_recall_conversation_returns_it_word_for_word(self, runtime: Any) -> None:
        from iris_harness.runtime.react_tools import builtin_react_tools

        _talk(runtime, FLAT_CONVERSATION)
        specs = {
            s.name: s
            for s in builtin_react_tools(
                semantic_index=runtime.semantic_index,
                wiki=runtime.wiki,
                repo_root=None,
                memory_store=runtime.memory_store,
            )
        }

        out = specs["recall_conversation"].call({"query": "deposit"})

        assert "180000" in out


class _CannedExtractor:
    """The fact-extraction model, answering what a model reads in the Northwind sentence.

    What is under test is the review gate after extraction, not the model's reading.
    """

    def invoke(self, *_args: Any, **_kwargs: Any) -> Any:
        from langchain_core.messages import AIMessage

        return AIMessage(content='[{"key": "bank", "value": "Northwind", "confidence": 0.9}]')


@pytest.fixture
def extractor_reads_northwind(offline_llm: None, monkeypatch: pytest.MonkeyPatch) -> None:
    # Requests offline_llm so this patch lands after (over) the module-wide dead model.
    from iris_harness.llm import client

    monkeypatch.setattr(client, "_default_model_factory", lambda **_kw: _CannedExtractor())


class TestNothingUnconfirmedReachesThePrompt:
    @pytest.mark.usefixtures("extractor_reads_northwind")
    def test_an_extracted_fact_waits_for_approval(self, runtime: Any) -> None:
        runtime.capture.extract_and_store_facts("i bank with Northwind and my deposit was 180000")

        proposals = runtime.memory_store.fetch_fact_proposals()
        prompt = _prompt(runtime, "where do i bank?")

        assert proposals, "extraction should propose something from a first-person statement"
        assert runtime.memory_store.fetch_all_user_facts(confirmed_only=True) == []
        assert "Confirmed facts about the user" not in prompt

    @pytest.mark.usefixtures("extractor_reads_northwind")
    def test_approving_it_puts_it_in_the_prompt(self, runtime: Any) -> None:
        runtime.capture.extract_and_store_facts("i bank with Northwind and my deposit was 180000")
        proposal = runtime.memory_store.fetch_fact_proposals()[0]

        FactCoordinator(runtime.memory_store, runtime.semantic_index).approve_proposal(proposal.id)
        prompt = _prompt(runtime, "where do i bank?")

        assert f"{proposal.key}={proposal.value}" in prompt

    def test_third_party_content_proposes_nothing(self, runtime: Any) -> None:
        """`employer=Department of Justice` was mined from a pasted article."""
        runtime.capture.extract_and_store_facts(
            "the Department of Justice said Russian intelligence agents were charged"
        )

        assert runtime.memory_store.fetch_fact_proposals() == []


class TestLessonsAndRuns:
    def test_an_approved_lesson_becomes_a_matchable_behavior(self, runtime: Any) -> None:
        curator = LessonCurator(runtime.memory_store)
        pid = curator.propose(
            trigger="asked about credit card dues",
            lesson="call finance_lookup before searching the inbox",
            source=SOURCE_CORRECTION,
        )
        assert pid is not None

        name = curator.approve(pid)

        from iris_harness.memory.identity import match_behavior

        matched = match_behavior("finance", "what are my credit card dues this month?")
        assert name is not None
        assert matched is not None and matched.name == name

    def test_a_success_report_is_not_a_lesson(self, runtime: Any) -> None:
        assert (
            LessonCurator(runtime.memory_store).propose(
                trigger="what's my net worth?",
                lesson="For 'finance' requests like this, the finance agent answered it cleanly.",
                source=SOURCE_CORRECTION,
            )
            is None
        )

    def test_a_playground_session_leaves_no_trace_in_memory(self, runtime: Any) -> None:
        runtime.sessions.record_turn("playground-acceptance", "scenario q", "scenario a")

        assert runtime.memory_store.fetch_turn_ids("playground-acceptance") == []

    def test_the_wiki_is_not_written_by_a_turn(self, runtime: Any) -> None:
        _talk(runtime, FLAT_CONVERSATION)

        assert runtime.wiki.ingest_enabled is False
        assert list((runtime.data_dir / "wiki").glob("entities/*.md")) == []


class TestTheContextIsInspectable:
    def test_every_block_reports_its_tokens_and_presence(self, runtime: Any) -> None:
        """The anti-'shipped unwired' check, on the same view the UI renders."""
        _talk(runtime, FLAT_CONVERSATION)
        ctx = runtime.sessions.build_memory_context("x", session_id=SID, intent="general")

        assert ctx.summary, "the rolling summary should exist after a rollover"
        assert ctx.recent_turns, "the verbatim window should not be empty"
        assert ctx.pointers, "L1 should say what is not in the prompt"

    def test_the_memory_block_stays_inside_its_budget(self, runtime: Any) -> None:
        from iris_harness.llm.budget import estimate_tokens

        _talk(runtime, FLAT_CONVERSATION)
        prompt = _prompt(runtime, "what do we know?")
        start = prompt.lower().index("you already know the following")
        block = prompt[start : prompt.index("--- Response rules", start)]

        assert estimate_tokens(block) <= BUDGET
