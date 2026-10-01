"""The L0/L1 context layers, pinned by a golden prompt snapshot.

Six features in this repo have shipped wired to nothing, and memory was the worst
case: the retriever fetched cross-session turns and learning signals every turn and
the prompt dropped them, `MemoryContext.summary` had no writer at all, and the
conversation reached the model as a fixed 3-line tail. A snapshot of the assembled
memory block is the check that catches that class of bug — a block that is not in
the prompt fails here, however well its producer is tested.

Update the golden file deliberately — delete it and re-run, then read the diff in
the commit — never to make a red test green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.agent.agentic_core import _build_react_prompt
from iris_harness.agent.context_budget import fill_recent_turns
from iris_harness.memory.retriever import MemoryContext

GOLDEN = Path(__file__).parent / "golden" / "react_memory_block.txt"

# One fixed context covering every L0 block and an L1 pointer.
CTX = MemoryContext(
    soul="# IRIS — Soul\n\nI am the test soul.",
    user_profile="# User Profile\n\n- Name: Test User\n- Role: maintainer",
    active="## Active\n- ship the memory PR",
    episodic_digest="## Patterns\n- asks for the day's summary each morning",
    behavior="## Behavior — Reminders\nTreat a reminder as a durable commitment.",
    behavior_name="reminders",
    summary="Goal: fix context memory. Decisions: retire the auto wiki. Open: retention.",
    related_turns=("user: what did we decide about Northwind dues?", "assistant: due on the 5th."),
    recent_turns=(
        "user: first question",
        "assistant: first answer",
        "user: second question",
        "assistant: second answer",
    ),
    pointers=("Earlier exchanges in this session are condensed in the summary above.",),
    linked=(
        "## What you remember about names in this message\n"
        "- the user: spouse of Petra (told 2026-09-16, confirmed)\n"
        "- Petra: works at Infosys (told 2026-09-16, confirmed)"
    ),
)


def _memory_block(prompt: str) -> str:
    """Just the assembled memory block — intro line up to the response rules."""
    start = prompt.lower().index("you already know the following")
    end = prompt.index("--- Response rules", start)
    return prompt[start:end].strip()


def test_memory_block_matches_golden() -> None:
    prompt = _build_react_prompt(
        "third question", tools=[], history=[], memory_context=CTX, memory_token_budget=4300
    )

    block = _memory_block(prompt)
    if not GOLDEN.exists():  # first run writes the snapshot
        GOLDEN.write_text(block + "\n", encoding="utf-8")
    assert block == GOLDEN.read_text(encoding="utf-8").strip()


@pytest.mark.parametrize(
    "needle",
    [
        "- Name: Test User",  # USER.md, curated head
        "ship the memory PR",  # active.md
        "asks for the day's summary each morning",  # episodic digest
        "Treat a reminder as a durable commitment.",  # matched behavior
        "## Earlier in this conversation",  # rolling summary — had NO renderer before
        "Goal: fix context memory.",
        "## From earlier sessions",  # cross-session recall — was cut by [-3:]
        "what did we decide about Northwind dues?",
        "Recent turns:",
        "Note: Earlier exchanges in this session",  # L1 pointer
    ],
)
def test_every_l0_and_l1_block_reaches_the_prompt(needle: str) -> None:
    prompt = _build_react_prompt(
        "third question", tools=[], history=[], memory_context=CTX, memory_token_budget=4300
    )
    assert needle in prompt


def test_block_order_is_priority_order() -> None:
    prompt = _build_react_prompt(
        "third question", tools=[], history=[], memory_context=CTX, memory_token_budget=4300
    )
    order = [
        prompt.index("- Name: Test User"),
        prompt.index("ship the memory PR"),
        prompt.index("asks for the day's summary each morning"),
        prompt.index("Treat a reminder as a durable commitment."),
        prompt.index("## Earlier in this conversation"),
        prompt.index("## From earlier sessions"),
        prompt.index("Recent turns:"),
        prompt.index("Note: Earlier exchanges"),
    ]
    assert order == sorted(order)


def test_memory_block_stays_within_its_budget() -> None:
    from iris_harness.llm.budget import estimate_tokens

    budget = 300  # deliberately tight: admission must evict, not overflow
    prompt = _build_react_prompt(
        "third question", tools=[], history=[], memory_context=CTX, memory_token_budget=budget
    )
    block = _memory_block(prompt)
    # Admission pins the user profile (index 0) even when it alone exceeds budget, so
    # allow that one block on top of the cap.
    assert estimate_tokens(block) <= budget + estimate_tokens(CTX.user_profile or "") + 50
    assert "- Name: Test User" in prompt  # identity is never the thing that gets dropped


class TestRecentTurnsBudget:
    """The conversation slice is token-filled, not a fixed 3-line tail."""

    def test_more_than_three_turns_when_the_budget_allows(self) -> None:
        turns = [f"user: question {i}" for i in range(12)]
        assert len(fill_recent_turns(turns, 4000)) == 12

    def test_fills_from_the_newest_end(self) -> None:
        turns = [f"user: question {i}" for i in range(12)]
        kept = fill_recent_turns(turns, 30)
        assert kept[-1] == "user: question 11"
        assert len(kept) < 12
        assert kept == turns[-len(kept) :]  # chronological order preserved

    def test_a_single_huge_turn_never_starves_continuity(self) -> None:
        turns = ["user: " + "x" * 40_000, "assistant: short answer"]
        assert len(fill_recent_turns(turns, 10)) == 2

    def test_no_budget_keeps_the_legacy_tail(self) -> None:
        turns = [f"user: question {i}" for i in range(12)]
        assert fill_recent_turns(turns, 0) == turns[-2:]

    def test_prompt_shows_more_than_three_lines_with_a_real_budget(self) -> None:
        ctx = MemoryContext(
            soul="soul",
            recent_turns=tuple(f"user: question {i}" for i in range(10)),
        )
        prompt = _build_react_prompt(
            "next", tools=[], history=[], memory_context=ctx, memory_token_budget=4300
        )
        assert "question 0" in prompt and "question 9" in prompt


class TestCapabilitiesLine:
    """L1 self-knowledge: generated, one line, and pointing at the detail."""

    def test_line_sits_with_the_identity_preamble_not_in_the_memory_block(self) -> None:
        line = 'Your current setup — plugins loaded: email. For the full list call iris_doc("CAPABILITIES").'
        prompt = _build_react_prompt(
            "hi",
            tools=[],
            history=[],
            memory_context=CTX,
            memory_token_budget=4300,
            capabilities_line=line,
        )

        assert line in prompt
        assert prompt.index(line) < prompt.lower().index("you already know the following")
        assert line not in _memory_block(prompt)

    def test_no_line_means_no_stray_text(self) -> None:
        prompt = _build_react_prompt(
            "hi", tools=[], history=[], memory_context=CTX, memory_token_budget=4300
        )

        assert "Your current setup" not in prompt
