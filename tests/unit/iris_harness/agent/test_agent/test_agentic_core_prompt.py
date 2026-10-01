"""Tests for ``_build_react_prompt`` identity injection.

The AgenticCore ReAct prompt assembly used to hardcode "You are IRIS,
a personal AI assistant." and ignore SOUL.md / USER.md / active /
episodic / behavior content even though the loader populated them on
``MemoryContext``. That meant the identity layer never reached the
LLM for any ``system``-intent turn (the default path when
IRIS_AGENTIC_CORE_ENABLED=on). These tests pin the corrected behavior.
"""

from __future__ import annotations

from iris_harness.agent.agentic_core import ToolSpec, _build_react_prompt
from iris_harness.memory.retriever import MemoryContext


def test_build_react_prompt_uses_soul_when_available() -> None:
    ctx = MemoryContext(soul="# IRIS — Soul\n\nI am the test soul.")

    prompt = _build_react_prompt("hello", tools=[], history=[], memory_context=ctx)

    # Soul replaces the hardcoded preamble verbatim.
    assert "# IRIS — Soul" in prompt
    assert "I am the test soul." in prompt
    assert "You are IRIS, a personal AI assistant." not in prompt


def test_build_react_prompt_falls_back_to_default_when_soul_missing() -> None:
    prompt = _build_react_prompt("hello", tools=[], history=[], memory_context=None)
    assert "You are IRIS, a personal AI assistant." in prompt


def test_build_react_prompt_injects_user_profile() -> None:
    ctx = MemoryContext(
        soul="soul body",
        user_profile="# User Profile\n\n- Name: Robin\n- Role: maintainer",
    )

    prompt = _build_react_prompt("hi", tools=[], history=[], memory_context=ctx)

    # Bracketed [Memory Context] marker removed 2026-05-19 — small models
    # were echoing it back. Check for the inline prose intro instead.
    assert "you already know the following" in prompt.lower()
    assert "- Name: Robin" in prompt
    assert "- Role: maintainer" in prompt
    assert "[Memory Context]" not in prompt  # explicit anti-regression


def test_build_react_prompt_shows_confirmed_facts_alongside_the_profile() -> None:
    """Both reach the prompt now, because a stored fact is owner-confirmed.

    This used to be "profile takes precedence": facts were whatever the extractor
    mined, so letting them sit next to the curated profile risked contradicting it.
    With confirmation, a stored fact IS the user's word — and gating it behind "only
    when USER.md is empty" meant everything IRIS learned stayed out of the prompt,
    since USER.md is never empty. A value that changes goes through the review queue
    as a "changed?" item instead of quietly landing beside the old one.
    """

    from datetime import UTC, datetime

    from iris_harness.memory.store import UserFact

    def _fact(key: str, value: str) -> UserFact:
        now = datetime.now(UTC)
        return UserFact(
            key=key,
            value=value,
            confidence=0.9,
            source="test",
            first_seen=now,
            last_confirmed=now,
        )

    ctx = MemoryContext(
        user_profile="Real curated profile content",
        user_facts=(_fact("bank", "Northwind"),),
    )

    prompt = _build_react_prompt("hi", tools=[], history=[], memory_context=ctx)

    assert "Real curated profile content" in prompt
    assert "Confirmed facts about the user: bank=Northwind" in prompt


def test_build_react_prompt_falls_back_to_user_facts_when_no_profile() -> None:
    from datetime import UTC, datetime

    from iris_harness.memory.store import UserFact

    def _fact(key: str, value: str) -> UserFact:
        now = datetime.now(UTC)
        return UserFact(
            key=key,
            value=value,
            confidence=0.9,
            source="test",
            first_seen=now,
            last_confirmed=now,
        )

    ctx = MemoryContext(
        user_facts=(
            _fact("name", "Robin"),
            _fact("region", "US"),
        ),
    )

    prompt = _build_react_prompt("hi", tools=[], history=[], memory_context=ctx)

    assert "name=Robin" in prompt
    assert "region=US" in prompt


def test_build_react_prompt_injects_active_and_episodic_and_behavior() -> None:
    ctx = MemoryContext(
        active="## Active\n- [ ] write tests",
        episodic_digest="## What I've noticed about you\n- prefers concise replies",
        behavior="## Behavior\nUse direct tone for engineering tasks.",
    )

    prompt = _build_react_prompt("hi", tools=[], history=[], memory_context=ctx)

    assert "write tests" in prompt
    assert "prefers concise replies" in prompt
    assert "Use direct tone for engineering tasks." in prompt


def test_build_react_prompt_omits_memory_block_when_nothing_to_inject() -> None:
    ctx = MemoryContext()  # all fields default / empty

    prompt = _build_react_prompt("hi", tools=[], history=[], memory_context=ctx)

    # No bracket marker, AND no inline intro phrase, when there's
    # nothing to inject.
    assert "[Memory Context]" not in prompt
    assert "you already know the following" not in prompt.lower()


def test_final_answer_regex_stops_at_trailing_react_keywords() -> None:
    """Anti-regression: when the LLM emits "Final Answer: <answer>" followed
    by hallucinated "Thought: ... Action: ..." text, the parser must
    extract ONLY the answer text, not the trailing ReAct leak."""

    from iris_harness.agent.agentic_core import _parse_react_step

    raw = (
        "Thought: I know the answer.\n"
        "Final Answer: You go by the name Robin.\n"
        "Thought: The user also asked about their stack.\n"
        "Action: memory_search\n"
        'Action Input: {"query": "stack"}\n'
        "Observation: <the tool's response>"
    )
    step = _parse_react_step(raw)

    assert step.is_terminal
    assert step.final_answer == "You go by the name Robin."
    # The bleed text must not be in the captured final answer.
    assert "memory_search" not in step.final_answer
    assert "Thought:" not in step.final_answer
    assert "Observation:" not in step.final_answer


def test_final_answer_regex_still_captures_multi_line_answers() -> None:
    """The lookahead-tightened regex must keep working for legitimate
    multi-line Final Answer text — only ReAct keywords end the capture."""

    from iris_harness.agent.agentic_core import _parse_react_step

    raw = (
        "Final Answer: Here is a multi-line answer:\n"
        "- Point one\n"
        "- Point two\n"
        "- Point three"
    )
    step = _parse_react_step(raw)
    assert step.is_terminal
    assert "Point three" in step.final_answer
    assert "multi-line answer" in step.final_answer


def test_build_react_prompt_drops_bracketed_section_markers() -> None:
    """Anti-regression: the [Example], [End Example], [Freshness
    Requirement] markers were removed because small/medium models
    echoed them into user-facing output. Inline prose stays."""

    from iris_harness.agent.agentic_core import ToolSpec

    ctx = MemoryContext(soul="agent soul", user_profile="user profile")
    tool = ToolSpec(name="research", description="search the web", call=lambda args: "")
    prompt = _build_react_prompt(
        "what is the latest news",
        tools=[tool],
        history=[],
        memory_context=ctx,
        require_retrieval_before_final=True,
    )

    # None of the bracket-tagged section labels remain.
    for marker in ("[Example]", "[End Example]", "[Memory Context]", "[Freshness Requirement]"):
        assert marker not in prompt, f"leaked marker {marker!r} in prompt"
    # But the actual content / instructions are still present.
    assert "examples" in prompt.lower()
    assert "Freshness requirement" in prompt
    assert "research" in prompt


def test_prompt_guides_direct_answer_when_no_tool_needed() -> None:
    # Regression: the ReAct prompt over-biased tool use (every few-shot example was a
    # research). It now tells the model to answer directly for known facts/math and
    # shows a no-tool example, so trivial questions don't trigger research.
    from iris_harness.agent.agentic_core import ToolSpec, _build_react_prompt

    tools = [ToolSpec(name="research", description="search the web", call=lambda args: "")]
    prompt = _build_react_prompt("what is 2 plus 2?", tools=tools, history=[], memory_context=None)

    # explicit guidance to skip tools for known answers
    assert "only call a tool when you genuinely need" in prompt.lower()
    # a no-tool example exists (Final Answer without an Action)
    assert "Final Answer: 4" in prompt
    # the tool example still exists for genuine external needs
    assert "Action: research" in prompt


def test_prompt_without_tools_has_no_tool_examples() -> None:
    from iris_harness.agent.agentic_core import _build_react_prompt

    prompt = _build_react_prompt("hi", tools=[], history=[], memory_context=None)
    assert "Action: research" not in prompt


def test_guidance_is_the_tools_own_and_shows_only_while_it_is_on_the_loop() -> None:
    """ADR-0110: the whole-day nudge (ADR-0077 P4) now rides on daily_plan's manifest
    declaration; the core renders it and authors none of it."""
    nudge = "For a whole-day question call daily_plan; calendar_lookup only for a SPECIFIC one."
    tools = [
        ToolSpec(
            name="daily_plan", description="day aggregator", call=lambda a: "", guidance=nudge
        ),
        ToolSpec(name="calendar_lookup", description="calendar", call=lambda a: ""),
    ]
    prompt = _build_react_prompt("how is my day", tools=tools, history=[], memory_context=None)
    assert nudge in prompt
    without = _build_react_prompt(
        "hi",
        tools=[ToolSpec(name="research", description="s", call=lambda a: "")],
        history=[],
        memory_context=None,
    )
    assert "daily_plan" not in without and "whole-day" not in without


def test_the_core_prompt_names_no_plugin_tool() -> None:
    """Routing prose belongs to the plugins (ADR-0110): with guidance-free tools on the loop
    the prompt carries no finance, email or planner sentence."""
    tools = [
        ToolSpec(name=n, description=n, call=lambda a: "")
        for n in ("finance_lookup", "upcoming_dues", "search_inbox", "daily_plan")
    ]
    prompt = _build_react_prompt("what do I owe", tools=tools, history=[], memory_context=None)
    body = prompt.split("Available tools:")[1]
    for phrase in ("call finance_lookup first", "call upcoming_dues first", "call daily_plan"):
        assert phrase not in body


def test_write_tools_are_marked_and_the_ask_first_rule_appears_once() -> None:
    tools = [
        ToolSpec(name="lookup", description="read it", call=lambda a: ""),
        ToolSpec(
            name="make_it",
            description="create it",
            call=lambda a: "",
            effect="write",
            confirm="once",
        ),
        ToolSpec(
            name="fix_it", description="edit it", call=lambda a: "", effect="write", confirm="never"
        ),
    ]
    prompt = _build_react_prompt("do things", tools=tools, history=[], memory_context=None)
    assert "- make_it (WRITES on the user's behalf): create it" in prompt
    assert "- fix_it (WRITES on the user's behalf): edit it" in prompt
    assert "- lookup: read it" in prompt
    assert prompt.count("ask the user ONCE with ask_user") == 1
    read_only = _build_react_prompt("do things", tools=tools[:1], history=[], memory_context=None)
    assert "ask the user ONCE" not in read_only


def test_a_write_that_is_not_held_brings_no_ask_first_rule() -> None:
    """restore_email is ``confirm: never``: telling the model to ask before it made the
    model ask before everything (2026-09-21)."""
    tools = [
        ToolSpec(
            name="fix_it", description="edit it", call=lambda a: "", effect="write", confirm="never"
        )
    ]
    prompt = _build_react_prompt("undo that", tools=tools, history=[], memory_context=None)
    assert "- fix_it (WRITES on the user's behalf): edit it" in prompt
    assert "ask the user ONCE" not in prompt


def test_duplicate_guidance_renders_once() -> None:
    same = "Prefer the local record over the web."
    tools = [
        ToolSpec(name="a", description="a", call=lambda a: "", guidance=same),
        ToolSpec(name="b", description="b", call=lambda a: "", guidance=same),
    ]
    prompt = _build_react_prompt("q", tools=tools, history=[], memory_context=None)
    assert prompt.count(same) == 1


# --- the scratchpad follows the question (multi-step loop plan, PR 4) --------------

_STEP = (
    "Thought: I need the dues first\nAction: upcoming_dues\n"
    'Action Input: {"within_days": 15}\nObservation: Wingtip Bank: INR 3,150.40, due 2026-09-30'
)


def test_no_history_means_no_scratchpad() -> None:
    prompt = _build_react_prompt("hi", tools=[], history=[], memory_context=None)
    assert prompt.endswith("User: hi\nAssistant:")
    assert "Your steps so far" not in prompt


def test_history_is_the_assistants_work_after_the_question() -> None:
    prompt = _build_react_prompt("what is due?", tools=[], history=[_STEP], memory_context=None)
    user = prompt.index("User: what is due?\nAssistant:")
    label = prompt.index("Your steps so far on this request")
    step = prompt.index("Observation: Wingtip Bank: INR 3,150.40, due 2026-09-30")
    cue = prompt.index("Continue from the last Observation. Do NOT repeat a completed action")
    assert user < label < step < cue
    # Nothing follows the cue: the model's next line is its next Thought.
    assert prompt.rstrip().endswith("or give the Final Answer.")


def test_nudges_land_inside_the_scratchpad() -> None:
    nudge = "System: Your last reply was internal reasoning or incomplete."
    prompt = _build_react_prompt("q", tools=[], history=[_STEP, nudge], memory_context=None)
    assert prompt.index("Your steps so far") < prompt.index(nudge) < prompt.index("Continue from")
