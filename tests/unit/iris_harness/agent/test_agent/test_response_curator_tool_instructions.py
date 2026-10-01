"""The user must never be told to run one of the model's own tools.

Cloud trial, 2026-09-19: "I don't have your electricity bill data loaded locally, so I
can't calculate your spending. To get that information, you can run the
``finance_lookup`` tool with the query 'electricity bill'." ``finance_lookup`` is an
internal tool the MODEL calls — there is no such command for the user, and the reply
reached a phone. The ReAct-leak guard only catches a whole scratchpad, not one sentence
of it.

The strip is narrow: a sentence must both instruct an action ("run", "use", "call") and
name a tool-shaped identifier (snake_case or backticked). Ordinary prose stays.
"""

from __future__ import annotations

from iris_harness.agent.agent_executor import AgentResult
from iris_harness.agent.response_curator import ResponseCurator, _strip_tool_instructions


def _curate(output: str) -> object:
    results = [AgentResult(agent_type="finance", output=output, success=True)]
    return ResponseCurator().curate(results, query="how much did I spend on electricity?")


def test_the_leaked_instruction_from_the_trial_is_removed() -> None:
    response = _curate(
        "I don't have your electricity bill data loaded locally, so I can't calculate "
        "your spending for the last month. To get that information, you can run the "
        "`finance_lookup` tool with the query 'electricity bill'."
    )

    assert "finance_lookup" not in response.text  # type: ignore[attr-defined]
    assert "don't have your electricity bill data" in response.text  # type: ignore[attr-defined]
    assert response.metadata["tool_instruction_stripped"]  # type: ignore[attr-defined]


def test_a_reply_that_was_only_an_instruction_gets_an_honest_answer() -> None:
    response = _curate("Try the inbox_digest tool to see your mail.")

    assert "inbox_digest" not in response.text  # type: ignore[attr-defined]
    assert "couldn't find that" in response.text  # type: ignore[attr-defined]


def test_ordinary_prose_is_untouched() -> None:
    for text in (
        "You spent $127.75 on electricity in the last billing cycle.",
        "Use the kitchen timer for 10 minutes.",
        "I called the bank and the payment cleared.",
        "Your next event is at 5:00 pm on Tuesday.",
    ):
        cleaned, removed = _strip_tool_instructions(text)
        assert cleaned == text
        assert removed == []


def test_user_facing_commands_are_not_stripped() -> None:
    """`iris auth gmail login` IS something the user runs; only tool ids go."""
    text = "Reconnect the account by running `iris auth gmail login --user you@example.com`."
    cleaned, removed = _strip_tool_instructions(text)

    assert cleaned == text
    assert removed == []


def test_several_instructions_all_go() -> None:
    cleaned, removed = _strip_tool_instructions(
        "Here is what I know. Run the finance_lookup tool first. "
        "Then call the inbox_digest tool for the rest."
    )

    assert "Here is what I know." in cleaned
    assert "finance_lookup" not in cleaned
    assert "inbox_digest" not in cleaned
    assert len(removed) == 2
