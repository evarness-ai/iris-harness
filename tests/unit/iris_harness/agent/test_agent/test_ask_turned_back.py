"""The approval card is the selection UI: asks are turned back until it is shown.

A destructive call already waits for the owner's approval card, which lists every item
for them to approve or reject, so "do you want me to delete these?" makes them confirm
twice and "which of these?" hands them a question with nothing to pick from. qwen2.5
asks anyway, and asked again repeats itself word for word (2026-09-22 probe: 5 of 7
runs ended on a bare question). The loop turns back every ask while a destructive tool
is on the menu and no card has been shown, twice at most, then ends the turn on a plain
account. These pin the rule; the loop-level cases (sync and stream) are in
test_card_is_selection_loop.py and tests/integration/test_gmail_trash_approval.py.
"""

from __future__ import annotations

from typing import Any

from iris_harness.agent.agentic_core import (
    _APPROVAL_APPROVED,
    _APPROVAL_REJECTED,
    _APPROVAL_SETTLED,
    _APPROVAL_UNREADABLE,
    _APPROVAL_WAITING,
    _FOUND_ITEMS_MAX_LINES,
    ASK_USER_ACTION,
    ReactStep,
    ToolSpec,
    _answer_as_question,
    _found_items,
    _guard_ask,
    _hold_back_answer,
    _run_asked_user,
    _with_found_items,
)


def _tool(name: str, effect: str, confirm: str = "never") -> ToolSpec:
    def call(args: dict[str, Any]) -> str:
        return ""

    return ToolSpec(name=name, description=name, call=call, effect=effect, confirm=confirm)


SEARCH = _tool("search_inbox", "read")
TRASH = _tool("trash_email", "destructive", "approval")
REMIND = _tool("create_reminder", "write", "once")


def _ask(observation: str | None = None) -> ReactStep:
    return ReactStep(thought="t", action=ASK_USER_ACTION, observation=observation)


def _do(name: str, observation: str = "ok") -> ReactStep:
    return ReactStep(thought="t", action=name, observation=observation)


# An attempt at the card tool that never reached the owner: the run the guard is for.
REFUSED = _do("trash_email", "Error: no email with id 'x1'. Nothing was sent for approval.")


def _turned_back(steps: list[ReactStep], tools: list[ToolSpec]) -> str | None:
    guard = _guard_ask(steps, tools)
    return guard.text if guard is not None and guard.kind == "turn_back" else None


def test_an_ask_after_an_attempt_at_the_card_tool_is_turned_back() -> None:
    back = _turned_back([REFUSED, _do("search_inbox"), _ask()], [SEARCH, TRASH])
    assert back is not None and back.startswith("Not asked yet.")
    assert "Call trash_email now" in back
    assert "confirmation and their selection" in back


def test_no_destructive_tool_means_no_turn_back() -> None:
    assert _guard_ask([_ask()], [SEARCH, REMIND]) is None


def test_a_card_tool_only_on_the_menu_leaves_the_question_alone() -> None:
    """A read-only turn ("read my email from John") has a legitimate "which one?"."""
    assert _guard_ask([_do("search_inbox"), _ask()], [SEARCH, TRASH]) is None
    assert _guard_ask([_do("search_inbox"), _final("Which one?")], [SEARCH, TRASH]) is None


def test_asking_again_is_turned_back_again_up_to_the_cap() -> None:
    first = _turned_back([REFUSED, _ask()], [TRASH])
    second = _turned_back([REFUSED, _ask(first), _ask()], [TRASH])
    assert second is not None


def test_past_the_cap_the_turn_ends_on_a_factual_account() -> None:
    first = _turned_back([REFUSED, _ask()], [TRASH])
    second = _turned_back([REFUSED, _ask(first), _ask()], [TRASH])
    guard = _guard_ask([REFUSED, _ask(first), _ask(second), _ask()], [TRASH])
    assert guard is not None and guard.kind == "end"
    assert guard.text == "I couldn't prepare the trash_email request, so nothing was changed."


def test_the_factual_account_shows_what_the_last_read_found() -> None:
    back = "Not asked yet. x"
    steps = [
        REFUSED,
        _do("search_inbox", "1. Deals [id a1]\n2. Sale [id b2]"),
        _ask(back),
        _ask(back),
    ]
    guard = _guard_ask([*steps, _ask()], [SEARCH, TRASH])
    assert guard is not None and guard.kind == "end"
    assert guard.text.endswith("What I found:\n1. Deals [id a1]\n2. Sale [id b2]")


def test_a_refused_destructive_call_does_not_count_as_the_card() -> None:
    """Step 0: a guessed-ids refusal switched the turn-back off in 4 of 7 runs. It is an
    attempt (so the guard applies) but no card (so the ask does not go through)."""
    assert _turned_back([REFUSED, _ask()], [TRASH]) is not None


def test_a_queued_card_lets_the_ask_through() -> None:
    for observation in (
        f"{_APPROVAL_WAITING} (ap-1).",
        f"{_APPROVAL_APPROVED} Results:\ntrash_email: done",
        f"{_APPROVAL_REJECTED} trash_email {{}}. Nothing was changed.",
        f"{_APPROVAL_SETTLED}expired, so nothing was changed: x.",
        f"{_APPROVAL_UNREADABLE} (ap-1). Nothing was changed.",
    ):
        assert _guard_ask([_do("trash_email", observation), _ask()], [TRASH]) is None


def test_the_confirmation_a_held_write_asked_for_is_not_turned_back() -> None:
    assert _guard_ask([_do("create_reminder"), _ask()], [REMIND, TRASH]) is None


def test_a_turned_back_ask_is_not_the_users_consent() -> None:
    """Else a write held for confirmation could pass with nobody asked."""
    back = _turned_back([REFUSED, _ask()], [TRASH])
    steps = [_do("create_reminder"), _ask(back)]
    assert _run_asked_user(steps, "create_reminder") is False
    assert _run_asked_user([_do("create_reminder"), _ask("yes")], "create_reminder") is True


def test_found_items_are_the_last_reads_result_bounded() -> None:
    long = "\n".join(f"{n}. item {n}" for n in range(1, 31))
    steps = [_do("search_inbox", "old"), _do("search_inbox", long), _ask()]
    found = _found_items(steps, [SEARCH])
    lines = found.splitlines()
    assert lines[0] == "1. item 1"
    assert len(lines) == _FOUND_ITEMS_MAX_LINES + 1
    assert lines[-1] == f"(+{30 - _FOUND_ITEMS_MAX_LINES} more lines not shown)"


def test_found_items_skip_errors_and_non_read_tools() -> None:
    assert _found_items([_do("search_inbox", "Error: boom")], [SEARCH]) == ""
    assert _found_items([_do("create_reminder", "Reminder set")], [SEARCH, REMIND]) == ""
    assert _found_items([], [SEARCH]) == ""


def test_found_items_are_bounded_by_characters_too() -> None:
    found = _found_items([_do("search_inbox", "x" * 5000)], [SEARCH])
    assert len(found) <= 1503 and found.endswith("...")


def test_a_question_carries_the_items_unless_it_already_shows_them() -> None:
    steps = [_do("search_inbox", "1. Deals\n2. Sale")]
    assert _with_found_items("Which ones?", steps, [SEARCH]) == "Which ones?\n\n1. Deals\n2. Sale"
    already = "Which ones?\n1. Deals\n2. Sale"
    assert _with_found_items(already, steps, [SEARCH]) == already
    assert _with_found_items("Which ones?", [], [SEARCH]) == "Which ones?"


# --- a final answer that ends on a question is an ask (owner, 2026-09-22) -----------


def _final(text: str, observation: str | None = None) -> ReactStep:
    return ReactStep(thought="", final_answer=text, is_terminal=True, observation=observation)


def test_a_final_answer_is_a_question_only_by_its_trailing_question_mark() -> None:
    assert _answer_as_question(_final("Which ones should I delete?  "), "") == (
        "Which ones should I delete?"
    )
    assert _answer_as_question(_final("Done. I trashed them."), "") == ""
    assert _answer_as_question(_final("Is it done? Yes, it is."), "") == ""
    plain = ReactStep()
    assert _answer_as_question(plain, "Could you give me the ids?") == "Could you give me the ids?"
    assert _answer_as_question(plain, "I need to check first?") == ""  # planning chatter
    acting = ReactStep(thought="t", action="search_inbox")
    assert _answer_as_question(acting, "Anything?") == ""


def test_asks_and_question_answers_share_one_budget() -> None:
    first = _turned_back([REFUSED, _ask()], [TRASH])
    second = _turned_back([REFUSED, _ask(first), _final("Which?")], [TRASH])
    assert second is not None
    steps = [REFUSED, _ask(first), _final("", observation=second), _final("Which?")]
    guard = _guard_ask(steps, [TRASH])
    assert guard is not None and guard.kind == "end"


def test_holding_back_an_answer_unfinals_it() -> None:
    step = ReactStep(thought="t", final_answer="Which?", is_terminal=True)
    entry = _hold_back_answer(step, "Which?", "Not asked yet. x")
    assert (step.is_terminal, step.final_answer, step.observation) == (
        False,
        None,
        "Not asked yet. x",
    )
    assert entry == "Thought: t\nFinal Answer: Which?\nObservation: Not asked yet. x"
