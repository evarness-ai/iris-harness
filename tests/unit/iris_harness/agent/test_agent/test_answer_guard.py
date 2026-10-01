"""A final answer that claims a change nothing made, or echoes the loop's own working,
is turned back once and then corrected — never shown as it is.

2026-09-27 tool eval (qwen3.5:4b, thinking off): "Marked the expense report task as
done." with no complete_task call, and a day-plan reply that was the loop's scratchpad
header. The rule is checked on its own and through both loops (sync ``run`` and
``run_stream``) with a scripted model and a real write tool.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from iris_harness.agent.agentic_core import AgenticCore, AgenticCoreConfig, ToolSpec
from iris_harness.agent.answer_guard import (
    SCAFFOLD_ECHO_NOTE,
    UNBACKED_CLAIM_ANSWER,
    UNBACKED_CLAIM_NOTE,
    check_final_answer,
    claims_a_write,
    load_claim_vocabulary,
)

# --- the claim vocabulary (config/write_claims.yaml) ---------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Marked the expense report task as done.",
        "I've added 'Renew passport' to your task list, due Friday.",
        "Done. Your expense report task has been marked complete.",
        "Your task to renew your passport is now set for Friday, October 2.",
        "The reminder has been cancelled.",
        "Done.",  # the whole answer (2026-09-27 rerun)
        'I will mark the "expense report" task as done now.',  # a promise, no call
        "You have two pending items: you've added a task to renew your passport.",
        # The conditional is in another sentence; the claim still stands.
        "I've added it. If you need anything else, let me know.",
    ],
)
def test_claims_of_a_change_are_recognised(text: str) -> None:
    assert claims_a_write(text)


@pytest.mark.parametrize(
    "text",
    [
        "It has not been added yet.",
        "I couldn't mark it as done.",
        "I haven't added it.",
        "Would you like me to add the task?",
        "Which task would you like to mark as done?",
        # How-to text (2026-09-28: a correct answer was replaced by "nothing was saved").
        "Once you give me the details, I'll add the reminder to your calendar.",
        "If you'd like, I'll set it for 9am.",
        "Just say 'remind me at 5pm' and I'll create it.",
        "I'll add 'Renew passport' due Friday. Shall I go ahead?",
        "Done with that — here is your plan for today.",
        "There are no bills due this week.",
        "I cannot delete all your tasks at once.",
        "Here is a 3-step plan: 1. Research the company.",
    ],
)
def test_questions_refusals_and_reads_are_not_claims(text: str) -> None:
    assert not claims_a_write(text)


def test_the_vocabulary_is_yaml_and_absent_means_no_check(tmp_path) -> None:
    path = tmp_path / "write_claims.yaml"
    path.write_text("claims: [filed ... away]\nnegations: [not]\nprior_markers: [already]\n")
    vocab = load_claim_vocabulary(path)
    assert claims_a_write("I filed the receipt away.", vocab)
    assert not claims_a_write("I did not file it away.", vocab)
    assert not claims_a_write("I've added it.", vocab)  # not in this file
    empty = load_claim_vocabulary(tmp_path / "absent.yaml")
    assert not claims_a_write("I've added it.", empty)


# --- the verdict -----------------------------------------------------------------------

_MARKERS = ("Your steps so far on this request",)


def _verdict(answer: str, effects: list[str] | None = None, retries: int = 0) -> Any:
    return check_final_answer(
        answer,
        effects_executed=effects or [],
        retries_used=retries,
        markers=_MARKERS,
        fallback="the last tool result",
    )


def test_an_unbacked_claim_is_turned_back_once_then_replaced() -> None:
    first = _verdict("Marked the expense report task as done.")
    assert (first.kind, first.note) == ("retry", UNBACKED_CLAIM_NOTE)
    second = _verdict("Marked the expense report task as done.", retries=1)
    assert (second.kind, second.text) == ("replace", UNBACKED_CLAIM_ANSWER)


def test_a_claim_backed_by_a_write_that_ran_stands() -> None:
    assert _verdict("I've added it to your tasks.", effects=["write"]).kind == "ok"


def test_after_the_turn_back_a_change_placed_on_an_earlier_request_stands() -> None:
    answer = "I already added it earlier, on your last request."
    assert _verdict(answer, retries=1).kind == "ok"


def test_a_scaffold_echo_is_turned_back_then_falls_back_to_the_last_result() -> None:
    for echo in ("Your steps so far on this request (every Observation ...)", "Thought: hm"):
        assert _verdict(echo).note == SCAFFOLD_ECHO_NOTE
        replaced = _verdict(echo, retries=1)
        assert (replaced.kind, replaced.text) == ("replace", "the last tool result")


# --- through the loops -------------------------------------------------------------------


class _Script:
    """The model: returns the scripted replies in order and records each prompt."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0)


def _core(script: _Script, done: list[dict[str, Any]]) -> AgenticCore:
    def complete_task(args: dict[str, Any]) -> str:
        done.append(args)
        return "✅ Done — Submit the expense report."

    tool = ToolSpec(
        name="complete_task",
        description="Mark one open to-do done.",
        call=complete_task,
        effect="write",
        confirm="never",
    )
    return AgenticCore(AgenticCoreConfig(max_iterations=5), llm_call=script, tools=[tool])


_CLAIM = "Thought: The user wants it done.\nFinal Answer: Marked the expense report task as done."
_CALL = 'Thought: call it\nAction: complete_task\nAction Input: {"task": "expense report"}'
_AFTER = "Thought: it ran\nFinal Answer: Done — the expense report task is complete."


def _stream_answer(chunks: Iterator[object]) -> str:
    return "".join(c for c in chunks if isinstance(c, str))


@pytest.mark.parametrize("mode", ["sync", "stream"])
def test_a_turned_back_claim_gets_the_tool_called(mode: str) -> None:
    done: list[dict[str, Any]] = []
    script = _Script(_CLAIM, _CALL, _AFTER)
    core = _core(script, done)
    if mode == "sync":
        answer = core.run("Mark the expense report task as done").final_answer
    else:
        answer = _stream_answer(core.run_stream("Mark the expense report task as done"))
    assert done == [{"task": "expense report"}]  # the write really happened
    assert answer == "Done — the expense report task is complete."
    assert UNBACKED_CLAIM_NOTE in script.prompts[1]  # the model was told why
    # ... and what this request asked, so it does not act on a recalled one.
    assert 'This request: "Mark the expense report task as done".' in script.prompts[1]


@pytest.mark.parametrize("mode", ["sync", "stream"])
def test_a_claim_repeated_after_the_turn_back_is_replaced(mode: str) -> None:
    done: list[dict[str, Any]] = []
    core = _core(_Script(_CLAIM, _CLAIM), done)
    if mode == "sync":
        answer = core.run("Mark the expense report task as done").final_answer
    else:
        answer = _stream_answer(core.run_stream("Mark the expense report task as done"))
    assert done == []
    assert answer == UNBACKED_CLAIM_ANSWER


@pytest.mark.parametrize("mode", ["sync", "stream"])
def test_an_honest_first_answer_is_left_alone(mode: str) -> None:
    script = _Script("Thought: nothing to do\nFinal Answer: There are no bills due this week.")
    core = _core(script, [])
    if mode == "sync":
        answer = core.run("What bills are due this week?").final_answer
    else:
        answer = _stream_answer(core.run_stream("What bills are due this week?"))
    assert answer == "There are no bills due this week."
    assert len(script.prompts) == 1  # no extra model call


def test_a_scaffold_echo_is_answered_from_the_tool_result_in_hand() -> None:
    echo = "Thought: done\nFinal Answer: Your steps so far on this request (every Observation"
    script = _Script(_CALL, echo, echo)
    trace = _core(script, []).run("Mark the expense report task as done")
    assert trace.final_answer == "✅ Done — Submit the expense report."


def test_a_bare_done_after_reads_closes_the_turn_but_not_after_nothing() -> None:
    assert (
        check_final_answer(
            "Done.", effects_executed=[], retries_used=0, markers=_MARKERS, any_tool_ran=True
        ).kind
        == "ok"
    )
    assert (
        check_final_answer(
            "Done.", effects_executed=[], retries_used=0, markers=_MARKERS, any_tool_ran=False
        ).kind
        == "retry"
    )


def test_a_write_replayed_by_a_resume_backs_the_claim() -> None:
    """A resumed run starts with no executed effects; the replayed write step counts."""
    from iris_harness.agent.agentic_core import ReactStep, ResumeSeed

    done: list[dict[str, Any]] = []
    script = _Script("Thought: it ran before the pause\nFinal Answer: I've marked it as done.")
    core = _core(script, done)
    ran = ReactStep(
        thought="call it",
        action="complete_task",
        action_input={"task": "expense report"},
        observation="✅ Done — Submit the expense report.",
    )
    seed = ResumeSeed(
        run_id="r1",
        query="Mark the expense report task as done",
        steps=[ran],
        history=["Action: complete_task\nObservation: ✅ Done — Submit the expense report."],
        start_iteration=1,
    )
    assert core.run_from_seed(seed).final_answer == "I've marked it as done."
    # A write the policy held does not count: the claim is turned back.
    held = ReactStep(
        thought="call it",
        action="complete_task",
        observation="Request needs approval by governance: tool_policy: not confirmed",
    )
    script2 = _Script(
        "Thought: t\nFinal Answer: I've marked it as done.",
        "Thought: t\nFinal Answer: I've marked it as done.",
    )
    seed2 = ResumeSeed(run_id="r2", query="q", steps=[held], history=["x"], start_iteration=1)
    assert _core(script2, []).run_from_seed(seed2).final_answer == UNBACKED_CLAIM_ANSWER


# --- a harness note is never the answer (2026-09-28 eval) ----------------------------


def _core_with_read(script: _Script, read_result: str = "No pending actions.") -> AgenticCore:
    def pending(_args: dict[str, Any]) -> str:
        return read_result

    read = ToolSpec(name="pending_actions", description="List pending actions.", call=pending)
    write = ToolSpec(
        name="add_task",
        description="Add a to-do.",
        call=lambda _a: "📝 Added.",
        effect="write",
        confirm="never",
    )
    return AgenticCore(AgenticCoreConfig(max_iterations=6), llm_call=script, tools=[read, write])


_READ = "Thought: look\nAction: pending_actions\nAction Input: {}"


def test_a_repeated_read_after_a_turn_back_answers_from_the_read_not_the_note() -> None:
    """The live leak: claim -> turned back -> the same read twice -> forced synthesis
    with no final answer -> "the last good observation", which was the guard's note."""
    claim = "Thought: t\nFinal Answer: I've added 'Renew passport' to your tasks."
    script = _Script(claim, _READ, _READ, "Thought: still thinking")
    answer = _core_with_read(script).run("Add a task to renew my passport").final_answer
    assert answer == "No pending actions."
    # When the read gave nothing usable, the note was the only "good" observation left.
    script = _Script(claim, _READ, _READ, "Thought: still thinking")
    core = _core_with_read(script, read_result="Error: pending actions unavailable")
    answer = core.run("Add a task to renew my passport").final_answer
    assert not answer.startswith(UNBACKED_CLAIM_NOTE[:40])


def test_a_claim_written_by_the_forced_synthesis_is_replaced() -> None:
    claim = "Thought: t\nFinal Answer: I've added 'Renew passport' to your tasks."
    script = _Script(claim, _READ, _READ, claim)
    answer = _core_with_read(script).run("Add a task to renew my passport").final_answer
    assert answer == UNBACKED_CLAIM_ANSWER


def test_no_harness_note_is_a_usable_observation() -> None:
    from iris_harness.agent.agentic_core import (
        _ASK_TURNED_BACK,
        ReactStep,
        _last_good_observation,
    )

    steps = [
        ReactStep(action="pending_actions", observation="No pending actions."),
        ReactStep(final_answer=None, observation=UNBACKED_CLAIM_NOTE + ' This request: "x".'),
        ReactStep(observation=SCAFFOLD_ECHO_NOTE),
        ReactStep(observation=f"{_ASK_TURNED_BACK} Call trash_email now."),
    ]
    assert _last_good_observation(steps) == "No pending actions."
