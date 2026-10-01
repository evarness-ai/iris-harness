"""ADR-0106 ``choice`` — a numbered list the user picks from, answered by the next reply.

The live bug: ``read_email`` showed "Which one should I read? 1. ... 2. ..." and asked
for a number. The reply "1" was rebuilt from the previous user turn by regex, which
dropped the number and re-ran the search; "just go with the 1 st one" was not
recognised at all, and the model searched the inbox for "1". Nothing had recorded
what was on screen, so nothing could resolve a pick against it.

A ``choice`` continuation records the options when they are shown. These tests pin
each leg: the store's shape rule, the pick parser, the classify claim, the single
task the plan stage hands the owner, the record stage leaving an explicit question
alone, and the next-reply-or-never rule on the intercept path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.agent.intent_router import IntentResult
from iris_harness.memory.retriever import MemoryContext
from iris_harness.memory.state.continuations import ContinuationStore
from iris_harness.runtime.continuations import (
    ContinuationRegistry,
    offered_choices,
    reads_as_choice,
)
from iris_harness.runtime.turn.stages import classify, plan, record
from iris_harness.runtime.turn.state import TurnRequest, TurnState
from iris_harness.runtime.turn_context import current_session_id, set_current_session_id
from iris_harness.runtime.types import ChatResult

_SHORTLIST = (
    "I found multiple emails matching 'Fall 2 Registration Links :)'. "
    "Which one should I read?\n"
    "1. 2026-09-14  Fall 2 Registration Links :)\n"
    "2. 2026-09-14  Fall 2 Registration Links :)\n"
    "Reply with the number (e.g. 1), or refine with from/subject."
)
_CHOICES = [
    {"message_id": "m-first", "subject": "Fall 2 Registration Links :)"},
    {"message_id": "m-second", "subject": "Fall 2 Registration Links :)"},
]


@pytest.fixture()
def registry(tmp_path: Path) -> ContinuationRegistry:
    return ContinuationRegistry(store=ContinuationStore(db_path=tmp_path / "checkpoints.db"))


def _offer(registry: ContinuationRegistry, session_id: str = "s1", **kw: object) -> None:
    registry.ask(
        session_id,
        "email",
        kind="choice",
        question=_SHORTLIST,
        intent="communication",
        payload={"choices": _CHOICES},
        **kw,  # type: ignore[arg-type]
    )


# ── the store: a choice carries options, never an action ──────────────────────


def test_a_choice_round_trips_its_options_in_order(registry: ContinuationRegistry) -> None:
    _offer(registry)
    pending = registry.pending("s1")
    assert pending is not None
    assert pending.kind == "choice"
    assert [c["message_id"] for c in pending.choices] == ["m-first", "m-second"]
    assert pending.is_executable is False  # nothing runs a pick; no approve button


def test_a_choice_without_options_is_refused(registry: ContinuationRegistry) -> None:
    with pytest.raises(ValueError, match="choices"):
        registry.ask("s1", "email", kind="choice", payload={"choices": []})
    with pytest.raises(ValueError, match="choices"):
        registry.ask("s1", "email", kind="choice")


def test_a_choice_refuses_an_executor(registry: ContinuationRegistry) -> None:
    with pytest.raises(ValueError, match="executor_kind"):
        registry.ask(
            "s1", "email", kind="choice", executor_kind="calendar_event", payload={"choices": [{}]}
        )


def test_an_approval_payload_still_needs_its_executor(registry: ContinuationRegistry) -> None:
    """The narrowing is to ``choice`` only — an action payload with no executor is
    still the half-built confirmation the payload seam refuses."""
    with pytest.raises(ValueError, match="executor_kind"):
        registry.ask("s1", "confirmation", kind="approval", payload={"a": 1})


# ── reads_as_choice ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("message", "index"),
    [
        ("1", 0),
        ("just go with the 1 st one", 0),  # the live reply, space before "st" included
        ("the 1st one", 0),
        ("first one please", 0),
        ("I'll take the first one", 0),
        ("#2", 1),
        ("2.", 1),
        ("the second email", 1),
        ("read number 2", 1),
        ("last", 1),
    ],
)
def test_reads_a_pick(registry: ContinuationRegistry, message: str, index: int) -> None:
    _offer(registry)
    pending = registry.pending("s1")
    assert pending is not None
    assert reads_as_choice(pending, message) == index


@pytest.mark.parametrize(
    "message",
    [
        "3",  # not an option that was shown
        "0",
        "1 and 2",  # two picks is not one choice
        "show my top 2 holdings",  # a new question that happens to hold a number
        "what's on my calendar today?",
        "yes",
        "",
    ],
)
def test_does_not_read_a_non_pick(registry: ContinuationRegistry, message: str) -> None:
    _offer(registry)
    pending = registry.pending("s1")
    assert pending is not None
    assert reads_as_choice(pending, message) is None


def test_other_kinds_have_no_options_to_pick(registry: ContinuationRegistry) -> None:
    registry.ask("s1", "planner", question="proceed?", intent="planner")
    pending = registry.pending("s1")
    assert pending is not None
    assert reads_as_choice(pending, "1") is None


# ── classify: a pick routes home with its option; anything else withdraws it ──


class _Runtime:
    def __init__(self, registry: ContinuationRegistry) -> None:
        self.continuations = registry

    def _resolve_supported_agent(self, candidate: str, *, fallback: str) -> str:
        return candidate if candidate in {"email", "system"} else fallback


def _state(message: str, *, intent: str = "general") -> TurnState:
    state = TurnState(request=TurnRequest(message=message, session_id="s1"))
    state.intent_result = IntentResult(
        intent=intent, agent_type=intent, confidence=0.4, source="fallback"
    )
    state.memory_ctx = MemoryContext()
    return state


def test_a_pick_routes_to_the_owner_with_the_option_it_named(
    registry: ContinuationRegistry,
) -> None:
    _offer(registry)
    state = _state("just go with the 1 st one")

    classify._honour_continuation(_Runtime(registry), state)

    assert state.continuation is not None
    assert state.continuation_choice == _CHOICES[0]
    assert state.intent_result is not None
    assert state.intent_result.agent_type == "email"  # not wherever "1" classified
    assert state.intent_result.intent == "communication"
    assert state.intent_result.source == "continuation"
    assert registry.pending("s1") is None  # answered


def test_a_reply_that_does_not_pick_withdraws_the_choice(
    registry: ContinuationRegistry,
) -> None:
    """A stale shortlist must not claim a bare number typed later in reply to
    something else, so the next reply either picks or ends it."""
    _offer(registry)
    state = _state("what's the weather?")

    classify._honour_continuation(_Runtime(registry), state)

    assert state.continuation is None
    assert state.continuation_choice is None
    assert registry.pending("s1") is None
    assert registry.history("s1")[-1].status == "superseded"  # kept on the record


# ── plan: one task, to the owner, carrying the option ─────────────────────────


def test_a_pick_is_one_task_for_the_owner_not_a_plan(registry: ContinuationRegistry) -> None:
    _offer(registry)
    runtime = _Runtime(registry)
    state = _state("1")
    classify._honour_continuation(runtime, state)

    task = plan._resume_task(runtime, state)  # type: ignore[arg-type]

    assert task is not None
    assert task.agent_type == "email"
    assert task.selected_choice == _CHOICES[0]
    assert task.resume_run_id is None  # a pick resumes no run
    assert task.session_id == "s1"


def test_an_ordinary_turn_still_plans(registry: ContinuationRegistry) -> None:
    state = _state("what's in my inbox?")
    assert plan._resume_task(_Runtime(registry), state) is None  # type: ignore[arg-type]


# ── record: an inferred offer never overwrites the shortlist this turn opened ─


class _MissionsOff:
    def mission_autocreate_enabled(self) -> bool:
        return False


class _RecordRuntime(_Runtime):
    mission_proposals = _MissionsOff()


def _answered(state: TurnState, response: str) -> TurnState:
    state.result = ChatResult(
        response=response,
        intent="communication",
        agent_type="email",
        has_errors=False,
        sources=[],
        error_summary="",
        metadata={},
    )
    return state


def test_the_shortlist_survives_its_own_should_i_ending(registry: ContinuationRegistry) -> None:
    """ "Which one should I read?" reads as an offer to proceed. Recording that
    inference would supersede the options the next reply is about to pick from."""
    state = _state("I need the one with subject Fall 2", intent="communication")
    _offer(registry)  # the tool, during this turn
    _answered(state, "Which one should I read?")

    list(record.run(_RecordRuntime(registry), state))  # type: ignore[arg-type]

    pending = registry.pending("s1")
    assert pending is not None
    assert pending.kind == "choice"


def test_an_older_choice_does_not_block_a_new_offer(registry: ContinuationRegistry) -> None:
    earlier = datetime.now(UTC) - timedelta(minutes=5)
    _offer(registry, now=earlier)
    state = _answered(_state("draft it"), "Shall I send it?")

    list(record.run(_RecordRuntime(registry), state))  # type: ignore[arg-type]

    pending = registry.pending("s1")
    assert pending is not None
    assert pending.kind == "approval"


# ── the intercept path keeps the next-reply rule ──────────────────────────────


def test_a_stale_choice_is_dropped_but_a_fresh_one_is_kept(
    registry: ContinuationRegistry,
) -> None:
    turn_start = datetime.now(UTC)
    _offer(registry, now=turn_start - timedelta(minutes=1))
    assert registry.drop_stale_inferred("s1", turn_start) is True
    assert registry.pending("s1") is None

    _offer(registry, now=turn_start + timedelta(seconds=1))
    assert registry.drop_stale_inferred("s1", turn_start) is False
    assert registry.pending("s1") is not None


def test_drop_stale_inferred_leaves_other_kinds_alone(registry: ContinuationRegistry) -> None:
    turn_start = datetime.now(UTC)
    registry.ask("s1", "planner", question="proceed?", now=turn_start - timedelta(minutes=1))
    assert registry.drop_stale_inferred("s1", turn_start) is False
    assert registry.pending("s1") is not None


# ── the session seam a plugin tool reads ──────────────────────────────────────


def test_the_turn_session_is_readable_from_a_tool() -> None:
    set_current_session_id("web-abc")
    try:
        assert current_session_id() == "web-abc"
    finally:
        set_current_session_id("")
    assert current_session_id() == ""


# ── a list the agent's own answer offers (2026-09-16) ─────────────────────────
#
# "Any breaking news today" was answered with five numbered headlines ending "Want me
# to open any of these ...?". Only a tool could open a choice, so the record stage
# inferred a yes/no approval, which cannot read "4". The model guessed item 1, the
# goal-drift guard halted a run whose task was "4", and "Just tell me more about the
# 4th news from your news list" was routed to the inbox. Headlines here are synthetic.

_NEWS_ANSWER = (
    "Today (2026-09-16) - top breaking items I found:\n"
    "1. Harbor bridge reopens after overnight repairs (Example Wire).\n"
    "2. City council approves the new transit budget (Example Times).\n"
    "3. Storm warning issued for the northern coast (Example Weather).\n"
    "4. Regional airline adds three routes (Example Business).\n"
    "5. Museum returns a borrowed painting (Example Arts).\n"
    "\n"
    "Want me to open any of these and pull the full article or a short explainer?"
)
_PLAN_ANSWER = (
    "Here is the plan:\n"
    "1. Search recent coverage of the topic.\n"
    "2. Select the three most cited sources.\n"
    "3. Summarise what they agree on.\n"
    "\n"
    "Would you like me to proceed with this plan?"
)


def test_an_answer_offering_a_pick_from_its_list_reads_as_choices() -> None:
    options = offered_choices(_NEWS_ANSWER)
    assert len(options) == 5
    assert options[3] == "Regional airline adds three routes (Example Business)."


@pytest.mark.parametrize(
    "answer",
    [
        _PLAN_ANSWER,  # the ADR-0106 incident's shape: steps to approve, not a menu
        "1. Only one headline today.\n\nWant me to open any of these?",
        "Want me to open any of these?",  # nothing listed
        _NEWS_ANSWER.rsplit("\n", 1)[0],  # a list with no offer
        "1. a\n3. b\n\nWhich one should I open?",  # numbering that is not a list
    ],
)
def test_no_choices_without_a_pick_offer_over_a_list(answer: str) -> None:
    assert offered_choices(answer) == ()


def _news_turn(registry: ContinuationRegistry) -> None:
    state = _state("Any breaking news today", intent="search")
    state.result = ChatResult(
        response=_NEWS_ANSWER,
        intent="search",
        agent_type="system",
        has_errors=False,
        sources=[],
        error_summary="",
        metadata={},
    )
    list(record.run(_RecordRuntime(registry), state))  # type: ignore[arg-type]


def test_the_record_stage_keeps_the_list_the_answer_offered(
    registry: ContinuationRegistry,
) -> None:
    _news_turn(registry)

    pending = registry.pending("s1")
    assert pending is not None and pending.kind == "choice"
    assert pending.owner == "system" and pending.intent == "search"
    assert [c["position"] for c in pending.choices] == [1, 2, 3, 4, 5]
    assert pending.question.startswith("Want me to open any of these")


def test_a_plan_offer_is_still_an_approval(registry: ContinuationRegistry) -> None:
    state = _answered(_state("research this"), _PLAN_ANSWER)
    list(record.run(_RecordRuntime(registry), state))  # type: ignore[arg-type]

    pending = registry.pending("s1")
    assert pending is not None and pending.kind == "approval"


@pytest.mark.parametrize(
    "reply",
    ["4", "the 4th one", "Just tell me more about the 4th news from your news list"],
)
def test_a_pick_from_the_answer_reaches_its_owner_as_a_real_task(
    registry: ContinuationRegistry, reply: str
) -> None:
    _news_turn(registry)
    runtime = _Runtime(registry)
    state = _state(reply, intent="communication")  # where the live rephrase was routed

    classify._honour_continuation(runtime, state)
    task = plan._resume_task(runtime, state)  # type: ignore[arg-type]

    assert task is not None
    assert task.agent_type == "system" and state.intent_result.intent == "search"  # type: ignore[union-attr]
    assert "Regional airline adds three routes" in task.query  # item 4, not item 1
    assert "Want me to open any of these" in task.query
    assert task.query.endswith(f"The user replied: {reply}")


@pytest.mark.parametrize(
    "reply",
    [
        "more on 3",
        "details on 3",
        "give me more details on 3",
        "expand on 3",
        "elaborate on 3",
        "summarize 3",
        "summarise 3",
        "what about 3",
        "let's do 3",
        "dig into 3",
        "can you open 3",
        "i want to know more about 3",
        "the 3rd headline",
        "give me a summary of the third story",
    ],
)
def test_the_same_pick_said_the_other_ways(registry: ContinuationRegistry, reply: str) -> None:
    """Every one of these is the user picking item 3 off the list the answer showed.

    A miss is not benign: the reply stops reading as a pick, the choice is withdrawn and
    the message is planned as a fresh question — the 2026-09-16 failure, where a pick off
    a news list was routed away from the list it was about.
    """
    _news_turn(registry)
    pending = registry.pending("s1")
    assert pending is not None
    assert reads_as_choice(pending, reply) == 2


@pytest.mark.parametrize(
    "reply",
    [
        "show me 5 more",
        "tell me more about those two",
        # A plural noun after the number is a quantity, not a pick — which is why the
        # filler set carries only the singular a picked item goes by.
        "give me 3 headlines",
        "send 3 emails to bob",
        "show my top 2 holdings",
    ],
)
def test_more_of_the_list_is_not_a_pick(registry: ContinuationRegistry, reply: str) -> None:
    _news_turn(registry)
    pending = registry.pending("s1")
    assert pending is not None
    assert reads_as_choice(pending, reply) is None


def test_a_tool_shortlist_reaches_its_owner_with_the_reply_untouched(
    registry: ContinuationRegistry,
) -> None:
    """Options a tool recorded carry ids its owner reads directly; the query is not
    rewritten for them."""
    _offer(registry)
    runtime = _Runtime(registry)
    state = _state("1")
    classify._honour_continuation(runtime, state)

    task = plan._resume_task(runtime, state)  # type: ignore[arg-type]

    assert task is not None and task.query == "1"


# ── an answer that ends on a question is owned, whatever its phrasing ─────────
#
# The 2026-09-16 news thread. Opening a continuation used to require an offer phrase, so
# these three closes opened nothing, every follow-up was classified from scratch, and
# "Key quotes" reached the inbox. A natural language has unbounded ways to ask a question
# and one way to punctuate it.

_LIVE_NEWS_ANSWER = """- Gaza - a war-damaged building collapsed; Reuters reports 12 killed.
- Ukraine - drone strikes hit a bus and a train, killing five.
- US legal action - five charged in alleged Russian-backed murder plots.
- DOJ allegation - DOJ accuses Russian intelligence agents of plotting a murder.

Do you want full articles, more sources, or a short briefing on any one story?"""


@pytest.mark.parametrize(
    "close",
    [
        "Do you want full articles, more sources, or a short briefing on any one story?",
        "Which item do you want more sources for?",
        "Reply with answers 1-5.?",
        "Tell me which and I'll fetch them?",
        "How many sources do you want?",
    ],
)
def test_any_closing_question_opens_a_continuation(
    registry: ContinuationRegistry, close: str
) -> None:
    state = _answered(_state("what's the breaking news today?"), close)
    list(record.run(_RecordRuntime(registry), state))  # type: ignore[arg-type]

    pending = registry.pending("s1")
    assert pending is not None, f"no continuation opened for: {close}"
    assert pending.kind == "question"


@pytest.mark.parametrize(
    "close",
    [
        "Let me know if you need anything else.",
        "That's the summary.",
        "I've filed the report and archived the thread.",
    ],
)
def test_an_answer_that_asks_nothing_opens_nothing(
    registry: ContinuationRegistry, close: str
) -> None:
    """Signing off is a pleasantry, not a question. No question mark, no continuation."""
    state = _answered(_state("summarise this"), close)
    list(record.run(_RecordRuntime(registry), state))  # type: ignore[arg-type]
    assert registry.pending("s1") is None


def test_a_yes_no_offer_is_still_an_approval(registry: ContinuationRegistry) -> None:
    """The offer phrasing still decides the KIND, it just no longer decides whether a
    continuation opens at all."""
    state = _answered(_state("research this"), "Would you like me to proceed with this plan?")
    list(record.run(_RecordRuntime(registry), state))  # type: ignore[arg-type]

    pending = registry.pending("s1")
    assert pending is not None and pending.kind == "approval"


def test_the_free_text_reply_reaches_the_owner_with_the_question(
    registry: ContinuationRegistry,
) -> None:
    """The whole point: "Key quotes" routes to the owner that offered them, carrying the
    question. Before this it was classified from scratch and ran an inbox search."""
    opening = _answered(_state("what's the breaking news today?"), _LIVE_NEWS_ANSWER)
    list(record.run(_RecordRuntime(registry), opening))  # type: ignore[arg-type]

    runtime = _Runtime(registry)
    reply = _state("Key quotes", intent="communication")  # where the live turn was routed
    classify._honour_continuation(runtime, reply)

    assert reply.intent_result is not None
    assert reply.intent_result.agent_type == "email"  # the owner of the answer above
    assert reply.intent_result.source == "continuation"

    task = plan._resume_task(runtime, reply)  # type: ignore[arg-type]
    assert task is not None
    assert "Do you want full articles" in task.query  # the subject the reply selects in
    assert task.query.endswith("The user replied: Key quotes")
    # No run to resume — this is a fresh task, not a checkpoint restart.
    assert task.resume_run_id is None


def test_an_inferred_question_claims_one_turn_and_no_more(
    registry: ContinuationRegistry,
) -> None:
    """A question that shielded the router indefinitely would be the worse bug."""
    opening = _answered(_state("what's the breaking news today?"), _LIVE_NEWS_ANSWER)
    list(record.run(_RecordRuntime(registry), opening))  # type: ignore[arg-type]

    runtime = _Runtime(registry)
    classify._honour_continuation(runtime, _state("Key quotes", intent="communication"))
    assert registry.pending("s1") is None  # closed by the reply that claimed it

    # The turn after that is classified normally again.
    later = _state("check my inbox", intent="communication")
    classify._honour_continuation(runtime, later)
    assert later.intent_result is not None
    assert later.intent_result.source == "fallback"


def test_an_intercepted_turn_withdraws_an_inferred_question(
    registry: ContinuationRegistry,
) -> None:
    """The one-turn rule holds on the intercept path too. Without this an inferred
    question survives the turn an intercept answered and claims the turn after it."""
    turn_start = datetime.now(UTC)
    registry.ask(
        "s1",
        "system",
        kind="question",
        question="Do you want full articles, more sources, or a short briefing?",
        intent="search",
        now=turn_start - timedelta(minutes=1),
    )
    assert registry.drop_stale_inferred("s1", turn_start) is True
    assert registry.pending("s1") is None


def test_a_resumable_question_survives_an_intercepted_turn(
    registry: ContinuationRegistry,
) -> None:
    """Tier B owns a halted run, not a guess. It is not the inference this drops."""
    turn_start = datetime.now(UTC)
    registry.ask(
        "s1",
        "system",
        kind="question",
        question="which database did you mean?",
        run_id="r-1",
        step_id=3,
        now=turn_start - timedelta(minutes=1),
    )
    assert registry.drop_stale_inferred("s1", turn_start) is False
    assert registry.pending("s1") is not None
