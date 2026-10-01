"""The card-is-the-selection rule through the real loop, sync and streaming (2026-09-22).

The real loop, kernel hooks, approval queue and checkpoint store; only the model and
the tools are scripted. Every case runs on both ``run`` and ``run_stream``, because the
two loop bodies are separate and have drifted before (ADR-0107).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agentic_core import (
    AgenticCore,
    AgenticCoreConfig,
    ToolSpec,
    resume_seed_from_checkpoint,
)
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.approvals import ApprovalQueue
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.plugins import DestructiveApprovalHook, ToolPolicyHook
from iris_harness.memory.state import CheckpointStore
from iris_harness.memory.state.store import CheckpointNotFoundError

_FOUND = "1. Weekly deals [id a1]\n2. Last chance sale [id b2]"
_SEARCH = 'Thought: look\nAction: search_items\nAction Input: {"query": "deals"}'
_GUESS = 'Thought: remove\nAction: remove_items\nAction Input: {"ids": ["zz9"]}'
_REMOVE = 'Thought: remove\nAction: remove_items\nAction Input: {"ids": ["a1", "b2"]}'
_ASK = (
    'Thought: ask\nAction: ask_user\nAction Input: {"question": "Which of these should I remove?"}'
)
_HOLD = 'Thought: note\nAction: add_note\nAction Input: {"text": "cleanup"}'
_DONE = "Thought: done\nFinal Answer: Done."


class _AllowHook:
    priority: int = 10

    def __init__(self, name: str, hook_point: HookPoint) -> None:
        self.name = name
        self.hook_point = hook_point

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="test")


class _LLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else _DONE


class _World:
    def __init__(self, tmp_path: Path, *, found: str = _FOUND) -> None:
        self.queue = ApprovalQueue(db_path=tmp_path / "approvals.db")
        self.store = CheckpointStore(db_path=tmp_path / "checkpoints.db")
        self.kernel = GovernanceKernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
        self.kernel.register(_AllowHook("allow_classify", HookPoint.PRE_CLASSIFY))
        self.kernel.register(_AllowHook("allow_llm", HookPoint.PRE_LLM_CALL))
        self.kernel.register(ToolPolicyHook())
        self.kernel.register(DestructiveApprovalHook(approval_queue=self.queue))
        self.kernel.init_lock()
        self.removed: list[list[str]] = []
        self.notes: list[str] = []
        self.found = found

    def tools(self, *, destructive: bool = True) -> list[ToolSpec]:
        def _remove(args: dict[str, Any]) -> str:
            self.removed.append(list(args["ids"]))
            return "removed"

        def _validate(args: dict[str, Any]) -> str | None:
            unknown = [i for i in args.get("ids", []) if i not in ("a1", "b2")]
            return f"No item with id {unknown[0]!r}." if unknown else None

        def _note(args: dict[str, Any]) -> str:
            self.notes.append(str(args.get("text")))
            return "noted"

        tools = [
            ToolSpec(
                name="search_items",
                description="read",
                call=lambda _a: self.found,
                effect="read",
            ),
            ToolSpec(
                name="add_note", description="write", call=_note, effect="write", confirm="once"
            ),
        ]
        if destructive:
            tools.append(
                ToolSpec(
                    name="remove_items",
                    description="destructive",
                    call=_remove,
                    effect="destructive",
                    confirm="approval",
                    validate=_validate,
                )
            )
        return tools

    def _read(self, approval_id: str) -> tuple[str, list[tuple[str, dict[str, Any]]]] | None:
        row = self.queue.get(approval_id)
        if row is None:
            return None
        return row.status, [(i.tool, i.args) for i in (row.items or ())]

    def core(self, llm: _LLM, *, destructive: bool = True) -> AgenticCore:
        return AgenticCore(
            config=AgenticCoreConfig(max_iterations=8, allow_ask_user=True),
            llm_call=llm,
            tools=self.tools(destructive=destructive),
            kernel=self.kernel,
            checkpoint_store=self.store,
            session_id="web-s1",
            origin_channel="web",
            link_approval_checkpoint=self.queue.set_checkpoint,
            read_approval=self._read,
            agent_type="chat",
        )


def _drive(core: AgenticCore, mode: str, query: str = "", seed: Any = None) -> dict[str, Any]:
    """One turn on either loop, as {final, halted, pending, steps, success}."""
    if mode == "sync":
        trace = core.run_from_seed(seed) if seed is not None else core.run(query)
        return {
            "final": trace.final_answer,
            "halted": trace.halted_by,
            "pending": trace.pending_approval_id,
            "steps": [(s.action, s.observation) for s in trace.steps],
            "success": trace.success,
            "run_id": trace.run_id,
        }
    chunks = list(core.run_stream(query, resume=seed))
    meta = [c for c in chunks if isinstance(c, dict)][-1]
    texts = [c for c in chunks if isinstance(c, str)]
    halted = {"awaiting_user_input": "ask_user", "awaiting_approval": "approval"}
    return {
        "final": texts[-1],
        "halted": halted.get(str(meta["reason"])),
        "pending": meta.get("pending_approval_id"),
        "steps": [(s["action"], s["observation"]) for s in meta["trace"]],  # type: ignore[index]
        "success": meta["success"],
        "run_id": meta["run_id"],
        "reason": meta["reason"],
    }


MODES = pytest.mark.parametrize("mode", ["sync", "stream"])


@MODES
def test_a_refused_guess_does_not_switch_the_turn_back_off(tmp_path: Path, mode: str) -> None:
    world = _World(tmp_path)
    llm = _LLM([_SEARCH, _GUESS, _ASK, _REMOVE])

    out = _drive(world.core(llm), mode, "remove the deals")

    assert out["halted"] == "approval" and out["pending"]  # the card, not the question
    assert (out["steps"][1][1] or "").startswith("Error: No item with id 'zz9'.")
    assert (out["steps"][2][1] or "").startswith("Not asked yet.")
    assert "Call remove_items now" in llm.prompts[3]
    assert world.removed == []


@MODES
def test_two_turn_backs_then_a_factual_ending(tmp_path: Path, mode: str) -> None:
    world = _World(tmp_path)
    llm = _LLM([_GUESS, _SEARCH, _ASK, _ASK, _ASK])

    out = _drive(world.core(llm), mode, "remove the deals")

    assert out["final"] == (
        "I couldn't prepare the remove_items request, so nothing was changed.\n\n"
        f"What I found:\n{_FOUND}"
    )
    assert out["halted"] is None and out["pending"] is None
    assert out["success"] is True
    assert [a for a, _ in out["steps"]] == [
        "remove_items",
        "search_items",
        "ask_user",
        "ask_user",
        "ask_user",
    ]
    assert world.queue.list_pending() == []
    with pytest.raises(CheckpointNotFoundError):  # nothing to resume: nobody was asked
        world.store.get_latest(out["run_id"])
    assert len(llm.prompts) == 5
    if mode == "stream":
        assert out["reason"] == "no_approval_card"


@MODES
def test_the_ending_names_the_tool_that_was_tried(tmp_path: Path, mode: str) -> None:
    world = _World(tmp_path, found="Error: search failed")
    out = _drive(world.core(_LLM([_GUESS, _ASK, _ASK, _ASK])), mode, "remove them")
    assert out["final"] == "I couldn't prepare the remove_items request, so nothing was changed."


@MODES
def test_once_the_card_was_shown_an_ask_reaches_the_owner_with_the_list(
    tmp_path: Path, mode: str
) -> None:
    world = _World(tmp_path)
    halted = _drive(world.core(_LLM([_SEARCH, _REMOVE])), mode, "remove the deals")
    assert halted["pending"]
    world.queue.respond(halted["pending"], status="rejected", actor="owner")
    seed = resume_seed_from_checkpoint(world.store.get_latest(halted["run_id"]))

    out = _drive(world.core(_LLM([_ASK])), mode, seed=seed)

    assert out["halted"] == "ask_user"
    assert out["final"] == f"Which of these should I remove?\n\n{_FOUND}"
    assert world.removed == []


@MODES
def test_a_question_after_a_read_carries_the_items(tmp_path: Path, mode: str) -> None:
    world = _World(tmp_path)
    out = _drive(world.core(_LLM([_SEARCH, _ASK]), destructive=False), mode, "tidy up")

    assert out["halted"] == "ask_user"
    assert out["final"] == f"Which of these should I remove?\n\n{_FOUND}"
    # The model's own record keeps the bare question; it already has the list.
    assert out["steps"][-1][1] == "Asked the user: Which of these should I remove?"


@MODES
def test_confirm_once_is_unaffected(tmp_path: Path, mode: str) -> None:
    """The ask a held write told the model to make still reaches the owner, and a
    turned-back ask before it is never taken for consent."""
    world = _World(tmp_path)
    confirm = _ASK.replace("Which of these should I remove?", "Add the note 'cleanup'?")
    out = _drive(world.core(_LLM([_GUESS, _ASK, _HOLD, confirm])), mode, "note it")

    assert (out["steps"][1][1] or "").startswith("Not asked yet.")
    assert "needs approval" in (out["steps"][2][1] or "")  # held: nobody was asked yet
    assert out["halted"] == "ask_user"
    assert out["final"].startswith("Add the note 'cleanup'?")
    assert world.notes == []

    seed = resume_seed_from_checkpoint(
        world.store.get_latest(out["run_id"]), user_reply="yes, add it"
    )
    _drive(world.core(_LLM([_HOLD])), mode, seed=seed)
    assert world.notes == ["cleanup"]


# --- a final answer that ends on a question is an ask (owner, 2026-09-22) -----------
#
# Probe run 2: after a refused guess qwen2.5 skipped ask_user and answered in plain text
# ending on a question, which the loop took as the answer.

_PLAIN_Q = "Could you give me the ids of the emails you want removed?"
_FINAL_Q = "Thought: ask\nFinal Answer: Which of these should I remove?"


@MODES
def test_a_plain_text_question_is_turned_back_and_the_card_follows(
    tmp_path: Path, mode: str
) -> None:
    world = _World(tmp_path)
    llm = _LLM([_GUESS, _PLAIN_Q, _SEARCH, _REMOVE])

    out = _drive(world.core(llm), mode, "remove the deals")

    assert out["halted"] == "approval" and out["pending"]
    assert out["steps"][1][0] is None
    assert (out["steps"][1][1] or "").startswith("Not asked yet.")
    assert f"Final Answer: {_PLAIN_Q}\nObservation: Not asked yet." in llm.prompts[2]


@MODES
def test_question_answers_and_asks_share_the_budget_then_end(tmp_path: Path, mode: str) -> None:
    world = _World(tmp_path)
    llm = _LLM([_GUESS, _SEARCH, _FINAL_Q, _ASK, _PLAIN_Q, _REMOVE])

    out = _drive(world.core(llm), mode, "remove the deals")

    assert out["final"] == (
        "I couldn't prepare the remove_items request, so nothing was changed.\n\n"
        f"What I found:\n{_FOUND}"
    )
    assert out["success"] is True and out["pending"] is None
    assert len(llm.prompts) == 5  # the third ask ended the turn; _REMOVE never ran
    assert world.queue.list_pending() == []


@MODES
def test_a_turned_back_final_answer_is_not_left_as_a_final_answer(
    tmp_path: Path, mode: str
) -> None:
    world = _World(tmp_path)
    script = [_GUESS, _SEARCH, _FINAL_Q, _REMOVE]
    out = _drive(world.core(_LLM(list(script))), mode, "remove the deals")
    core_steps = out["steps"]
    assert out["pending"]
    assert core_steps[2][0] is None and (core_steps[2][1] or "").startswith("Not asked yet.")
    if mode == "sync":
        trace = world.core(_LLM(list(script))).run("remove the deals")
        held = trace.steps[2]
        assert (held.is_terminal, held.final_answer) == (False, None)


@MODES
def test_without_a_card_tool_a_question_answer_passes_untouched(tmp_path: Path, mode: str) -> None:
    world = _World(tmp_path)
    for reply in (_FINAL_Q, _PLAIN_Q):
        out = _drive(world.core(_LLM([_SEARCH, reply]), destructive=False), mode, "tidy up")
        assert out["final"] == reply.removeprefix("Thought: ask\nFinal Answer: ")
        assert out["success"] is True


@MODES
def test_once_the_card_was_shown_a_question_answer_passes(tmp_path: Path, mode: str) -> None:
    world = _World(tmp_path)
    halted = _drive(world.core(_LLM([_SEARCH, _REMOVE])), mode, "remove the deals")
    world.queue.respond(halted["pending"], status="rejected", actor="owner")
    seed = resume_seed_from_checkpoint(world.store.get_latest(halted["run_id"]))

    out = _drive(world.core(_LLM([_FINAL_Q])), mode, seed=seed)

    assert out["final"] == "Which of these should I remove?"


@MODES
def test_a_held_writes_confirmation_in_a_final_answer_passes(tmp_path: Path, mode: str) -> None:
    """Confirm-once: the question a held write asked for is not the card's business."""
    world = _World(tmp_path)
    out = _drive(
        world.core(_LLM([_HOLD, "Final Answer: Add the note 'cleanup'?"])), mode, "note it"
    )
    assert out["final"] == "Add the note 'cleanup'?"
    assert world.notes == []


# --- the guard needs an ATTEMPT at the card tool, not just one on the menu ----------
#
# Owner, 2026-09-22: a read-only turn ("read my email from John") has a legitimate
# "which one?", and the card tool sitting unused on the menu must not take it away.


@MODES
def test_an_unattempted_card_tool_leaves_the_question_alone(tmp_path: Path, mode: str) -> None:
    world = _World(tmp_path)
    out = _drive(world.core(_LLM([_SEARCH, _ASK])), mode, "which of my deals mails are these")

    assert out["halted"] == "ask_user"
    assert out["final"] == f"Which of these should I remove?\n\n{_FOUND}"
    assert world.removed == [] and world.queue.list_pending() == []


@MODES
def test_an_unattempted_card_tool_leaves_a_question_answer_alone(tmp_path: Path, mode: str) -> None:
    world = _World(tmp_path)
    out = _drive(world.core(_LLM([_SEARCH, _FINAL_Q])), mode, "which of these are deals")

    assert out["final"] == "Which of these should I remove?"
    assert out["success"] is True and out["pending"] is None


@MODES
def test_an_attempt_that_was_refused_turns_the_next_ask_back(tmp_path: Path, mode: str) -> None:
    world = _World(tmp_path)
    out = _drive(world.core(_LLM([_GUESS, _SEARCH, _ASK, _REMOVE])), mode, "remove the deals")

    assert out["halted"] == "approval" and out["pending"]
    assert (out["steps"][2][1] or "").startswith("Not asked yet.")
