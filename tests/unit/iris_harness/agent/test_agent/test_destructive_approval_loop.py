"""A destructive call waits for the owner, then runs exactly what they approved (ADR-0118).

End to end through the real loop, the real kernel hooks, a real approval queue and a
real checkpoint store — only the model and the tool are scripted. The halt raises one
itemised approval and checkpoints the run; the owner's answer is recorded on the row;
the resumed run settles it before the model says another word.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iris_harness.agent.agentic_core import (
    AgenticCore,
    AgenticCoreConfig,
    ToolSpec,
    resume_seed_from_checkpoint,
)
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.approvals import ApprovalItem, ApprovalQueue
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.plugins import DestructiveApprovalHook, ToolPolicyHook
from iris_harness.kernel.governance.side_effects import SideEffectLedger
from iris_harness.kernel.governance.wiring import register_side_effect_ledger
from iris_harness.memory.state import CheckpointStore

_TRASH = 'Thought: trash them\nAction: trash_email\nAction Input: {"ids": ["m1", "m2"]}'
_TRASH_MORE = 'Thought: and this one\nAction: trash_email\nAction Input: {"ids": ["m3"]}'
_DONE = "Thought: done\nFinal Answer: Done."


class _AllowHook:
    priority: int = 10

    def __init__(self, name: str, hook_point: HookPoint) -> None:
        self.name = name
        self.hook_point = hook_point

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="test")


class _ScriptedLLM:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else _DONE


class _Mailbox:
    def __init__(self) -> None:
        self.trashed: list[list[str]] = []

    def tool(self) -> ToolSpec:
        def _trash(args: dict[str, Any]) -> str:
            self.trashed.append(list(args["ids"]))
            return f"trashed {len(args['ids'])}"

        return ToolSpec(
            name="trash_email",
            description="Move emails to the trash.",
            call=_trash,
            effect="destructive",
            confirm="approval",
        )


class _World:
    """One queue, one checkpoint store and one kernel, shared by the halt and the resume."""

    def __init__(self, tmp_path: Path) -> None:
        self.queue = ApprovalQueue(db_path=tmp_path / "approvals.db")
        self.store = CheckpointStore(db_path=tmp_path / "checkpoints.db")
        self.kernel = GovernanceKernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
        self.kernel.register(_AllowHook("allow_classify", HookPoint.PRE_CLASSIFY))
        self.kernel.register(_AllowHook("allow_llm", HookPoint.PRE_LLM_CALL))
        self.kernel.register(ToolPolicyHook())
        self.kernel.register(DestructiveApprovalHook(approval_queue=self.queue))
        # A destructive call runs only once its row is in the side-effect ledger (#73).
        self.ledger = SideEffectLedger(tmp_path / "side_effects.db")
        register_side_effect_ledger(self.kernel, self.ledger)
        self.kernel.init_lock()
        self.mailbox = _Mailbox()

    def _read(self, approval_id: str) -> tuple[str, list[tuple[str, dict[str, Any]]]] | None:
        row = self.queue.get(approval_id)
        if row is None:
            return None
        return row.status, [(i.tool, i.args) for i in (row.items or ())]

    def core(self, responses: list[str], *, resumable: bool = True) -> AgenticCore:
        return AgenticCore(
            config=AgenticCoreConfig(max_iterations=6),
            llm_call=_ScriptedLLM(responses),
            tools=[self.mailbox.tool()],
            kernel=self.kernel,
            checkpoint_store=self.store if resumable else None,
            session_id="web-s1",
            origin_channel="web",
            link_approval_checkpoint=self.queue.set_checkpoint if resumable else None,
            read_approval=self._read if resumable else None,
            agent_type="chat",
        )

    def resume_seed(self, run_id: str) -> Any:
        return resume_seed_from_checkpoint(self.store.get_latest(run_id))


def _halt(world: _World) -> Any:
    trace = world.core([_TRASH]).run("trash the two promos")
    assert trace.pending_approval_id is not None
    return trace


def test_a_destructive_call_halts_on_one_itemised_approval(tmp_path: Path) -> None:
    world = _World(tmp_path)
    trace = _halt(world)

    assert world.mailbox.trashed == []  # nothing ran
    assert trace.halted_by == "approval"
    assert trace.success is True  # waiting on the owner is a complete turn
    assert "needs your approval, so nothing has changed yet" in trace.final_answer
    assert "[Review it in Activity](/actions)" in trace.final_answer
    assert trace.effects_executed == []

    row = world.queue.get(trace.pending_approval_id)
    assert row is not None and row.status == "pending"
    assert row.items == (ApprovalItem.of("trash_email", {"ids": ["m1", "m2"]}),)
    assert '"ids": ["m1", "m2"]' in row.context_summary  # the card shows exactly what
    assert (row.channel, row.session_id) == ("web", "web-s1")
    # Linked to the checkpoint the resume starts from.
    assert row.checkpoint_id == trace.checkpoint_id
    assert world.store.get_latest(trace.run_id).payload["pending_approval_id"] == str(
        trace.pending_approval_id
    )


def test_approving_runs_exactly_the_pinned_call_once(tmp_path: Path) -> None:
    world = _World(tmp_path)
    halted = _halt(world)
    world.queue.respond(halted.pending_approval_id, status="approved", actor="owner")

    llm = _ScriptedLLM([_DONE])
    core = world.core([])
    core._llm = llm
    resumed = core.run_from_seed(world.resume_seed(halted.run_id))

    assert world.mailbox.trashed == [["m1", "m2"]]
    assert resumed.effects_executed == ["destructive"]
    assert resumed.final_answer == "Done."
    # The call left its row before it ran, settled once it returned (#73).
    rows = world.ledger.list_by_run(halted.run_id)
    assert [(r.tool, r.status, r.error) for r in rows] == [("trash_email", "completed", None)]
    assert rows[0].probe_metadata["pre_recorded"] is True
    # The model continued from what really happened.
    assert "The owner approved. Results:" in llm.prompts[0]
    assert "trashed 2" in llm.prompts[0]


def test_an_approved_call_with_no_ledger_row_runs_nothing(tmp_path: Path) -> None:
    """Fail closed (#73): the resumed, approved call is denied when its row cannot be
    written before it runs -- here, a kernel with no side-effect ledger."""
    world = _World(tmp_path)
    halted = _halt(world)
    world.queue.respond(halted.pending_approval_id, status="approved", actor="owner")
    world.kernel = GovernanceKernel(audit_log=AuditLog(db_path=tmp_path / "audit2.db"))
    world.kernel.register(_AllowHook("allow_classify", HookPoint.PRE_CLASSIFY))
    world.kernel.register(_AllowHook("allow_llm", HookPoint.PRE_LLM_CALL))
    world.kernel.register(ToolPolicyHook())
    world.kernel.register(DestructiveApprovalHook(approval_queue=world.queue))
    register_side_effect_ledger(world.kernel, None)
    world.kernel.init_lock()

    llm = _ScriptedLLM(["Thought: ok\nFinal Answer: I could not do it."])
    core = world.core([])
    core._llm = llm
    resumed = core.run_from_seed(world.resume_seed(halted.run_id))

    assert world.mailbox.trashed == []
    assert resumed.effects_executed == []
    assert "needs a durable record before it runs" in llm.prompts[0]


def test_rejecting_runs_nothing_and_the_model_is_told(tmp_path: Path) -> None:
    world = _World(tmp_path)
    halted = _halt(world)
    world.queue.respond(halted.pending_approval_id, status="rejected", actor="owner")

    llm = _ScriptedLLM(["Thought: ok\nFinal Answer: I did not delete them."])
    core = world.core([])
    core._llm = llm
    resumed = core.run_from_seed(world.resume_seed(halted.run_id))

    assert world.mailbox.trashed == []
    assert resumed.effects_executed == []
    assert "The owner rejected this" in llm.prompts[0]
    assert resumed.final_answer == "I did not delete them."


def test_an_unanswered_approval_runs_nothing(tmp_path: Path) -> None:
    world = _World(tmp_path)
    halted = _halt(world)  # still pending

    llm = _ScriptedLLM([_DONE])
    core = world.core([])
    core._llm = llm
    core.run_from_seed(world.resume_seed(halted.run_id))

    assert world.mailbox.trashed == []
    assert "The approval is pending, so nothing was changed" in llm.prompts[0]


def test_one_approval_is_not_consent_to_the_next_delete(tmp_path: Path) -> None:
    """After an approved resume the model proposes another delete: that one waits too."""
    world = _World(tmp_path)
    halted = _halt(world)
    world.queue.respond(halted.pending_approval_id, status="approved", actor="owner")

    resumed = world.core([_TRASH_MORE]).run_from_seed(world.resume_seed(halted.run_id))

    assert world.mailbox.trashed == [["m1", "m2"]]  # only the approved one
    assert resumed.pending_approval_id not in (None, halted.pending_approval_id)
    second = world.queue.get(resumed.pending_approval_id)
    assert second is not None and second.items == (ApprovalItem.of("trash_email", {"ids": ["m3"]}),)


def test_a_forged_or_changed_claim_is_denied_by_the_kernel(tmp_path: Path) -> None:
    """The loop's claim is checked against the row, not trusted."""
    world = _World(tmp_path)
    halted = _halt(world)
    world.queue.respond(halted.pending_approval_id, status="approved", actor="owner")
    core = world.core([])

    changed = core._execute_tool(
        "trash_email", {"ids": ["m1", "m2", "m9"]}, approved_by=halted.pending_approval_id
    ).observation
    forged = core._execute_tool(
        "trash_email", {"ids": ["m1", "m2"]}, approved_by="no-such-id"
    ).observation

    assert world.mailbox.trashed == []
    assert "not what approval" in changed
    assert "is missing" in forged


def test_the_streaming_loop_halts_and_resumes_the_same_way(tmp_path: Path) -> None:
    world = _World(tmp_path)
    chunks = list(world.core([_TRASH]).run_stream("trash the two promos"))
    final = [c for c in chunks if isinstance(c, dict)][-1]
    approval_id = final["pending_approval_id"]
    assert final["reason"] == "awaiting_approval" and approval_id
    assert world.mailbox.trashed == []

    world.queue.respond(approval_id, status="approved", actor="owner")
    run_id = final["run_id"]
    resumed = list(world.core([_DONE]).run_stream("x", resume=world.resume_seed(run_id)))

    assert world.mailbox.trashed == [["m1", "m2"]]
    assert [c for c in resumed if isinstance(c, dict)][-1]["effects_executed"] == ["destructive"]


def test_a_run_that_cannot_resume_is_refused_not_queued(tmp_path: Path) -> None:
    world = _World(tmp_path)
    trace = world.core([_TRASH], resumable=False).run("trash the two promos")

    assert world.mailbox.trashed == []
    assert trace.pending_approval_id is None
    assert world.queue.list_pending() == []  # no approval nobody could act on
    assert "cannot pause" in (trace.steps[0].observation or "")


def test_with_no_approval_queue_it_is_denied(tmp_path: Path) -> None:
    kernel = GovernanceKernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
    kernel.register(_AllowHook("allow_classify", HookPoint.PRE_CLASSIFY))
    kernel.register(_AllowHook("allow_llm", HookPoint.PRE_LLM_CALL))
    kernel.register(DestructiveApprovalHook(approval_queue=None))
    kernel.init_lock()
    mailbox = _Mailbox()
    core = AgenticCore(
        llm_call=_ScriptedLLM([_TRASH]), tools=[mailbox.tool()], kernel=kernel, agent_type="chat"
    )
    trace = core.run("trash them")
    assert mailbox.trashed == []
    assert "no approval queue" in (trace.steps[0].observation or "")


# --- ADR-0118 step 4: the card reads in the owner's words ----------------------------


def _described(world: _World, describe: Any) -> Any:
    tool = world.mailbox.tool()._replace(
        describe=describe, undo="restore_email", undo_window_days=30
    )
    core = world.core([_TRASH])
    core._tools, core._tool_index = [tool], {tool.name: tool}
    return core


def test_the_card_carries_the_plugins_words_the_undo_and_the_request(tmp_path: Path) -> None:
    from iris_harness.agent.agentic_core import ToolDescription

    world = _World(tmp_path)
    seen: list[dict[str, Any]] = []

    def describe(args: dict[str, Any]) -> ToolDescription:
        seen.append(args)
        return ToolDescription(
            title="Trash 2 emails",
            lines=("Your weekly deals — Store X", "Last chance: 40% off — Shop Y"),
        )

    trace = _described(world, describe).run("clean up the promo emails")
    row = world.queue.get(trace.pending_approval_id)

    assert seen == [{"ids": ["m1", "m2"]}]  # described once, from the real arguments
    assert row is not None and row.card is not None
    assert row.card.title == "Trash 2 emails"
    assert row.card.lines == ("Your weekly deals — Store X", "Last chance: 40% off — Shop Y")
    assert (row.card.undo_tool, row.card.undo_window_days) == ("restore_email", 30)
    assert row.card.asked == "clean up the promo emails"
    assert row.signal == "Trash 2 emails"  # the phone banner's words
    assert row.context_summary.startswith("Trash 2 emails\n- Your weekly deals — Store X")
    assert "Reversible for 30 days (undo: restore_email)." in row.context_summary
    assert 'Exact call: trash_email {"ids": ["m1", "m2"]}' in row.context_summary
    assert trace.final_answer.startswith("Trash 2 emails: this needs your approval")


def test_a_describe_that_raises_falls_back_to_the_raw_call(tmp_path: Path) -> None:
    world = _World(tmp_path)

    def describe(args: dict[str, Any]) -> Any:
        raise RuntimeError("mailbox offline")

    trace = _described(world, describe).run("clean up the promo emails")
    row = world.queue.get(trace.pending_approval_id)

    assert world.mailbox.trashed == []
    assert row is not None and row.card is not None
    assert row.card.title == "trash_email wants to delete or overwrite your data"
    assert row.card.lines == ('trash_email {"ids": ["m1", "m2"]}',)
    # The chat halt message words the fallback the same way the card does (#109).
    assert trace.final_answer.startswith(
        "trash_email wants to delete or overwrite your data: this needs your approval"
    )
    assert row.card.undo_tool == "restore_email"  # the manifest's part still holds


def test_a_tool_with_no_undo_says_it_cannot_be_undone(tmp_path: Path) -> None:
    world = _World(tmp_path)
    trace = world.core([_TRASH]).run("delete them")
    row = world.queue.get(trace.pending_approval_id)
    assert row is not None and row.card is not None
    assert row.card.undo_tool is None
    assert "This cannot be undone." in row.context_summary


# --- The plugin checks the arguments before the owner is ever asked ----------------------

_TRASH_INVENTED = (
    'Thought: trash them\nAction: trash_email\nAction Input: {"ids": ["17vq6d3-120928"]}'
)


def _validated(world: _World, responses: list[str], validate: Any) -> tuple[AgenticCore, Any]:
    tool = world.mailbox.tool()._replace(validate=validate)
    llm = _ScriptedLLM(responses)
    core = world.core([])
    core._llm = llm
    core._tools, core._tool_index = [tool], {tool.name: tool}
    return core, llm


def _real_ids_only(args: dict[str, Any]) -> str | None:
    unknown = [i for i in args.get("ids", []) if i not in {"m1", "m2", "m3"}]
    return f"not in the owner's mail: {', '.join(unknown)}." if unknown else None


def test_invented_ids_never_reach_the_owner(tmp_path: Path) -> None:
    # Phone test 2026-09-21: the model trashed ids it made up, the owner approved a card
    # of "(not in your mail any more: …)" lines, and nothing was trashed.
    world = _World(tmp_path)
    core, llm = _validated(world, [_TRASH_INVENTED], _real_ids_only)

    trace = core.run("trash all my promo emails this month")

    assert trace.pending_approval_id is None
    assert world.queue.list_pending() == []  # no card, no banner, no Telegram message
    assert world.mailbox.trashed == []
    assert "not in the owner's mail: 17vq6d3-120928." in llm.prompts[-1]
    assert "Nothing was sent for approval and nothing changed." in llm.prompts[-1]


def test_the_model_can_correct_itself_and_then_the_owner_is_asked(tmp_path: Path) -> None:
    world = _World(tmp_path)
    core, _llm = _validated(world, [_TRASH_INVENTED, _TRASH], _real_ids_only)

    trace = core.run("trash the two promos")

    assert trace.pending_approval_id is not None
    [row] = world.queue.list_pending()
    assert row.items == (ApprovalItem.of("trash_email", {"ids": ["m1", "m2"]}),)


def test_a_validate_that_raises_refuses_the_call(tmp_path: Path) -> None:
    world = _World(tmp_path)

    def validate(args: dict[str, Any]) -> str | None:
        raise RuntimeError("store offline")

    core, llm = _validated(world, [_TRASH], validate)
    trace = core.run("trash the two promos")

    assert trace.pending_approval_id is None
    assert world.queue.list_pending() == []
    assert "trash_email could not check these arguments." in llm.prompts[-1]


def test_the_approved_call_is_not_revalidated_into_a_refusal(tmp_path: Path) -> None:
    # The owner approved what the card showed; the resume runs it. If the mail changed
    # meanwhile, the tool itself reports what it could not trash.
    world = _World(tmp_path)
    core, _llm = _validated(world, [_TRASH], _real_ids_only)
    trace = core.run("trash the two promos")
    world.queue.respond(trace.pending_approval_id, status="approved", actor="owner")

    calls: list[dict[str, Any]] = []

    def now_says_no(args: dict[str, Any]) -> str | None:
        calls.append(args)
        return "gone"

    resumed, _ = _validated(world, [_DONE], now_says_no)
    resumed.run_from_seed(world.resume_seed(trace.run_id))

    assert world.mailbox.trashed == [["m1", "m2"]]
    assert calls == []


# --- the label an approved call's result earned is a floor on the resumed run ---------


class _Label:
    """Sets a fixed classification at one hook point (the question's, or the result's)."""

    priority: int = 10

    def __init__(self, name: str, hook_point: HookPoint, label: str) -> None:
        self.name = name
        self.hook_point = hook_point
        self._label = label

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="test", set_classification=self._label)  # type: ignore[arg-type]


class _EgressLabels:
    """Stands in for the egress gate: the label each PRE_LLM_CALL was decided against."""

    name = "egress_labels"
    hook_point = HookPoint.PRE_LLM_CALL
    priority = 50

    def __init__(self) -> None:
        self.seen: list[str | None] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.seen.append(ctx.classification)
        return HookDecision(outcome="allow", reason="test")


class _LabelledWorld(_World):
    """The question classifies ``public``; the approved call's result is ``personal``."""

    def __init__(self, tmp_path: Path) -> None:
        self.queue = ApprovalQueue(db_path=tmp_path / "approvals.db")
        self.store = CheckpointStore(db_path=tmp_path / "checkpoints.db")
        self.egress = _EgressLabels()
        self.kernel = GovernanceKernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
        self.kernel.register(_Label("question_label", HookPoint.PRE_CLASSIFY, "public"))
        self.kernel.register(_Label("result_label", HookPoint.POST_TOOL_USE, "personal"))
        self.kernel.register(self.egress)
        self.kernel.register(ToolPolicyHook())
        self.kernel.register(DestructiveApprovalHook(approval_queue=self.queue))
        # A destructive call runs only once its row is in the side-effect ledger (#73).
        self.ledger = SideEffectLedger(tmp_path / "side_effects.db")
        register_side_effect_ledger(self.kernel, self.ledger)
        self.kernel.init_lock()
        self.mailbox = _Mailbox()


def test_a_resumed_run_egresses_under_the_label_the_approved_result_earned(
    tmp_path: Path,
) -> None:
    """The resumed core has no label yet, so PRE_CLASSIFY derives ``public`` from the
    question -- but the approved call already put a ``personal`` result in the prompt.
    The earned label is a floor under the derived one, never dropped."""
    world = _LabelledWorld(tmp_path)
    halted = _halt(world)
    world.queue.respond(halted.pending_approval_id, status="approved", actor="owner")
    world.egress.seen.clear()

    world.core([_DONE]).run_from_seed(world.resume_seed(halted.run_id))

    assert world.mailbox.trashed == [["m1", "m2"]]
    assert world.egress.seen and set(world.egress.seen) == {"personal"}, world.egress.seen


def test_the_streaming_resume_egresses_under_the_earned_label_too(tmp_path: Path) -> None:
    world = _LabelledWorld(tmp_path)
    chunks = list(world.core([_TRASH]).run_stream("trash the two promos"))
    final = [c for c in chunks if isinstance(c, dict)][-1]
    world.queue.respond(final["pending_approval_id"], status="approved", actor="owner")
    world.egress.seen.clear()

    list(world.core([_DONE]).run_stream("x", resume=world.resume_seed(final["run_id"])))

    assert world.mailbox.trashed == [["m1", "m2"]]
    assert world.egress.seen and set(world.egress.seen) == {"personal"}, world.egress.seen
