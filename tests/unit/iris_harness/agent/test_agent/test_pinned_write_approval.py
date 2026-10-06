"""A write declared ``approval: pinned`` takes the destructive tools' approval path
(ADR-0118 amendment), without being called data loss.

The manifest form, the loop's gate, the hook's queue and card, the resume that runs
exactly the pinned call, and the refusals where no approval can be taken — each checked
for a ``write`` whose gate is ``approval``, through the real loop, kernel hooks, approval
queue and checkpoint store. Only the model and the tool are scripted.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agentic_core import (
    AgenticCore,
    AgenticCoreConfig,
    ToolDescription,
    ToolSpec,
    _build_react_prompt,
    resume_seed_from_checkpoint,
)
from iris_harness.agent.tool_runner import approved_per_call
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.approvals import ApprovalItem, ApprovalQueue
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.plugins import DestructiveApprovalHook, ToolPolicyHook
from iris_harness.kernel.governance.side_effects import SideEffectLedger
from iris_harness.kernel.governance.wiring import register_side_effect_ledger
from iris_harness.memory.state import CheckpointStore
from iris_harness.runtime.plugin_host.manifest import PluginManifest

_ARGS = {"to": ["bob@example.com"], "subject": "Lunch", "body": "Friday at 1?"}
_SEND = (
    "Thought: send it\nAction: send_email\nAction Input: "
    '{"to": ["bob@example.com"], "subject": "Lunch", "body": "Friday at 1?"}'
)
_DONE = "Thought: done\nFinal Answer: Done."


# --- the manifest form ---------------------------------------------------------------


def _manifest(tools: dict[str, dict[str, object]]) -> PluginManifest:
    return PluginManifest.model_validate({"name": "x", "provides": ["tool"], "tools": tools})


def test_a_pinned_write_has_the_approval_gate_and_stays_a_write() -> None:
    decl = _manifest({"send": {"effect": "write", "approval": "pinned"}}).tools["send"]
    assert (decl.effect, decl.confirm_mode) == ("write", "approval")
    # A plain write still asks once; nothing else changed.
    assert _manifest({"w": {"effect": "write"}}).tools["w"].confirm_mode == "once"


def test_approval_pinned_is_for_writes_only_and_excludes_confirm() -> None:
    for effect in ("read", "destructive"):
        with pytest.raises(ValueError, match="'approval' only applies"):
            _manifest({"t": {"effect": effect, "approval": "pinned"}})
    for confirm in ("once", "never"):
        with pytest.raises(ValueError, match="cannot both be declared"):
            _manifest({"t": {"effect": "write", "approval": "pinned", "confirm": confirm}})
    with pytest.raises(ValueError):
        _manifest({"t": {"effect": "write", "approval": "sometimes"}})


def test_a_pinned_write_cannot_be_an_undo_tool() -> None:
    """An undo is a write that does not ask; a pinned write asks every time."""
    with pytest.raises(ValueError, match="confirm: never"):
        _manifest(
            {
                "trash": {"effect": "destructive", "undo": "send"},
                "send": {"effect": "write", "approval": "pinned"},
            }
        )


# --- the loop, the hook and the resume -------------------------------------------------


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


class _Outbox:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def tool(self, *, validate: Any = None, titled: bool = True) -> ToolSpec:
        def _send(args: dict[str, Any]) -> str:
            self.sent.append(dict(args))
            return "sent"

        return ToolSpec(
            name="send_email",
            description="Send an email.",
            call=_send,
            effect="write",
            confirm="approval",
            describe=(
                (
                    lambda a: ToolDescription(
                        title=f"Send email to {a['to'][0]} — {a['subject']}", lines=(a["body"],)
                    )
                )
                if titled
                else None
            ),
            validate=validate,
        )


class _World:
    def __init__(self, tmp_path: Path, *, with_hook: bool = True) -> None:
        self.queue = ApprovalQueue(db_path=tmp_path / "approvals.db")
        self.store = CheckpointStore(db_path=tmp_path / "checkpoints.db")
        self.kernel = GovernanceKernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
        self.kernel.register(_AllowHook("allow_classify", HookPoint.PRE_CLASSIFY))
        self.kernel.register(_AllowHook("allow_llm", HookPoint.PRE_LLM_CALL))
        self.kernel.register(ToolPolicyHook())
        if with_hook:
            self.kernel.register(DestructiveApprovalHook(approval_queue=self.queue))
        # A pinned write runs only once its row is in the side-effect ledger (#73).
        self.ledger = SideEffectLedger(tmp_path / "side_effects.db")
        register_side_effect_ledger(self.kernel, self.ledger)
        self.kernel.init_lock()
        self.outbox = _Outbox()

    def _read(self, approval_id: str) -> Any:
        row = self.queue.get(approval_id)
        return None if row is None else (row.status, [(i.tool, i.args) for i in row.items or ()])

    def core(
        self,
        responses: list[str],
        *,
        kernel: bool = True,
        validate: Any = None,
        titled: bool = True,
    ) -> AgenticCore:
        return AgenticCore(
            config=AgenticCoreConfig(max_iterations=6),
            llm_call=_ScriptedLLM(responses),
            tools=[self.outbox.tool(validate=validate, titled=titled)],
            kernel=self.kernel if kernel else None,
            checkpoint_store=self.store,
            session_id="web-s1",
            origin_channel="web",
            link_approval_checkpoint=self.queue.set_checkpoint,
            read_approval=self._read,
            agent_type="chat",
        )


def test_the_predicate_covers_destructive_and_pinned_writes_only() -> None:
    def spec(effect: str, confirm: str) -> ToolSpec:
        return ToolSpec(name="t", description="", call=str, effect=effect, confirm=confirm)

    assert approved_per_call(spec("destructive", "approval"))
    assert approved_per_call(spec("write", "approval"))
    assert not approved_per_call(spec("write", "once"))
    assert not approved_per_call(spec("write", "never"))
    assert not approved_per_call(spec("read", "never"))


def test_a_pinned_write_halts_on_a_card_that_does_not_say_it_deletes(tmp_path: Path) -> None:
    world = _World(tmp_path)
    trace = world.core([_SEND]).run("email Bob about lunch")

    assert world.outbox.sent == []  # nothing ran
    assert trace.halted_by == "approval"
    assert trace.effects_executed == []
    row = world.queue.get(trace.pending_approval_id)
    assert row is not None and row.status == "pending"
    assert row.items == (ApprovalItem.of("send_email", _ARGS),)  # the whole message, pinned
    assert row.card is not None
    assert row.card.title == "Send email to bob@example.com — Lunch"
    assert row.card.lines == ("Friday at 1?",)
    assert row.card.effect == "write"
    assert row.card.undo_sentence() == "This cannot be undone."
    assert "delete" not in row.context_summary.lower()
    assert "Send email to bob@example.com — Lunch: this needs your approval" in trace.final_answer


def test_a_pinned_write_without_a_describe_title_does_not_say_it_deletes(tmp_path: Path) -> None:
    """The halt message words the fallback by effect, like the queue card (#109)."""
    world = _World(tmp_path)
    core = world.core([_SEND], titled=False)
    trace = core.run("email Bob about lunch")

    assert trace.halted_by == "approval"
    row = world.queue.get(trace.pending_approval_id)
    assert row is not None and row.card is not None
    assert row.card.title == "send_email wants to act on your behalf"
    assert trace.final_answer.startswith(
        "send_email wants to act on your behalf: this needs your approval"
    )
    assert "delete" not in trace.final_answer.lower()
    assert "overwrite" not in trace.final_answer.lower()


def test_approving_sends_exactly_the_pinned_call_once(tmp_path: Path) -> None:
    world = _World(tmp_path)
    halted = world.core([_SEND]).run("email Bob about lunch")
    world.queue.respond(halted.pending_approval_id, status="approved", actor="owner")

    # The model would change the message on the resume; it never gets the chance.
    changed = _SEND.replace("Friday at 1?", "Actually, never mind")
    resumed = world.core([changed, _DONE]).run_from_seed(
        resume_seed_from_checkpoint(world.store.get_latest(halted.run_id))
    )

    assert world.outbox.sent[0] == _ARGS
    assert resumed.effects_executed[0] == "write"
    assert (resumed.steps[0].observation or "").startswith("The owner approved. Results:")
    # A pinned write is high-risk: its row was written before it ran, then settled (#73).
    rows = world.ledger.list_by_run(halted.run_id)
    assert [(r.tool, r.status) for r in rows] == [("send_email", "completed")]
    assert rows[0].probe_metadata["pre_recorded"] is True


def test_rejecting_sends_nothing_and_says_so(tmp_path: Path) -> None:
    world = _World(tmp_path)
    halted = world.core([_SEND]).run("email Bob about lunch")
    world.queue.respond(halted.pending_approval_id, status="rejected", actor="owner")

    resumed = world.core([_DONE]).run_from_seed(
        resume_seed_from_checkpoint(world.store.get_latest(halted.run_id))
    )

    assert world.outbox.sent == []
    assert resumed.effects_executed == []
    assert "nothing was changed, sent or deleted" in (resumed.steps[0].observation or "")


def test_a_changed_message_cannot_claim_the_approval(tmp_path: Path) -> None:
    world = _World(tmp_path)
    halted = world.core([_SEND]).run("email Bob about lunch")
    world.queue.respond(halted.pending_approval_id, status="approved", actor="owner")
    hook = DestructiveApprovalHook(approval_queue=world.queue)

    def claim(args: dict[str, Any]) -> str:
        ctx = HookContext(
            hook_point=HookPoint.PRE_TOOL_USE,
            run_id="r",
            agent_type="chat",
            payload={"tool_name": "send_email", "args": args},
            metadata={
                "tool_effect": "write",
                "tool_confirm": "approval",
                "approved_by": str(halted.pending_approval_id),
            },
        )
        return asyncio.run(hook(ctx)).outcome

    assert claim(_ARGS) == "allow"
    assert claim({**_ARGS, "to": ["eve@example.com"]}) == "deny"


def test_invalid_arguments_are_refused_before_any_card(tmp_path: Path) -> None:
    world = _World(tmp_path)
    trace = world.core([_SEND, _DONE], validate=lambda a: "not an email address: bob").run("x")

    assert world.queue.list_pending() == []
    assert world.outbox.sent == []
    assert "not an email address" in (trace.steps[0].observation or "")


def test_refused_with_governance_off_in_plain_words(tmp_path: Path) -> None:
    world = _World(tmp_path)
    trace = world.core([_SEND, _DONE], kernel=False).run("email Bob")

    assert world.outbox.sent == []
    observation = trace.steps[0].observation or ""
    assert observation.startswith("Refused: each 'send_email' call needs the owner's approval")
    assert "nothing was sent" in observation
    assert "deletes" not in observation


def test_refused_when_the_approval_hook_is_missing(tmp_path: Path) -> None:
    world = _World(tmp_path, with_hook=False)
    trace = world.core([_SEND, _DONE]).run("email Bob")

    assert world.outbox.sent == []
    assert (trace.steps[0].observation or "").startswith("Refused: each 'send_email'")


def test_a_plain_write_is_not_queued(tmp_path: Path) -> None:
    """The hook's other half: a write without the gate never becomes a card."""
    queue = ApprovalQueue(db_path=tmp_path / "a.db")
    hook = DestructiveApprovalHook(approval_queue=queue)
    for confirm in ("once", "never"):
        ctx = HookContext(
            hook_point=HookPoint.PRE_TOOL_USE,
            run_id="r",
            agent_type="chat",
            payload={"tool_name": "remind", "args": {}},
            metadata={"tool_effect": "write", "tool_confirm": confirm, "resumable": True},
        )
        assert asyncio.run(hook(ctx)).outcome == "allow"
    assert queue.list_pending() == []


def test_the_prompt_marks_it_as_a_write_with_a_card_not_a_delete() -> None:
    prompt = _build_react_prompt("email Bob", [_Outbox().tool()], [], None)
    assert "send_email (WRITES on the user's behalf; each call shows the owner an approval" in (
        prompt
    )
    assert "DELETES" not in prompt
    # It is not held for a confirm-once question either.
    assert "ask the user ONCE" not in prompt
