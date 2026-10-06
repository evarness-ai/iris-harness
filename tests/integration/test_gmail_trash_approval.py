"""The first destructive tool, end to end (ADR-0118 step 5).

"Trash the promo emails" through the real loop, the real kernel hooks, a real approval
queue and checkpoint store, the real ``trash_email`` over a real mail store — declared
exactly as the email plugin's manifest declares it — with only the model and Gmail
faked. The run halts on an approval whose card names each email; approving trashes
exactly those; "undo that" brings them back.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import iris_personal.plugins.email_workflows as email_workflows
from iris_harness.agent.agentic_core import (
    AgenticCore,
    AgenticCoreConfig,
    resume_seed_from_checkpoint,
)
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.approvals import ApprovalQueue
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.plugins import DestructiveApprovalHook, ToolPolicyHook
from iris_harness.kernel.governance.side_effects import SideEffectLedger
from iris_harness.kernel.governance.wiring import register_side_effect_ledger
from iris_harness.memory.state import CheckpointStore
from iris_harness.runtime.plugin_host.manifest import load_manifest
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore
from iris_personal.email.trash_tools import build_trash_tools

pytestmark = pytest.mark.integration

ACCT = "gmail:owner@gmail.com"
_TRASH = 'Thought: trash the promos\nAction: trash_email\nAction Input: {"ids": ["m1", "m2"]}'
# An attempt the validator refuses: no card, but it is what puts the ask guard on.
_GUESS = 'Thought: trash them\nAction: trash_email\nAction Input: {"ids": ["zz9"]}'
_RESTORE = "Thought: undo\nAction: restore_email\nAction Input: {}"
_CONFIRM = (
    "Thought: I should confirm first\nAction: ask_user\n"
    'Action Input: {"question": "Do you want me to delete these 2 emails?"}'
)


class _Allow:
    priority = 10

    def __init__(self, name: str, point: HookPoint) -> None:
        self.name, self.hook_point = name, point

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="test")


class _Gmail:
    def __init__(self, messages: list[EmailMessage]) -> None:
        self.by_id = {m.id: m for m in messages}
        self.trash: list[str] = []

    def trash_messages(self, account_id: str, ids: Any) -> list[str]:
        self.trash.extend(ids)
        return list(ids)

    def restore_messages(
        self, account_id: str, ids: Any, *, labels_before: Any = None
    ) -> list[EmailMessage]:
        self.trash = [i for i in self.trash if i not in ids]
        return [self.by_id[i] for i in ids]


class _LLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "Thought: done\nFinal Answer: Done."


def _msg(mid: str, subject: str, sender: str) -> EmailMessage:
    return EmailMessage(
        id=mid,
        provider="gmail",  # type: ignore[arg-type]
        account_id=ACCT,
        from_address=sender,
        subject=subject,
        received_at=datetime(2026, 9, 20, 9, 0, tzinfo=UTC),
    )


@pytest.fixture()
def world(tmp_path: Path) -> Any:
    messages = [
        _msg("m1", "Your weekly deals are here", "Store X <deals@storex.com>"),
        _msg("m2", "Last chance: 40% off", "Shop Y <hi@shopy.com>"),
        _msg("m3", "Your flight itinerary", "Airline <no-reply@air.com>"),
    ]
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert_many(messages)
    gmail = _Gmail(messages)

    # Declared exactly as the email plugin's manifest declares them.
    manifest = load_manifest(Path(email_workflows.__file__).parent / "manifest.yaml")
    tools = []
    for tool in build_trash_tools(data_dir=tmp_path, provider_for=lambda a: gmail):
        decl = manifest.tools[tool.name]
        tools.append(
            tool._replace(
                effect=decl.effect,
                confirm=decl.confirm_mode,
                undo=decl.undo,
                undo_window_days=decl.undo_window_days,
            )
        )

    queue = ApprovalQueue(db_path=tmp_path / "approvals.db")
    checkpoints = CheckpointStore(db_path=tmp_path / "checkpoints.db")
    kernel = GovernanceKernel(audit_log=AuditLog(db_path=tmp_path / "audit.db"))
    kernel.register(_Allow("classify", HookPoint.PRE_CLASSIFY))
    kernel.register(_Allow("llm", HookPoint.PRE_LLM_CALL))
    kernel.register(ToolPolicyHook())
    kernel.register(DestructiveApprovalHook(approval_queue=queue))
    # The approved call runs only once its row is in the side-effect ledger (#73).
    register_side_effect_ledger(kernel, SideEffectLedger(tmp_path / "side_effects.db"))
    kernel.init_lock()

    def read(approval_id: str) -> Any:
        row = queue.get(approval_id)
        return None if row is None else (row.status, [(i.tool, i.args) for i in (row.items or ())])

    def core(llm: _LLM) -> AgenticCore:
        return AgenticCore(
            config=AgenticCoreConfig(max_iterations=6, allow_ask_user=True),
            llm_call=llm,
            tools=tools,
            kernel=kernel,
            checkpoint_store=checkpoints,
            session_id="web-s1",
            origin_channel="web",
            link_approval_checkpoint=queue.set_checkpoint,
            read_approval=read,
            agent_type="chat",
        )

    return store, gmail, queue, checkpoints, core


def test_trash_waits_for_approval_then_runs_exactly_what_the_card_showed(world: Any) -> None:
    store, gmail, queue, checkpoints, core = world

    halted = core(_LLM([_TRASH])).run("trash the promo emails from this week")

    assert gmail.trash == [] and store.get("m1") is not None  # nothing yet
    row = queue.get(halted.pending_approval_id)
    assert row is not None and row.card is not None
    assert row.card.title == "Trash 2 emails"
    assert row.card.lines == (
        "Your weekly deals are here — Store X <deals@storex.com> · 20 Sep",
        "Last chance: 40% off — Shop Y <hi@shopy.com> · 20 Sep",
    )
    assert (row.card.undo_tool, row.card.undo_window_days) == ("restore_email", 30)
    assert row.card.asked == "trash the promo emails from this week"

    queue.respond(halted.pending_approval_id, status="approved", actor="web")
    llm = _LLM(["Thought: done\nFinal Answer: Trashed the 2 promos."])
    resumed = core(llm).run_from_seed(
        resume_seed_from_checkpoint(checkpoints.get_latest(halted.run_id))
    )

    assert gmail.trash == ["m1", "m2"]
    assert store.get("m1") is None and store.get("m3") is not None
    assert resumed.effects_executed == ["destructive"]
    assert "Moved 2 emails to Trash" in llm.prompts[0]


def test_undo_that_brings_them_back_without_asking(world: Any) -> None:
    store, gmail, queue, checkpoints, core = world
    halted = core(_LLM([_TRASH])).run("trash the promo emails")
    queue.respond(halted.pending_approval_id, status="approved", actor="web")
    core(_LLM([])).run_from_seed(resume_seed_from_checkpoint(checkpoints.get_latest(halted.run_id)))
    assert gmail.trash == ["m1", "m2"]

    undo = core(_LLM([_RESTORE])).run("undo that")

    assert undo.pending_approval_id is None  # a write with confirm: never — no approval
    assert gmail.trash == []
    assert store.get("m1") is not None and store.get("m2") is not None
    assert undo.effects_executed == ["write"]


def test_rejecting_trashes_nothing(world: Any) -> None:
    store, gmail, queue, checkpoints, core = world
    halted = core(_LLM([_TRASH])).run("trash the promo emails")
    queue.respond(halted.pending_approval_id, status="rejected", actor="web")
    core(_LLM([])).run_from_seed(resume_seed_from_checkpoint(checkpoints.get_latest(halted.run_id)))
    assert gmail.trash == [] and store.get("m1") is not None


# --- the route the owner's request actually takes -----------------------------------
#
# The tests above drive a loop directly. The owner's "trash the promo emails" was routed
# to the `email` agent, which until 2026-09-21 was the email plugin's own loop: it could
# not pause for an approval, so the call was refused and the turn fell back to the inbox
# digest. These go through a real runtime, with the email plugin mounted, to the agent
# the request is routed to.


class _FakeGmailProvider:
    name = "gmail"

    def __init__(self) -> None:
        self.trashed: list[str] = []

    def trash_messages(self, account_id: str, ids: Any) -> list[str]:
        self.trashed.extend(ids)
        return list(ids)

    def restore_messages(
        self, account_id: str, ids: Any, *, labels_before: Any = None
    ) -> list[EmailMessage]:
        return []

    def __getattr__(self, name: str) -> Any:  # the read methods are not used here
        raise AttributeError(name)


def _model(self: Any, *, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
    if "The owner approved. Results:" in user_prompt:
        return "Thought: done\nFinal Answer: Trashed the two promos."
    if "trash_email" in user_prompt:
        return _TRASH
    return "Thought: nothing\nFinal Answer: ok"


@pytest.fixture(params=["no_model", "model"])
def runtime_world(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """A real runtime with the email plugin mounted, at the default 12-tool shortlist.

    ``no_model`` is a fresh offline install (the hosted runner): no MiniLM on disk, so
    the shortlist ranks without an embedder — the email agent's own tools first. With
    the menu left to pool order every email tool fell off it and the fake model below
    never saw ``trash_email``. ``model`` ranks with the embedder, where it is on disk.
    """
    from iris_harness.kernel.governance.evaluator import embeddings
    from iris_harness.llm.client import CodingLLMClient
    from iris_harness.runtime import build_runtime, routine_authoring
    from iris_personal.email.providers import register_mail_provider

    if request.param == "no_model":
        monkeypatch.setenv("IRIS_TEST_NULL_EMBEDDINGS", "1")
        monkeypatch.setattr(embeddings, "default_model_on_disk", lambda: False)
    elif not embeddings.default_model_on_disk():
        pytest.skip(f"needs the all-MiniLM-L6-v2 ONNX model at {embeddings.default_model_path()}")
    # A router built by an earlier test holds whichever embedder that test had.
    monkeypatch.setattr(routine_authoring, "_semantic_router_singleton", None)
    monkeypatch.setattr(routine_authoring, "_semantic_router_init_failed", False)
    monkeypatch.delenv("IRIS_GOVERNANCE_ENABLED", raising=False)
    monkeypatch.setattr(CodingLLMClient, "invoke", _model)
    config_dir, data_dir = tmp_path / "config", tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    store = EmailStore(db_path=data_dir / "email.db")
    store.ensure_schema()
    store.upsert_many(
        [
            _msg("m1", "Your weekly deals are here", "Store X <deals@storex.com>"),
            _msg("m2", "Last chance: 40% off", "Shop Y <hi@shopy.com>"),
        ]
    )
    runtime = build_runtime(
        config_dir=config_dir, data_dir=data_dir, use_background_scheduler=False
    )
    gmail = _FakeGmailProvider()
    register_mail_provider(gmail)  # type: ignore[arg-type]
    return runtime, store, gmail


def test_an_email_turn_runs_on_the_loop_that_can_pause(runtime_world: Any) -> None:
    runtime, _store, _gmail = runtime_world
    assert "email" in runtime._loop_intents
    assert "email" in runtime._react_fallbacks  # the plugin's agent is the degrade path


def test_trash_routed_to_the_email_agent_halts_then_approving_trashes(runtime_world: Any) -> None:
    from iris_harness.agent.agent_executor import AgentTask
    from iris_harness.kernel.governance.approvals.service import respond_to_approval

    runtime, store, gmail = runtime_world
    result = runtime.agent_executor.execute(
        AgentTask(
            query="trash the promo emails from this week",
            agent_type="email",
            session_id="web-s1",
            params={"intent": "communication"},
        )
    )

    approval_id = result.metadata.get("pending_approval_id")
    assert approval_id, result.output  # not the inbox digest
    assert gmail.trashed == []
    row = ApprovalQueue().get(approval_id)
    assert row is not None and row.card is not None
    assert row.card.lines[0].startswith("Your weekly deals are here — Store X")

    outcome = respond_to_approval(approval_id, status="approved", actor="web", resumer=runtime)

    assert outcome.resumed is True
    assert gmail.trashed == ["m1", "m2"]
    assert store.get("m1") is None


# --- the owner confirms once, on the card (2026-09-21) ------------------------------
#
# The owner said "yes" to "Do you want to allow IRIS to manage your Gmail account for
# deleting emails?" and then had the approval card to answer as well. qwen2.5 asks first
# whatever the prompt says, so once it has tried the tool the loop turns the ask back
# until the card is shown.


def test_a_confirm_first_question_is_turned_back_and_the_card_is_raised(world: Any) -> None:
    _store, gmail, queue, _checkpoints, core = world
    llm = _LLM([_GUESS, _CONFIRM, _TRASH])

    halted = core(llm).run("trash the promo emails from this week")

    assert halted.halted_by != "ask_user"
    assert halted.pending_approval_id is not None  # one confirmation: the card
    assert gmail.trash == []
    assert (halted.steps[1].observation or "").startswith("Not asked yet.")
    assert "card is both their confirmation and their selection" in llm.prompts[2]


def test_the_streaming_loop_turns_it_back_too(world: Any) -> None:
    _store, gmail, queue, _checkpoints, core = world
    chunks = list(core(_LLM([_GUESS, _CONFIRM, _TRASH])).run_stream("trash the promo emails"))
    final = [c for c in chunks if isinstance(c, dict)][-1]
    assert final.get("pending_approval_id")
    assert gmail.trash == []


def test_asking_again_and_again_ends_on_a_plain_account(world: Any) -> None:
    """The card is the selection UI (2026-09-22): a "which ones?" never reaches the owner
    with nothing to pick from. Twice turned back, the third ask ends the turn."""
    _store, gmail, queue, _checkpoints, core = world
    which = _CONFIRM.replace("Do you want me to delete these 2 emails?", "Which ones?")

    ended = core(_LLM([_GUESS, which, which, which])).run("delete some of my emails")

    assert ended.halted_by is None
    assert (
        ended.final_answer == "I couldn't prepare the trash_email request, so nothing was changed."
    )
    assert queue.list_pending() == []
    assert gmail.trash == []
