"""``send_email`` end to end (ADR-0118 amendment: a write approved per call).

"Email Bob about lunch" through the real loop, the real kernel hooks, a real approval
queue and checkpoint store, the real ``send_email`` over a real mail store — declared
exactly as the email plugin's manifest declares it — with only the model and Gmail
faked. The run halts on a card that shows the whole message; approving sends exactly
the pinned message; rejecting sends nothing and the model is told nothing was sent.
The second half drives the route the owner's request takes: a real runtime with the
email plugin mounted, the ``email`` agent, and ``respond_to_approval``. Nothing here
reaches Gmail.
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
from iris_harness.kernel.governance.approvals import ApprovalItem, ApprovalQueue
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.plugins import DestructiveApprovalHook, ToolPolicyHook
from iris_harness.kernel.governance.side_effects import SideEffectLedger
from iris_harness.kernel.governance.wiring import register_side_effect_ledger
from iris_harness.memory.state import CheckpointStore
from iris_harness.runtime.plugin_host.manifest import load_manifest
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.send_tools import build_send_tools
from iris_personal.email.store import EmailStore

pytestmark = pytest.mark.integration

ACCT = "gmail:owner@gmail.com"
_ARGS = {"to": ["bob@example.com"], "subject": "Lunch", "body": "Friday at 1?"}
_SEND = (
    "Thought: send it\nAction: send_email\nAction Input: "
    '{"to": ["bob@example.com"], "subject": "Lunch", "body": "Friday at 1?"}'
)
_REPLY = (
    "Thought: reply\nAction: send_email\nAction Input: "
    '{"reply_to_id": "m1", "body": "Count me in."}'
)


class _Allow:
    priority = 10

    def __init__(self, name: str, point: HookPoint) -> None:
        self.name, self.hook_point = name, point

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="test")


class _Gmail:
    name = "gmail"

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def send_message(self, account_id: str, **kwargs: Any) -> str:
        self.sent.append({"account_id": account_id, **kwargs})
        return f"sent-{len(self.sent)}"

    def __getattr__(self, name: str) -> Any:  # the read methods are not used here
        raise AttributeError(name)


class _LLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies, self.prompts = list(replies), []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "Thought: done\nFinal Answer: Done."


def _msg(mid: str) -> EmailMessage:
    return EmailMessage(
        id=mid,
        provider="gmail",  # type: ignore[arg-type]
        account_id=ACCT,
        thread_id="thread-" + mid,
        from_address="Alice <alice@example.com>",
        subject="Dinner plans",
        received_at=datetime(2026, 9, 20, 9, 0, tzinfo=UTC),
        headers_subset={"Message-ID": "<abc@mail.example.com>"},
    )


@pytest.fixture()
def world(tmp_path: Path) -> Any:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert_many([_msg("m1")])
    gmail = _Gmail()

    # Declared exactly as the email plugin's manifest declares it.
    decl = load_manifest(Path(email_workflows.__file__).parent / "manifest.yaml").tools[
        "send_email"
    ]
    assert (decl.effect, decl.confirm_mode) == ("write", "approval")
    (tool,) = build_send_tools(
        data_dir=tmp_path,
        provider_for=lambda a: gmail,
        current_query=lambda: "email bob@example.com about lunch",
    )
    tools = [tool._replace(effect=decl.effect, confirm=decl.confirm_mode)]

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

    def resume(halted: Any, llm: _LLM) -> Any:
        return core(llm).run_from_seed(
            resume_seed_from_checkpoint(checkpoints.get_latest(halted.run_id))
        )

    return gmail, queue, core, resume


def test_send_waits_then_sends_exactly_the_pinned_message(world: Any) -> None:
    gmail, queue, core, resume = world

    halted = core(_LLM([_SEND])).run("email Bob about lunch on Friday at 1")

    assert gmail.sent == []  # nothing yet
    row = queue.get(halted.pending_approval_id)
    assert row is not None and row.card is not None
    assert row.items == (ApprovalItem.of("send_email", _ARGS),)
    assert row.card.title == "Send email to bob@example.com — Lunch"
    assert row.card.lines == (
        "From: owner@gmail.com",
        "To: bob@example.com",
        "Subject: Lunch",
        "Message:",
        "Friday at 1?",
    )
    assert row.card.effect == "write"
    assert row.card.undo_sentence() == "This cannot be undone."

    queue.respond(halted.pending_approval_id, status="approved", actor="web")
    llm = _LLM(["Thought: ok\nFinal Answer: Sent."])
    resumed = resume(halted, llm)

    assert len(gmail.sent) == 1
    assert gmail.sent[0]["to"] == ["bob@example.com"]
    assert gmail.sent[0]["body"] == "Friday at 1?"
    # The send ran before the model said a word: its result is the first thing it reads.
    assert "Sent to bob@example.com: Lunch." in llm.prompts[0]
    assert resumed.effects_executed[0] == "write"


def test_rejecting_sends_nothing_and_the_model_hears_nothing_was_sent(world: Any) -> None:
    gmail, queue, core, resume = world
    halted = core(_LLM([_SEND])).run("email Bob about lunch")
    queue.respond(halted.pending_approval_id, status="rejected", actor="web")

    llm = _LLM(["Thought: ok\nFinal Answer: I did not send it."])
    resumed = resume(halted, llm)

    assert gmail.sent == []
    assert resumed.effects_executed == []
    assert "nothing was changed, sent or deleted" in llm.prompts[0]


def test_a_reply_is_threaded_on_the_card_and_in_the_send(world: Any) -> None:
    gmail, queue, core, resume = world
    halted = core(_LLM([_REPLY])).run("reply to Alice that I'm in")
    row = queue.get(halted.pending_approval_id)
    assert row is not None and row.card is not None
    assert row.card.title == "Send email to Alice <alice@example.com> — Re: Dinner plans"

    queue.respond(halted.pending_approval_id, status="approved", actor="web")
    resume(halted, _LLM([]))

    (sent,) = gmail.sent
    assert (sent["thread_id"], sent["in_reply_to"]) == ("thread-m1", "<abc@mail.example.com>")


def test_a_bad_address_never_becomes_a_card(world: Any) -> None:
    gmail, queue, core, _resume = world
    bad = _SEND.replace("bob@example.com", "Bob")
    trace = core(_LLM([bad])).run("email Bob")
    assert queue.list_pending() == []
    assert gmail.sent == []
    assert "not an email address: Bob" in (trace.steps[0].observation or "")


# --- the route the owner's request actually takes -----------------------------------


def _model(self: Any, *, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
    if "The owner approved. Results:" in user_prompt:
        return "Thought: done\nFinal Answer: Sent it."
    if "The owner rejected this" in user_prompt:
        return "Thought: done\nFinal Answer: I did not send it."
    # The loop's prompt (it carries the ReAct format) for the owner's request. The
    # relevance shortlist may leave send_email off this test's menu (no embedder here);
    # the loop admits a pool tool on its first call either way.
    if "Action Input" in user_prompt and "email bob@example.com about lunch" in user_prompt:
        return _SEND
    return "Thought: nothing\nFinal Answer: ok"


@pytest.fixture()
def runtime_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    from iris_harness.llm.client import CodingLLMClient
    from iris_harness.runtime import build_runtime
    from iris_personal.email.providers import register_mail_provider

    monkeypatch.delenv("IRIS_GOVERNANCE_ENABLED", raising=False)
    monkeypatch.setattr(CodingLLMClient, "invoke", _model)
    config_dir, data_dir = tmp_path / "config", tmp_path / "data"
    config_dir.mkdir()
    data_dir.mkdir()
    store = EmailStore(db_path=data_dir / "email.db")
    store.ensure_schema()
    store.upsert_many([_msg("m1")])
    runtime = build_runtime(
        config_dir=config_dir, data_dir=data_dir, use_background_scheduler=False
    )
    gmail = _Gmail()
    register_mail_provider(gmail)  # type: ignore[arg-type]
    return runtime, gmail


def _ask(runtime: Any) -> str:
    from iris_harness.agent.agent_executor import AgentTask

    result = runtime.agent_executor.execute(
        AgentTask(
            query="email bob@example.com about lunch on Friday at 1",
            agent_type="email",
            session_id="web-s1",
            params={"intent": "communication"},
        )
    )
    approval_id = result.metadata.get("pending_approval_id")
    assert approval_id, result.output
    return str(approval_id)


def test_routed_to_the_email_agent_it_halts_then_approving_sends(runtime_world: Any) -> None:
    from iris_harness.kernel.governance.approvals.service import respond_to_approval

    runtime, gmail = runtime_world
    approval_id = _ask(runtime)
    assert gmail.sent == []
    row = ApprovalQueue().get(approval_id)
    assert row is not None and row.card is not None and row.card.effect == "write"
    assert row.items == (ApprovalItem.of("send_email", _ARGS),)

    outcome = respond_to_approval(approval_id, status="approved", actor="web", resumer=runtime)

    assert outcome.resumed is True
    assert [(s["account_id"], s["to"], s["body"]) for s in gmail.sent] == [
        (ACCT, ["bob@example.com"], "Friday at 1?")
    ]


def test_routed_to_the_email_agent_rejecting_sends_nothing(runtime_world: Any) -> None:
    from iris_harness.kernel.governance.approvals.service import respond_to_approval

    runtime, gmail = runtime_world
    approval_id = _ask(runtime)

    outcome = respond_to_approval(approval_id, status="rejected", actor="web", resumer=runtime)

    assert outcome.resumed is True  # the conversation closes instead of going silent
    assert gmail.sent == []
