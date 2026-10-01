"""The chat `email` agent's degrade path under the harness's governed loop.

The loop answers `email` (the plugin claims it with `api.register_loop_intent`); these pin
what the fallback answers without a model. Moved here from tests/unit/test_runtime/test_react_handler.py at M6.1b, with the
handler itself: the email library and its agent left the core (OSS plan M6,
decision 2), and the tests mirror the source tree (decision 9).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agent_executor import AgentTask
from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.memory.retriever import MemoryContext
from iris_harness.runtime.tool_service import BoundTools
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.agent import make_email_fallback_handler


@pytest.fixture(autouse=True)
def _gmail_provider_mounted() -> None:
    """These tools reach the mailbox through the provider registry (M5.7 track A);
    the real Gmail provider is registered so the ``gf.*`` patches below are what runs."""
    from iris_personal.email.providers import clear_mail_providers, register_mail_provider
    from iris_personal.plugins.gmail.provider import GmailProvider

    clear_mail_providers()
    register_mail_provider(GmailProvider())
    yield  # type: ignore[misc]
    clear_mail_providers()


class _StubTierRouter:
    def get_llm_config(self, intent: str):
        from iris_harness.llm.client import CodingLLMConfig

        return CodingLLMConfig(
            provider="github",
            model="stub",
            base_url="http://localhost",
            api_key_env="STUB",
            temperature=0.0,
            max_tokens=64,
            timeout_seconds=10,
        )


def _patch_llm_invoke(monkeypatch, responses: list[str]) -> None:
    iterator = iter(responses)

    def fake_invoke(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        stop: object = None,  # added 2026-05-19 (commit 1aa9e47) — ignored by the stub
    ) -> str:
        return next(iterator)

    from iris_harness.llm.client import CodingLLMClient

    monkeypatch.setattr(CodingLLMClient, "invoke", fake_invoke)


class _Spy:
    """Records every PRE_TOOL_USE context the kernel fires; allows each call."""

    name = "spy"
    priority = 1
    hook_point = HookPoint.PRE_TOOL_USE

    def __init__(self, outcome: str = "allow") -> None:
        self.outcome = outcome
        self.seen: list[HookContext] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.seen.append(ctx)
        return HookDecision(outcome=self.outcome, reason="spy")  # type: ignore[arg-type]


def _tools(
    data_dir: Path,
    *,
    spy: _Spy | None = None,
    continuations: object = None,
    session_id: str = "",
) -> BoundTools:
    """The email tools as the plugin's ``api.tools`` sees them: the pool the plugin
    registers, run through a governed ``ToolService`` as ``plugin:email_workflows``."""
    from iris_harness.llm.narrate import make_narrative_llm_call
    from iris_harness.runtime.tool_service import ToolService
    from iris_personal.email.agent_tools import build_email_tools

    summarize = make_narrative_llm_call(
        _StubTierRouter(), intent="communication", max_tokens=512  # type: ignore[arg-type]
    )
    pool = build_email_tools(
        data_dir=data_dir,
        llm_call=None,
        summarize_llm=summarize,
        current_session_id=lambda: session_id,
        continuations=continuations,
    )
    kernel = GovernanceKernel(audit_log=None)
    kernel.register(spy or _Spy())
    kernel.init_lock()
    return ToolService(tools=lambda: pool, kernel=lambda: kernel).for_caller(
        "plugin:email_workflows"
    )


def _fallback(data_dir: Path, **kw: Any):  # type: ignore[no-untyped-def]
    return make_email_fallback_handler(_tools(data_dir, **kw), data_dir=data_dir, llm_call=None)


def test_the_fallback_reads_an_explicit_sender_subject_request(tmp_path: Path, monkeypatch) -> None:
    import iris_personal.plugins.gmail.gmail_fetch as gf

    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(
        EmailMessage(
            id="rh1",
            provider="gmail",  # type: ignore[arg-type]
            account_id="gmail:user@gmail.com",
            thread_id=None,
            from_address="noreply@robinhood.com",
            to=("user@gmail.com",),
            subject="Payment reminder: Interest charged",
            received_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
            snippet="Your monthly interest payment is due.",
            attachments=(),
        )
    )
    monkeypatch.setattr(
        gf,
        "fetch_message_body",
        lambda _account, _message, **_kwargs: "Robinhood says the payment reminder is for margin interest charged this month.",
    )
    _patch_llm_invoke(
        monkeypatch, ["It says this is a Robinhood margin-interest payment reminder."]
    )

    handler = _fallback(tmp_path)
    text, meta = handler(
        AgentTask(
            query='can you read the emails from noreply@robinhood.com, specifically with subjects containing "Payment reminder:" and see what is in the email body',
            agent_type="email",
        )
    )

    assert "Payment reminder: Interest charged" in text
    assert "Robinhood margin-interest payment reminder" in text
    assert meta["read_email_recall"] is True


def test_the_fallback_carries_a_sender_subject_request_across_a_yes_followup(
    tmp_path: Path, monkeypatch
) -> None:
    import iris_personal.plugins.gmail.gmail_fetch as gf

    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(
        EmailMessage(
            id="rh1",
            provider="gmail",  # type: ignore[arg-type]
            account_id="gmail:user@gmail.com",
            thread_id=None,
            from_address="noreply@robinhood.com",
            to=("user@gmail.com",),
            subject="Payment reminder: Interest charged",
            received_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
            snippet="Your monthly interest payment is due.",
            attachments=(),
        )
    )
    monkeypatch.setattr(
        gf,
        "fetch_message_body",
        lambda _account, _message, **_kwargs: "Robinhood says the payment reminder is for margin interest charged this month.",
    )
    _patch_llm_invoke(
        monkeypatch, ["It says this is a Robinhood margin-interest payment reminder."]
    )

    handler = _fallback(tmp_path)
    text, meta = handler(
        AgentTask(
            query="yes please do it",
            agent_type="email",
            memory_context=MemoryContext(
                recent_turns=(
                    'user: can you read the emails from noreply@robinhood.com, specifically with subjects containing "Payment reminder:" and see what is in the email body',
                    'assistant: The user wants specific emails from Robinhood containing "Payment reminder:". I need to fetch and read these emails directly.',
                    "user: yes please do it",
                )
            ),
        )
    )

    assert "Payment reminder: Interest charged" in text
    assert "Robinhood margin-interest payment reminder" in text
    assert meta["read_email_recall"] is True
    assert "noreply@robinhood.com" in str(meta["effective_query"])


def test_a_pick_from_the_shortlist_reads_the_email_that_was_on_screen(
    tmp_path: Path, monkeypatch
) -> None:
    """The live bug, end to end on the agent. Two emails share a subject; the agent
    shows a numbered shortlist; the user replies "just go with the 1 st one". Before,
    the reply was rebuilt from the previous turn by regex and either re-showed the
    shortlist or searched the inbox for "1". Now the shortlist is a ``choice``
    continuation and the pick arrives resolved on ``AgentTask.selected_choice``."""
    import iris_personal.plugins.gmail.gmail_fetch as gf
    from iris_harness.memory.state.continuations import ContinuationStore
    from iris_harness.runtime.continuations import ContinuationRegistry, reads_as_choice

    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    now = datetime.now(UTC)
    for message_id in ("reg-a", "reg-b"):
        store.upsert(
            EmailMessage(
                id=message_id,
                provider="gmail",  # type: ignore[arg-type]
                account_id="gmail:user@gmail.com",
                thread_id=None,
                from_address='"Coach" <coach@example.org>',
                to=("user@gmail.com",),
                subject="Fall 2 Registration Links :)",
                received_at=now,
                snippet="Registration is open.",
                attachments=(),
            )
        )
    monkeypatch.setattr(
        gf,
        "fetch_message_body",
        lambda _account, message_id, **_kwargs: f"Body of {message_id}: links inside.",
    )
    registry = ContinuationRegistry(store=ContinuationStore(db_path=tmp_path / "cp.db"))
    # The pool's session is the harness's per-turn one (services.current_session_id).
    handler = _fallback(tmp_path, continuations=registry, session_id="web-1")

    shortlist, _meta = handler(
        AgentTask(
            query='read the email with subject containing "Fall 2 Registration Links"',
            agent_type="email",
            session_id="web-1",
        )
    )
    assert "Which one should I read?" in shortlist

    pending = registry.pending("web-1")
    assert pending is not None and pending.kind == "choice"
    index = reads_as_choice(pending, "just go with the 1 st one")
    assert index == 0
    picked = pending.choices[index]

    _patch_llm_invoke(monkeypatch, [f"Summary of {picked['message_id']}."])
    text, meta = handler(
        AgentTask(
            query="just go with the 1 st one",
            agent_type="email",
            session_id="web-1",
            selected_choice=picked,
        )
    )

    assert "Fall 2 Registration Links" in text
    assert f"Summary of {picked['message_id']}." in text
    assert meta["selected_choice"] is True
    assert "multiple emails" not in text.lower()


def test_a_bare_number_is_not_rebuilt_from_the_previous_turn() -> None:
    """With no shortlist on record there is nothing for "1" to pick, so the agent must
    not quietly re-run the previous ask — the regex carry that dropped the number."""
    from iris_personal.plugins.email_workflows.agent import _effective_email_query

    task = AgentTask(
        query="1",
        agent_type="email",
        memory_context=MemoryContext(
            recent_turns=(
                'user: I need the one with subject "Fall 2 Registration Links :)"',
                "user: 1",
            )
        ),
    )
    assert _effective_email_query(task) == "1"


def _msg(msg_id: str, sender: str, subject: str, received: datetime) -> EmailMessage:
    return EmailMessage(
        id=msg_id,
        provider="gmail",  # type: ignore[arg-type]
        account_id="gmail:user@gmail.com",
        thread_id=None,
        from_address=sender,
        to=("user@gmail.com",),
        subject=subject,
        received_at=received,
        snippet=f"{subject} (snippet)",
        attachments=(),
    )


def _seed_judged_inbox(tmp_path: Path) -> None:
    """Today's statement (judged bill), an older one, a bank alert and a newsletter
    (judged fyi), a bill the owner corrected into the bucket, and one not judged yet."""
    from iris_personal.plugins.email_workflows.judgments import JudgmentStore

    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    now = datetime.now(UTC)
    rows = [
        ("stmt-today", "alerts@bank.example", "Card statement for Jul-2026", now, "bill"),
        (
            "stmt-old",
            "alerts@bank.example",
            "Card statement for Jun-2026",
            now - timedelta(days=3),
            "bill",
        ),
        ("alert-today", "alerts@bank.example", "Transaction alert: card used", now, "fyi"),
        ("news-today", "news@example.com", "AI newsletter", now, "fyi"),
        ("fixed-today", "billing@utility.example", "Your utility bill", now, "fyi"),
    ]
    judgments = JudgmentStore(db_path=store.db_path)
    judgments.ensure_schema()
    for msg_id, sender, subject, received, bucket in rows:
        store.upsert(_msg(msg_id, sender, subject, received))
        judgments.record(
            message_id=msg_id, account_id="gmail:user@gmail.com", bucket=bucket, confidence=0.9
        )
    judgments.correct("fixed-today", "bill", source="chat")  # the owner: "that's a bill"
    store.upsert(
        _msg(
            "new-today",
            "alerts@bank.example",
            "Card statement for Aug-2026",
            now,
        )
    )
    judgments.mark_waiting("gmail:user@gmail.com", ["new-today"])


def test_bill_emails_come_from_the_judges_bill_bucket(tmp_path: Path) -> None:
    _seed_judged_inbox(tmp_path)
    handler = _fallback(tmp_path)
    text, meta = handler(AgentTask(query="any bill emails today?", agent_type="email"))

    assert "bill email" in text
    assert "Card statement for Jul-2026" in text  # judged bill, today
    assert "Your utility bill" in text  # the owner's correction wins
    assert "Card statement for Jun-2026" not in text  # outside "today"
    assert "Transaction alert" not in text and "AI newsletter" not in text  # not bills
    assert "Aug-2026" not in text  # held for the judge: never shown...
    assert "1 newer email(s)" in text  # ...but counted, so "no bill" is never implied
    assert meta["bill_email_recall"] is True
    assert meta["being_sorted_count"] == 1


def test_with_no_bill_judged_there_is_no_bill_answer(tmp_path: Path) -> None:
    from iris_personal.plugins.email_workflows.agent import _bill_email_digest

    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(_msg("news", "news@example.com", "AI newsletter", datetime.now(UTC)))
    assert _bill_email_digest("any bill emails today?", data_dir=tmp_path) is None


def test_the_email_slice_imports_nothing_from_finance() -> None:
    """Email stands on its own (it ships publicly in release 1); finance depends on email
    as one of its sources, never the reverse."""
    import re

    pattern = re.compile(
        r"^\s*(?:from|import)\s+iris_personal\.(?:finance|plugins\.finance_workflows)\b",
        re.M,
    )
    roots = ("email", "connections", "plugins/email_workflows", "plugins/gmail")
    offenders = [
        str(path)
        for root in roots
        for path in Path("src/iris_personal", root).rglob("*.py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_anything_else_is_the_digest_with_no_model_call(tmp_path: Path, monkeypatch) -> None:
    """The fallback is the floor under the loop, not a second loop: a plain inbox ask is
    the deterministic digest, and no model is asked to pick a tool (email slice 5)."""
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(_msg("d1", "deals@shop.example", "Weekend sale", datetime.now(UTC)))
    calls: list[str] = []

    def fake_invoke(self, *, system_prompt: str, user_prompt: str, stop: object = None) -> str:
        calls.append(user_prompt)
        return "Thought: digest.\nAction: inbox_digest\nAction Input: {}"

    from iris_harness.llm.client import CodingLLMClient

    monkeypatch.setattr(CodingLLMClient, "invoke", fake_invoke)
    handler = _fallback(tmp_path)
    text, meta = handler(AgentTask(query="summarize my latest emails", agent_type="email"))

    assert "Weekend sale" in text
    assert meta["email_capability_registered"] is True and "agentic_core" not in meta
    assert calls == []


@pytest.mark.parametrize(
    ("query", "tool"),
    [
        ('read the email with subject containing "Weekend sale"', "read_email"),
        ("find my boarding pass pdf in my email", "find_attachment"),
    ],
)
def test_every_tool_the_fallback_runs_goes_through_the_governed_runner(
    tmp_path: Path, query: str, tool: str
) -> None:
    """No direct ``ToolSpec.call``: PRE_TOOL_USE fires for each read, as the plugin."""
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(_msg("d1", "deals@shop.example", "Weekend sale", datetime.now(UTC)))
    spy = _Spy()
    _fallback(tmp_path, spy=spy)(AgentTask(query=query, agent_type="email"))

    assert [(c.payload["tool_name"], c.metadata["caller"]) for c in spy.seen] == [
        (tool, "plugin:email_workflows")
    ]


def test_a_pick_is_read_through_the_governed_runner(tmp_path: Path) -> None:
    spy = _Spy()
    _fallback(tmp_path, spy=spy)(
        AgentTask(query="1", agent_type="email", selected_choice={"message_id": "x1"})
    )
    assert [(c.payload["tool_name"], c.payload["args"]) for c in spy.seen] == [
        ("read_email", {"message_id": "x1"})
    ]
    assert spy.seen[0].metadata["caller"] == "plugin:email_workflows"


def test_a_read_governance_refuses_is_said_then_the_digest(tmp_path: Path) -> None:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(_msg("d1", "deals@shop.example", "Weekend sale", datetime.now(UTC)))
    text, meta = _fallback(tmp_path, spy=_Spy("deny"))(
        AgentTask(query='read the email with subject containing "Weekend sale"', agent_type="email")
    )
    assert text.startswith("I couldn't run read_email for this:")
    assert "Weekend sale" in text  # the digest follows, from the local store
    assert meta["tool_held"] is True and "read_email_recall" not in meta
