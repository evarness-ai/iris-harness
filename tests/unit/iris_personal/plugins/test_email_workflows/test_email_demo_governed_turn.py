"""R14 on a real chat turn: the email profile, the demo mailbox, the scripted model.

``iris email demo`` proves every model call of fetch -> judge -> digest is audited, but
it never runs a chat turn. This does, through ``iris_harness.testing.harness`` -- the
composition root, not a look-alike: the ``email`` profile's plugins mount, the demo
mailbox is supplied in-process, and one question goes through the whole pipeline
(PRE_TURN screen, routing, the email agent's governed loop, a tool call, the curator,
PRE_RESPONSE). The invariant (OSS plan R14): every model call and every answer has an
audit row.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from iris_harness.sdk import PluginAPI
from iris_harness.testing import harness, plugin
from iris_personal.email.providers import clear_mail_providers

QUESTION = "Which emails need a reply about the carpool?"
ANSWER = "Aisha Patel is waiting on your reply about the carpool swap this week."

# The email agent's loop, scripted: search the inbox, then answer from what it found.
# No default rule: a model call the script does not expect fails the turn instead of
# being answered by something the test never looked at.
SCRIPT: dict[str, Any] = {
    "rules": [
        {
            "name": "email loop / answer from what the tool found",
            "match": {"user": r"(?s)Observation:.*Carpool swap"},
            "reply": {"content": f"Thought: The search found it.\nFinal Answer: {ANSWER}"},
        },
        {
            "name": "email loop / search the inbox",
            "match": {"user": r"User: Which emails need a reply about the carpool\?"},
            "reply": {
                "content": "Thought: I should search the inbox.\n"
                "Action: search_inbox\n"
                'Action Input: {"query": "carpool"}'
            },
        },
    ]
}


def _demo_mailbox(api: PluginAPI) -> None:
    """The demo mailbox as a plugin: its provider, its account, its mail fetched."""
    from iris_personal.email.accounts import EmailAccountStore
    from iris_personal.email.providers import register_mail_provider
    from iris_personal.plugins.email_workflows.demo.corpus import OWNER_ADDRESS
    from iris_personal.plugins.email_workflows.demo.provider import (
        DEMO_ACCOUNT,
        DEMO_PROVIDER,
        DemoMailProvider,
    )

    provider = DemoMailProvider()
    register_mail_provider(provider)
    accounts = EmailAccountStore()
    accounts.ensure_schema()
    if accounts.get(DEMO_ACCOUNT) is None:
        accounts.add(provider=DEMO_PROVIDER, address=OWNER_ADDRESS)
    provider.fetch_new(DEMO_ACCOUNT, max_messages=1000)


@pytest.fixture(autouse=True)
def _no_leftover_providers() -> Iterator[None]:
    clear_mail_providers()
    yield
    clear_mail_providers()


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_governed_email_turn_audits_every_model_call_and_the_answer(entry: str) -> None:
    demo = plugin(_demo_mailbox, name="demo_mailbox")
    with harness(profile="email", plugins=[demo], fake_model=SCRIPT) as h:
        assert h.plugin_loaded("email_workflows")
        assert h.plugin_loaded("demo_mailbox")

        if entry == "chat":
            result = h.chat(QUESTION)
        else:
            result = h.chat_stream(QUESTION)
            assert result.events[-1].kind == "done", result.error
        session = h.turns[-1].session_id

        assert result.text == ANSWER
        assert result.agent == "email"
        # Both scripted calls ran, on the local fake, and nothing else reached a model.
        calls = h.model_calls()
        assert [c.rule for c in calls] == [
            "email loop / search the inbox",
            "email loop / answer from what the tool found",
        ]

        # An LLM-call row per model call: each loop step fires PRE_LLM_CALL under its
        # own (run_id, step_id).
        llm_rows = h.audit_rows(hook_point="pre_llm_call", session_id=session)
        assert len({(r.run_id, r.step_id) for r in llm_rows}) == len(calls)
        # The inbox the tool read lifted the second call's label: personal, still local.
        second = [r for r in llm_rows if r.step_id == 1]
        assert second and {r.classification for r in second} == {"personal"}
        assert all(r.decision == "allow" for r in llm_rows)

        # The tool call itself went through the kernel.
        tool_rows = h.audit_rows(hook_point="pre_tool_use", session_id=session)
        assert any(r.tool == "search_inbox" for r in tool_rows)

        # The answer row: the model-free response check, at PRE_RESPONSE.
        answer_rows = h.audit_rows(hook_point="pre_response", session_id=session)
        assert any(r.plugin == "response_safety" for r in answer_rows)

        assert h.audit_gaps() == []
