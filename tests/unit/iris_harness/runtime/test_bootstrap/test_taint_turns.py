"""A write after an external read waits for the owner on a real chat turn (issue #149).

``rt.chat`` and ``rt.chat_stream`` share one pipeline (the REPL uses ``/chat/stream``), so
each is driven here through the governed harness: the composition root, a scripted model, a
plugin tool that declares ``content: external`` and a write tool the taint policy lists. The
write is held for approval when the turn read the page first, runs when it did not, and the
card the owner sees says why.
"""

from __future__ import annotations

from typing import Any

import pytest

from iris_harness.kernel.governance.approvals import ApprovalQueue
from iris_harness.kernel.governance.taint_policy import TAINT_REASON
from iris_harness.sdk import PluginAPI
from iris_harness.testing import harness, plugin

FETCH_ASK = "Please fetch the page and save a note"
SAVE_ONLY = "Please just save a note"

_SCRIPT: dict[str, Any] = {
    "rules": [
        {
            "name": "answer after the save",
            "match": {"user": r"(?s)(Observation: saved|approved.*saved)"},
            "reply": {"content": "Thought: Done.\nFinal Answer: Saved."},
        },
        {
            "name": "save after the page",
            "match": {"user": r"(?s)Observation:.*Weather in Oslo is mild"},
            "reply": {
                "content": 'Thought: Save it.\nAction: save_note\nAction Input: {"text": "mild"}'
            },
        },
        {
            "name": "fetch the page",
            "match": {"user": r"User: Please fetch the page and save a note"},
            "reply": {"content": "Thought: Fetch it.\nAction: fetch_page\nAction Input: {}"},
        },
        {
            "name": "save directly",
            "match": {"user": r"User: Please just save a note"},
            "reply": {
                "content": 'Thought: Save it.\nAction: save_note\nAction Input: {"text": "hi"}'
            },
        },
    ]
}


def _plugin(saved: list[str]) -> Any:
    def setup(api: PluginAPI) -> None:
        api.register_tool(
            "fetch_page", "Fetch a page. Args: {}.", lambda a: "Weather in Oslo is mild."
        )

        def save(args: dict[str, Any]) -> str:
            saved.append(str(args.get("text")))
            return "saved"

        api.register_tool("save_note", 'Save a note. Args: {"text": str}.', save)

    return plugin(
        setup,
        manifest={
            "name": "notes",
            "provides": ["tool"],
            "tools": {
                "fetch_page": {"effect": "read", "content": "external"},
                "save_note": {"effect": "write", "confirm": "never"},
            },
        },
    )


@pytest.fixture(autouse=True)
def _policy_lists_the_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "iris_harness.agent.tool_runner.taint_gated_tools", lambda: frozenset({"save_note"})
    )


def _ask(entry: str, h: Any, question: str) -> str:
    if entry == "chat":
        return str(h.chat(question).text)
    turn = h.chat_stream(question)
    assert turn.answered
    return str(turn.text)


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_write_after_the_page_is_held_for_approval_on_both_entries(entry: str) -> None:
    saved: list[str] = []
    with harness(plugins=[_plugin(saved)], fake_model=_SCRIPT) as h:
        text = _ask(entry, h, FETCH_ASK)
        pending = ApprovalQueue().list_pending()

    assert saved == []  # the note was not written
    assert "needs your approval" in text
    assert len(pending) == 1
    assert pending[0].items[0].tool == "save_note"
    assert TAINT_REASON in pending[0].context_summary


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_the_same_write_with_no_outside_read_runs_on_both_entries(entry: str) -> None:
    saved: list[str] = []
    with harness(plugins=[_plugin(saved)], fake_model=_SCRIPT) as h:
        text = _ask(entry, h, SAVE_ONLY)
        pending = ApprovalQueue().list_pending()

    assert saved == ["hi"] and pending == []
    assert "Saved." in text


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_the_approved_write_runs_once_when_the_owner_answers(entry: str) -> None:
    saved: list[str] = []
    with harness(plugins=[_plugin(saved)], fake_model=_SCRIPT) as h:
        _ask(entry, h, FETCH_ASK)
        (pending,) = ApprovalQueue().list_pending()
        assert saved == []

        h.respond_to_approval(str(pending.approval_id), approve=True)

    assert saved == ["mild"]  # exactly the call the card showed, once
