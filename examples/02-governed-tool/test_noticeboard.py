"""The noticeboard's tools, run through the governed runner of a real IRIS, offline.

The model is scripted (``SCRIPT``): each rule matches what the loop sends and replies
the way a model would -- an ``Action`` to call a tool, or a ``Final Answer``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from noticeboard import Board, setup

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.approvals import ApprovalQueue
from iris_harness.testing import harness, plugin

MANIFEST = Path(__file__).with_name("manifest.yaml")

# Rules are tried in order. The router's call comes first (it picks the turn's intent),
# then the later steps of a run -- which carry an Observation -- before the first step.
SCRIPT: dict[str, Any] = {
    "rules": [
        {
            "name": "route",
            "match": {"system": "request router"},
            "reply": {"json": {"intent": "general"}},
        },
        {
            "name": "answer after the removal ran",
            "match": {"user": r"(?s)Observation:.*Removed n2"},
            "reply": {"content": "Thought: Done.\nFinal Answer: The dentist note is gone."},
        },
        {
            "name": "answer after the owner said no",
            "match": {"user": r"(?s)Observation: The owner rejected this"},
            "reply": {"content": "Thought: Leave it.\nFinal Answer: Understood, the note stays."},
        },
        {
            "name": "answer from the list",
            "match": {"user": r"(?s)Observation:.*Bake sale"},
            "reply": {
                "content": "Thought: I have the notes.\n"
                "Final Answer: There are 3 notes: a bake sale, the dentist and the printer."
            },
        },
        {
            "name": "remove the dentist note",
            "match": {"user": r"User: Remove the dentist note"},
            "reply": {
                "content": "Thought: That is note n2.\nAction: remove_note\n"
                'Action Input: {"note_id": "n2"}'
            },
        },
        {
            "name": "read the board",
            "match": {"user": r"User: What is on the noticeboard"},
            "reply": {"content": "Thought: Read the board.\nAction: list_notes\nAction Input: {}"},
        },
    ]
}


def _plugin(board: Board, keep: list[PluginAPI] | None = None) -> Any:
    def setup_with(api: PluginAPI) -> None:
        if keep is not None:
            keep.append(api)
        setup(api, board)

    return plugin(setup_with, manifest=MANIFEST)


def test_a_read_tool_call_is_checked_before_and_after_it_runs() -> None:
    board = Board.sample()
    with harness(plugins=[_plugin(board)], fake_model=SCRIPT) as h:
        result = h.chat("What is on the noticeboard?")

        assert result.text.startswith("There are 3 notes")
        session = result.session_id
        before = h.audit_rows(hook_point="pre_tool_use", session_id=session)
        after = h.audit_rows(hook_point="post_tool_use", session_id=session)
        # One row per governance check, each naming the tool; all of them allowed it. The
        # one exception is the floor's row on the result: ``list_notes`` is ``external``, so
        # the floor marks the result untrusted (a ``transform``) before the model reads it.
        assert before and {row.tool for row in before} == {"list_notes"}
        assert after and {row.tool for row in after} == {"list_notes"}
        assert all(row.decision == "allow" for row in before)
        for row in after:
            expected = "transform" if row.plugin == "external_content_floor" else "allow"
            assert row.decision == expected, (row.plugin, row.decision)
        assert any(row.plugin == "external_content_floor" for row in after)
        assert h.audit_gaps() == []


def test_a_destructive_call_waits_for_the_owner_then_runs() -> None:
    board = Board.sample()
    with harness(plugins=[_plugin(board)], fake_model=SCRIPT) as h:
        result = h.chat("Remove the dentist note, please.")

        # The turn stopped at the approval: nothing has changed yet.
        assert "needs your approval" in result.text
        assert "n2" in board.notes
        [pending] = ApprovalQueue().list_pending()
        assert pending.card is not None
        assert pending.card.title == "Remove 1 note"

        told = h.respond_to_approval(pending.approval_id, approve=True)

        # The approved call ran, and the resumed run answered.
        assert "n2" not in board.notes
        assert told == "The dentist note is gone."
        assert ApprovalQueue().list_pending() == []
        tool_rows = h.audit_rows(hook_point="pre_tool_use")
        assert {row.tool for row in tool_rows} == {"remove_note"}
        assert h.audit_gaps() == []


def test_a_rejected_call_never_runs() -> None:
    board = Board.sample()
    with harness(plugins=[_plugin(board)], fake_model=SCRIPT) as h:
        h.chat("Remove the dentist note, please.")
        [pending] = ApprovalQueue().list_pending()

        told = h.respond_to_approval(pending.approval_id, approve=False)

        # The run resumed to tell the owner, and nothing ran.
        assert told == "Understood, the note stays."
        assert "n2" in board.notes
        assert ApprovalQueue().list_pending() == []


def test_plugin_code_calls_a_tool_through_the_same_governance() -> None:
    board = Board.sample()
    kept: list[PluginAPI] = []
    with harness(plugins=[_plugin(board, keep=kept)], fake_model=SCRIPT) as h:
        [api] = kept
        assert api.tools is not None

        # A write from code has nobody to ask, so it is held for the owner's approval.
        held = api.tools.call("pin_note", {"note_id": "n3"})
        assert held.held and held.approval_id
        assert not board.notes["n3"].pinned

        h.respond_to_approval(held.approval_id, approve=True)

        assert board.notes["n3"].pinned
