"""One ULID call id per call attempt, on every audit row of the call (issue #134, stage 1).

Driven through the real governed harness on both entry points (``chat`` and
``chat_stream``, the REPL's and the web's path), reading the real ledgers: every
``PRE_TOOL_USE`` / ``POST_TOOL_USE`` row of a tool call carries the same minted
``call_id``; the session log's ``tool.invoke.*`` events and the side-effect ledger key carry
it too; no caller can choose it; and a held call's id reaches the approval row and the
approved re-execution (a new call with its own id) as ``held_call_id``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from iris_harness.foundation.ids import is_ulid
from iris_harness.kernel.governance.approvals import ApprovalQueue
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.side_effects import SideEffectLedger
from iris_harness.sdk import PluginAPI
from iris_harness.testing import harness, plugin

_ECHO_SCRIPT: dict[str, Any] = {
    "rules": [
        {
            "name": "answer from the tool",
            "match": {"user": r"(?s)Observation:.*echo: banana"},
            "reply": {"content": "Thought: Done.\nFinal Answer: The tool said banana."},
        },
        {
            "name": "call the tool",
            "match": {"user": r"User: Please echo the word banana"},
            "reply": {
                "content": "Thought: Use the tool.\nAction: echo_back\n"
                # A model (or anyone upstream of it) trying to name its own call.
                'Action Input: {"word": "banana", "call_id": "FORGED-1", '
                '"tool_call_id": "FORGED-2"}'
            },
        },
    ]
}


def _echo_plugin() -> Any:
    def setup(api: PluginAPI) -> None:
        api.register_tool(
            "echo_back",
            'Repeat the given word back. Args: {"word": str}.',
            lambda args: f"echo: {args.get('word', '')}",
        )

    return plugin(
        setup,
        manifest={"name": "echo", "provides": ["tool"], "tools": {"echo_back": {"effect": "read"}}},
    )


def _rows(h: Any) -> list[tuple[Any, dict[str, Any]]]:
    return [(r, json.loads(r.payload_json)) for r in AuditLog(db_path=h.audit_db).query()]


def _tool_rows(h: Any, tool: str) -> list[tuple[Any, dict[str, Any]]]:
    return [
        (r, p)
        for r, p in _rows(h)
        if r.hook_point in ("pre_tool_use", "post_tool_use") and p.get("tool_name") == tool
    ]


def _chat(entry: str, h: Any, message: str) -> Any:
    if entry == "chat":
        return h.chat(message)
    return h.chat_stream(message)


def _session_events(h: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in (h.home / "logs").glob("session-*.jsonl"):
        events += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return events


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_every_row_of_a_tool_call_carries_one_ulid_and_other_rows_none(entry: str) -> None:
    with harness(plugins=[_echo_plugin()], fake_model=_ECHO_SCRIPT) as h:
        _chat(entry, h, "Please echo the word banana")
        mine = _tool_rows(h, "echo_back")
        everything = _rows(h)
        events = _session_events(h)
    # Both tool hook points fired, several hooks each (one row per hook firing).
    assert {r.hook_point for r, _ in mine} == {"pre_tool_use", "post_tool_use"}
    assert len(mine) >= 4
    ids = {p.get("call_id") for _, p in mine}
    assert len(ids) == 1, f"every row of one call names the same id, got {ids}"
    (call_id,) = ids
    assert is_ulid(call_id)
    assert call_id not in ("FORGED-1", "FORGED-2")  # a value in the arguments is never read
    # Only tool-call rows are about a call: model-call and turn rows name none (stage 2).
    for row, payload in everything:
        if row.hook_point not in ("pre_tool_use", "post_tool_use"):
            assert "call_id" not in payload, (row.hook_point, payload)
    # The old key is an alias for the session log's readers, with the same value ...
    invoke = [e for e in events if str(e.get("kind", "")).startswith("tool.invoke.")]
    assert {e["kind"] for e in invoke} == {"tool.invoke.start", "tool.invoke.end"}
    for event in invoke:
        assert event["payload"]["call_id"] == call_id
        assert event["payload"]["tool_call_id"] == call_id
    # ... and it is never in the audit rows (which have no reader of it).
    assert all("tool_call_id" not in p for _, p in mine)


def test_two_calls_have_two_ids_and_ascending_ones() -> None:
    script = {
        "rules": [
            {
                "name": "answer",
                "match": {"user": r"(?s)Observation:.*echo: banana"},
                "reply": {"content": "Thought: Done.\nFinal Answer: banana."},
            },
            {
                "name": "call",
                "match": {"user": r"User: Please echo the word banana"},
                "reply": {
                    "content": "Thought: Use it.\nAction: echo_back\n"
                    'Action Input: {"word": "banana"}'
                },
            },
        ]
    }
    with harness(plugins=[_echo_plugin()], fake_model=script) as h:
        h.chat("Please echo the word banana")
        h.chat("Please echo the word banana")
        ids = []
        for _, p in _tool_rows(h, "echo_back"):
            if p["call_id"] not in ids:
                ids.append(p["call_id"])
    assert len(ids) == 2 and ids == sorted(ids)


# ---------------------------------------------------------------------- held -> approved

_SHRED_SCRIPT = {
    "rules": [
        {
            "name": "route",
            "match": {"system": "request router"},
            "reply": {"json": {"intent": "general"}},
        },
        {
            "name": "done",
            "match": {"user": r"(?s)Observation: The owner approved.*shredded memo"},
            "reply": {"content": "Thought: Done.\nFinal Answer: The memo is shredded."},
        },
        {
            "name": "shred",
            "match": {"user": r"User: Shred the memo"},
            "reply": {
                "content": 'Thought: Shred it.\nAction: shred_doc\nAction Input: {"doc": "memo"}'
            },
        },
    ]
}


def _shredder() -> Any:
    def setup(api: PluginAPI) -> None:
        api.register_tool(
            "shred_doc",
            'Shred a document. Args: {"doc": str}.',
            lambda args: f"shredded {args.get('doc')}",
        )

    return plugin(
        setup,
        manifest={
            "name": "shredder",
            "provides": ["tool"],
            "tools": {"shred_doc": {"effect": "destructive"}},
        },
    )


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_held_call_keeps_its_id_and_the_approved_attempt_names_it(entry: str) -> None:
    with harness(plugins=[_shredder()], fake_model=_SHRED_SCRIPT) as h:
        result = _chat(entry, h, "Shred the memo")
        assert "needs your approval" in result.text
        [pending] = ApprovalQueue().list_pending()
        held_rows = _tool_rows(h, "shred_doc")
        queue_rows = [
            p for r, p in _rows(h) if r.hook_point == "approval_queue" and r.decision != "x"
        ]
        held_ids = {p["call_id"] for _, p in held_rows}
        assert len(held_ids) == 1
        (held_id,) = held_ids
        assert is_ulid(held_id)
        # The approval row (the one new column) and its audit rows carry the held id.
        assert ApprovalQueue().get(pending.approval_id).call_id == held_id  # type: ignore[union-attr]
        assert queue_rows and all(p.get("call_id") == held_id for p in queue_rows)

        told = h.respond_to_approval(pending.approval_id, approve=True)
        assert "shredded" in told
        after = _tool_rows(h, "shred_doc")
        approved_rows = [(r, p) for r, p in after if p["call_id"] != held_id]
        approved_ids = {p["call_id"] for _, p in approved_rows}
        # The approved re-execution is a NEW call: its own ULID, joined to the held one.
        assert len(approved_ids) == 1
        (approved_id,) = approved_ids
        assert is_ulid(approved_id) and approved_id > held_id
        assert approved_rows and all(p.get("held_call_id") == held_id for _, p in approved_rows)
        # The held attempt's rows never name a held id (they were not a re-execution).
        assert all("held_call_id" not in p for _, p in held_rows)
        # Both attempts' queue rows after the answer still name the held attempt.
        answered = [p for r, p in _rows(h) if r.hook_point == "approval_queue" and "actor" in p]
        assert answered and all(p.get("call_id") == held_id for p in answered)
        # The side-effect ledger row of the executed attempt is keyed by the approved id.
        run_id = approved_rows[0][0].run_id
        keys = [
            row.side_effect_id
            for row in SideEffectLedger(h.home / "governance" / "side_effects.db").list_by_run(
                run_id
            )
        ]
        assert keys and all(k.endswith(f":{approved_id}") for k in keys)
        assert not any(k.endswith(f":{held_id}") for k in keys)


# --------------------------------------------- stage 2: parent, attempt, turn, resumed run


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_every_row_of_a_turn_names_the_turn_and_two_turns_differ(entry: str) -> None:
    with harness(plugins=[_echo_plugin()], fake_model=_ECHO_SCRIPT) as h:
        _chat(entry, h, "Please echo the word banana")
        _chat(entry, h, "Please echo the word banana")
        rows = _rows(h)
        tool_rows = _tool_rows(h, "echo_back")
        typed = h.audit_rows(hook_point="pre_tool_use")
    turn_ids = {
        p.get("turn_id") for r, p in rows if r.hook_point in ("pre_tool_use", "pre_llm_call")
    }
    assert None not in turn_ids and len(turn_ids) == 2  # both hook kinds, two turns
    tool_turns = {p["turn_id"] for _, p in tool_rows}
    assert tool_turns <= turn_ids and len(tool_turns) == 2
    # A top-level model call has no parent, is attempt 1, and the stable row says the same.
    mine = [r for r in typed if r.tool == "echo_back"]
    assert mine and all(r.parent_call_id is None and r.attempt == 1 for r in mine)
    assert all(r.replay_of is None and r.held_call_id is None for r in mine)
    assert all(is_ulid(r.call_id) and r.turn_id in turn_ids for r in mine)


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_the_approved_attempt_is_attempt_two_replaying_the_held_one(entry: str) -> None:
    with harness(plugins=[_shredder()], fake_model=_SHRED_SCRIPT) as h:
        _chat(entry, h, "Shred the memo")
        [pending] = ApprovalQueue().list_pending()
        held_rows = _tool_rows(h, "shred_doc")
        held_ids = {p["call_id"] for _, p in held_rows}
        (held_id,) = held_ids
        run_id = held_rows[0][0].run_id
        h.respond_to_approval(pending.approval_id, approve=True)
        after = _tool_rows(h, "shred_doc")
        approved = [(r, p) for r, p in after if p["call_id"] != held_id]
        typed = [r for r in h.audit_rows(hook_point="pre_tool_use") if r.tool == "shred_doc"]
    assert approved
    assert all(p["attempt"] == 2 and p["replay_of"] == held_id for _, p in approved)
    assert all(p["attempt"] == 1 and "replay_of" not in p for _, p in held_rows)
    # The held attempt was written before the run was re-entered; the rows the resumed run
    # wrote say so, with the run id (which survives the halt).
    assert all("resumed_from_run" not in p for _, p in held_rows)
    assert all(p.get("resumed_from_run") == run_id for _, p in approved)
    by_call = {r.call_id: r for r in typed}
    assert by_call[held_id].attempt == 1 and by_call[held_id].replay_of is None
    (replay,) = [r for cid, r in by_call.items() if cid != held_id]
    assert replay.attempt == 2 and replay.replay_of == held_id and replay.held_call_id == held_id
    assert replay.resumed_from_run == run_id


# ----------------------------------------------- stage 3: the identity is in real columns


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_every_audit_row_has_its_own_record_id_and_the_identity_in_columns(entry: str) -> None:
    with harness(plugins=[_echo_plugin()], fake_model=_ECHO_SCRIPT) as h:
        _chat(entry, h, "Please echo the word banana")
        rows = AuditLog(db_path=h.audit_db).query()
        tool_rows = [r for r in rows if r.hook_point in ("pre_tool_use", "post_tool_use")]
        mine = [r for r in tool_rows if json.loads(r.payload_json).get("tool_name") == "echo_back"]
    assert rows and all(r.record_id and is_ulid(r.record_id) for r in rows)
    assert len({r.record_id for r in rows}) == len(rows)  # one per row, none shared
    assert mine
    for row in mine:  # the columns say what the payload says, from one source
        payload = json.loads(row.payload_json)
        assert row.call_id == payload["call_id"] and is_ulid(row.call_id)
        assert row.session_id == payload["session_id"] and row.turn_id == payload["turn_id"]
        assert row.attempt == 1 and row.parent_call_id is None
    assert len({r.call_id for r in mine}) == 1 and len({r.turn_id for r in mine}) == 1
    assert all(r.session_id for r in rows if r.hook_point == "pre_llm_call")


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_the_approval_row_names_the_step_and_turn_and_the_ledger_row_the_call(
    entry: str,
) -> None:
    import sqlite3

    with harness(plugins=[_shredder()], fake_model=_SHRED_SCRIPT) as h:
        _chat(entry, h, "Shred the memo")
        [pending] = ApprovalQueue().list_pending()
        held_rows = _tool_rows(h, "shred_doc")
        (held_id,) = {p["call_id"] for _, p in held_rows}
        held_turn = held_rows[0][1]["turn_id"]
        queue = ApprovalQueue().get(pending.approval_id)
        h.respond_to_approval(pending.approval_id, approve=True)
        approved = [
            r
            for r in AuditLog(db_path=h.audit_db).query()
            if r.call_id and r.call_id != held_id and r.hook_point == "pre_tool_use"
        ]
        run_id = approved[0].run_id
        ledger_db = h.home / "governance" / "side_effects.db"
        conn = sqlite3.connect(ledger_db)
        ledger = conn.execute(
            "SELECT side_effect_id, call_id, attempt, replay_of FROM side_effect_ledger "
            "WHERE run_id = ?",
            (run_id,),
        ).fetchall()
        events = conn.execute(
            "SELECT status FROM side_effect_events WHERE call_id = ? ORDER BY ts, event_id",
            (approved[0].call_id,),
        ).fetchall()
        conn.close()
    # The approval says where it was raised: the loop step and the turn of the held attempt.
    assert queue is not None and queue.call_id == held_id
    assert queue.step_id is not None and queue.turn_id == held_turn
    # The approved attempt is attempt 2 in real columns, and its ledger row carries the call.
    assert {r.attempt for r in approved} == {2} and {r.replay_of for r in approved} == {held_id}
    ((key, call_id, attempt, replay_of),) = ledger
    assert key.endswith(f":{call_id}") and call_id == approved[0].call_id  # key format kept
    assert (attempt, replay_of) == (2, held_id)
    assert events == [("pending",), ("completed",)]
