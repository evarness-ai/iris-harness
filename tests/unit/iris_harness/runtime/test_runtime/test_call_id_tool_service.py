"""``ToolService`` calls and the code caller's approved call carry minted call ids (#134).

``api.tools.call`` (``plugin:<name>``), the core's ``core:<workflow>`` and an MCP client's
``mcp:<client>`` all pass the one runner: each call is a new attempt with its own ULID on
every audit row, and the caller can neither supply nor omit it (``ToolCall`` has no such
field; a ``call_id`` argument is only an argument). An approved code-caller call is a NEW
call that records the held attempt's id as ``held_call_id``, joined through the approval row.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.tool_runner import ToolCall
from iris_harness.foundation.ids import is_ulid
from iris_harness.kernel.governance.caller_policy import register_caller_policy

from .test_approved_call_executor import _World


@pytest.fixture()
def world(tmp_path: Path) -> Iterator[_World]:
    w = _World(tmp_path)
    yield w
    register_caller_policy(None)


def _rows(w: _World) -> list[tuple[Any, dict[str, Any]]]:
    return [
        (r, json.loads(r.payload_json))
        for r in w.audit.query()
        if r.hook_point in ("pre_tool_use", "post_tool_use")
    ]


def test_toolcall_has_no_field_a_caller_could_use_to_name_a_call() -> None:
    names = {f.name for f in dataclasses.fields(ToolCall)}
    assert not names & {"call_id", "tool_call_id"}
    with pytest.raises(TypeError):
        ToolCall(call_id="mine")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ToolCall(tool_call_id="mine")  # type: ignore[call-arg]


@pytest.mark.parametrize("caller", ["plugin:p", "core:digest"])
def test_a_tool_call_from_code_gets_a_fresh_ulid_every_time(world: _World, caller: str) -> None:
    world.tools.append(world._tool("look_up", "read", "never"))
    for _ in range(2):
        result = world.service.for_caller(caller).call(
            "look_up", {"v": 1, "call_id": "FORGED", "tool_call_id": "FORGED"}
        )
        assert result.ok
    rows = _rows(world)
    by_run: dict[str, set[str]] = {}
    for row, payload in rows:
        assert is_ulid(payload.get("call_id")), payload
        by_run.setdefault(row.run_id, set()).add(payload["call_id"])
    assert len(by_run) == 2 and all(len(v) == 1 for v in by_run.values())
    ids = {next(iter(v)) for v in by_run.values()}
    assert len(ids) == 2 and "FORGED" not in ids


def test_a_held_code_call_keeps_its_id_on_the_approval_row(world: _World) -> None:
    approval_id = world.queued("wipe")
    row = world.queue.get(approval_id)
    assert row is not None and is_ulid(row.call_id)
    held_ids = {p["call_id"] for _, p in _rows(world)}
    assert held_ids == {row.call_id}
    # The approval's own audit row names the held attempt too.
    enq = [
        json.loads(r.payload_json) for r in world.audit.query() if r.hook_point == "approval_queue"
    ]
    assert enq and all(p.get("call_id") == row.call_id for p in enq)


def test_the_approved_code_call_is_a_new_call_that_names_the_held_one(world: _World) -> None:
    approval_id = world.queued("wipe")
    held = world.queue.get(approval_id)
    assert held is not None and held.call_id
    held_id = held.call_id
    before = len(_rows(world))
    outcome = world.approve(approval_id, executor=world.service)
    assert outcome.executed and world.ran == [("wipe", {"text": "milk"})]

    approved = _rows(world)[before:]
    assert approved
    approved_ids = {p["call_id"] for _, p in approved}
    assert len(approved_ids) == 1
    (approved_id,) = approved_ids
    assert is_ulid(approved_id) and approved_id != held_id and approved_id > held_id
    assert all(p.get("held_call_id") == held_id for _, p in approved)
    # The approval's claim and outcome rows (queue rows) still name the held attempt.
    queue_rows = [
        json.loads(r.payload_json) for r in world.audit.query() if r.hook_point == "approval_queue"
    ]
    assert len(queue_rows) >= 3 and all(p.get("call_id") == held_id for p in queue_rows)
    # The ledger row of what actually ran is keyed by the approved attempt's id.
    run_id = approved[0][0].run_id
    keys = [r.side_effect_id for r in world.ledger.list_by_run(run_id)]
    assert keys == [f"{run_id}:0:{approved_id}"]
