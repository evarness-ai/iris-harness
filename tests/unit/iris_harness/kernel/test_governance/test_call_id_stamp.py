"""The kernel stamps the call id; nothing else can (#134, stage 1).

``kernel._audit`` writes ``call_id`` (and ``held_call_id`` on an approved re-execution) from
the context's kernel-stamped metadata only. A value in the payload, in the arguments or in a
hook's ``audit_metadata`` never becomes the row's call id; the old ``tool_call_id`` metadata
key is read as an alias; and the ledger key keeps its ``run:step:call`` shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iris_harness.kernel.governance import GovernanceKernel, HookContext, HookDecision, HookPoint
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.hooks.tool_payload import (
    CALL_ID,
    TOOL_CALL_ID,
    call_id_of,
    pre_tool_payload,
    tool_post_metadata,
)
from iris_harness.kernel.governance.plugins.post_tool_use_ledger import side_effect_key


class _Allow:
    priority = 1

    def __init__(self, point: HookPoint, **audit_metadata: Any) -> None:
        self.name = f"allow_{point.value}"
        self.hook_point = point
        self._meta = audit_metadata

    async def __call__(self, ctx: HookContext) -> HookDecision:
        return HookDecision(outcome="allow", reason="ok", audit_metadata=dict(self._meta))


def _fire(tmp_path: Path, ctx: HookContext, **audit_metadata: Any) -> dict[str, Any]:
    audit = AuditLog(db_path=tmp_path / "audit.db")
    kernel = GovernanceKernel(audit_log=audit)
    kernel.register(_Allow(ctx.hook_point, **audit_metadata))
    kernel.init_lock()
    kernel.fire_sync(ctx.hook_point, ctx)
    [row] = audit.query()
    return json.loads(row.payload_json)


def _ctx(point: HookPoint, payload: dict[str, Any], metadata: dict[str, Any]) -> HookContext:
    return HookContext(
        hook_point=point, run_id="r", agent_type="t", payload=payload, metadata=metadata
    )


def test_the_row_takes_call_id_from_the_metadata(tmp_path: Path) -> None:
    ctx = _ctx(
        HookPoint.PRE_TOOL_USE,
        pre_tool_payload("t", {}),
        {CALL_ID: "01CALL", TOOL_CALL_ID: "01CALL", "held_call_id": "01HELD"},
    )
    payload = _fire(tmp_path, ctx)
    assert (payload["call_id"], payload["held_call_id"]) == ("01CALL", "01HELD")


def test_a_payload_or_arguments_value_is_never_the_call_id(tmp_path: Path) -> None:
    ctx = _ctx(
        HookPoint.PRE_TOOL_USE,
        {**pre_tool_payload("t", {"call_id": "FORGED"}), "call_id": "FORGED2"},
        {},
    )
    assert "call_id" not in _fire(tmp_path, ctx)


def test_a_hooks_audit_metadata_cannot_name_the_call_either(tmp_path: Path) -> None:
    ctx = _ctx(HookPoint.PRE_TOOL_USE, pre_tool_payload("t", {}), {CALL_ID: "01REAL"})
    assert _fire(tmp_path, ctx, call_id="FORGED")["call_id"] == "01REAL"


def test_a_row_with_no_call_in_metadata_drops_a_hooks_call_id(tmp_path: Path) -> None:
    ctx = _ctx(HookPoint.PRE_LLM_CALL, {"model": "m"}, {})
    assert "call_id" not in _fire(tmp_path, ctx, call_id="FORGED")


def test_a_hooks_audit_metadata_cannot_name_the_held_call_either(tmp_path: Path) -> None:
    ctx = _ctx(HookPoint.PRE_TOOL_USE, pre_tool_payload("t", {}), {CALL_ID: "01REAL"})
    assert "held_call_id" not in _fire(tmp_path, ctx, held_call_id="FORGED")
    held = _ctx(
        HookPoint.PRE_TOOL_USE,
        pre_tool_payload("t", {}),
        {CALL_ID: "01REAL", "held_call_id": "01HELD"},
    )
    (tmp_path / "second").mkdir()
    assert _fire(tmp_path / "second", held, held_call_id="FORGED")["held_call_id"] == "01HELD"


def test_the_old_key_alone_is_read_as_the_call_id(tmp_path: Path) -> None:
    assert call_id_of({TOOL_CALL_ID: "01OLD"}) == "01OLD"
    assert call_id_of({CALL_ID: "01NEW", TOOL_CALL_ID: "01OLD"}) == "01NEW"
    assert call_id_of({}) is None and call_id_of({CALL_ID: 7}) is None
    ctx = _ctx(HookPoint.PRE_TOOL_USE, pre_tool_payload("t", {}), {TOOL_CALL_ID: "01OLD"})
    assert _fire(tmp_path, ctx)["call_id"] == "01OLD"


def test_post_metadata_carries_both_keys_with_one_value() -> None:
    meta = tool_post_metadata(
        effect="read", content="internal", verify=None, tool_call_id="01A", held_call_id="01H"
    )
    assert (meta[CALL_ID], meta[TOOL_CALL_ID], meta["held_call_id"]) == ("01A", "01A", "01H")


def test_the_ledger_key_keeps_its_shape_with_the_ulid_as_the_call_part() -> None:
    assert side_effect_key("run-1", 2, "01JABCDEFGHJKMNPQRSTVWXYZ0") == (
        "run-1:2:01JABCDEFGHJKMNPQRSTVWXYZ0"
    )
    assert side_effect_key("run-1", 2, None) == "run-1:2:-"
