"""An MCP tool the operator declares destructive is held for approval and written ahead
(issue #102).

An MCP server declares no effect of its own, so its calls got no pending ledger row and no
approval gate. ``governance.tools`` in the server's config now lets the operator say what a
tool does; the bridge hands that effect to the same hooks a plugin tool meets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from iris_harness.kernel.governance import build_default_kernel
from iris_harness.kernel.governance.approvals import ApprovalQueue
from iris_harness.kernel.governance.approvals.store import ApprovalItem
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.plugins.mcp_allowlist import (
    MCPServerGovernance,
    MCPToolGovernance,
)
from iris_harness.kernel.governance.side_effects import shared_side_effect_ledger
from iris_harness.kernel.governor.audit import GovernorAuditLogger
from iris_harness.kernel.governor.policy import GovernorPolicyEngine, load_governor_policy
from iris_harness.kernel.governor.service import IRISGovernorService
from iris_harness.tools.mcp_bridge import MCPBridge, MCPBridgeConfig, MCPServerConfig

ROUTE = "mcp/files/delete_file"
RUN = "mcp-files"
ARGS = {"path": "notes.txt"}


def _governor(root: Path) -> IRISGovernorService:
    policy_dir = root / "config" / "governor"
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "policy.yaml").write_text(
        "version: '1'\nroutes:\n  - route: coding/mcp\n    allowed_actions:\n"
        "      - session_open\n      - call_tool\n      - invoke_server\n"
        "    requires_approval: true\n    rate_limit:\n      requests: 10\n"
        "      window_seconds: 3600\n",
        encoding="utf-8",
    )
    return IRISGovernorService(
        policy_engine=GovernorPolicyEngine(load_governor_policy(root)),
        audit_logger=GovernorAuditLogger(root / "data" / "audit.db"),
    )


class Rig:
    def __init__(self, root: Path, tools: dict[str, str], *, undeclared: str | None = None) -> None:
        self.queue = ApprovalQueue(db_path=root / "approvals.db")
        self.ledger = shared_side_effect_ledger(root / "ledger.db")
        kernel = build_default_kernel(
            audit_log=AuditLog(root / "audit.db"),
            approval_queue=self.queue,
            side_effect_ledger_db_path=root / "ledger.db",
        )
        governance = MCPServerGovernance(
            tools={n: MCPToolGovernance(effect=e) for n, e in tools.items()},  # type: ignore[arg-type]
            **({"undeclared_tools": undeclared} if undeclared else {}),  # type: ignore[arg-type]
        )
        config = MCPBridgeConfig(
            enabled=True,
            servers=(
                MCPServerConfig(
                    name="files", enabled=True, command="python", governance=governance
                ),
            ),
        )
        self.bridge = MCPBridge(
            root, config=config, governor_service=_governor(root), governance_kernel=kernel
        )
        self.reached: list[str] = []
        self.rows_when_reached: list[list[Any]] = []

    def approval(self, args: dict[str, Any] = ARGS) -> str:
        approval_id = self.queue.enqueue(
            RUN, None, "delete", "delete notes.txt", items=(ApprovalItem.of(ROUTE, args),)
        )
        self.queue.respond(approval_id, status="approved", actor="owner")
        return str(approval_id)

    def call(
        self, tool: str = "delete_file", *, approved_by: str | None = None, fail: bool = False
    ):
        def executor(_server: Any, _tool: str, _args: dict[str, Any]) -> str:
            self.reached.append(_tool)
            self.rows_when_reached.append(self.ledger.pending(RUN))
            if fail:
                raise RuntimeError("transport broke")
            return "deleted"

        return self.bridge.invoke_external_tool(
            "files",
            tool,
            dict(ARGS),
            approval_granted=True,
            executor=executor,
            approved_by=approved_by,
        )


def test_a_declared_destructive_call_with_an_approval_leaves_a_pending_row_before_it_runs(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, {"delete_file": "destructive"})

    result = rig.call(approved_by=rig.approval())

    assert result.result is not None and rig.reached == ["delete_file"]
    (row,) = rig.rows_when_reached[0]  # the row existed when the transport ran
    assert row.tool == ROUTE and row.status == "pending"
    assert rig.ledger.pending(RUN) == []  # the post hook settled it


def test_a_declared_destructive_call_with_no_approval_never_reaches_the_server(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, {"delete_file": "destructive"})

    with pytest.raises(PermissionError):
        rig.call()

    assert rig.reached == [] and rig.ledger.pending(RUN) == []


def test_an_approval_for_other_arguments_does_not_open_the_call(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"delete_file": "destructive"})

    with pytest.raises(PermissionError):
        rig.call(approved_by=rig.approval({"path": "other.txt"}))

    assert rig.reached == []


def test_a_transport_that_fails_leaves_the_row_pending_for_resume(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"delete_file": "destructive"})

    with pytest.raises(RuntimeError):
        rig.call(approved_by=rig.approval(), fail=True)

    (row,) = rig.ledger.pending(RUN)
    assert row.tool == ROUTE and row.status == "pending"


def test_a_taken_ledger_key_denies_the_call_before_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rig = Rig(tmp_path, {"delete_file": "destructive"})
    monkeypatch.setattr("iris_harness.tools.mcp_bridge.new_ulid", lambda: "FIXEDCALLID")
    rig.ledger.record(
        side_effect_id=f"{RUN}:0:FIXEDCALLID",
        run_id=RUN,
        step_id=0,
        tool="other",
        verification_probe="",
    )

    with pytest.raises(PermissionError):
        rig.call(approved_by=rig.approval())

    assert rig.reached == []


def test_an_undeclared_tool_with_no_approval_never_reaches_the_server(tmp_path: Path) -> None:
    """Fail closed (#180): a tool nobody declared is treated as destructive."""
    rig = Rig(tmp_path, {"delete_file": "destructive"})

    with pytest.raises(PermissionError):
        rig.call("read_file")

    assert rig.reached == [] and rig.ledger.pending(RUN) == []


def test_an_undeclared_tool_with_an_approval_pinning_the_call_runs_with_a_row(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, {})
    approval_id = rig.queue.enqueue(
        RUN,
        None,
        "read",
        "read notes.txt",
        items=(ApprovalItem.of("mcp/files/read_file", ARGS),),
    )
    rig.queue.respond(approval_id, status="approved", actor="owner")

    result = rig.call("read_file", approved_by=str(approval_id))

    assert result.result == "deleted" and rig.reached == ["read_file"]
    (row,) = rig.rows_when_reached[0]  # written ahead, like a declared destructive tool
    assert row.tool == "mcp/files/read_file" and row.status == "pending"


def test_a_server_that_opts_out_runs_its_undeclared_tools_unchanged(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"delete_file": "destructive"}, undeclared="read")

    result = rig.call("read_file")

    assert result.result == "deleted" and rig.reached == ["read_file"]
    assert rig.rows_when_reached == [[]] and rig.ledger.pending(RUN) == []
    with pytest.raises(PermissionError):  # the declared destructive tool is still gated
        rig.call("delete_file")


def test_undeclared_tools_rejects_an_unknown_value() -> None:
    with pytest.raises(ValidationError):
        MCPServerGovernance.model_validate({"undeclared_tools": "write"})
    assert MCPServerGovernance().undeclared_tools == "destructive"


def test_a_server_with_no_governance_block_fails_closed_too(tmp_path: Path) -> None:
    from iris_harness.tools.mcp_bridge import effective_effect

    bare = MCPServerConfig(name="files", enabled=True, command="python")

    assert effective_effect(bare, "anything") == "destructive"


def test_discovery_warns_naming_the_undeclared_tools_and_the_default(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rig = Rig(tmp_path, {"list_files": "read"})
    rig.bridge._invoke_transport = lambda *_a, **_k: {  # type: ignore[method-assign]
        "tools": [
            {"name": "list_files", "description": "d", "inputSchema": {}},
            {"name": "delete_file", "description": "d", "inputSchema": {}},
            {"name": "read_file", "description": "d", "inputSchema": {}},
        ]
    }

    with caplog.at_level("WARNING", logger="iris_harness.tools.mcp_bridge"):
        rig.bridge.list_external_tools("files", approval_granted=True)

    (record,) = [r for r in caplog.records if "no declared effect" in r.getMessage()]
    message = record.getMessage()
    assert "delete_file, read_file" in message and "list_files" not in message
    assert "treated as destructive" in message


def test_a_declared_read_tool_needs_no_approval_and_writes_no_row(tmp_path: Path) -> None:
    rig = Rig(tmp_path, {"list_files": "read"})

    rig.call("list_files")

    assert rig.reached == ["list_files"] and rig.rows_when_reached == [[]]


def test_the_config_rejects_an_unknown_effect_and_an_empty_tool_name() -> None:
    with pytest.raises(ValidationError):
        MCPServerGovernance.model_validate({"tools": {"x": {"effect": "explode"}}})
    with pytest.raises(ValidationError):
        MCPServerGovernance.model_validate({"tools": {"  ": {"effect": "read"}}})
    with pytest.raises(ValidationError):
        MCPServerGovernance.model_validate({"tools": {"x": {"effect": "read", "extra": 1}}})
    ok = MCPServerGovernance.model_validate({"tools": {"x": {"effect": "destructive"}}})
    assert ok.tools["x"].effect == "destructive"


def test_a_pre_tool_hook_that_requires_approval_never_reaches_the_server(
    tmp_path: Path,
) -> None:
    """The bridge refused only ``deny`` before the call; ``require_approval`` ran it (#181)."""
    from iris_harness.kernel.governance.hooks.types import (
        HookContext,
        HookDecision,
        HookPoint,
    )
    from iris_harness.kernel.governance.kernel import GovernanceKernel

    class AsksForApproval:
        name = "asks_for_approval"
        hook_point = HookPoint.PRE_TOOL_USE
        priority = 10

        async def __call__(self, ctx: HookContext) -> HookDecision:
            return HookDecision(outcome="require_approval", reason="owner must confirm")

    kernel = GovernanceKernel()
    kernel.register(AsksForApproval())
    kernel.init_lock()
    config = MCPBridgeConfig(
        enabled=True,
        servers=(MCPServerConfig(name="files", enabled=True, command="python"),),
    )
    bridge = MCPBridge(
        tmp_path, config=config, governor_service=_governor(tmp_path), governance_kernel=kernel
    )
    reached: list[str] = []

    def executor(_server: Any, tool: str, _args: dict[str, Any]) -> str:
        reached.append(tool)
        return "ok"

    with pytest.raises(PermissionError, match="needs approval"):
        bridge.invoke_external_tool(
            "files", "read_file", dict(ARGS), approval_granted=True, executor=executor
        )

    assert reached == []
