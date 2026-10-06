"""``iris mcp serve``, governed (runtime/mcp_serve.py).

Every tool call an MCP client makes runs through the governed runner as ``mcp:<client>``:
PRE_TOOL_USE and POST_TOOL_USE fire and are audited with keyed digests; a call that needs
the owner's answer (destructive, pinned, confirm-once) is refused at PRE_TOOL_USE, never
queued; no audit key or no kernel, nothing runs; a POST_TOOL_USE deny withholds the
result; the session is one label; ingress and egress are logged. The served set is the
registered tool catalogue with each tool's manifest declaration, plus skill reads.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.kernel.governance import (
    DataClassification,
    GovernanceKernel,
    HookContext,
    HookDecision,
    HookPoint,
)
from iris_harness.kernel.governance.approvals import ApprovalQueue
from iris_harness.kernel.governance.approvals.store import ApprovalStore
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.mcp_clients import mcp_client_is_local
from iris_harness.kernel.governance.plugins import DestructiveApprovalHook, ToolPolicyHook
from iris_harness.kernel.governance.plugins.caller_policy import CallerPolicyHook
from iris_harness.kernel.governance.plugins.mcp_client_egress import McpClientEgressHook
from iris_harness.runtime.harness_services import HarnessServices
from iris_harness.runtime.mcp_serve import (
    McpServeError,
    ServedTool,
    client_is_local,
    load_mcp_clients,
    mcp_caller,
    select_tools,
    serve,
)
from iris_harness.runtime.plugin_host.api import PluginAPI
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.runtime.tool_service import ToolService
from iris_harness.tools.mcp.server import SkillTool


class _Spy:
    """Records what a tool hook point was shown; optionally denies or labels."""

    priority = 1

    def __init__(
        self,
        point: HookPoint,
        *,
        deny: bool = False,
        label: DataClassification | None = None,
    ) -> None:
        self.name = f"spy_{point.value}"
        self.hook_point = point
        self.deny = deny
        self.label = label
        self.seen: list[HookContext] = []

    async def __call__(self, ctx: HookContext) -> HookDecision:
        self.seen.append(ctx)
        if self.label is not None:
            return HookDecision(
                outcome="allow", reason="spy: labelled", set_classification=self.label
            )
        if self.deny:
            return HookDecision(outcome="deny", reason="spy: withheld", severity="warn")
        return HookDecision(outcome="allow", reason="spy: seen")


class _World:
    def __init__(self, tmp_path: Path, *hooks: Any, kernel: bool = True) -> None:
        self.ran: list[str] = []
        self.audit = AuditLog(db_path=tmp_path / "audit.db")
        self.queue = ApprovalQueue(
            store=ApprovalStore(db_path=tmp_path / "approvals.db"), audit_log=self.audit
        )
        self.kernel: GovernanceKernel | None = None
        if kernel:
            self.kernel = GovernanceKernel(audit_log=self.audit)
            for hook in (
                CallerPolicyHook(),
                ToolPolicyHook(),
                DestructiveApprovalHook(approval_queue=self.queue),
                McpClientEgressHook(),  # as production registers it (kernel wiring)
                *hooks,
            ):
                self.kernel.register(hook)
            self.kernel.init_lock()
        self.tools = [
            self.tool("look_up"),
            self.tool("set_pref", "write", "never"),
            self.tool("add_note", "write", "once"),
            self.tool("send_it", "write", "approval"),
            self.tool("wipe", "destructive", "approval"),
        ]
        self.service = ToolService(tools=lambda: self.tools, kernel=lambda: self.kernel)

    def tool(self, name: str, effect: str = "read", confirm: str = "never") -> ToolSpec:
        def call(args: dict[str, Any]) -> str:
            self.ran.append(name)
            return f"{name} ran"

        return ToolSpec(name, f"{name} tool", call, effect=effect, confirm=confirm)

    def serve(
        self, *calls: tuple[str, dict[str, Any]], client: str = "desk", local: bool = False
    ) -> list[Any]:
        """Serve every registered tool and make ``calls``; the ``tools/call`` replies."""
        served = select_tools(self.tools, tools=[t.name for t in self.tools], skill_tools=[])
        lines = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"clientInfo": {"name": "plugin:admin"}},
            }
        ]
        lines += [
            {
                "jsonrpc": "2.0",
                "id": i + 2,
                "method": "tools/call",
                "params": {"name": name, "arguments": args},
            }
            for i, (name, args) in enumerate(calls)
        ]
        stdout = io.StringIO()
        stdin = io.StringIO("\n".join(json.dumps(line) for line in lines) + "\n")
        serve(self.service, served.served, client=client, local=local, stdin=stdin, stdout=stdout)
        replies = [json.loads(line) for line in stdout.getvalue().splitlines()]
        return [reply["result"] for reply in replies[1:]]

    def rows(self, hook_point: str) -> list[Any]:
        return [row for row in self.audit.query() if row.hook_point == hook_point]


def _text(result: dict[str, Any]) -> str:
    return "".join(part["text"] for part in result["content"])


def test_a_served_call_is_governed_and_audited_as_the_operators_client(
    tmp_path: Path,
) -> None:
    # A POST_TOOL_USE hook, as production has (the output classifier, the ledger).
    world = _World(tmp_path, _Spy(HookPoint.POST_TOOL_USE))
    [result] = world.serve(("look_up", {"query": "x"}))

    assert result["isError"] is False and _text(result) == "look_up ran"
    pre, post = world.rows("pre_tool_use"), world.rows("post_tool_use")
    assert pre and post
    # The caller is the operator's name for the client, never the name the client sent.
    assert {row.agent_type for row in pre + post} == {"mcp:desk"}
    assert {row.run_id for row in pre} == {row.run_id for row in post}
    # Keyed digests, never the arguments or the result themselves.
    pre_payload = json.loads(pre[0].payload_json)
    assert pre_payload.get("args_digest") and "x" not in json.dumps(pre_payload)
    assert json.loads(post[0].payload_json).get("result_digest")
    # Whose tool it is, stamped from the registry's record (a test tool: the core's).
    assert {json.loads(r.payload_json)["tool_plugin"] for r in pre + post} == {"system"}


@pytest.mark.parametrize("name", ["wipe", "send_it"])
def test_a_call_that_needs_the_owners_approval_is_refused_not_queued(
    tmp_path: Path, name: str
) -> None:
    world = _World(tmp_path)
    [result] = world.serve((name, {"ids": ["a"]}))

    assert result["isError"] is True
    assert "needs the owner's approval" in _text(result)
    assert world.ran == []
    assert world.queue.list_pending() == []  # nothing queued for later
    [denied] = [r for r in world.rows("pre_tool_use") if r.decision == "deny"]
    assert (denied.plugin, denied.agent_type) == ("destructive_approval", "mcp:desk")
    assert world.rows("post_tool_use") == []


def test_a_confirm_once_write_is_refused_the_client_cannot_confirm(tmp_path: Path) -> None:
    world = _World(tmp_path)
    [result] = world.serve(("add_note", {"query": "x"}))

    assert result["isError"] is True and "outside IRIS" in _text(result)
    assert "ask_user" not in _text(result)  # the loop's instruction, not an MCP client's
    assert world.ran == []
    assert any(
        r.plugin == "tool_policy" and r.decision == "deny" for r in world.rows("pre_tool_use")
    )


def test_a_write_the_owner_needs_not_confirm_runs(tmp_path: Path) -> None:
    world = _World(tmp_path)
    [result] = world.serve(("set_pref", {"query": "dark"}))
    assert result["isError"] is False and world.ran == ["set_pref"]


@pytest.mark.usefixtures("no_vault_master_key")
def test_no_audit_key_no_call(tmp_path: Path) -> None:
    world = _World(tmp_path)
    [result] = world.serve(("look_up", {}))

    assert result["isError"] is True and "no vault master key" in _text(result)
    assert world.ran == [] and not world.audit.query()


def test_no_kernel_no_call(tmp_path: Path) -> None:
    world = _World(tmp_path, kernel=False)
    [result] = world.serve(("look_up", {}))
    assert result["isError"] is True and "never run ungoverned" in _text(result)
    assert world.ran == []


def test_a_post_tool_use_deny_withholds_the_result(tmp_path: Path) -> None:
    world = _World(tmp_path, _Spy(HookPoint.POST_TOOL_USE, deny=True))
    [result] = world.serve(("look_up", {}))

    assert world.ran == ["look_up"]  # it ran; the client does not get what it returned
    assert result["isError"] is True
    assert "look_up ran" not in _text(result) and "withheld" in _text(result)
    assert any(r.decision == "deny" for r in world.rows("post_tool_use"))


def test_the_session_is_one_label_a_result_raises_it_for_later_calls(tmp_path: Path) -> None:
    pre = _Spy(HookPoint.PRE_TOOL_USE)
    world = _World(tmp_path, pre, _Spy(HookPoint.POST_TOOL_USE, label="personal"))
    world.serve(("look_up", {}), ("look_up", {}))

    assert [ctx.classification for ctx in pre.seen] == [None, "personal"]


def test_each_call_is_logged_in_and_out_never_its_content(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    world = _World(tmp_path)
    with caplog.at_level(logging.INFO, logger="iris"):
        world.serve(("look_up", {"query": "my secret question"}), ("wipe", {}))

    ingress = [r.getMessage() for r in caplog.records if r.name == "iris.ingress"]
    egress = [r.getMessage() for r in caplog.records if r.name == "iris.egress"]
    assert any("tools/call look_up" in m and "from=mcp:desk" in m for m in ingress)
    assert any("EGRESS mcp" in m and "tool=look_up" in m and "status=ok" in m for m in egress)
    assert any("tool=wipe" in m and "status=refused" in m for m in egress)
    assert not any("secret question" in m or "look_up ran" in m for m in ingress + egress)


def test_the_client_name_is_the_operators_and_checked() -> None:
    assert mcp_caller("claude-desktop") == "mcp:claude-desktop"
    for bad in ("", "Has Space", "plugin:x", "a" * 65):
        with pytest.raises(McpServeError):
            mcp_caller(bad)


# -- what is served -----------------------------------------------------------------------


def _skill(name: str, route: str) -> SkillTool:
    class _Instance:
        def invoke(self, args: dict[str, Any]) -> str:
            return f"{name} skill ran"

    return SkillTool(name, f"{name}.", {"type": "object"}, route, "a-skill", _Instance())


def test_by_default_the_read_tools_are_served(tmp_path: Path) -> None:
    world = _World(tmp_path)
    selection = select_tools(world.tools, skill_tools=[])
    assert [s.spec.name for s in selection.served] == ["look_up"]


def test_named_tools_are_served_whatever_their_effect_and_unknown_names_fail(
    tmp_path: Path,
) -> None:
    world = _World(tmp_path)
    selection = select_tools(world.tools, tools=["wipe", "look_up"], skill_tools=[])
    assert [(s.spec.name, s.spec.effect) for s in selection.served] == [
        ("wipe", "destructive"),
        ("look_up", "read"),
    ]
    with pytest.raises(McpServeError, match="no_such"):
        select_tools(world.tools, tools=["no_such"], skill_tools=[])


def test_only_a_skills_reads_are_served_and_never_over_a_registered_tool(
    tmp_path: Path,
) -> None:
    world = _World(tmp_path)
    selection = select_tools(
        world.tools,
        skill_tools=[
            _skill("list_events", "calendar/read"),
            _skill("create_event", "calendar/write"),
            _skill("look_up", "system/read"),
        ],
    )
    served = {s.spec.name: s for s in selection.served}
    assert set(served) == {"look_up", "list_events"}
    assert served["look_up"].source == "plugin"
    assert (served["list_events"].source, served["list_events"].spec.effect) == (
        "skill:a-skill",
        "read",
    )
    assert any("create_event" in why and "not a read" in why for why in selection.skipped)


def test_a_skill_read_is_governed_like_any_tool(tmp_path: Path) -> None:
    world = _World(tmp_path)
    [skill] = select_tools([], skill_tools=[_skill("list_events", "calendar/read")]).served
    stdout = io.StringIO()
    call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list_events"}}
    serve(world.service, [skill], client="desk", stdin=io.StringIO(json.dumps(call)), stdout=stdout)

    assert json.loads(stdout.getvalue())["result"]["isError"] is False
    assert {r.agent_type for r in world.rows("pre_tool_use")} == {"mcp:desk"}
    pre_plugins = {json.loads(r.payload_json)["tool_plugin"] for r in world.rows("pre_tool_use")}
    assert pre_plugins == {"skill:a-skill"}


def test_a_plugin_tool_is_served_with_its_manifest_declaration(tmp_path: Path) -> None:
    world = _World(tmp_path)
    registry = PluginRegistry()
    manifest = PluginManifest.model_validate(
        {
            "name": "notes",
            "provides": ["tool"],
            "tools": {
                "find_note": {"effect": "read"},
                "drop_note": {"effect": "destructive"},
            },
        }
    )
    registry.add_plugin(
        PluginRecord(name="notes", source="test", status=PluginStatus.LOADED, manifest=manifest)
    )
    services = HarnessServices(
        config_dir=tmp_path,
        data_dir=tmp_path,
        tier_router=None,  # type: ignore[arg-type]
        agent_executor=None,  # type: ignore[arg-type]
        heartbeats=None,  # type: ignore[arg-type]
        channels=None,  # type: ignore[arg-type]
        deterministic_reply=lambda **kw: None,
        tools=world.service,
    )
    api = PluginAPI(plugin="notes", services=services, registry=registry)
    api.register_tool("find_note", "Find a note.", lambda args: "found")
    api.register_tool("drop_note", "Drop a note.", lambda args: "dropped")
    world.tools = registry.tools()

    by_default = select_tools(world.tools, skill_tools=[])
    assert [s.spec.name for s in by_default.served] == ["find_note"]
    named = select_tools(world.tools, tools=["drop_note"], skill_tools=[])
    [drop] = named.served
    assert drop.spec.effect == "destructive"  # the manifest's word, not a default
    stdout = io.StringIO()
    call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "drop_note"}}
    served: list[ServedTool] = list(named.served)
    serve(world.service, served, client="desk", stdin=io.StringIO(json.dumps(call)), stdout=stdout)
    assert json.loads(stdout.getvalue())["result"]["isError"] is True
    # The registry stamped the owner at registration; the served call's row names it.
    owners = {json.loads(r.payload_json)["tool_plugin"] for r in world.rows("pre_tool_use")}
    assert owners == {"notes"}


def test_a_plugin_tool_that_raises_is_an_error_to_the_client(tmp_path: Path) -> None:
    """Regression: the plugin's fault boundary answered with a sentence, so a client was
    told a tool that raised had succeeded (``isError: false``)."""
    world = _World(tmp_path)
    registry = PluginRegistry()
    registry.add_plugin(PluginRecord(name="notes", source="test", status=PluginStatus.LOADED))

    def boom(args: dict[str, Any]) -> str:
        raise RuntimeError("index missing")

    registry.add_tool("notes", ToolSpec("find_note", "Find a note.", boom))
    world.tools = registry.tools()

    [result] = world.serve(("find_note", {}))

    assert result["isError"] is True
    assert _text(result).startswith("find_note is unavailable (plugin 'notes' raised RuntimeError")


# -- where a result may go (owner's decision, 2026-09-30) ----------------------------------


def test_a_personal_result_is_withheld_from_a_client_not_declared_local(tmp_path: Path) -> None:
    world = _World(tmp_path, _Spy(HookPoint.POST_TOOL_USE, label="personal"))
    [result] = world.serve(("look_up", {}))

    assert world.ran == ["look_up"]
    assert result["isError"] is True and "look_up ran" not in _text(result)
    assert "not declared local" in _text(result)
    [deny] = [r for r in world.rows("post_tool_use") if r.decision == "deny"]
    assert (deny.plugin, deny.agent_type) == ("mcp_client_egress", "mcp:desk")


def test_a_personal_result_reaches_a_client_declared_local(tmp_path: Path) -> None:
    world = _World(tmp_path, _Spy(HookPoint.POST_TOOL_USE, label="personal"))
    [result] = world.serve(("look_up", {}), local=True)
    assert result["isError"] is False and _text(result) == "look_up ran"
    assert not mcp_client_is_local("mcp:desk")  # withdrawn when the server stops


def test_a_secret_result_reaches_no_client_not_even_a_local_one(tmp_path: Path) -> None:
    world = _World(tmp_path, _Spy(HookPoint.POST_TOOL_USE, label="secret"))
    [result] = world.serve(("look_up", {}), local=True)
    assert result["isError"] is True and "secret" in _text(result)


def test_the_sessions_label_is_the_floor_for_what_may_leave(tmp_path: Path) -> None:
    class _FirstPersonal(_Spy):
        async def __call__(self, ctx: HookContext) -> HookDecision:
            self.seen.append(ctx)
            if len(self.seen) == 1:
                return HookDecision(outcome="allow", reason="spy", set_classification="personal")
            return HookDecision(outcome="allow", reason="spy: nothing private in this one")

    spy = _FirstPersonal(HookPoint.POST_TOOL_USE)
    world = _World(tmp_path, spy)
    first, second = world.serve(("look_up", {}), ("look_up", {}))
    # The second result raised nothing itself, but the session already holds personal data.
    assert spy.seen[1].classification == "personal"
    assert first["isError"] is True and second["isError"] is True


def test_a_public_result_reaches_any_client(tmp_path: Path) -> None:
    world = _World(tmp_path, _Spy(HookPoint.POST_TOOL_USE, label="public"))
    [result] = world.serve(("look_up", {}))
    assert result["isError"] is False


def test_the_shipped_config_lists_stdio_as_not_local(tmp_path: Path) -> None:
    clients = load_mcp_clients(home=tmp_path)
    assert clients == {"stdio": False}
    assert client_is_local("stdio", clients) is False
    with pytest.raises(McpServeError, match="not in the owner's MCP serve config"):
        client_is_local("claude-desktop", clients)


def test_the_owners_overlay_adds_and_overrides_clients(tmp_path: Path) -> None:
    (tmp_path / "mcp-serve.yaml").write_text(
        "clients:\n  claude-desktop:\n    local: true\n  stdio: {}\n", encoding="utf-8"
    )
    assert load_mcp_clients(home=tmp_path) == {"stdio": False, "claude-desktop": True}


@pytest.mark.parametrize(
    "text",
    [
        "clients: [a, b]\n",
        "clients:\n  desk:\n    local: yes-please\n",
        "clients:\n  desk:\n    remote: true\n",
        "clients:\n  Bad Name: {}\n",
        "clients: {\n",
    ],
)
def test_a_malformed_config_is_an_error_never_a_wider_list(tmp_path: Path, text: str) -> None:
    (tmp_path / "mcp-serve.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(McpServeError):
        load_mcp_clients(home=tmp_path)


def test_research_and_code_execution_are_served_only_when_named(tmp_path: Path) -> None:
    world = _World(tmp_path)
    research = world.tool("research")._replace(sends_to="search_engine", content="external")
    mail = world.tool("read_mail")._replace(content="external")
    code = world.tool("code_exec")._replace(executes_code=True)
    catalogue = [world.tools[0], research, mail, code]

    by_default = select_tools(catalogue, skill_tools=[])
    assert [s.spec.name for s in by_default.served] == ["look_up"]
    skipped = {why.split("'")[1] for why in by_default.skipped}
    assert skipped == {"research", "read_mail", "code_exec"}
    named = select_tools(catalogue, tools=["research", "code_exec"], skill_tools=[])
    assert [s.spec.name for s in named.served] == ["research", "code_exec"]


def test_no_skill_is_served_unless_named(tmp_path: Path) -> None:
    world = _World(tmp_path)
    assert [s.spec.name for s in select_tools(world.tools).served] == ["look_up"]


def test_each_served_call_gets_its_own_minted_call_id_on_every_row(tmp_path: Path) -> None:
    """#134: an inbound MCP call is a call like any other; its rows carry one ULID, two
    calls carry two, and the client cannot name it (its argument is just an argument)."""
    from iris_harness.foundation.ids import is_ulid

    world = _World(tmp_path, _Spy(HookPoint.POST_TOOL_USE))
    first, second = world.serve(
        ("look_up", {"query": "x", "call_id": "FORGED", "tool_call_id": "FORGED"}),
        ("look_up", {"query": "y"}),
    )
    assert first["isError"] is False and second["isError"] is False
    by_run: dict[str, set[str]] = {}
    for row in world.rows("pre_tool_use") + world.rows("post_tool_use"):
        payload = json.loads(row.payload_json)
        assert is_ulid(payload.get("call_id")), payload
        assert payload["call_id"] != "FORGED"
        by_run.setdefault(row.run_id, set()).add(payload["call_id"])
    # Two calls, two runs, one id each; the run id is not the call id.
    assert len(by_run) == 2 and all(len(ids) == 1 for ids in by_run.values())
    assert not set(by_run) & {i for ids in by_run.values() for i in ids}
