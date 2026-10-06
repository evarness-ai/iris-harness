"""A code caller's approved call runs once, governed, as the caller that asked.

docs/architecture/plugin-capabilities.md, decision 1. A plugin's ``api.tools.call`` of a
``confirm: once`` write, a pinned write or a destructive tool is queued for the owner and
comes back held with the approval's id. When the owner approves, the harness runs the
pinned call through the same governed runner — the caller policy re-checked at execution,
the approval verified and claimed once, POST_TOOL_USE, the audit — and tells the caller
on ``approval.call_completed``. Rejected or expired, it never runs. These tests use the
real kernel hooks, a real approval queue and a real audit log.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iris_harness.agent.agentic_core import ToolSpec
from iris_harness.foundation.auth import auth_headers
from iris_harness.foundation.eventbus import EventBus
from iris_harness.foundation.observability.session_log import session_scope
from iris_harness.kernel.governance import GovernanceKernel
from iris_harness.kernel.governance.approvals import ApprovalAlreadyAnsweredError, ApprovalQueue
from iris_harness.kernel.governance.approvals.events import (
    APPROVAL_CALL_COMPLETED,
    ApprovalCallCompletedPayload,
)
from iris_harness.kernel.governance.approvals.service import respond_to_approval, sweep_expired
from iris_harness.kernel.governance.approvals.store import ApprovalStore
from iris_harness.kernel.governance.audit import AuditLog
from iris_harness.kernel.governance.caller_policy import register_caller_policy
from iris_harness.kernel.governance.plugins import DestructiveApprovalHook, ToolPolicyHook
from iris_harness.kernel.governance.plugins.caller_policy import CallerPolicyHook
from iris_harness.runtime.harness_services import HarnessServices
from iris_harness.runtime.plugin_host.api import PluginAPI
from iris_harness.runtime.plugin_host.harness_topics import GuardedEventBus
from iris_harness.runtime.plugin_host.registry import PluginRecord, PluginRegistry, PluginStatus
from iris_harness.runtime.tool_service import ToolService
from iris_harness.server.iris_api.main import create_app

CALLER = "plugin:p"


class _World:
    """The pieces one approved call touches, with what each saw."""

    def __init__(self, tmp_path: Path) -> None:
        self.ran: list[tuple[str, dict[str, Any]]] = []
        self.notices: list[tuple[str, str]] = []
        self.denied: set[tuple[str, str]] = set()
        self.audit = AuditLog(db_path=tmp_path / "audit.db")
        self.queue = ApprovalQueue(
            store=ApprovalStore(db_path=tmp_path / "approvals.db"), audit_log=self.audit
        )
        self.bus = EventBus()
        self.kernel = GovernanceKernel(audit_log=self.audit)
        for hook in (
            CallerPolicyHook(),
            ToolPolicyHook(),
            DestructiveApprovalHook(approval_queue=self.queue),
        ):
            self.kernel.register(hook)
        self.kernel.init_lock()
        register_caller_policy(
            lambda caller, tool: "narrowed" if (caller, tool) in self.denied else None
        )
        self.service = ToolService(
            tools=lambda: self.tools,
            kernel=lambda: self.kernel,
            events=lambda: self.bus,
            deliver_in_chat=lambda session, text: self.notices.append((session, text)),
        )
        self.tools = [
            self._tool("add_note", "write", "once"),
            self._tool("send_it", "write", "approval"),
            self._tool("wipe", "destructive", "approval"),
            self._tool("broken", "write", "once", fail=True),
        ]

    def _tool(self, name: str, effect: str, confirm: str, *, fail: bool = False) -> ToolSpec:
        def call(args: dict[str, Any]) -> str:
            self.ran.append((name, dict(args)))
            if fail:
                raise RuntimeError("disk full for someone@example.com")
            return f"{name} done for someone@example.com"

        return ToolSpec(name, f"{name} tool", call, effect=effect, confirm=confirm)

    def plugin_api(self, plugin: str, registry: PluginRegistry) -> PluginAPI:
        registry.add_plugin(PluginRecord(name=plugin, source="test", status=PluginStatus.LOADED))
        services = HarnessServices(
            config_dir=Path("."),
            data_dir=Path("."),
            tier_router=None,  # type: ignore[arg-type]
            agent_executor=None,  # type: ignore[arg-type]
            heartbeats=None,  # type: ignore[arg-type]
            channels=None,  # type: ignore[arg-type]
            deterministic_reply=lambda **kw: None,
            # As production wires it: plugins see a guarded view of the runtime's bus.
            events=GuardedEventBus(self.bus),
            tools=self.service,
        )
        return PluginAPI(plugin=plugin, services=services, registry=registry)

    def queued(self, tool: str = "add_note", args: dict[str, Any] | None = None) -> str:
        with session_scope("chat-1"):
            result = self.service.for_caller(CALLER).call(tool, args or {"text": "milk"})
        assert result.held and not result.ok and result.approval_id is not None
        return result.approval_id

    def approve(self, approval_id: str, **kw: Any) -> Any:
        return respond_to_approval(
            approval_id, status="approved", actor="test", queue=self.queue, **kw
        )

    def outcomes(self) -> list[dict[str, Any]]:
        return [
            json.loads(row.payload_json)
            for row in self.audit.query()
            if row.decision.startswith("call_")
        ]


@pytest.fixture()
def world(tmp_path: Path) -> Iterator[_World]:
    w = _World(tmp_path)
    yield w
    register_caller_policy(None)


# ── queued, not run ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("tool", ["add_note", "send_it", "wipe"])
def test_a_code_caller_s_gated_call_is_queued_with_its_caller(world: _World, tool: str) -> None:
    approval_id = world.queued(tool)
    row = world.queue.get(approval_id)
    assert row is not None and row.status == "pending"
    assert row.caller == CALLER and row.is_deferred_call and row.session_id == "chat-1"
    assert [(i.tool, i.args) for i in row.items or ()] == [(tool, {"text": "milk"})]
    assert world.ran == []


def test_the_held_result_names_the_approval_in_no_surface_s_words(world: _World) -> None:
    result = world.service.for_caller(CALLER).call("add_note", {"text": "milk"})
    assert result.text.startswith(f"Queued for approval {result.approval_id};")
    assert "iris approvals" not in result.text  # no one surface's wording


# ── approved: runs once, governed, as the caller ────────────────────────────────


def test_approving_runs_the_pinned_call_once_as_the_caller(world: _World) -> None:
    approval_id = world.queued()
    outcome = world.approve(approval_id, executor=world.service)

    assert world.ran == [("add_note", {"text": "milk"})]
    assert outcome.executed and outcome.row.executed_at is not None
    assert "add_note done" in outcome.detail
    assert [o["status"] for o in world.outcomes()] == ["ran"]
    # The governed call itself left kernel audit rows, stamped with the caller.
    assert any(r.agent_type == CALLER for r in world.audit.query())


def test_the_result_reaches_only_the_calling_plugin_and_the_conversation(world: _World) -> None:
    registry = PluginRegistry()
    mine: list[ApprovalCallCompletedPayload] = []
    theirs: list[ApprovalCallCompletedPayload] = []
    world.plugin_api("p", registry).on_approved_call(mine.append)
    world.plugin_api("q", registry).on_approved_call(theirs.append)

    approval_id = world.queued()
    world.approve(approval_id, executor=world.service)

    assert [(e.approval_id, e.caller, e.tool, e.status) for e in mine] == [
        (approval_id, CALLER, "add_note", "ran")
    ]
    assert theirs == []
    # Redacted: the address in the tool's output is display-masked.
    assert "someone@example.com" not in mine[0].summary and "so***ne@example.com" in (
        mine[0].summary
    )
    assert [s for s, _ in world.notices] == ["chat-1"]
    assert "someone@example.com" not in world.notices[0][1]
    # The subscription is on the registry, where the drift report and `iris plugins` see it.
    assert ("p", APPROVAL_CALL_COMPLETED, "runtime") in registry.subscriptions()


def test_double_approve_runs_nothing_twice(world: _World) -> None:
    approval_id = world.queued()
    world.approve(approval_id, executor=world.service)
    with pytest.raises(ApprovalAlreadyAnsweredError):
        world.approve(approval_id, executor=world.service)
    assert len(world.ran) == 1


def test_replaying_the_approved_call_is_denied_by_the_claim(world: _World) -> None:
    approval_id = world.queued()
    world.approve(approval_id, executor=world.service)
    row = world.queue.get(approval_id)
    assert row is not None
    replay = world.service.execute_approved_call(row)
    assert replay.status == "denied" and "already run" in replay.summary
    assert len(world.ran) == 1


def test_the_caller_policy_is_rechecked_at_execution(world: _World) -> None:
    approval_id = world.queued()
    world.denied.add((CALLER, "add_note"))  # the operator narrowed it after the ask
    outcome = world.approve(approval_id, executor=world.service)
    assert world.ran == [] and not outcome.executed
    assert [o["status"] for o in world.outcomes()] == ["denied"]


def test_another_caller_cannot_use_the_approval(world: _World) -> None:
    from dataclasses import replace

    approval_id = world.queued()
    world.queue.respond(approval_id, status="approved", actor="test")
    row = world.queue.get(approval_id)
    assert row is not None
    forged = world.service.execute_approved_call(replace(row, caller="plugin:other"))
    assert forged.status == "denied" and world.ran == []


def test_a_changed_argument_is_not_what_was_approved(world: _World) -> None:
    from dataclasses import replace

    from iris_harness.kernel.governance.approvals.store import ApprovalItem

    approval_id = world.queued()
    world.queue.respond(approval_id, status="approved", actor="test")
    row = world.queue.get(approval_id)
    assert row is not None
    swapped = replace(row, items=(ApprovalItem.of("add_note", {"text": "everything"}),))
    assert world.service.execute_approved_call(swapped).status == "denied"
    assert world.ran == []


def test_a_tool_that_raises_is_reported_failed(world: _World) -> None:
    approval_id = world.queued("broken")
    outcome = world.approve(approval_id, executor=world.service)
    assert outcome.executed and [o["status"] for o in world.outcomes()] == ["failed"]
    assert "someone@example.com" not in outcome.detail


# ── rejected, expired, no runtime: never runs ───────────────────────────────────


def test_rejecting_runs_nothing_and_tells_the_caller(world: _World) -> None:
    events: list[Any] = []
    world.bus.on(APPROVAL_CALL_COMPLETED, events.append)
    approval_id = world.queued()
    outcome = respond_to_approval(
        approval_id, status="rejected", actor="t", queue=world.queue, executor=world.service
    )
    assert world.ran == [] and not outcome.executed
    assert [e.status for e in events] == ["rejected"]
    assert [o["status"] for o in world.outcomes()] == ["rejected"]


def test_an_expired_approval_never_runs_and_the_caller_hears(world: _World) -> None:
    events: list[Any] = []
    world.bus.on(APPROVAL_CALL_COMPLETED, events.append)
    approval_id = world.queued()
    with world.queue._store._connect() as conn:  # lapse it
        conn.execute("UPDATE approval_queue SET timeout_at = '2000-01-01T00:00:00+00:00'")
        conn.commit()
    lapsed = sweep_expired(queue=world.queue, executor=world.service)
    assert [r.approval_id for r in lapsed] == [approval_id]
    assert [e.status for e in events] == ["expired"] and world.ran == []
    assert world.queue._store.claim_execution(approval_id) is None


def test_approving_with_no_executor_leaves_it_waiting(world: _World) -> None:
    approval_id = world.queued()
    outcome = world.approve(approval_id)
    assert outcome.row.status == "pending" and "still waiting" in outcome.detail
    row = world.queue.get(approval_id)
    assert row is not None and row.status == "pending" and world.ran == []


# ── the loop is unchanged ───────────────────────────────────────────────────────


def test_a_loop_call_of_a_confirm_once_write_is_still_turned_back_to_ask(world: _World) -> None:
    from iris_harness.agent.tool_runner import GovernedToolRunner, ToolCall

    runner = GovernedToolRunner(kernel=world.kernel, agent_type="email", resumable=True)
    outcome = runner.execute(world.tools[0], {"text": "milk"}, ToolCall())
    assert outcome.status == "held" and outcome.decision is not None
    assert outcome.decision.approval_request_id is None  # tool policy: ask first
    assert world.queue.list_pending() == []


# ── every surface that answers an approval reaches the executor ─────────────────


class _Runtime(SimpleNamespace):
    """What the API and the in-process poller ask of a runtime: resume, and the tools."""

    def resume_halted_run(self, **kw: Any) -> Any:
        raise AssertionError("a code caller's approval resumes no run")


def _api(world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from iris_harness.kernel.governance.approvals import service

    monkeypatch.setattr(service, "_queue", lambda given: given or world.queue)
    runtime = _Runtime(data_dir=tmp_path, tool_service=world.service)
    return TestClient(create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers())


def test_surface_api_and_web_run_the_approved_call(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    approval_id = world.queued()
    body = (
        _api(world, tmp_path, monkeypatch)
        .post(f"/governance/approvals/{approval_id}/respond", json={"status": "approved"})
        .json()
    )
    assert body["executed"] is True and body["caller"] == CALLER
    assert world.ran == [("add_note", {"text": "milk"})]


def test_surface_gateway_telegram_runs_it_through_the_api(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.server.channel_gateway.telegram import ApiApprovalBackend
    from iris_harness.services.channels.approval_commands import handle_approval_command

    client = _api(world, tmp_path, monkeypatch)
    backend = ApiApprovalBackend(iris_api_url="http://testserver", client=client)
    approval_id = world.queued()
    reply = handle_approval_command(
        f"/approve {approval_id}", "42", backend, allowed_users=frozenset()
    )
    assert reply is not None and reply.text.startswith("Approved:")
    assert len(world.ran) == 1


def test_surface_in_process_telegram_runs_it(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.runtime.channel_wiring import _telegram_approval_commands

    monkeypatch.setattr(
        "iris_harness.kernel.governance.approvals.ApprovalQueue", lambda **kw: world.queue
    )
    handler = _telegram_approval_commands(
        _Runtime(data_dir=tmp_path, tool_service=world.service)  # type: ignore[arg-type]
    )
    approval_id = world.queued()
    reply = handler(f"/approve {approval_id}", "42")
    assert reply is not None and reply.text.startswith("Approved:")
    assert world.ran == [("add_note", {"text": "milk"})]


def _api_down(cli_api: Any) -> None:
    import httpx

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    cli_api(refused)


def test_surface_cli_answers_through_the_running_api(
    world: _World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cli_api: Any
) -> None:
    """The call runs in the server, so approval.call_completed reaches the plugin there."""
    import httpx
    from typer.testing import CliRunner

    from iris_harness.cli.approvals import approvals_app

    client = _api(world, tmp_path, monkeypatch)
    posted: list[str] = []

    def via_server(request: httpx.Request) -> httpx.Response:
        posted.append(request.url.path)
        answer = client.post(request.url.path, json=json.loads(request.content))
        return httpx.Response(answer.status_code, json=answer.json())

    cli_api(via_server)

    def no_local_runtime() -> Any:
        raise AssertionError("answered through the API; no local runtime is built")

    monkeypatch.setattr("iris_harness.cli.approvals._queue", lambda: world.queue)
    monkeypatch.setattr("iris_harness.runtime.bootstrap.build_runtime", no_local_runtime)
    events: list[Any] = []
    world.bus.on(APPROVAL_CALL_COMPLETED, events.append)  # the server's bus
    approval_id = world.queued()
    result = CliRunner().invoke(approvals_app, ["approve", approval_id])
    assert result.exit_code == 0, result.output
    assert "The call ran" in result.output and "stays local" not in result.output
    assert posted == [f"/governance/approvals/{approval_id}/respond"]
    assert len(world.ran) == 1 and [e.status for e in events] == ["ran"]


def test_surface_cli_with_the_api_down_runs_it_locally_and_says_so(
    world: _World, monkeypatch: pytest.MonkeyPatch, cli_api: Any
) -> None:
    from typer.testing import CliRunner

    from iris_harness.cli.approvals import approvals_app

    _api_down(cli_api)
    monkeypatch.setattr("iris_harness.cli.approvals._queue", lambda: world.queue)
    monkeypatch.setattr(
        "iris_harness.runtime.bootstrap.build_runtime",
        lambda: SimpleNamespace(tool_service=world.service),
    )
    approval_id = world.queued()
    result = CliRunner().invoke(approvals_app, ["approve", approval_id])
    assert result.exit_code == 0 and "The call ran" in result.output
    assert "stays local" in " ".join(result.output.split())
    assert len(world.ran) == 1


def test_surface_cli_no_resume_leaves_a_code_call_waiting(
    world: _World, monkeypatch: pytest.MonkeyPatch, cli_api: Any
) -> None:
    from typer.testing import CliRunner

    from iris_harness.cli.approvals import approvals_app

    def no_answer(*_a: Any, **_kw: Any) -> Any:
        raise AssertionError("--no-resume must not answer a code caller's approval")

    cli_api(no_answer)
    monkeypatch.setattr("iris_harness.cli.approvals._queue", lambda: world.queue)
    monkeypatch.setattr("iris_harness.runtime.bootstrap.build_runtime", no_answer)
    approval_id = world.queued()
    result = CliRunner().invoke(approvals_app, ["approve", approval_id, "--no-resume"])
    assert result.exit_code == 0 and "still waiting" in " ".join(result.output.split())
    row = world.queue.get(approval_id)
    assert row is not None and row.status == "pending" and world.ran == []


def test_surface_cli_with_no_runtime_leaves_it_waiting(
    world: _World, monkeypatch: pytest.MonkeyPatch, cli_api: Any
) -> None:
    from typer.testing import CliRunner

    from iris_harness.cli.approvals import approvals_app

    def no_runtime() -> Any:
        raise RuntimeError("no models here")

    _api_down(cli_api)
    monkeypatch.setattr("iris_harness.cli.approvals._queue", lambda: world.queue)
    monkeypatch.setattr("iris_harness.runtime.bootstrap.build_runtime", no_runtime)
    approval_id = world.queued()
    result = CliRunner().invoke(approvals_app, ["approve", approval_id])
    assert result.exit_code == 0 and "Not answered" in result.output
    row = world.queue.get(approval_id)
    assert row is not None and row.status == "pending" and world.ran == []


def test_surface_cli_channel_does_not_prompt_for_a_code_call(
    world: _World, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An inline yes would run inside the call's own PreToolUse and execute nothing."""
    from iris_harness.kernel.governance.approvals.channels.cli_channel import CLIChannel

    def no_input(_prompt: str) -> str:
        raise AssertionError("prompted for a code caller's approval")

    monkeypatch.setattr("builtins.input", no_input)
    approval_id = world.queued()
    row = world.queue.get(approval_id)
    assert row is not None
    CLIChannel(world.queue, force_interactive=True).notify(row)
    assert f"iris approvals approve {approval_id}" in capsys.readouterr().err
    assert world.queue.get(approval_id).status == "pending"  # type: ignore[union-attr]


def test_surface_timeout_heartbeat_settles_a_lapsed_code_call(
    world: _World, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.runtime.confirmations import Confirmations
    from iris_harness.services.heartbeat import HeartbeatDefinition

    events: list[Any] = []
    world.bus.on(APPROVAL_CALL_COMPLETED, events.append)
    approval_id = world.queued()
    with world.queue._store._connect() as conn:
        conn.execute("UPDATE approval_queue SET timeout_at = '2000-01-01T00:00:00+00:00'")
        conn.commit()
    lapse_notices: list[Any] = []
    host = SimpleNamespace(
        tool_service=world.service,
        _activity_notices=lambda: SimpleNamespace(
            deliver_lapse_notice=lambda **kw: lapse_notices.append(kw)
        ),
    )
    monkeypatch.setattr(
        "iris_harness.kernel.governance.approvals.ApprovalQueue", lambda **kw: world.queue
    )
    confirmations = Confirmations(host)  # type: ignore[arg-type]
    confirmations.approval_timeout_heartbeat(
        HeartbeatDefinition(name="approval_timeout_tick", handler="x", schedule="interval:60")
    )
    assert [(e.approval_id, e.status) for e in events] == [(approval_id, "expired")]
    assert lapse_notices == [] and world.ran == []


# ── the event is the harness's: only filtered delivery, no forging ─────────────


def test_a_plugin_cannot_subscribe_to_every_plugin_s_outcomes(world: _World) -> None:
    api = world.plugin_api("p", PluginRegistry())
    with pytest.raises(PermissionError, match="on_approved_call"):
        api.subscribe(APPROVAL_CALL_COMPLETED, lambda payload: None)
    with pytest.raises(PermissionError):
        api.publish(
            APPROVAL_CALL_COMPLETED,
            ApprovalCallCompletedPayload("x", "plugin:q", "add_note", "ran", "forged"),
        )
    api.subscribe("task.created", lambda payload: None)  # any other topic is fine


def test_a_lapsed_code_call_is_not_described_as_a_halted_run(world: _World) -> None:
    from iris_harness.kernel.governance.approvals.service import lapse_notice

    approval_id = world.queued()
    row = world.queue.get(approval_id)
    assert row is not None
    text = lapse_notice(row)
    assert "add_note" in text and "was not run" in text
    assert "run " + row.run_id not in text and "halted" not in row.lapse_consequence()
    # With no executor to settle it, the sweep sends that notice rather than a run's.
    with world.queue._store._connect() as conn:
        conn.execute("UPDATE approval_queue SET timeout_at = '2000-01-01T00:00:00+00:00'")
        conn.commit()
    delivered: list[str] = []
    sweep_expired(
        queue=world.queue,
        notifier=SimpleNamespace(deliver_lapse_notice=lambda **kw: delivered.append(kw["text"])),
    )
    assert len(delivered) == 1 and "was not run" in delivered[0]
    assert "still stopped" not in delivered[0]


def test_services_events_can_neither_hear_nor_forge_the_topic(world: _World) -> None:
    """The second way onto the bus: ``services.events`` itself, not just ``api.subscribe``."""
    api = world.plugin_api("p", PluginRegistry())
    events = api.services.events
    assert events is not None
    heard: list[Any] = []
    forged = ApprovalCallCompletedPayload("x", "plugin:q", "add_note", "ran", "forged")
    with pytest.raises(PermissionError, match="on_approved_call"):
        events.on(APPROVAL_CALL_COMPLETED, heard.append)
    with pytest.raises(PermissionError):
        events.emit_sync(APPROVAL_CALL_COMPLETED, forged)
    with pytest.raises(PermissionError):
        asyncio.run(events.emit(APPROVAL_CALL_COMPLETED, forged))  # type: ignore[attr-defined]
    with pytest.raises(PermissionError):
        events.off(APPROVAL_CALL_COMPLETED, heard.append)  # type: ignore[attr-defined]

    # The real outcome still reaches the calling plugin through the sanctioned path.
    mine: list[Any] = []
    api.on_approved_call(mine.append)
    world.approve(world.queued(), executor=world.service)
    assert heard == [] and [e.status for e in mine] == ["ran"]

    # Every other topic passes straight through to the runtime's bus.
    other: list[Any] = []
    events.on("task.created", other.append)
    events.emit_sync("task.created", "t1")
    world.bus.emit_sync("task.created", "t2")
    assert other == ["t1", "t2"]


def test_a_built_runtime_hands_plugins_the_guarded_bus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import iris_harness.runtime.plugin_host as plugin_host
    from iris_harness.runtime.bootstrap import build_runtime

    seen: list[Any] = []
    real = plugin_host.load_plugins

    def capture(profile: Any, *, services: Any, registry: Any, **kwargs: Any) -> Any:
        seen.append(services)
        return real(profile, services=services, registry=registry, **kwargs)

    monkeypatch.setattr(plugin_host, "load_plugins", capture)
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    (tmp_path / "data").mkdir()
    runtime = build_runtime(
        config_dir=Path("config").resolve(),
        data_dir=tmp_path / "data",
        use_background_scheduler=False,
    )
    assert len(seen) == 1 and isinstance(seen[0].events, GuardedEventBus)
    with pytest.raises(PermissionError):
        seen[0].events.on(APPROVAL_CALL_COMPLETED, lambda payload: None)
    # The emitter keeps the raw bus, so the outcome is still delivered.
    assert runtime.tool_service is not None
    assert runtime.tool_service._events is not None
    assert runtime.tool_service._events() is runtime._event_bus


def test_api_tools_is_the_bound_entry_and_gets_the_approval_path(world: _World) -> None:
    """With the plugin view of ``services.tools`` a describe-only catalogue (capabilities
    step 3b), ``api.tools`` is a plugin's one way to call: it is still bound to the plugin
    and still gets the held-for-approval result with its ``approval_id``."""
    api = world.plugin_api("p", PluginRegistry())
    assert not hasattr(api.services.tools, "for_caller")
    assert api.tools is not None and api.tools.caller == CALLER
    with session_scope("chat-1"):
        result = api.tools.call("add_note", {"text": "milk"})
    assert result.held and not result.ok and result.approval_id is not None
    assert world.ran == []


# ── a plugin's tool that raises: through the fault boundary, as production mounts it ──


def _plugin_tool(world: _World, name: str, effect: str, confirm: str) -> PluginRegistry:
    """``name`` registered the way a mounted plugin's tool is: behind the registry's fault
    boundary, which catches the raise and records it against the plugin."""
    registry = PluginRegistry()
    registry.add_plugin(PluginRecord(name="flaky", source="test", status=PluginStatus.LOADED))
    registry.add_tool("flaky", world._tool(name, effect, confirm, fail=True))
    world.tools.append(registry.tools()[-1])
    return registry


def test_a_plugin_tool_that_raises_is_reported_failed_not_ran(world: _World) -> None:
    """Regression: the boundary answered with a sentence, so the approved call that raised
    settled as ``ran``."""
    registry = _plugin_tool(world, "flaky_write", "write", "once")
    approval_id = world.queued("flaky_write")
    outcome = world.approve(approval_id, executor=world.service)
    assert outcome.executed and [o["status"] for o in world.outcomes()] == ["failed"]
    record = next(r for r in registry.plugins() if r.name == "flaky")
    assert record.failure_count == 1 and record.status is PluginStatus.DEGRADED


def test_a_plugin_tool_that_raises_is_not_ok_for_a_code_caller(world: _World) -> None:
    _plugin_tool(world, "flaky_read", "read", "never")
    result = world.service.for_caller(CALLER).call("flaky_read", {})
    assert not result.ok and not result.held
    # The caller is told the boundary's sentence, as before; only ``ok`` changed.
    assert result.text.startswith("flaky_read is unavailable (plugin 'flaky' raised RuntimeError")
