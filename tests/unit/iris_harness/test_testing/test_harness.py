"""``iris_harness.testing.harness``: a real governed IRIS in a throwaway home (R16, L2).

The harness builds the runtime through the composition root, so these drive real turns
and read the real audit ledger; only the model (scripted), the home, the keyring and the
network are replaced.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import pytest

from iris_harness.sdk import PluginAPI
from iris_harness.sdk.audit import AuditLog
from iris_harness.testing import (
    NetworkBlockedError,
    ScriptedChatModel,
    TurnAuditRow,
    TurnEvent,
    harness,
    plugin,
)

_SCRIPT = {"default": {"content": "Scripted answer."}}


def _tree(root: Path) -> dict[str, tuple[int, int]]:
    """Every file under ``root``: relative path -> (size, mtime_ns)."""
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_turn_runs_the_governed_pipeline_and_is_audited(entry: str) -> None:
    with harness(fake_model=_SCRIPT) as h:
        if entry == "chat":
            result = h.chat("hello there, how are you?")
            assert result.events == ()
        else:
            result = h.chat_stream("hello there, how are you?")
            assert result.events[-1] == TurnEvent(kind="done", text="Scripted answer.")
        assert result.answered and result.error is None
        assert result.text == "Scripted answer."
        session = h.turns[-1].session_id
        assert result.session_id == session
        assert h.turns[-1].result is result
        # The turn's audit refs are exactly the ledger rows written in its session.
        assert result.audit_refs
        assert result.audit_refs == tuple(r.id for r in h.audit_rows(session_id=session))
        assert h.audit_rows(hook_point="pre_turn", session_id=session)
        assert h.audit_rows(hook_point="pre_response", session_id=session)
        calls = h.model_calls()
        assert calls, "the general lane answers with the model"
        # Every call went to the scripted model; a row that names a provider names it.
        assert {c.model for c in calls} and all(c.rule == "default" for c in calls)
        # (The provider is in the ledger's raw payload, which TurnAuditRow leaves out.)
        for raw in AuditLog(db_path=h.audit_db).query():
            if raw.hook_point == "pre_llm_call":
                payload = raw.payload_json
                assert '"provider"' not in payload or '"provider": "fake"' in payload
        assert h.audit_gaps() == []


def test_audit_gaps_names_a_model_call_without_its_row() -> None:
    with harness(fake_model=_SCRIPT) as h:
        h.chat("hello there")
        # A call the ledger never saw: the transport answered it with no client around
        # it, so no hook fired (a bare CodingLLMClient still gets the default kernel).
        from langchain_core.messages import HumanMessage

        ScriptedChatModel(model="m").invoke([HumanMessage(content="off the books")])
        gaps = h.audit_gaps()
    assert len(gaps) == 1
    assert "audited at pre_llm_call" in gaps[0]


def _greeter(api: PluginAPI) -> None:
    services = api.services

    def greet(message: str, *, session_id: str, span: Any = None) -> Any:
        if "ping the greeter" not in message:
            return None
        return services.deterministic_reply(
            message=message, session_id=session_id, response="pong", metadata={}, span=span
        )

    api.register_intercept("greeter_ping", greet)


def test_a_plugin_supplied_in_process_mounts_and_its_answer_is_audited() -> None:
    greeter = plugin(_greeter, manifest={"name": "greeter", "provides": ["intercept"]})
    with harness(plugins=[greeter], fake_model=_SCRIPT) as h:
        assert h.plugin_loaded("greeter")
        assert h.plugin_loaded("system")
        result = h.chat("please ping the greeter")
        assert result.text == "pong"
        # A deterministic answer passes PRE_RESPONSE like a generated one (R15), and its
        # row says so, with the handler's name.
        rows = h.audit_rows(hook_point="pre_response", session_id=h.turns[-1].session_id)
        assert rows and all(isinstance(r, TurnAuditRow) for r in rows)
        marked = [r for r in rows if r.deterministic]
        assert [(r.plugin, r.handler, r.decision) for r in marked] == [
            ("response_safety", "greeter_ping", "allow")
        ]
        assert h.model_calls() == ()
        assert h.audit_gaps() == []


def test_an_in_process_plugin_gets_the_manifest_checks() -> None:
    def registers_an_undeclared_tool(api: PluginAPI) -> None:
        api.register_tool("undeclared", "d", lambda args: "x")

    sneaky = plugin(registers_an_undeclared_tool, name="sneaky")
    with harness(plugins=[sneaky], fake_model=_SCRIPT) as h:
        # Refused at registration (ADR-0110) and recorded against the plugin, which
        # stays mounted for whatever else it registered.
        assert h.plugins()["sneaky"][0] == "degraded"
        assert not h.plugin_loaded("sneaky")


def test_an_in_process_plugin_never_stands_in_for_a_profile_one() -> None:
    with pytest.raises(ValueError, match="already in profile"):
        with harness(plugins=[plugin(_greeter, name="system")]):
            pass


def test_a_plugin_needs_a_name() -> None:
    with pytest.raises(ValueError, match="needs a name"):
        plugin(_greeter)
    with pytest.raises(ValueError, match="disagrees"):
        plugin(_greeter, name="a", manifest={"name": "b"})


def test_the_harness_is_isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import keyring

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    # The suite's own relocated home and data dir stand in for the owner's.
    outer = [Path(os.environ["IRIS_HOME"]), Path(os.environ["IRIS_DATA_DIR"])]
    before = [_tree(root) for root in outer]
    env_before = dict(os.environ)
    keyring_before = keyring.get_keyring()
    home = tmp_path / "home"

    with harness(fake_model=_SCRIPT, home=home) as h:
        h.chat("hello there")
        h.chat_stream("and again")
        assert h.home == home.resolve()
        assert os.environ["IRIS_HOME"] == str(home.resolve())
        # A throwaway master key in the environment; the keyring refuses every call.
        assert os.environ["IRIS_VAULT_MASTER_KEY"] != env_before.get("IRIS_VAULT_MASTER_KEY")
        assert type(keyring.get_keyring()).__module__ == "keyring.backends.fail"
        with pytest.raises(NetworkBlockedError):
            socket.create_connection(("192.0.2.1", 80), timeout=1)
        assert h.audit_db.is_file()

    assert [_tree(root) for root in outer] == before
    assert list(cwd.iterdir()) == []
    assert dict(os.environ) == env_before
    assert keyring.get_keyring() is keyring_before
    assert _tree(home), "the harness wrote its stores into its own home"


def test_the_owner_settings_do_not_reach_the_harness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_PROFILE", "default")
    monkeypatch.setenv("GITHUB_TOKEN", "owner-token")
    with harness(fake_model=_SCRIPT, env={"IRIS_EMAIL_JUDGE": "1"}) as h:
        assert os.environ["IRIS_PROFILE"] == "minimal"
        assert "GITHUB_TOKEN" not in os.environ
        assert os.environ["IRIS_EMAIL_JUDGE"] == "1"
        assert set(h.plugins()) == {"system"}
    assert os.environ["GITHUB_TOKEN"] == "owner-token"


def test_without_a_script_a_model_call_fails_instead_of_reaching_a_model() -> None:
    with harness() as h:
        result = h.chat_stream("hello there")
    assert result.events[-1].kind in {"done", "error"}
    assert result.answered == (result.events[-1].kind == "done")
    assert h.model_calls() == ()


def _echo(api: PluginAPI) -> None:
    api.register_tool(
        "echo_back",
        'Repeat the given word back. Args: {"word": str}.',
        lambda args: f"echo: {args.get('word', '')}",
    )


_ECHO_SCRIPT = {
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
                'Action Input: {"word": "banana"}'
            },
        },
    ]
}


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_each_loop_step_is_its_own_audited_model_call(entry: str) -> None:
    echo = plugin(
        _echo,
        manifest={"name": "echo", "provides": ["tool"], "tools": {"echo_back": {"effect": "read"}}},
    )
    with harness(plugins=[echo], fake_model=_ECHO_SCRIPT) as h:
        if entry == "chat":
            result = h.chat("Please echo the word banana")
        else:
            result = h.chat_stream("Please echo the word banana")
            assert result.answered, result.error
        assert result.text == "The tool said banana."
        assert [c.rule for c in h.model_calls()] == ["call the tool", "answer from the tool"]
        # Two model calls of one loop run: one key per step (they used to share one).
        # (The router's call is audited too; the script has no rule for it, so it fails
        # and the router falls back to keywords -- an audited attempt, never answered.)
        rows = h.audit_rows(hook_point="pre_llm_call")
        loop_run = {r.run_id for r in rows if r.step_id is not None}
        assert len(loop_run) == 1
        assert {r.step_id for r in rows if r.run_id in loop_run} == {0, 1}
        assert h.audit_gaps() == []


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
            "name": "declined",
            "match": {"user": r"(?s)Observation: The owner rejected this"},
            "reply": {"content": "Thought: Fine.\nFinal Answer: The memo stays."},
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


@pytest.mark.parametrize("approve", [True, False])
def test_respond_to_approval_resumes_the_halted_turn(approve: bool) -> None:
    from iris_harness.sdk.approvals import ApprovalQueue

    shredded: list[str] = []

    def shredder(api: PluginAPI) -> None:
        def shred(args: dict[str, Any]) -> str:
            shredded.append(str(args.get("doc", "")))
            return f"shredded {args.get('doc')}"

        api.register_tool("shred_doc", 'Shred a document. Args: {"doc": str}.', shred)

    manifest = {
        "name": "shredder",
        "provides": ["tool"],
        "tools": {"shred_doc": {"effect": "destructive"}},
    }
    with harness(plugins=[plugin(shredder, manifest=manifest)], fake_model=_SHRED_SCRIPT) as h:
        result = h.chat("Shred the memo")
        assert "needs your approval" in result.text
        assert shredded == []
        [pending] = ApprovalQueue().list_pending()

        told = h.respond_to_approval(pending.approval_id, approve=approve)

        # Approved: the pinned call ran and the resumed run answered. Rejected: nothing
        # ran, and the resumed run said so.
        assert shredded == (["memo"] if approve else [])
        assert told == ("The memo is shredded." if approve else "The memo stays.")
        assert ApprovalQueue().list_pending() == []


def test_the_health_watcher_installed_before_the_harness_is_put_back() -> None:
    """The system plugin installs a process-wide health watcher whose store lives in the
    harness's temp home. Left installed after the home is deleted, it sent a later
    `/health/connectors` read to a database that no longer exists."""
    from iris_harness.services.health.watch import current_watcher, install_watcher

    sentinel = object()
    install_watcher(sentinel)  # type: ignore[arg-type]
    try:
        with harness(fake_model=_SCRIPT):
            assert current_watcher() is not sentinel  # the harness's runtime installed its own
        assert current_watcher() is sentinel
    finally:
        install_watcher(None)


_LEAK_TOPIC = "harness.leak_probe"


def _leaky(api: PluginAPI) -> None:
    """Fills process-wide seams a second harness run must not inherit."""
    api.register_api_router("leaky", lambda: None)
    api.register_credential_check("leaky", lambda net_probe: [])
    api.register_footer_line("leaky", lambda start, end: "leaky line")
    api.subscribe(_LEAK_TOPIC, lambda payload: None, scope="process")


def _what_leaks() -> dict[str, bool]:
    from iris_harness.foundation.eventbus import bus as eventbus
    from iris_harness.runtime.api_routes import registered_api_routers
    from iris_harness.services.digest import footer
    from iris_harness.services.health import credentials

    return {
        "api_router": "leaky" in registered_api_routers(),
        "credential_check": "leaky" in credentials._registered,
        "footer_line": "leaky" in footer._LINES,
        # Read the bus as it is, never through get_default_bus(): that creates one when
        # none exists yet, and the probe would itself change the state this test compares.
        "process_subscription": bool(
            eventbus._default_bus is not None and eventbus._default_bus._handlers.get(_LEAK_TOPIC)
        ),
    }


def _state_differences(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    return sorted(name for name in before if before[name] != after[name])


def test_a_second_run_does_not_inherit_the_first_runs_process_state() -> None:
    from iris_harness.foundation.process_state import snapshot_process_state

    before = snapshot_process_state().values
    leaky = plugin(_leaky, name="leaky")
    with harness(plugins=[leaky], fake_model=_SCRIPT) as h:
        assert h.plugin_loaded("leaky")
        assert all(_what_leaks().values()), _what_leaks()
    assert not any(_what_leaks().values()), _what_leaks()

    with harness(plugins=[plugin(_greeter, name="greeter")], fake_model=_SCRIPT) as h:
        assert set(h.plugins()) == {"system", "greeter"}
        assert not any(_what_leaks().values()), _what_leaks()
        h.chat("hello there")

    # Every declared piece of process state is exactly what it was before either run.
    assert _state_differences(before, snapshot_process_state().values) == []
