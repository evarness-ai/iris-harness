"""The approvals/confirmation carve (OSS plan M5.7 track C, slice 14).

The confirmation turn, the approvals timeout sweep and the deterministic reply builder
left ``IrisRuntime`` as two collaborators: ``Confirmations`` (``runtime.confirmations``)
and ``DeterministicReplies`` (``runtime.replies``). What mypy cannot check about that:

* **The intercept chain still reaches the turn.** ``config/intercepts.yaml`` names the
  handler by string and the chain skips a name that does not resolve with only a
  warning — so a row left pointing at the old runtime method would silently drop
  in-chat confirmation, and every "approve" would route as a fresh query.
* **Both host declarations are exact**, measured by AST from both sides.
* **The three seams are bound to the runtime's own objects**: the heartbeat handler,
  ``HarnessServices.deterministic_reply`` and the chain's handler. A second
  ``Confirmations`` would sweep with its own queue and never the one a test seeds.
* **Nothing reaches for the moved names on the runtime.** A dataclass will not raise on
  ``runtime._approvals_queue_cache`` being assigned; the seeded queue would be ignored.
* **The turn still works end to end** through the YAML name: approve dispatches to the
  registered executor by kind, reject cancels, an unknown kind acknowledges.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from iris_harness.runtime import build_runtime
from iris_harness.runtime.confirmations import Confirmations
from iris_harness.runtime.replies import DeterministicReplies

CONFIRMATIONS = Path("src/iris_harness/runtime/confirmations.py")
REPLIES = Path("src/iris_harness/runtime/replies.py")
SWEPT = ("src", "services", "tests", "scripts")

# Renamed public on a collaborator: the old name is stale wherever it is read.
RENAMED = {
    "_handle_confirmation_turn",
    "_pending_confirmation_options",
    "_approval_timeout_heartbeat",
    "_approvals_queue",
    "_approvals_router",
    "_system_chat_result",
    "_reminder_chat_result",
}
# Moved from the runtime, same names: valid only off ``.confirmations``.
MOVED_STATE = {"_approvals_queue_cache", "_approvals_router_cache"}
MOVED = RENAMED | MOVED_STATE


@pytest.fixture()
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    (tmp_path / "data").mkdir()
    rt = build_runtime(
        config_dir=Path("config").resolve(),
        data_dir=tmp_path / "data",
        use_background_scheduler=False,
    )
    rt.startup()
    try:
        yield rt
    finally:
        rt.shutdown()


# ── the seams are the runtime's own collaborators ────────────────────────────


def test_the_chain_dispatches_the_confirmation_turn_to_the_runtimes_collaborator(
    runtime,
) -> None:
    handlers = {spec.name: handler for spec, handler in runtime.intercepts.effective_chain()}

    assert handlers["confirmation"].__self__ is runtime.confirmations
    assert handlers["confirmation"].__func__ is Confirmations.handle_confirmation_turn


def test_the_approval_timeout_tick_is_the_collaborators_job(runtime) -> None:
    handler = runtime.heartbeats._handlers["approval_timeout_tick"]

    assert handler.__self__ is runtime.confirmations
    assert handler.__func__ is Confirmations.approval_timeout_heartbeat


def test_the_deterministic_reply_service_is_the_runtimes_replies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`HarnessServices.deterministic_reply` is bound in `_mount_plugins`. Bound to a
    second collaborator it would still build a ChatResult, but record the turn and the
    signal through whatever host that one was handed."""
    # Patch the module `bootstrap._mount_plugins` imports FROM, not the facade that
    # re-exports it: a monkeypatch on the wrong module is a no-op the test cannot see.
    import iris_harness.runtime.plugin_host as plugins_pkg

    captured: dict[str, object] = {}
    real_load = plugins_pkg.load_plugins

    def spy(profile, *, services, registry, **kw):  # type: ignore[no-untyped-def]
        captured["services"] = services
        return real_load(profile, services=services, registry=registry, **kw)

    monkeypatch.setattr(plugins_pkg, "load_plugins", spy)
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    (tmp_path / "data").mkdir()
    rt = build_runtime(
        config_dir=Path("config").resolve(),
        data_dir=tmp_path / "data",
        use_background_scheduler=False,
    )
    reply = captured["services"].deterministic_reply  # type: ignore[attr-defined]

    assert reply.__self__ is rt.replies
    assert reply.__func__ is DeterministicReplies.system_chat_result


# ── the turn, end to end through the YAML name ───────────────────────────────


def _task_store(runtime) -> Any:  # type: ignore[no-untyped-def]
    from iris_harness.services.tasks import TaskStore

    store = TaskStore(db_path=runtime.data_dir / "tasks.db")
    store.ensure_schema()
    return store


def _action_center_task(runtime, dedup_key: str) -> Any:  # type: ignore[no-untyped-def]
    return _task_store(runtime).upsert(
        dedup_key=dedup_key,
        title="Approve it",
        description="carve",
        source_kind="other",
        source_id=dedup_key,
    )


# The question is asked as owner "confirmation" — the intercept's own name — because the
# ADR-0106 shield only lets a confirmation-resolving intercept answer its own owner's
# question. That is how the calendar plugin asks (plugins_builtin/calendar/handlers.py).


def test_approve_dispatches_to_the_registered_executor_by_kind(runtime) -> None:
    seen: list[dict[str, Any]] = []

    def executor(pending: dict[str, Any], **kw: Any) -> Any:
        seen.append(pending)
        return runtime.replies.system_chat_result(
            message=kw["message"],
            session_id=kw["session_id"],
            response="done",
            metadata={"confirmation": "approved", "kind": pending["kind"]},
        )

    runtime.plugin_registry.add_confirmation_executor("t", "carve_kind", executor)
    runtime.continuations.ask(
        "s-approve", "confirmation", question="do it?", executor_kind="carve_kind", payload={"n": 1}
    )
    assert runtime.confirmations.pending_options("s-approve") == ["approve", "reject"]

    out = runtime.chat("approve", session_id="s-approve")

    assert seen == [{"n": 1, "kind": "carve_kind"}]
    assert out.response == "done" and out.metadata["confirmation"] == "approved"
    assert runtime.continuations.pending("s-approve") is None


def test_reject_cancels_without_running_anything(runtime) -> None:
    def executor(pending: dict[str, Any], **kw: Any) -> Any:  # pragma: no cover
        raise AssertionError("a rejected action must not run")

    runtime.plugin_registry.add_confirmation_executor("t", "carve_reject", executor)
    task = _action_center_task(runtime, "carve:reject")
    runtime.continuations.ask(
        "s-reject",
        "confirmation",
        question="do it?",
        executor_kind="carve_reject",
        payload={"task_dedup": task.dedup_key},
    )

    out = runtime.chat("reject", session_id="s-reject")

    assert out.metadata["confirmation"] == "rejected"
    assert out.intent == "calendar" and out.sources == ("reminders",)
    assert runtime.continuations.pending("s-reject") is None
    # The Action Center's pending action is dropped with it (dropped tasks don't surface).
    assert _task_store(runtime).get_by_dedup_key(task.dedup_key) is None
    # The deterministic reply is a recorded turn like any other.
    assert [t.content for t in runtime.sessions.conversations["s-reject"]] == [
        "reject",
        out.response,
    ]


def test_an_unknown_kind_is_acknowledged_without_acting(runtime) -> None:
    task = _action_center_task(runtime, "carve:unknown")
    runtime.continuations.ask(
        "s-unknown",
        "confirmation",
        question="do it?",
        executor_kind="nobody_registered",
        payload={"task_dedup": task.dedup_key},
    )

    out = runtime.chat("approve", session_id="s-unknown")

    assert out.response == "Approved."
    assert out.metadata["confirmation"] == "approved"
    assert _task_store(runtime).get_by_dedup_key(task.dedup_key).status == "done"


def test_a_plain_question_offers_no_buttons(runtime) -> None:
    runtime.continuations.ask("s-plain", "planner", question="which one?", intent="planner")

    assert runtime.confirmations.pending_options("s-plain") is None


# ── the host declarations are exact ───────────────────────────────────────────


def _host_reached(module: Path) -> set[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "_host"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
    }


def _host_declared(module: Path, protocol: str) -> set[str]:
    tree = ast.parse(module.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == protocol)
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)} | {
        n.name for n in cls.body if isinstance(n, ast.FunctionDef)
    }


def test_confirmations_host_declares_exactly_what_the_module_reaches() -> None:
    declared = _host_declared(CONFIRMATIONS, "ConfirmationsHost")
    assert _host_reached(CONFIRMATIONS) == declared
    assert declared == {
        "continuations",
        "data_dir",
        "plugin_registry",
        "replies",
        "tool_service",
        "_activity_notices",
    }


def test_replies_host_declares_exactly_what_the_module_reaches() -> None:
    declared = _host_declared(REPLIES, "RepliesHost")
    assert _host_reached(REPLIES) == declared
    assert declared == {"capture", "sessions"}


# ── nothing reaches for the moved names off a runtime ────────────────────────


def _via_confirmations(owner: ast.expr) -> bool:
    """``<anything>.confirmations`` — the collaborator, where the moved state now lives."""
    return isinstance(owner, ast.Attribute) and owner.attr == "confirmations"


def _stale(name: str, owner: ast.expr) -> bool:
    return name in RENAMED or (name in MOVED_STATE and not _via_confirmations(owner))


def test_nothing_reaches_for_the_moved_confirmation_members_off_a_runtime() -> None:
    """The two caches kept their names on the collaborator, so they may be read off
    ``.confirmations``; the renamed methods' old names are stale everywhere. Inside the
    two modules both are ``self``'s."""
    leftovers: list[str] = []
    for path in sorted(p for root in SWEPT for p in Path(root).rglob("*.py")):
        if path in (CONFIRMATIONS, REPLIES):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and _stale(node.attr, node.value):
                leftovers.append(f"{path}:{node.lineno} .{node.attr}")
            # monkeypatch.setattr(runtime, "<moved name>", ...) is an attribute by string.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in MOVED
                and _stale(node.args[1].value, node.args[0])
            ):
                leftovers.append(f"{path}:{node.lineno} setattr {node.args[1].value!r}")
    assert leftovers == []
