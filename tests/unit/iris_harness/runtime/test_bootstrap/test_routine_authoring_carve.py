"""The routine-authoring carve (OSS plan M5.7 track C, slice 10).

``RoutineAuthoringMixin`` became ``RoutineAuthoring``, held as ``runtime.routines``.
What mypy cannot check about that:

* **The intercept chain still reaches the handlers.** ``config/intercepts.yaml`` names
  core handlers by string and the chain skips a name that does not resolve with only a
  warning — so a row left pointing at the old runtime method would silently drop routine
  authoring from chat. Every declared core row must resolve on a built runtime.
* **The host declaration is exact**, measured by AST from both sides.
* **Nothing reaches for the moved names on the runtime.** A dataclass will not raise on
  ``runtime._pending_capability_swaps`` being assigned, and ``monkeypatch.setattr(runtime,
  "_routine_authoring_llm_caller", ...)`` would stub nothing.
* **The plugin service reads the live state.** ``conversation_in_flight`` must be bound
  to the runtime's own collaborator, not a second one with empty dicts.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from iris_harness.runtime import build_runtime
from iris_harness.runtime.intercepts import (
    DEFAULT_INTERCEPTS,
    load_intercept_chain,
    resolve_runtime_handler,
)

MODULE = Path("src/iris_harness/runtime/routine_authoring.py")
SWEPT = ("src", "services", "tests", "scripts")

# Renamed public on the collaborator: the old name is stale wherever it is read.
RENAMED = {
    "_handle_routine_management_turn",
    "_handle_routine_authoring_turn",
    "_routine_authoring_llm_caller",
    "_has_active_routine_conversation",
}
# Moved from the runtime's fields, same names: valid only off ``.routines``.
MOVED_STATE = {
    "_pending_refinement_picks",
    "_pending_capability_swaps",
    "_pending_capability_clarifications",
    "_recent_refinement_snapshots",
}
MOVED = RENAMED | MOVED_STATE


def _stale(name: str, owner: ast.expr) -> bool:
    return name in RENAMED or (name in MOVED_STATE and not _via_routines(owner))


def test_a_dotted_handler_name_walks_to_a_collaborator_method() -> None:
    def handler() -> str:
        return "answered"

    host = SimpleNamespace(routines=SimpleNamespace(handle=handler), plain=handler)

    assert resolve_runtime_handler(host, "routines.handle") is handler
    assert resolve_runtime_handler(host, "plain") is handler
    assert resolve_runtime_handler(host, "routines.missing") is None
    assert resolve_runtime_handler(host, "missing.handle") is None


@pytest.fixture()
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    (tmp_path / "data").mkdir()
    return build_runtime(
        config_dir=Path("config").resolve(),
        data_dir=tmp_path / "data",
        use_background_scheduler=False,
    )


@pytest.mark.parametrize(
    "chain",
    [load_intercept_chain(Path("config/intercepts.yaml")), DEFAULT_INTERCEPTS],
    ids=["config/intercepts.yaml", "DEFAULT_INTERCEPTS"],
)
def test_every_declared_core_intercept_resolves_on_a_built_runtime(runtime, chain) -> None:
    unresolved = [
        f"{spec.name} -> {spec.handler}"
        for spec in chain
        if not spec.handler.startswith("plugin:")
        and not callable(resolve_runtime_handler(runtime, spec.handler))
    ]
    assert unresolved == []


def test_the_chain_dispatches_routine_turns_to_the_runtimes_collaborator(runtime) -> None:
    handlers = {spec.name: handler for spec, handler in runtime.intercepts.effective_chain()}

    for name in ("routine_management", "routine_authoring"):
        assert handlers[name].__self__ is runtime.routines, name


def test_the_planner_service_sees_the_live_routine_conversation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`HarnessServices.conversation_in_flight` is bound in `_mount_plugins`. Bound to a
    second collaborator it would always answer False, and the planner's brief intercept
    would answer a turn that belongs to a routine clarification."""
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
    in_flight = captured["services"].conversation_in_flight  # type: ignore[attr-defined]

    assert in_flight("s-live") is False
    rt.routines._pending_capability_swaps["s-live"] = {"routine_id": "r1"}
    assert in_flight("s-live") is True


def _host_reached() -> set[str]:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "_host"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
    }


def _host_declared() -> set[str]:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RoutineHost")
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)} | {
        n.name for n in cls.body if isinstance(n, ast.FunctionDef)
    }


def test_routine_host_declares_exactly_what_the_module_reaches() -> None:
    assert _host_reached() == _host_declared()
    assert len(_host_declared()) == 6  # the plan doc's slice-10 section says six


def _via_routines(owner: ast.expr) -> bool:
    """``<anything>.routines`` — the collaborator, where the moved state now lives."""
    return isinstance(owner, ast.Attribute) and owner.attr == "routines"


def test_nothing_reaches_for_the_moved_routine_members_off_a_runtime() -> None:
    """The state dicts kept their names on the collaborator, so they may be read off
    ``.routines``; the renamed methods' old names are stale everywhere. Inside the module
    both are ``self``'s."""
    leftovers: list[str] = []
    for path in sorted(p for root in SWEPT for p in Path(root).rglob("*.py")):
        if path == MODULE:
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
