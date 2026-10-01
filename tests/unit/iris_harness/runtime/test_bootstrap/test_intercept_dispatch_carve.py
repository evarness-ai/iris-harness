"""The intercept-dispatch carve (OSS plan M5.7 track C, slice 15).

``_dispatch_intercepts``, ``effective_intercept_chain``, ``_pre_intercept_activity_hint``
and ``_log_chat_result`` left ``IrisRuntime`` as ``InterceptDispatch``, held as
``runtime.intercepts``. ``TurnHost`` lost two private members for it. What mypy cannot
check about that:

* **Core rows still resolve against the runtime.** A ``handler:`` row names where the
  handler lives on the runtime (``confirmations.handle_confirmation_turn``); resolving it
  against the collaborator instead would drop every core row with only a warning.
* **The host declaration is exact**, measured by AST from both sides.
* **A plugin's activity hint reaches the stream.** The stage asks the collaborator, and
  ``chat_stream`` is the surface the web UI reads the "working" line from.
* **Nothing reaches for the moved names on the runtime.**
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from iris_harness.runtime import build_runtime
from iris_harness.runtime.intercepts import DEFAULT_INTERCEPTS, InterceptSpec, load_intercept_chain

MODULE = Path("src/iris_harness/runtime/intercept_dispatch.py")
SWEPT = ("src", "services", "tests", "scripts")
# Renamed on the collaborator: the old name is stale wherever it is read.
RENAMED = {"_dispatch_intercepts", "effective_intercept_chain", "_pre_intercept_activity_hint"}


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


@pytest.mark.parametrize(
    "chain",
    [load_intercept_chain(Path("config/intercepts.yaml")), DEFAULT_INTERCEPTS],
    ids=["config/intercepts.yaml", "DEFAULT_INTERCEPTS"],
)
def test_every_declared_core_row_is_in_the_effective_chain(runtime, chain) -> None:
    """Slice 10's test resolves each row against the runtime directly; this one goes
    through the collaborator, which is the only place the resolution now happens."""
    runtime.intercept_chain = tuple(chain)
    effective = {spec.name for spec, _handler in runtime.intercepts.effective_chain()}
    missing = [
        spec.name
        for spec in chain
        if not spec.handler.startswith("plugin:") and spec.name not in effective
    ]
    assert missing == []


def test_a_plugin_hint_reaches_the_stream_before_its_intercept_runs(runtime) -> None:
    def slow_intercept(message: str, *, session_id: str, span: Any = None) -> Any:
        return runtime.replies.system_chat_result(
            message=message, session_id=session_id, response="counted", metadata={}
        )

    runtime.plugin_registry.add_intercept(
        "t",
        InterceptSpec("carve_slow", "plugin:t"),
        slow_intercept,
        activity_hint=lambda m: "scanning…" if "carve" in m else None,
    )

    assert runtime.intercepts.activity_hint("carve this folder") == "scanning…"
    assert runtime.intercepts.activity_hint("what time is it") is None

    events = list(runtime.chat_stream("carve this folder", session_id="s-hint"))
    kinds = [(e.kind, e.text) for e in events if e.kind in ("activity", "done")]
    assert kinds[0] == ("activity", "scanning…")
    assert kinds[-1][0] == "done" and events[-1].result.response == "counted"


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
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InterceptDispatchHost"
    )
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)} | {
        n.name for n in cls.body if isinstance(n, ast.FunctionDef)
    }


def test_intercept_dispatch_host_declares_exactly_what_the_module_reaches() -> None:
    assert _host_reached() == _host_declared()
    assert _host_declared() == {
        "continuations",
        "intercept_chain",
        "openers",
        "plugin_registry",
        "profile",
    }


def test_nothing_reaches_for_the_moved_dispatch_members_off_a_runtime() -> None:
    leftovers: list[str] = []
    for path in sorted(p for root in SWEPT for p in Path(root).rglob("*.py")):
        if path == MODULE:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr in RENAMED:
                leftovers.append(f"{path}:{node.lineno} .{node.attr}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in RENAMED
            ):
                leftovers.append(f"{path}:{node.lineno} setattr {node.args[1].value!r}")
    assert leftovers == []
