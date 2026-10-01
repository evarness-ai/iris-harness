"""The self-learning loop carve (OSS plan M5.7 track C, slice 11).

``HeartbeatRunnersMixin`` became ``SelfLearningLoop``, held as ``runtime.learning_loop``.
What mypy cannot check about that:

* **The host declaration is exact**, measured by AST from both sides.
* **The scheduler fires the collaborator's jobs.** The heartbeats are registered by name;
  a registration left on a stale bound method, or on a second loop, raises nothing.
* **Startup still loads the escalation priors**, the one call outside registration.
* **Nothing reaches for the old names.** They were private methods on the runtime; a
  leftover call on a dataclass fails only when that line runs.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from iris_harness.runtime import build_runtime
from iris_harness.runtime.self_learning_loop import SelfLearningLoop

MODULE = Path("src/iris_harness/runtime/self_learning_loop.py")
SWEPT = ("src", "services", "tests", "scripts")

# heartbeat name -> the collaborator method the scheduler must hold
JOBS = {
    "escalation_priors_tick": "escalation_priors_heartbeat",
    "crystallize_tick": "crystallize_heartbeat",
    "routine_reflection_tick": "routine_reflection_heartbeat",
    "experiment_remeasure_tick": "experiment_remeasure_heartbeat",
    "sandbox_preflight_tick": "sandbox_preflight_heartbeat",
}
RENAMED = {f"_{public}" for public in JOBS.values()} | {"_refresh_escalation_priors"}


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
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SelfLearningHost")
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}


def test_self_learning_host_declares_exactly_what_the_module_reaches() -> None:
    assert _host_reached() == _host_declared()
    assert len(_host_declared()) == 7  # the plan doc's slice-11 section says seven


@pytest.fixture()
def started(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls: list[SelfLearningLoop] = []

    def record(self: SelfLearningLoop) -> int:
        calls.append(self)
        return 0

    monkeypatch.setattr(SelfLearningLoop, "refresh_escalation_priors", record)
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    (tmp_path / "config").mkdir()
    (tmp_path / "data").mkdir()
    rt = build_runtime(
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
        use_background_scheduler=False,
    )
    rt.startup()
    try:
        yield rt, calls
    finally:
        rt.shutdown()


def test_the_scheduler_holds_the_runtimes_own_loop_jobs(started) -> None:
    runtime, _ = started
    for name, method in JOBS.items():
        handler = runtime.heartbeats._handlers[name]
        assert handler.__self__ is runtime.learning_loop, name
        assert handler.__func__ is getattr(SelfLearningLoop, method), name


def test_startup_loads_the_escalation_priors_on_the_runtimes_loop(started) -> None:
    runtime, calls = started
    assert calls == [runtime.learning_loop]


def test_nothing_calls_the_old_private_names() -> None:
    leftovers: list[str] = []
    for path in sorted(p for root in SWEPT for p in Path(root).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and node.attr in RENAMED:
                leftovers.append(f"{path}:{node.lineno} .{node.attr}")
    assert leftovers == []
