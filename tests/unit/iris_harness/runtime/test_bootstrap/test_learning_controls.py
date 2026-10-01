"""The learning controls carve (OSS plan M5.7 track C, slice 9).

Three things mypy cannot check about ``runtime/learning_controls.py``:

* **The host declaration is exact.** mypy fails a read of an undeclared member, but not
  a declared member nothing reads any more. Measured by AST from both sides, the way
  ``test_turn_host_contract.py`` measures ``TurnHost``.
* **Nothing still reaches for the moved state on the runtime.** ``IrisRuntime`` is a
  plain dataclass, so ``runtime.behavior_miner = stub`` after the move raises nothing: it
  creates a dead attribute that no run reads, and the test around it passes for the
  wrong reason. ``src/iris_harness/server/iris_api`` reads the runtime as ``Any``, so mypy is blind
  there too. Every ``*.py`` under src, services, tests and scripts is swept.
* **The wiring.** The heartbeats the scheduler fires and the compaction buffer must
  reach the collaborator's live state, not a copy taken at build time.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from iris_harness.runtime import build_runtime
from iris_harness.services.heartbeat.config import load_heartbeats

MODULE = Path("src/iris_harness/runtime/learning_controls.py")
# Everywhere Python that calls the runtime lives. Not `config/`: the crystallizer's
# `sandbox_script.py` artifacts there are prose, not Python.
SWEPT = ("src", "services", "tests", "scripts")

# State that lived on ``IrisRuntime`` and is now the collaborator's.
MOVED_STATE = {
    "behavior_miner",
    "intention_analyst",
    "learning_analyst",
    "learning_analyst_model",
    "_self_management_override",
}
# Methods that lived on ``IrisRuntime`` or ``HeartbeatRunnersMixin``, under their old names.
MOVED_METHODS = {
    "learning_flags",
    "self_management_enabled",
    "_learning_capability_enabled",
    "set_learning_flag",
    "run_learning_now",
    "_learning_invoke_for_preview",
    "preview_behavior_mining",
    "preview_intention_rollup",
    "_run_learning_analysis",
    "_learning_analysis_heartbeat",
    "_run_behavior_mining",
    "_behavior_mining_heartbeat",
    "_run_intention_rollup",
    "_intention_rollup_heartbeat",
    "_dedupe_intentions_semantically",
    "_gather_intention_context",
}


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
    } | {
        # ``from_env`` reads the host through its parameter, not ``self._host``.
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "host"
    }


def _host_declared() -> set[str]:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LearningHost")
    return {
        n.target.id
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    }


def test_learning_host_declares_exactly_what_the_module_reaches() -> None:
    assert _host_reached() == _host_declared()
    assert len(_host_declared()) == 8  # the plan doc's slice-9 section says eight


def test_nothing_reaches_for_the_moved_learning_members_off_a_runtime() -> None:
    """A moved name may be read off ``.learning`` or, inside the module, off ``self``.
    Anywhere else it is a leftover from before the carve. (A same-named attribute on an
    unrelated object would trip this too; none exists, and one would be worth a look.)"""
    moved = MOVED_STATE | MOVED_METHODS
    leftovers: list[str] = []
    for path in sorted(p for root in SWEPT for p in Path(root).rglob("*.py")):
        if path == MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr not in moved:
                continue
            owner = node.value
            if isinstance(owner, ast.Attribute) and owner.attr == "learning":
                continue
            leftovers.append(f"{path}:{node.lineno} .{node.attr}")
    assert leftovers == []


@pytest.fixture()
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
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
        yield rt
    finally:
        rt.shutdown()


def _stub_miner(system: str, user: str) -> str:
    return (
        '[{"pattern": "reviews finances every morning", "confidence": "high", '
        '"evidence": ["checked balance"]}]'
    )


def test_the_scheduled_tick_runs_on_the_live_learning_state(runtime) -> None:
    """The scheduler holds the handler ``startup`` registered. It must be the
    collaborator's, reading the miner as it is when the tick fires — here, switched on
    after the runtime was built."""
    for i in range(4):  # 8 turns, above the miner's min_turns gate
        runtime.sessions.record_turn("s-tick", f"message {i}", f"response {i}")
    runtime.learning.behavior_miner = _stub_miner
    definition = {d.name: d for d in load_heartbeats(Path("config/heartbeats.yaml"))}[
        "behavior_mining_tick"
    ]

    run = runtime.heartbeats.trigger_now(definition)

    assert run.output == '{"proposed": 1}'


def test_compaction_buffers_for_the_miner_only_while_it_is_on(runtime) -> None:
    """Compaction's "is the miner on" check is the one read of the moved state from
    outside the topic. Off: nothing is buffered. Toggled on through the console path:
    the next compaction buffers."""
    big = "alpha beta gamma delta. " * 80  # ~500 tokens/turn, forces a token compaction
    # Deterministic summarizer: this case is about what compaction BUFFERS for the
    # miner, not the model's prose. A real tier-1 roll takes seconds and, under the
    # parallel gate, contends with every other live-model test.
    runtime.compactor._llm = lambda prompt: "Goal: the session under test."

    assert runtime.learning.behavior_miner is None
    for i in range(8):
        runtime.sessions.record_turn("s-off", f"q {i}: {big}", f"a {i}: {big}")
    # The roll runs on a worker thread now (the reply must not wait on a summarizer),
    # so settle before reading what it buffered.
    runtime.sessions.wait_for_compaction("s-off")
    assert runtime.sessions._compaction_archive_buffer == []

    runtime.learning.set_learning_flag("behavior_miner", True)
    if runtime.learning.behavior_miner is None:
        pytest.skip("no LLM client could be built for the miner on this box")
    for i in range(8):
        runtime.sessions.record_turn("s-on", f"q {i}: {big}", f"a {i}: {big}")
    runtime.sessions.wait_for_compaction("s-on")
    assert runtime.sessions._compaction_archive_buffer
