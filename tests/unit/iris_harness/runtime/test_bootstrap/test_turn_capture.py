"""The per-turn capture carve (OSS plan M5.7 track C, slice 12).

``FactCaptureMixin`` became ``TurnCapture``, held as ``runtime.capture`` — the last mixin.
What mypy cannot check about that:

* **The host declaration is exact**, measured by AST from both sides.
* **Both turn paths reach the runtime's own capture.** The intercept stage attributes
  the prior turn's outcome and the curate step records facts and the turn's signal; all
  three read or write per-session state the capture owns, so a call on a second
  ``TurnCapture`` would lose the prior turn's descriptor without raising.
* **No mixin comes back.** ``IrisRuntime`` inherits nothing after this slice.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from iris_harness.runtime import build_runtime
from iris_harness.runtime.bootstrap import IrisRuntime
from iris_harness.runtime.turn_capture import TurnCapture

# No model server in tests (tests/conftest.py network guard): runtime turns here
# reach the LLM on their degrade paths, so the model is a stubbed dead server.
pytestmark = pytest.mark.usefixtures("offline_llm")

MODULE = Path("src/iris_harness/runtime/turn_capture.py")


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
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TurnCaptureHost")
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}


def test_turn_capture_host_declares_exactly_what_the_module_reaches() -> None:
    assert _host_reached() == _host_declared()
    # Back to five: a plainly-stated fact is confirmed during the turn, so capture
    # reaches the recall index again (waiting for the startup resync would mean IRIS
    # knew something it could not recall).
    assert len(_host_declared()) == 5


def test_iris_runtime_inherits_nothing() -> None:
    assert IrisRuntime.__bases__ == (object,)
    assert sorted(Path("src/iris_harness/runtime").glob("*_mixin.py")) == []


def test_both_turn_paths_capture_through_the_runtimes_own_collaborator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    calls: list[tuple[str, Any]] = []

    def recorder(name: str):  # type: ignore[no-untyped-def]
        def record(self: TurnCapture, *args: Any, **kwargs: Any) -> None:
            calls.append((name, self))

        return record

    expected = {"evaluate_prior_turn_outcome", "extract_and_store_facts", "record_signal"}
    for name in expected:
        monkeypatch.setattr(TurnCapture, name, recorder(name))
    config_dir = Path(__file__).resolve().parents[5] / "config"
    rt = build_runtime(
        config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False
    )

    rt.chat("what can you do for me?", session_id="cap-sync")
    after_sync = len(calls)
    list(rt.chat_stream("what can you do for me?", session_id="cap-stream"))

    assert {name for name, _ in calls[:after_sync]} == expected
    assert {name for name, _ in calls[after_sync:]} == expected
    assert all(owner is rt.capture for _, owner in calls)
