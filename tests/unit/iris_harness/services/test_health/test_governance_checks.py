"""Health rows for the model guard (issue #136).

The model guard is opt-in and fails open. Two situations used to be invisible: it is on but
its classifier cannot run, and it is off while a tool that returns third-party text is
mounted. Each is a yellow ``governance`` row; a sound posture is silent. The probe never
loads the model.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iris_harness.kernel.governance.threat import availability
from iris_harness.kernel.governance.threat.availability import model_guard_state
from iris_harness.services.health import service
from iris_harness.services.health.governance import model_guard_provider
from iris_harness.services.health.models import CheckKind, HealthState, alerts

_REPO = Path(__file__).resolve().parents[5]


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    service.clear_check_providers()
    monkeypatch.delenv("IRIS_GOVERNANCE_PROMPT_GUARD", raising=False)
    monkeypatch.setattr(availability, "prompt_guard_load_failed", lambda: False)


def _installed(monkeypatch: pytest.MonkeyPatch, *, weights: bool, packages: bool = True) -> None:
    monkeypatch.setattr(
        availability.importlib.util,
        "find_spec",
        lambda name: object() if packages else None,
    )
    monkeypatch.setattr(availability, "_weights_cached", lambda model_id: weights)


def _none_mounted() -> list[tuple[str, str]]:
    return []


def _mounted() -> list[tuple[str, str]]:
    return [
        ("email_workflows", "read_email"),
        ("email_workflows", "search_inbox"),
        ("core", "wiki_search"),
    ]


def test_flag_on_with_the_classifier_packages_missing_is_a_yellow_row_with_the_fix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_PROMPT_GUARD", "1")
    _installed(monkeypatch, weights=False, packages=False)

    (row,) = model_guard_provider(_none_mounted)()

    assert row.kind is CheckKind.GOVERNANCE and row.state is HealthState.YELLOW
    assert "transformers and torch not installed" in row.detail
    assert row.action is not None and "iris-harness[ml]" in row.action
    assert "floor" in row.detail  # the deterministic floor is named as still running


def test_flag_on_with_the_weights_absent_is_a_yellow_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_PROMPT_GUARD", "1")
    _installed(monkeypatch, weights=False)

    (row,) = model_guard_provider(_none_mounted)()

    assert row.state is HealthState.YELLOW
    assert "weights" in row.detail and "Llama-Prompt-Guard-2-86M" in row.detail


def test_flag_on_after_a_live_load_failure_is_a_yellow_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_PROMPT_GUARD", "1")
    _installed(monkeypatch, weights=True)
    monkeypatch.setattr(availability, "prompt_guard_load_failed", lambda: True)

    (row,) = model_guard_provider(_none_mounted)()

    assert "tried and could not load" in row.detail


def test_a_real_classifier_that_fails_to_load_is_seen_without_loading_anything_new(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iris_harness.kernel.governance.threat import backends

    monkeypatch.undo()  # keep the real prompt_guard_load_failed
    guard = backends.PromptGuardClassifier(model_id="x/none")
    monkeypatch.setitem(sys.modules, "transformers", None)  # the import fails: unavailable
    guard._ensure_pipe()

    assert guard.load_failed and backends.prompt_guard_load_failed()


def test_flag_on_and_able_to_run_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_PROMPT_GUARD", "1")
    _installed(monkeypatch, weights=True)

    assert model_guard_provider(_mounted)() == []
    assert model_guard_state().classifier == "available"


def test_flag_off_with_external_tools_mounted_is_a_yellow_row_naming_them() -> None:
    (row,) = model_guard_provider(_mounted)()

    assert row.kind is CheckKind.GOVERNANCE and row.state is HealthState.YELLOW
    assert "3 external-content tool(s)" in row.detail
    assert "email_workflows" in row.detail and "core" in row.detail
    assert row.action is not None and "IRIS_GOVERNANCE_PROMPT_GUARD" in row.action
    assert alerts(SimpleNamespace(checks=(row,))) == []  # yellow: shown, never an alert


def test_flag_off_with_no_external_tool_is_silent() -> None:
    assert model_guard_provider(_none_mounted)() == []


def test_the_row_reaches_the_snapshot_every_surface_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_harness.services.health import render_text
    from iris_harness.services.health.models import HealthSnapshot

    bare = HealthSnapshot(checks=(), sampled_at="2026-10-06T00:00:00+00:00")
    monkeypatch.setattr(service, "_cached", None)
    monkeypatch.setattr(service, "build_snapshot", lambda **_: bare)
    service.register_check_provider("model_guard", model_guard_provider(_mounted))

    snapshot = service.refresh()

    rows = [c for c in snapshot.checks if c.kind is CheckKind.GOVERNANCE]
    assert [c.target for c in rows] == ["Model guard"]
    # GET /health and the web screen read as_dict(); chat reads the system_health text.
    assert any(
        c["target"] == "Model guard" and c["kind"] == "governance"
        for c in snapshot.as_dict()["checks"]
    )
    assert "Model guard [yellow]" in render_text(snapshot)


def test_the_probe_loads_neither_torch_nor_transformers() -> None:
    code = (
        "import os, sys\n"
        "os.environ['IRIS_GOVERNANCE_PROMPT_GUARD'] = '1'\n"
        "from iris_harness.kernel.governance.threat.availability import model_guard_state\n"
        "model_guard_state()\n"
        "bad = [m for m in ('torch', 'transformers') if m in sys.modules]\n"
        "print('LOADED', bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    done = subprocess.run(  # noqa: S603 - a fixed argv: this interpreter, a literal script
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(_REPO / "src"), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_every_check_kind_has_a_group_on_the_web_health_screen() -> None:
    """The screen renders a row only under a group for its kind, so a kind with no group is
    invisible there (plugin rows were, before issue #136)."""
    screen = (_REPO / "webui/src/screens/Health.tsx").read_text()
    groups = set(re.findall(r'\{ label: "[^"]+", kind: "(\w+)" \}', screen))
    types = (_REPO / "webui/src/lib/control.ts").read_text()
    union = re.search(r"interface HealthCheck \{\s*target: string;\s*kind: ([^;]+);", types)
    assert union is not None
    declared = set(re.findall(r'"(\w+)"', union.group(1)))
    values = {k.value for k in CheckKind}
    assert values <= groups, f"no web group for {sorted(values - groups)}"
    assert values <= declared, f"not in the web HealthCheck type: {sorted(values - declared)}"


def _runtime_stub(**parts: Any) -> Any:
    return SimpleNamespace(**parts)


def test_mounted_external_tools_lists_core_plugin_skill_and_capability_sources(
    tmp_path: Path,
) -> None:
    from iris_harness.agent.agentic_core import ToolSpec
    from iris_harness.foundation import capabilities as catalogue
    from iris_harness.runtime.external_tools import mounted_external_tools
    from iris_harness.tools.skills.registry import SkillRegistry

    skills = SkillRegistry(repo_root=_REPO)
    skills.discover()

    class Registry:
        def tools(self) -> list[Any]:
            return [
                ToolSpec("fetch_page", "d", lambda a: "", content="external", plugin="pagefetch"),
                ToolSpec("note", "d", lambda a: "", plugin="notes"),
            ]

        def capability_providers(self, name: str) -> tuple[str, ...]:
            return ("weather",) if name == "weather.forecast" else ()

    found = mounted_external_tools(_runtime_stub(plugin_registry=Registry(), skill_registry=skills))

    assert ("core", "wiki_search") in found  # a core tool
    assert ("pagefetch", "fetch_page") in found  # a plugin tool
    assert ("notes", "note") not in found  # an internal plugin tool
    assert any(owner.startswith("skill:") for owner, _ in found)  # a skill tool
    assert ("weather", catalogue.capability_tool_name("weather.forecast", "forecast")) in found
