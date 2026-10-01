"""Tests for the iris-core skill tools."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from iris_harness.services.routines import RoutineApprovalStatus, RoutineStore, create_routine_spec


def _load_iris_core_tools():
    repo_root = Path(__file__).resolve().parents[5]
    module_path = repo_root / "config" / "skills" / "builtin" / "iris-core" / "tools.py"
    spec = importlib.util.spec_from_file_location("iris_core_tools_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_list_active_items_tool_returns_only_undone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active_md = tmp_path / "active.md"
    active_md.write_text(
        "- [ ] write tests\n- [x] ship phase 1\n- [ ] verify suite\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("iris_harness.memory.identity.loader.ACTIVE_MD_PATH", active_md)
    module = _load_iris_core_tools()

    result = module.ListActiveItemsTool()._run()

    assert result == [{"text": "write tests"}, {"text": "verify suite"}]


def test_list_approved_routines_tool_returns_executable_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "routines.db"
    store = RoutineStore(db_path)
    store.save(
        create_routine_spec(
            title="Daily brief",
            goal="Send brief",
            schedule="0 7 * * *",
            template="skill_brief",
            approval_status=RoutineApprovalStatus.APPROVED,
        )
    )
    store.save(
        create_routine_spec(
            title="Repo brief",
            goal="Send repos",
            schedule="interval:3600",
            template="skill_brief",
            approval_status=RoutineApprovalStatus.SCHEDULED,
        )
    )
    store.save(
        create_routine_spec(
            title="Draft",
            goal="Skip me",
            schedule="interval:60",
            template="skill_brief",
            approval_status=RoutineApprovalStatus.DRAFT,
        )
    )
    monkeypatch.setenv("IRIS_ROUTINES_DB", str(db_path))
    module = _load_iris_core_tools()

    result = module.ListApprovedRoutinesTool()._run()

    titles = {entry["title"] for entry in result}
    assert titles == {"Daily brief", "Repo brief"}
    for entry in result:
        assert set(entry.keys()) == {"title", "schedule"}


def test_learned_yesterday_tool_is_the_footer_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never empty: a quiet day reads ``learned yesterday: nothing`` (loop-proof V36)."""
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("IRIS_TZ", "UTC")
    # No D13 footer lines: a plugin set up by another test may have registered one.
    monkeypatch.setattr("iris_harness.services.digest.footer._LINES", {})
    module = _load_iris_core_tools()

    assert module.LearnedYesterdayTool()._run() == "learned yesterday: nothing"
    assert "learned_yesterday" in {cls().name for cls in module.SKILL_TOOLS}


def test_learned_yesterday_tool_closes_with_the_footer_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loop-proof D13: each registered footer line follows as its own paragraph."""
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("IRIS_TZ", "UTC")
    monkeypatch.setattr(
        "iris_harness.services.digest.footer._LINES",
        {"jobs": lambda start, end: "Email jobs: sweep 3/3", "quiet": lambda start, end: None},
    )
    module = _load_iris_core_tools()

    assert module.LearnedYesterdayTool()._run() == (
        "learned yesterday: nothing\n\nEmail jobs: sweep 3/3"
    )
