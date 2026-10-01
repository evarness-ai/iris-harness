"""Baseline snapshot + diff tests."""

from __future__ import annotations

from pathlib import Path

from iris_harness.playground.baseline import diff, read_baseline, snapshot, write_baseline
from iris_harness.playground.models import ScenarioResult, SuiteResult


def _suite(*results: ScenarioResult) -> SuiteResult:
    return SuiteResult(suite_name="s", results=results)


def _r(name: str, **kw: object) -> ScenarioResult:
    base: dict[str, object] = {
        "scenario_name": name,
        "passed": True,
        "intent": "general",
        "agent_type": "general",
        "handler": None,
        "sources": (),
    }
    base.update(kw)
    return ScenarioResult(**base)  # type: ignore[arg-type]


def test_snapshot_tracks_behavioral_fields() -> None:
    snap = snapshot(_suite(_r("a", intent="finance", handler="dues_request")))
    assert snap["a"] == {
        "passed": True,
        "intent": "finance",
        "agent_type": "general",
        "handler": "dues_request",
        "sources": [],
    }


def test_diff_detects_no_change() -> None:
    snap = snapshot(_suite(_r("a")))
    assert diff(snap, snap) == []


def test_diff_detects_changed_handler() -> None:
    before = snapshot(_suite(_r("a", handler="dues_request")))
    after = snapshot(_suite(_r("a", handler=None)))
    deltas = diff(before, after)
    assert len(deltas) == 1
    assert deltas[0].status == "changed"
    assert ("handler", "dues_request", None) in deltas[0].changes


def test_diff_detects_added_and_removed() -> None:
    before = snapshot(_suite(_r("a"), _r("b")))
    after = snapshot(_suite(_r("a"), _r("c")))
    statuses = {d.name: d.status for d in diff(before, after)}
    assert statuses == {"b": "removed", "c": "added"}


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "baseline.json"
    result = _suite(_r("a", intent="finance"))
    write_baseline(path, result)
    assert path.is_file()
    loaded = read_baseline(path)
    assert loaded["a"]["intent"] == "finance"
    # Round-trips through diff with no change.
    assert diff(loaded, snapshot(result)) == []
