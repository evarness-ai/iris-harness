"""Synthesis brief slots render but aren't section-selectable (Planner P3 / ADR-0059)."""

from __future__ import annotations

from pathlib import Path

from iris_harness.services.routines.authoring import brief_tool_slot_keys
from iris_harness.tools.skills.loader import load_skill_manifest, load_skill_package

REPO_ROOT = Path(__file__).resolve().parents[5]
MORNING_BRIEFING = REPO_ROOT / "config" / "skills" / "builtin" / "morning-briefing"


def test_todays_plan_slot_exists_and_is_synthesis() -> None:
    brief = load_skill_manifest(MORNING_BRIEFING).brief
    assert brief is not None
    slot = brief.slots["todays_plan"]
    assert slot.kind == "tool" and slot.synthesis is True  # type: ignore[union-attr]
    assert "{{todays_plan}}" in brief.layout  # it DOES render in the brief


def test_synthesis_slot_excluded_from_selectable_keys() -> None:
    pkg = load_skill_package(REPO_ROOT, MORNING_BRIEFING)
    keys = brief_tool_slot_keys(pkg)
    # The headline synthesis slot is not a user-selectable section …
    assert "todays_plan" not in keys
    # … but ordinary tool slots still are.
    assert "due_today" in keys and "bills_due" in keys
