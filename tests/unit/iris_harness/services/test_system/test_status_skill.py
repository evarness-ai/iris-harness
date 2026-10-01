"""The system-status skill loads and matches its manifest (System agent S0)."""

from __future__ import annotations

from pathlib import Path

from iris_harness.tools.skills.loader import load_skill_package

REPO_ROOT = Path(__file__).resolve().parents[5]
SKILL_DIR = REPO_ROOT / "config" / "skills" / "system" / "system-status"


def test_system_status_skill_loads() -> None:
    pkg = load_skill_package(REPO_ROOT, SKILL_DIR)
    assert pkg.manifest.name == "system-status"
    assert pkg.manifest.default_enabled is True
    assert {t.name for t in pkg.manifest.tools} == {"system_status"}
    assert {c.model_fields["name"].default for c in pkg.tool_classes} == {"system_status"}
    assert pkg.manifest.tools[0].governor_route == "system/read"
    assert pkg.missing_prerequisites == ()
