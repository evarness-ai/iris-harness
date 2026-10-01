"""Render the morning-briefing brief end-to-end (Phase 2 manual test).

Usage:
    IRIS_AUTH_SECRET=test-secret-for-testing IRIS_DISABLE_WARMUP=1 \\
        poetry run python scripts/phase2_render_brief.py

If web-fetch slots fail (no network, no cached repos/news data) the
whole render aborts. Use scripts/phase2_render_brief_slots.py to
exercise just the Phase 2 slots in isolation.
"""

from __future__ import annotations

from pathlib import Path

from iris_harness.runtime.handlers.skill_brief import render_brief_package
from iris_harness.tools.skills.registry import SkillRegistry


def main() -> None:
    registry = SkillRegistry(Path("."))
    registry.discover()
    package = next(
        p
        for p in registry.list_packages(only_loadable=True)
        if p.manifest.name == "morning-briefing"
    )
    print(render_brief_package(package, registry))


if __name__ == "__main__":
    main()
