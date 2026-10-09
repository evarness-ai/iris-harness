"""Skill availability for System Health (issue #110).

A skill that declares a prerequisite this install lacks (a package an optional extra
supplies, an env var, a credential) loads as blocked. That is not a fault, so it is one
INFO line in the log; but a log line is not somewhere to look, so the same fact is a grey
row here, and a skill that failed to load (a tools module that raises) is a yellow one.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from iris_harness.services.health.models import CheckKind, HealthCheck, HealthState


class SkillAvailability(Protocol):
    """The two views of the skill registry this provider reads."""

    @property
    def load_failures(self) -> dict[Path, str]: ...

    def unavailable(self) -> list[tuple[str, str, str | None]]: ...


def skills_provider(registry: SkillAvailability) -> Callable[[], list[HealthCheck]]:
    """A ``register_check_provider`` callable. Silent when every skill is usable."""

    def provider() -> list[HealthCheck]:
        checks = [
            HealthCheck(
                target=f"skill:{name}",
                kind=CheckKind.SKILL,
                state=HealthState.GREY,
                detail=f"unavailable: missing {missing}",
                action=fix,
            )
            for name, missing, fix in registry.unavailable()
        ]
        checks.extend(
            HealthCheck(
                target=f"skill:{skill_dir.name}",
                kind=CheckKind.SKILL,
                state=HealthState.YELLOW,
                detail=f"failed to load: {reason}",
            )
            for skill_dir, reason in registry.load_failures.items()
        )
        return checks

    return provider
