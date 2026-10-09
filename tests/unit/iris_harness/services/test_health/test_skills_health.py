"""Skills in System Health (issue #110): a blocked skill is a row, not only a log line."""

from __future__ import annotations

from pathlib import Path

from iris_harness.services.health.models import CheckKind, HealthState
from iris_harness.services.health.skills import skills_provider


class _Registry:
    def __init__(
        self,
        blocked: list[tuple[str, str, str | None]],
        failures: dict[Path, str] | None = None,
    ) -> None:
        self._blocked = blocked
        self.load_failures = failures or {}

    def unavailable(self) -> list[tuple[str, str, str | None]]:
        return self._blocked


def test_blocked_skill_is_a_grey_row_naming_the_fix() -> None:
    registry = _Registry(
        [("gmail-inbox", "package google-api-python-client", "pip install 'iris-harness[email]'")]
    )
    (check,) = skills_provider(registry)()
    assert check.target == "skill:gmail-inbox"
    assert check.kind is CheckKind.SKILL
    assert check.state is HealthState.GREY
    assert "google-api-python-client" in check.detail
    assert check.action == "pip install 'iris-harness[email]'"


def test_skill_that_failed_to_load_is_yellow() -> None:
    registry = _Registry([], {Path("config/skills/x/broken"): "ImportError: boom"})
    (check,) = skills_provider(registry)()
    assert check.target == "skill:broken"
    assert check.state is HealthState.YELLOW
    assert "boom" in check.detail


def test_nothing_to_report_is_silent() -> None:
    assert skills_provider(_Registry([]))() == []
