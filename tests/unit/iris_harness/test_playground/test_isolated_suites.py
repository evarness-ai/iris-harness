"""An isolated suite runs on a throwaway memory, never the owner's.

A suite that tells the assistant things ("my bank is Barclays") would otherwise have
them remembered in the real store and projected into the real USER.md.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from iris_harness.memory.identity import loader
from iris_harness.playground import service
from iris_harness.playground.models import Scenario, ScenarioSuite


def test_the_identity_files_and_data_move_for_the_run_and_come_back() -> None:
    real_user_md, real_home = loader.user_md_path(), loader.iris_home()

    with service._isolated_memory() as env:
        data = Path(env["IRIS_DATA_DIR"])
        moved = loader.user_md_path()
        assert not moved.is_relative_to(real_home)
        assert moved.name == "USER.md" and data.parent == moved.parents[2]
        assert loader.iris_home() == real_home  # the owner's profile + plugins still apply

    assert loader.user_md_path() == real_user_md
    # The override slots are put back to "resolve from IRIS_HOME", not frozen.
    assert all(getattr(loader, name) is None for name in loader.IDENTITY_PATH_ACCESSORS)
    assert not data.parent.exists()  # thrown away


def test_run_suite_isolates_only_when_the_suite_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    @contextmanager
    def fake_runtime(env: dict[str, str], *, isolated: bool = False) -> Iterator[Any]:
        seen.append(isolated)

        class _Result:
            response, intent, agent_type = "ok", "general", "general"
            sources: tuple[str, ...] = ()
            metadata: dict[str, Any] = {}
            has_errors = False

        class _Runtime:
            def chat(self, message: str, **kw: Any) -> _Result:
                return _Result()

        yield _Runtime()

    monkeypatch.setattr(service, "_built_runtime", fake_runtime)
    scenario = Scenario(name="s", message="hi")

    service.run_suite(ScenarioSuite(name="plain", scenarios=(scenario,)))
    service.run_suite(ScenarioSuite(name="sandboxed", isolated=True, scenarios=(scenario,)))

    assert seen == [False, True]


@pytest.mark.parametrize("name", ["memory-graph", "memory-continuity"])
def test_the_memory_suites_run_isolated(name: str) -> None:
    from iris_harness.playground.loader import load_suite

    assert load_suite(Path("config/playground") / f"{name}.yaml").isolated is True
