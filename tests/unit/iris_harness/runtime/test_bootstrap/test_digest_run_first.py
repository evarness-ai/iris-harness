"""The digest sweeps and judges mail first, while the judge is on (loop-proof PR 5).

Owner decision 2026-09-26: the digest runs after the email judge job, only if the judge
is enabled. ``digest.yaml`` ``run_first`` names the jobs; the seeded ``morning-digest``
runs them — scheduled and manual runs alike, never a preview — through the heartbeat
scheduler's Run-now path, then renders and delivers whatever they did.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iris_harness.foundation.settings.catalog import SettingDeclaration
from iris_harness.runtime.facade import IrisRuntime
from iris_harness.runtime.handlers.ticks import execute_routine
from iris_harness.runtime.plugin_host.manifest import PluginManifest
from iris_harness.services.digest import settings as digest_settings
from iris_harness.services.digest.settings import DigestSettings
from iris_harness.services.heartbeat import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatScheduler,
    HeartbeatStatus,
)
from iris_harness.services.heartbeat.run_store import HeartbeatRunStore
from iris_harness.services.routines import RoutineExecutionStatus, RoutineStore
from iris_harness.services.routines.seeded import MORNING_DIGEST_ROUTINE_ID

_REPO_DIGEST = Path(__file__).resolve().parents[5] / "config" / "digest.yaml"
_JOBS = ("email_sweep", "email_judge")


@pytest.fixture(autouse=True)
def _settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        digest_settings,
        "load_digest_settings",
        lambda data_dir, config_dir=None: DigestSettings(),
    )
    monkeypatch.setenv("IRIS_TZ", "America/Chicago")
    monkeypatch.delenv("IRIS_EMAIL_JUDGE", raising=False)


# The setting ``digest.yaml``'s ``run_first`` reads, declared the way the email plugin
# declares it (on by default). That the real plugin still declares it so is asserted
# beside the plugin: tests/unit/iris_personal/plugins/test_email_workflows/
# test_digest_run_first_setting.py.
_EMAIL_PLUGIN = PluginManifest(
    name="email_workflows",
    settings={
        "IRIS_EMAIL_JUDGE": SettingDeclaration(
            kind="bool",
            default=True,
            applies="next_run",
            label="Judge new email",
            description="When on (default), the email judge sorts each new email.",
            tab="agents",
            agent="email",
        )
    },
)


class _Registry:
    """The loaded plugins, as ``registry_catalog`` reads them: an email plugin whose
    manifest declares ``IRIS_EMAIL_JUDGE`` on by default."""

    def plugins(self) -> list[Any]:
        return [SimpleNamespace(manifest=_EMAIL_PLUGIN)]


def _runtime(
    tmp_path: Path,
    order: list[str],
    *,
    fail: str | None = None,
    raises: str | None = None,
    budget: int | None = None,
    jobs: tuple[str, ...] = _JOBS,
) -> SimpleNamespace:
    config = tmp_path / "config"
    config.mkdir()
    shutil.copy(_REPO_DIGEST, config / "digest.yaml")
    if budget is not None:
        text = (config / "digest.yaml").read_text(encoding="utf-8")
        (config / "digest.yaml").write_text(
            text.replace("run_first_budget_seconds: 240", f"run_first_budget_seconds: {budget}"),
            encoding="utf-8",
        )
    scheduler = HeartbeatScheduler(run_store=HeartbeatRunStore(db_path=tmp_path / "hb.db"))

    def job(definition: HeartbeatDefinition) -> HeartbeatRun:
        order.append(definition.name)
        if definition.name == raises:
            raise RuntimeError("Mac unreachable")
        status = HeartbeatStatus.FAILED if definition.name == fail else HeartbeatStatus.SUCCESS
        return HeartbeatRun(name=definition.name, status=status, output="done")

    for name in jobs:
        scheduler.register_handler(name, job)
        scheduler.register(
            HeartbeatDefinition(name=name, handler=name, schedule="15 6,12,18 * * *")
        )

    def skill_brief(definition: HeartbeatDefinition) -> HeartbeatRun:
        order.append("render")
        return HeartbeatRun(name=definition.name, status=HeartbeatStatus.SUCCESS, output="sent")

    scheduler.register_handler("skill_brief", skill_brief)
    brief_pkg = SimpleNamespace(
        manifest=SimpleNamespace(kind="brief", name="morning-briefing"), is_loadable=True
    )
    runtime = SimpleNamespace(
        routine_store=RoutineStore(tmp_path / "routines.db"),
        heartbeats=scheduler,
        skill_registry=SimpleNamespace(
            list_packages=lambda *, agent_name=None, only_loadable=False: (brief_pkg,)
        ),
        plugin_registry=_Registry(),
        data_dir=tmp_path,
        config_dir=config,
    )
    IrisRuntime._seed_core_routines(runtime)  # type: ignore[arg-type]
    return runtime


def _run(runtime: SimpleNamespace, *, trigger: str = "schedule") -> Any:
    digest = runtime.routine_store.load(MORNING_DIGEST_ROUTINE_ID)
    return execute_routine(
        runtime,  # type: ignore[arg-type]
        digest,
        checked_at=datetime.now(UTC).replace(microsecond=0),
        record=True,
        trigger=trigger,
    )


def test_sweep_then_judge_then_render_with_the_judge_on_by_default(tmp_path: Path) -> None:
    order: list[str] = []
    record = _run(_runtime(tmp_path, order))

    assert order == ["email_sweep", "email_judge", "render"]
    assert record.status == RoutineExecutionStatus.SUCCESS


@pytest.mark.parametrize("value", ["0", "off"])
def test_judge_off_runs_neither_job_and_the_digest_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("IRIS_EMAIL_JUDGE", value)
    order: list[str] = []
    record = _run(_runtime(tmp_path, order))

    assert order == ["render"]
    assert record.status == RoutineExecutionStatus.SUCCESS


def test_judge_explicitly_on_runs_both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_EMAIL_JUDGE", "1")
    order: list[str] = []
    _run(_runtime(tmp_path, order))
    assert order == ["email_sweep", "email_judge", "render"]


@pytest.mark.parametrize("how", ["fail", "raises"])
def test_a_failing_job_never_blocks_the_digest(tmp_path: Path, how: str) -> None:
    order: list[str] = []
    runtime = _runtime(tmp_path, order, **{how: "email_sweep"})
    record = _run(runtime)

    assert order == ["email_sweep", "email_judge", "render"]
    assert record.status == RoutineExecutionStatus.SUCCESS
    assert runtime.heartbeats.run_store.last("email_sweep").status == "failed"


def test_budget_spent_skips_the_rest_and_the_digest_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.services.digest import run_first as run_first_module

    clock = iter([0.0, 0.0, 30.0])  # start, before the sweep, before the judge
    real = run_first_module.run_first

    def with_clock(*args: Any, **kwargs: Any) -> Any:
        return real(*args, **kwargs, monotonic=lambda: next(clock))

    monkeypatch.setattr(run_first_module, "run_first", with_clock)
    order: list[str] = []
    _run(_runtime(tmp_path, order, budget=20))

    assert order == ["email_sweep", "render"]


def test_an_unregistered_job_is_skipped_and_the_digest_renders(tmp_path: Path) -> None:
    order: list[str] = []
    _run(_runtime(tmp_path, order, jobs=("email_sweep",)))
    assert order == ["email_sweep", "render"]


def test_a_turned_off_job_is_skipped(tmp_path: Path) -> None:
    order: list[str] = []
    runtime = _runtime(tmp_path, order)
    runtime.heartbeats.update("email_sweep", enabled=False, actor="test")
    _run(runtime)
    assert order == ["email_judge", "render"]


def test_the_runs_are_kept_like_a_run_now(tmp_path: Path) -> None:
    order: list[str] = []
    runtime = _runtime(tmp_path, order)
    _run(runtime)

    store = runtime.heartbeats.run_store
    for name in _JOBS:
        kept = store.last(name)
        assert kept is not None and kept.ok
        assert kept.trigger == "digest"


def test_a_manual_digest_run_sweeps_and_judges_first_too(tmp_path: Path) -> None:
    order: list[str] = []
    runtime = _runtime(tmp_path, order)

    record = IrisRuntime.run_routine(runtime, MORNING_DIGEST_ROUTINE_ID)  # type: ignore[arg-type]

    assert record is not None and record.status == RoutineExecutionStatus.SUCCESS
    assert order == ["email_sweep", "email_judge", "render"]


def test_a_dry_run_does_not_run_the_jobs(tmp_path: Path) -> None:
    order: list[str] = []
    runtime = _runtime(tmp_path, order)
    digest = runtime.routine_store.load(MORNING_DIGEST_ROUTINE_ID)

    execute_routine(runtime, digest, checked_at=datetime.now(UTC), record=False)  # type: ignore[arg-type]

    assert order == ["render"]


def test_another_routine_never_runs_the_jobs(tmp_path: Path) -> None:
    order: list[str] = []
    runtime = _runtime(tmp_path, order)
    digest = runtime.routine_store.load(MORNING_DIGEST_ROUTINE_ID)
    other = digest.model_copy(update={"id": "evening-brief", "metadata": {}})

    execute_routine(runtime, other, checked_at=datetime.now(UTC), record=True)  # type: ignore[arg-type]

    assert order == ["render"]
