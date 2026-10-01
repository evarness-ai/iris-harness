"""The owner's heartbeat edits: live, saved, and back after a restart (ADR-0120).

These run a real APScheduler (started paused, so no job fires) because "applies now"
means the job's trigger changed, and only the real scheduler can say that.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from iris_harness.foundation.settings import SETTINGS_DB_NAME, SettingsStore
from iris_harness.services.heartbeat import (
    SETTINGS_SECTION,
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatScheduler,
    HeartbeatStatus,
)

SWEEP = HeartbeatDefinition(name="email_sweep", handler="sweep", schedule="interval:600")
PHOTOS = HeartbeatDefinition(
    name="filemanager_photos", handler="photos", schedule="30 4 * * *", enabled=False
)


def _ok(definition: HeartbeatDefinition) -> HeartbeatRun:
    return HeartbeatRun(
        name=definition.name, status=HeartbeatStatus.SUCCESS, finished_at=datetime.now(UTC)
    )


@pytest.fixture
def settings(tmp_path: Path) -> SettingsStore:
    return SettingsStore(db_path=tmp_path / SETTINGS_DB_NAME)


@pytest.fixture
def aps() -> Iterator[BackgroundScheduler]:
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.start(paused=True)
    yield scheduler
    scheduler.shutdown(wait=False)


def _harness(aps: BackgroundScheduler | None, settings: SettingsStore) -> HeartbeatScheduler:
    heartbeats = HeartbeatScheduler(
        scheduler=aps, handlers={"sweep": _ok, "photos": _ok}, settings=settings
    )
    heartbeats.register_all([SWEEP, PHOTOS])
    return heartbeats


def _interval_seconds(aps: BackgroundScheduler, name: str) -> float:
    trigger = aps.get_job(name).trigger
    assert isinstance(trigger, IntervalTrigger)
    return trigger.interval.total_seconds()


def test_disabled_heartbeats_are_listed_but_not_scheduled(
    aps: BackgroundScheduler, settings: SettingsStore
) -> None:
    heartbeats = _harness(aps, settings)

    assert {d.name for d in heartbeats.all_definitions()} == {"email_sweep", "filemanager_photos"}
    assert [d.name for d in heartbeats.list_definitions()] == ["email_sweep"]
    assert aps.get_job("filemanager_photos") is None


def test_a_new_schedule_applies_now_and_is_saved(
    aps: BackgroundScheduler, settings: SettingsStore
) -> None:
    heartbeats = _harness(aps, settings)

    updated = heartbeats.update("email_sweep", schedule="interval:300", actor="device:a")

    assert updated.schedule == "interval:300"
    assert _interval_seconds(aps, "email_sweep") == 300
    assert settings.get(SETTINGS_SECTION, "email_sweep") == {"schedule": "interval:300"}
    change = settings.history()[0]
    assert change.old == {"schedule": "interval:600", "enabled": True}
    assert change.new == {"schedule": "interval:300", "enabled": True}
    assert change.actor == "device:a"


def test_turning_one_on_schedules_it_and_off_removes_the_job(
    aps: BackgroundScheduler, settings: SettingsStore
) -> None:
    heartbeats = _harness(aps, settings)

    heartbeats.update("filemanager_photos", enabled=True, actor="s")
    assert isinstance(aps.get_job("filemanager_photos").trigger, CronTrigger)
    assert heartbeats.next_run_at("filemanager_photos") is not None

    heartbeats.update("email_sweep", enabled=False, actor="s")
    assert aps.get_job("email_sweep") is None
    assert "email_sweep" not in {d.name for d in heartbeats.list_definitions()}
    assert heartbeats.next_run_at("email_sweep") is None


def test_edits_come_back_after_a_restart(aps: BackgroundScheduler, settings: SettingsStore) -> None:
    """A restart or a new container: a new scheduler over the same settings file."""
    first = _harness(aps, settings)
    first.update("email_sweep", schedule="interval:1800", actor="s")
    first.update("filemanager_photos", enabled=True, actor="s")
    aps.remove_all_jobs()

    again = _harness(aps, SettingsStore(db_path=settings.db_path))

    assert _interval_seconds(aps, "email_sweep") == 1800
    assert aps.get_job("filemanager_photos") is not None
    assert again.declared("email_sweep") == SWEEP


def test_editing_back_to_the_default_clears_the_override(
    aps: BackgroundScheduler, settings: SettingsStore
) -> None:
    heartbeats = _harness(aps, settings)
    heartbeats.update("email_sweep", schedule="interval:300", actor="s")

    heartbeats.update("email_sweep", schedule="interval:600", actor="s")

    assert settings.get(SETTINGS_SECTION, "email_sweep") is None
    assert [c.action for c in settings.history()] == ["reset", "set"]


def test_reset_returns_to_the_declared_definition(
    aps: BackgroundScheduler, settings: SettingsStore
) -> None:
    heartbeats = _harness(aps, settings)
    heartbeats.update("filemanager_photos", schedule="0 5 * * *", enabled=True, actor="s")

    back = heartbeats.reset("filemanager_photos", actor="device:b")

    assert back == PHOTOS
    assert aps.get_job("filemanager_photos") is None
    assert settings.get(SETTINGS_SECTION, "filemanager_photos") is None
    assert settings.history()[0].new == {"schedule": "30 4 * * *", "enabled": False}


def test_a_later_deploy_still_changes_fields_the_owner_never_touched(
    aps: BackgroundScheduler, settings: SettingsStore
) -> None:
    """Only the edited field is pinned: turning a job off does not freeze its schedule."""
    _harness(aps, settings).update("email_sweep", enabled=False, actor="s")
    aps.remove_all_jobs()
    shipped = HeartbeatDefinition(name="email_sweep", handler="sweep", schedule="interval:900")

    heartbeats = HeartbeatScheduler(scheduler=aps, handlers={"sweep": _ok}, settings=settings)
    heartbeats.register(shipped)

    (effective,) = heartbeats.all_definitions()
    assert effective.schedule == "interval:900"
    assert effective.enabled is False


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"schedule": "interval:5"}, ValueError),
        ({"schedule": "not a schedule"}, ValueError),
        ({"schedule": "0 99 * * *"}, ValueError),
    ],
)
def test_a_bad_edit_changes_nothing(
    aps: BackgroundScheduler,
    settings: SettingsStore,
    change: dict[str, str],
    error: type[Exception],
) -> None:
    heartbeats = _harness(aps, settings)

    with pytest.raises(error):
        heartbeats.update("email_sweep", actor="s", **change)

    assert _interval_seconds(aps, "email_sweep") == 600
    assert settings.history() == []


def test_unknown_heartbeat_is_a_key_error(
    aps: BackgroundScheduler, settings: SettingsStore
) -> None:
    heartbeats = _harness(aps, settings)
    with pytest.raises(KeyError):
        heartbeats.update("nope", enabled=True, actor="s")
    with pytest.raises(KeyError):
        heartbeats.reset("nope", actor="s")


def test_a_heartbeat_this_harness_cannot_run_cannot_be_turned_on(
    aps: BackgroundScheduler, settings: SettingsStore
) -> None:
    heartbeats = HeartbeatScheduler(scheduler=aps, handlers={}, settings=settings)
    heartbeats.register(PHOTOS)

    with pytest.raises(ValueError, match="cannot run on this harness"):
        heartbeats.update("filemanager_photos", enabled=True, actor="s")
    assert settings.history() == []


def test_a_no_op_edit_records_nothing(aps: BackgroundScheduler, settings: SettingsStore) -> None:
    heartbeats = _harness(aps, settings)
    heartbeats.update("email_sweep", schedule="interval:600", enabled=True, actor="s")
    heartbeats.reset("email_sweep", actor="s")
    assert settings.history() == []


def test_a_saved_value_that_no_longer_validates_falls_back_to_the_default(
    aps: BackgroundScheduler, settings: SettingsStore
) -> None:
    settings.set(SETTINGS_SECTION, "email_sweep", {"schedule": "interval:1"}, old=None, actor="s")

    heartbeats = _harness(aps, settings)

    assert _interval_seconds(aps, "email_sweep") == 600
    assert heartbeats.all_definitions()[0].schedule == "interval:600"


def test_without_a_store_edits_still_apply_live(aps: BackgroundScheduler) -> None:
    heartbeats = HeartbeatScheduler(scheduler=aps, handlers={"sweep": _ok})
    heartbeats.register(SWEEP)

    heartbeats.update("email_sweep", schedule="interval:120", actor="s")

    assert _interval_seconds(aps, "email_sweep") == 120


MAC_ONLY = HeartbeatDefinition(
    name="apple_calendar_sync",
    handler="photos",
    schedule="interval:900",
    enabled=False,
    platforms=("darwin",),
)


def test_a_mac_only_heartbeat_is_listed_but_locked_on_linux(
    aps: BackgroundScheduler, settings: SettingsStore
) -> None:
    """The VM loads the handler, so only the declared platform can say "not here"."""
    heartbeats = HeartbeatScheduler(
        scheduler=aps, handlers={"photos": _ok}, settings=settings, platform="linux"
    )
    heartbeats.register(MAC_ONLY)

    reason = heartbeats.unavailable_reason(MAC_ONLY)
    assert reason == "needs darwin; this harness runs on linux"
    with pytest.raises(ValueError, match="needs darwin"):
        heartbeats.update("apple_calendar_sync", enabled=True, actor="s")
    assert aps.get_job("apple_calendar_sync") is None
    assert settings.history() == []
    # Its schedule can still be edited, so it is ready if it ever runs where it can.
    heartbeats.update("apple_calendar_sync", schedule="interval:1800", actor="s")
    assert aps.get_job("apple_calendar_sync") is None


def test_a_mac_only_heartbeat_left_on_in_yaml_is_not_scheduled_on_linux(
    aps: BackgroundScheduler, settings: SettingsStore
) -> None:
    left_on = HeartbeatDefinition(
        name="apple_calendar_sync", handler="photos", schedule="interval:900", platforms=("darwin",)
    )
    heartbeats = HeartbeatScheduler(scheduler=aps, handlers={"photos": _ok}, platform="linux")

    assert heartbeats.register(left_on) is False
    assert aps.get_job("apple_calendar_sync") is None


def test_on_its_own_platform_a_mac_only_heartbeat_runs(
    aps: BackgroundScheduler, settings: SettingsStore
) -> None:
    heartbeats = HeartbeatScheduler(
        scheduler=aps, handlers={"photos": _ok}, settings=settings, platform="darwin"
    )
    heartbeats.register(MAC_ONLY)

    assert heartbeats.unavailable_reason(MAC_ONLY) is None
    heartbeats.update("apple_calendar_sync", enabled=True, actor="s")
    assert aps.get_job("apple_calendar_sync") is not None
