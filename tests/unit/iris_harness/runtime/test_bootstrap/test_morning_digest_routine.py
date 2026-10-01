"""The seeded ``morning-digest`` routine end to end through the routine tick.

Loop-proof D4/D5/D13: seeded on first boot, scheduled from Settings → Digest ``time``
in ``IRIS_TZ``, rendered with the digest's sections/caps/channel, and every run is an
``ActivityStore`` row.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iris_harness.runtime.facade import IrisRuntime
from iris_harness.runtime.handlers import ticks
from iris_harness.runtime.handlers.ticks import build_routine_tick_handler, execute_routine
from iris_harness.services.activities import ActivityStore
from iris_harness.services.digest import settings as digest_settings
from iris_harness.services.digest.settings import DigestSettings
from iris_harness.services.heartbeat import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatScheduler,
    HeartbeatStatus,
)
from iris_harness.services.routines import RoutineApprovalStatus, RoutineStore
from iris_harness.services.routines.seeded import MORNING_DIGEST_ROUTINE_ID

_TICK = HeartbeatDefinition(name="routine_tick", handler="routine_tick", schedule="interval:60")


class _Settings:
    """Mutable holder the patched ``load_digest_settings`` reads — the settings store."""

    def __init__(self, value: DigestSettings) -> None:
        self.value = value
        self.calls: list[tuple[Path, Path | None]] = []

    def load(self, data_dir: Path, config_dir: Path | None = None) -> DigestSettings:
        self.calls.append((data_dir, config_dir))
        return self.value


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> _Settings:
    holder = _Settings(DigestSettings())
    monkeypatch.setattr(digest_settings, "load_digest_settings", holder.load)
    monkeypatch.setenv("IRIS_TZ", "America/Chicago")
    return holder


def _freeze(monkeypatch: pytest.MonkeyPatch, moment: datetime) -> None:
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
            return moment.astimezone(tz) if tz is not None else moment

    monkeypatch.setattr(ticks, "datetime", _Frozen)


def _runtime(tmp_path: Path, *, ok: bool = True) -> tuple[SimpleNamespace, list[dict[str, Any]]]:
    store = RoutineStore(tmp_path / "routines.db")
    scheduler = HeartbeatScheduler()
    seen: list[dict[str, Any]] = []

    def skill_brief(definition: HeartbeatDefinition) -> HeartbeatRun:
        seen.append(dict(definition.params))
        if not ok:
            return HeartbeatRun(
                name=definition.name, status=HeartbeatStatus.FAILED, error="bills tool down"
            )
        return HeartbeatRun(name=definition.name, status=HeartbeatStatus.SUCCESS, output="sent")

    scheduler.register_handler("skill_brief", skill_brief)
    brief_pkg = SimpleNamespace(
        manifest=SimpleNamespace(kind="brief", name="morning-briefing"), is_loadable=True
    )
    registry = SimpleNamespace(
        list_packages=lambda *, agent_name=None, only_loadable=False: (brief_pkg,)
    )
    runtime = SimpleNamespace(
        routine_store=store,
        heartbeats=scheduler,
        skill_registry=registry,
        data_dir=tmp_path,
        config_dir=tmp_path / "config",
    )
    return runtime, seen


def _seed(runtime: SimpleNamespace) -> None:
    IrisRuntime._seed_core_routines(runtime)  # type: ignore[arg-type]


def _activities(tmp_path: Path) -> list[Any]:
    store = ActivityStore(db_path=tmp_path / "activities.db")
    store.ensure_schema()
    return store.list(origin="routine")


# --- seeding (IrisRuntime.startup) ---------------------------------------------------


def test_startup_seed_creates_the_digest_once(tmp_path: Path, settings: _Settings) -> None:
    runtime, _ = _runtime(tmp_path)
    settings.value = DigestSettings(time="06:45", channel="all", sections=("bills_due",))

    _seed(runtime)
    _seed(runtime)  # a restart

    rows = runtime.routine_store.list_all()
    assert [r.id for r in rows] == [MORNING_DIGEST_ROUTINE_ID]
    digest = rows[0]
    assert digest.approval_status == RoutineApprovalStatus.SCHEDULED
    assert digest.template == "morning-briefing"
    assert digest.schedule == "daily:06:40"
    assert digest.delivery_channel == "all"
    assert digest.metadata["timezone"] == "America/Chicago"
    assert settings.calls[0] == (tmp_path, tmp_path / "config")


def test_startup_seed_failure_never_breaks_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(data_dir: Path, config_dir: Path | None = None) -> DigestSettings:
        raise RuntimeError("settings.db locked")

    monkeypatch.setattr(digest_settings, "load_digest_settings", boom)
    runtime, _ = _runtime(tmp_path)

    _seed(runtime)  # logs, does not raise

    assert runtime.routine_store.list_all() == []


# --- scheduling in IRIS_TZ -----------------------------------------------------------


def test_digest_fires_at_settings_time_in_iris_tz(
    tmp_path: Path, settings: _Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, seen = _runtime(tmp_path)
    _seed(runtime)
    handler = build_routine_tick_handler(runtime)

    _freeze(monkeypatch, datetime(2026, 9, 28, 11, 54, 30, tzinfo=UTC))  # 06:54:30 CDT
    assert "due=0" in handler(_TICK).output
    _freeze(monkeypatch, datetime(2026, 9, 28, 11, 55, 20, tzinfo=UTC))  # 06:55:20 CDT, 5 min lead
    run = handler(_TICK)
    assert run.status is HeartbeatStatus.SUCCESS
    assert "due=1" in run.output and "success=1" in run.output
    _freeze(monkeypatch, datetime(2026, 9, 28, 11, 56, 20, tzinfo=UTC))  # next tick
    assert "due=0" in handler(_TICK).output
    assert len(seen) == 1


def test_settings_time_change_applies_at_the_next_tick(
    tmp_path: Path, settings: _Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, seen = _runtime(tmp_path)
    _seed(runtime)
    handler = build_routine_tick_handler(runtime)

    settings.value = DigestSettings(time="08:15")  # owner edits Settings → Digest; no restart
    _freeze(monkeypatch, datetime(2026, 9, 28, 12, 0, 20, tzinfo=UTC))  # 07:00 CDT
    assert "due=0" in handler(_TICK).output
    _freeze(monkeypatch, datetime(2026, 9, 28, 13, 10, 20, tzinfo=UTC))  # 08:10 CDT, 5 min lead
    assert "due=1" in handler(_TICK).output
    assert runtime.routine_store.load(MORNING_DIGEST_ROUTINE_ID).schedule == "daily:08:10"
    assert len(seen) == 1


def test_disabled_digest_is_not_dispatched(
    tmp_path: Path, settings: _Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, seen = _runtime(tmp_path)
    _seed(runtime)
    settings.value = DigestSettings(enabled=False)
    _freeze(
        monkeypatch, datetime(2026, 9, 28, 11, 55, 20, tzinfo=UTC)
    )  # 06:55 CDT, 5 min before 07:00

    run = build_routine_tick_handler(runtime)(_TICK)

    assert "due=0" in run.output
    assert seen == []


def test_owner_paused_digest_stays_paused(
    tmp_path: Path, settings: _Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, seen = _runtime(tmp_path)
    _seed(runtime)
    store = runtime.routine_store
    store.save(store.load(MORNING_DIGEST_ROUTINE_ID).with_status(RoutineApprovalStatus.PAUSED))
    _freeze(
        monkeypatch, datetime(2026, 9, 28, 11, 55, 20, tzinfo=UTC)
    )  # 06:55 CDT, 5 min before 07:00

    assert "due=0" in build_routine_tick_handler(runtime)(_TICK).output
    _seed(runtime)  # restart
    assert store.load(MORNING_DIGEST_ROUTINE_ID).approval_status == RoutineApprovalStatus.PAUSED
    assert seen == []


# --- render params come from Settings → Digest ---------------------------------------


def test_scheduled_digest_renders_with_digest_settings(
    tmp_path: Path, settings: _Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, seen = _runtime(tmp_path)
    _seed(runtime)
    settings.value = DigestSettings(
        channel="telegram",
        sections=("bills_due", "todays_events", "focus"),
        section_config={"focus": {"max_items": 5}, "bills_due": {"within_days": 7}},
    )
    _freeze(
        monkeypatch, datetime(2026, 9, 28, 11, 55, 20, tzinfo=UTC)
    )  # 06:55 CDT, 5 min before 07:00

    build_routine_tick_handler(runtime)(_TICK)

    [params] = seen
    assert params["skill_id"] == "morning-briefing"
    assert params["routine_id"] == MORNING_DIGEST_ROUTINE_ID
    assert params["channel"] == "telegram"
    assert params["source_preferences"] == ["bills_due", "todays_events", "focus"]
    assert params["section_order"] == ["bills_due", "todays_events", "focus"]
    assert params["section_line_caps"] == {"focus": 5}
    assert params["digest"] is True


def test_manual_run_reads_current_settings(tmp_path: Path, settings: _Settings) -> None:
    runtime, seen = _runtime(tmp_path)
    _seed(runtime)
    settings.value = DigestSettings(sections=("news",), channel="web")
    digest = runtime.routine_store.load(MORNING_DIGEST_ROUTINE_ID)

    record = execute_routine(runtime, digest, checked_at=datetime.now(UTC), record=True)

    assert record.status == "success"
    assert seen[0]["source_preferences"] == ["news"]
    assert seen[0]["channel"] == "web"


# --- D13: every run is an ActivityStore row ------------------------------------------


def test_scheduled_run_is_an_activity_row(
    tmp_path: Path, settings: _Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, _ = _runtime(tmp_path)
    _seed(runtime)
    _freeze(
        monkeypatch, datetime(2026, 9, 28, 11, 55, 20, tzinfo=UTC)
    )  # 06:55 CDT, 5 min before 07:00

    build_routine_tick_handler(runtime)(_TICK)

    [row] = _activities(tmp_path)
    assert row.kind == f"routine.{MORNING_DIGEST_ROUTINE_ID}"
    assert row.status == "completed"
    assert row.result_summary == "sent"
    assert row.metadata["trigger"] == "schedule"
    assert row.metadata["channel"] == "all"
    assert row.started_at is not None and row.finished_at is not None


def test_failed_run_is_a_failed_activity_row(
    tmp_path: Path, settings: _Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, _ = _runtime(tmp_path, ok=False)
    _seed(runtime)
    _freeze(
        monkeypatch, datetime(2026, 9, 28, 11, 55, 20, tzinfo=UTC)
    )  # 06:55 CDT, 5 min before 07:00

    run = build_routine_tick_handler(runtime)(_TICK)

    assert run.status is HeartbeatStatus.FAILED
    [row] = _activities(tmp_path)
    assert row.status == "failed"
    assert row.error == "bills tool down"


def test_manual_run_is_an_activity_row_but_a_dry_run_is_not(
    tmp_path: Path, settings: _Settings
) -> None:
    runtime, _ = _runtime(tmp_path)
    _seed(runtime)
    digest = runtime.routine_store.load(MORNING_DIGEST_ROUTINE_ID)

    execute_routine(runtime, digest, checked_at=datetime.now(UTC), record=False)
    assert _activities(tmp_path) == []
    execute_routine(runtime, digest, checked_at=datetime.now(UTC), record=True)

    [row] = _activities(tmp_path)
    assert row.metadata["trigger"] == "manual"


# --- expiry: the digest's run closes what aged out before it renders -------------------


def test_the_digest_run_expires_aged_out_tasks_first_but_a_dry_run_does_not(
    tmp_path: Path, settings: _Settings
) -> None:
    from iris_harness.services.tasks.store import TaskStore

    tasks = TaskStore(db_path=tmp_path / "tasks.db")
    tasks.ensure_schema()
    prep = tasks.create(
        title="Prep: Swim lesson",
        due_at=datetime(2026, 9, 22, 22, tzinfo=UTC),
        source_kind="calendar-prep",
    )
    runtime, _ = _runtime(tmp_path)
    _seed(runtime)
    digest = runtime.routine_store.load(MORNING_DIGEST_ROUTINE_ID)

    execute_routine(runtime, digest, checked_at=datetime.now(UTC), record=False)
    assert tasks.get(prep.id).status == "open"  # type: ignore[union-attr]

    execute_routine(runtime, digest, checked_at=datetime.now(UTC), record=True)
    assert tasks.get(prep.id).status == "expired"  # type: ignore[union-attr]


def test_a_real_digest_run_ages_missed_reminders_but_a_dry_run_does_not(
    tmp_path: Path, settings: _Settings
) -> None:
    """D18: a missed reminder is carried into ``reminder_missed_digests`` (1) real
    digests, then expires; a manual re-run neither counts nor expires."""
    import sqlite3

    from iris_harness.services.notifications.store import ReminderStore

    reminders = ReminderStore(db_path=tmp_path / "tasks.db")
    reminders.ensure_schema()
    row = reminders.create(
        target_kind="task",
        target_id="t1",
        remind_at=datetime(2026, 9, 21, 13, tzinfo=UTC),
        note="Take out the recycling",
    )
    with sqlite3.connect(tmp_path / "tasks.db") as conn:
        conn.execute("UPDATE notification_reminders SET status = 'failed' WHERE id = ?", (row.id,))
    runtime, _ = _runtime(tmp_path)
    _seed(runtime)
    digest = runtime.routine_store.load(MORNING_DIGEST_ROUTINE_ID)

    execute_routine(runtime, digest, checked_at=datetime.now(UTC), record=False)
    assert reminders.get(row.id).missed_digests == 0  # type: ignore[union-attr]

    execute_routine(runtime, digest, checked_at=datetime.now(UTC), record=True)
    shown = reminders.get(row.id)
    assert shown is not None and (shown.status, shown.missed_digests) == ("failed", 1)

    execute_routine(runtime, digest, checked_at=datetime.now(UTC), record=True)
    assert reminders.get(row.id).status == "expired"  # type: ignore[union-attr]


def test_a_real_digest_run_lists_a_delivered_unacknowledged_reminder_once_then_expires_it(
    tmp_path: Path, settings: _Settings
) -> None:
    """D18 + PR 3b: a delivered (``sent``) reminder from before today that the owner never
    answered is counted into the next real digest (its Missed line), and the real run
    after that expires it; a dry run changes nothing."""
    from iris_harness.services.notifications.store import ReminderStore

    reminders = ReminderStore(db_path=tmp_path / "tasks.db")
    reminders.ensure_schema()
    at = datetime(2026, 9, 21, 13, tzinfo=UTC)
    row = reminders.create(
        target_kind="task", target_id="t1", remind_at=at, note="Call the dentist"
    )
    reminders.mark_sent(row.id, delivered_channels=["telegram"], now=at)
    runtime, _ = _runtime(tmp_path)
    _seed(runtime)
    digest = runtime.routine_store.load(MORNING_DIGEST_ROUTINE_ID)

    execute_routine(runtime, digest, checked_at=datetime.now(UTC), record=False)
    assert reminders.get(row.id).missed_digests == 0  # type: ignore[union-attr]

    execute_routine(runtime, digest, checked_at=datetime.now(UTC), record=True)
    shown = reminders.get(row.id)
    assert shown is not None and shown.status == "sent" and shown.missed_digests == 1

    execute_routine(runtime, digest, checked_at=datetime.now(UTC), record=True)
    done = reminders.get(row.id)
    assert done is not None and done.status == "expired"
    assert done.closed_reason == "expired: delivered, not acknowledged"


def test_a_failing_expiry_sweep_never_stops_the_digest(
    tmp_path: Path, settings: _Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.services.digest import expiry

    def boom(*_: Any, **__: Any) -> list[Any]:
        raise RuntimeError("tasks.db locked")

    monkeypatch.setattr(expiry, "sweep_expired_tasks", boom)
    runtime, seen = _runtime(tmp_path)
    _seed(runtime)
    digest = runtime.routine_store.load(MORNING_DIGEST_ROUTINE_ID)

    record = execute_routine(runtime, digest, checked_at=datetime.now(UTC), record=True)
    assert record.status == "success" and seen
