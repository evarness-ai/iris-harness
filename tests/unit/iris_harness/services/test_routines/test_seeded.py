"""The seeded ``morning-digest`` core routine (ADR-0122 §3, loop-proof D4)."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

from iris_harness.services.digest.settings import DigestSettings
from iris_harness.services.routines import RoutineApprovalStatus, RoutineStore, create_routine_spec
from iris_harness.services.routines.seeded import (
    MORNING_DIGEST_ROUTINE_ID,
    MORNING_DIGEST_TEMPLATE,
    is_core_routine,
    seed_morning_digest,
    sync_morning_digest,
)

CHICAGO = ZoneInfo("America/Chicago")


def _store(tmp_path: Path) -> RoutineStore:
    return RoutineStore(tmp_path / "routines.db")


def test_seed_creates_scheduled_digest_with_stable_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    settings = DigestSettings(
        time="07:00",
        channel="all",
        sections=("bills_due", "todays_events"),
        section_config={"bills_due": {"max_items": 4}},
    )

    seeded = seed_morning_digest(store, settings, CHICAGO)

    assert seeded is not None
    assert seeded.id == MORNING_DIGEST_ROUTINE_ID == "morning-digest"
    loaded = store.load("morning-digest")
    assert loaded == seeded
    assert loaded.template == MORNING_DIGEST_TEMPLATE == "morning-briefing"
    assert loaded.approval_status == RoutineApprovalStatus.SCHEDULED
    assert loaded.is_approved_for_execution
    assert loaded.schedule == "daily:06:55"
    assert loaded.delivery_channel == "all"
    assert loaded.source_preferences == ("bills_due", "todays_events")
    assert loaded.metadata["timezone"] == "America/Chicago"
    assert loaded.metadata["section_order"] == ["bills_due", "todays_events"]
    assert loaded.metadata["section_line_caps"] == {"bills_due": 4}
    assert loaded.metadata["core"] is True
    assert loaded.metadata["digest"] is True  # rendered grouped (skill_brief.DIGEST_PARAM)
    assert is_core_routine(loaded)


def test_seed_is_idempotent_across_restarts(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = seed_morning_digest(store, DigestSettings(), CHICAGO)
    # A "restart": a new store over the same file, seeding again.
    second = seed_morning_digest(RoutineStore(tmp_path / "routines.db"), DigestSettings(), CHICAGO)

    assert first is not None
    assert second is None
    assert [r.id for r in store.list_all()] == ["morning-digest"]


def test_seed_never_overwrites_an_owner_edited_or_disabled_digest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    seeded = seed_morning_digest(store, DigestSettings(), CHICAGO)
    assert seeded is not None
    edited = seeded.model_copy(
        update={
            "title": "My digest",
            "approval_status": RoutineApprovalStatus.PAUSED,
            "metadata": {**seeded.metadata, "header": "Hi"},
        }
    )
    store.save(edited)

    assert seed_morning_digest(store, DigestSettings(time="06:30"), CHICAGO) is None
    assert store.load("morning-digest") == edited


def test_seed_leaves_the_owners_existing_routines_alone(tmp_path: Path) -> None:
    store = _store(tmp_path)
    retired = [
        store.save(
            create_routine_spec(
                title=title,
                goal="old brief",
                schedule="daily:07:00",
                template="morning-briefing",
                source_preferences=(section,),
                approval_status=RoutineApprovalStatus.RETIRED,
            )
        )
        for title, section in (("Stocks", "stocks"), ("AI news", "ai_news"))
    ]

    seed_morning_digest(store, DigestSettings(), CHICAGO)

    for spec in retired:
        again = store.load(spec.id)
        assert again == spec
        assert again.approval_status == RoutineApprovalStatus.RETIRED
        assert again.run_count == 0
    assert len(store.list_all()) == 3


def test_sync_applies_settings_and_keeps_owner_fields(tmp_path: Path) -> None:
    store = _store(tmp_path)
    seeded = seed_morning_digest(store, DigestSettings(), CHICAGO)
    assert seeded is not None
    store.save(
        seeded.model_copy(
            update={
                "title": "My digest",
                "approval_status": RoutineApprovalStatus.PAUSED,
                "metadata": {**seeded.metadata, "footer": "bye"},
            }
        )
    )
    settings = DigestSettings(
        time="6:45",
        channel="telegram",
        sections=("focus", "bills_due"),
        section_config={
            "focus": {"line_cap": 5},
            "bills_due": {"within_days": 7},  # a knob, not a cap
            "news": {"max_items": 0},  # non-positive: ignored
        },
    )

    synced = sync_morning_digest(store, settings, CHICAGO)

    assert synced is not None
    assert synced == store.load("morning-digest")
    assert synced.schedule == "daily:06:40"
    assert synced.delivery_channel == "telegram"
    assert synced.source_preferences == ("focus", "bills_due")
    assert synced.metadata["section_order"] == ["focus", "bills_due"]
    assert synced.metadata["section_line_caps"] == {"focus": 5}
    # Owner-owned fields untouched: status (paused stays paused), title, footer.
    assert synced.approval_status == RoutineApprovalStatus.PAUSED
    assert synced.title == "My digest"
    assert synced.metadata["footer"] == "bye"


def test_sync_is_a_read_when_nothing_changed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    seeded = seed_morning_digest(store, DigestSettings(), CHICAGO)
    assert seeded is not None

    again = sync_morning_digest(store, DigestSettings(), CHICAGO)

    assert again == seeded  # same updated_at: no write


def test_sync_follows_the_timezone(tmp_path: Path) -> None:
    store = _store(tmp_path)
    seed_morning_digest(store, DigestSettings(), ZoneInfo("UTC"))

    synced = sync_morning_digest(store, DigestSettings(), CHICAGO)

    assert synced is not None
    assert synced.metadata["timezone"] == "America/Chicago"


def test_sync_without_a_row_does_not_seed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert sync_morning_digest(store, DigestSettings(), CHICAGO) is None
    assert store.list_all() == []


def test_malformed_time_keeps_the_default(tmp_path: Path) -> None:
    store = _store(tmp_path)
    seeded = seed_morning_digest(store, DigestSettings(time="7am"), CHICAGO)
    assert seeded is not None
    assert seeded.schedule == "daily:06:55"
    synced = sync_morning_digest(store, DigestSettings(time="25:00"), CHICAGO)
    assert synced is not None
    assert synced.schedule == "daily:06:55"


def test_user_routines_are_not_core(tmp_path: Path) -> None:
    spec = create_routine_spec(
        title="Brief", goal="g", schedule="daily:07:00", template="morning-briefing"
    )
    assert not is_core_routine(spec)


def test_lead_wraps_past_midnight(tmp_path: Path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    seed_morning_digest(store, DigestSettings(), CHICAGO)
    synced = sync_morning_digest(store, DigestSettings(time="00:03"), CHICAGO)

    assert synced is not None
    assert synced.schedule == "daily:23:58"
