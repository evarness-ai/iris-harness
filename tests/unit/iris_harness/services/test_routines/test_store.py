"""Tests for routine lifecycle contracts and storage."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from iris_harness.services.routines import (
    RoutineApprovalRequestStatus,
    RoutineApprovalStatus,
    RoutineSpec,
    RoutineStore,
    create_routine_spec,
)


def _local(year: int, month: int, day: int, hour: int, minute: int, second: int = 0) -> datetime:
    """A timezone-aware datetime at the given wall-clock time in the MACHINE-local zone.

    ``daily:``/``cron:`` schedules are matched in local time, so due-logic tests must
    feed local-aware ``now`` values (not UTC) to stay deterministic across CI zones.
    """
    return datetime(year, month, day, hour, minute, second).astimezone()


def test_create_routine_spec_defaults_to_draft() -> None:
    spec = create_routine_spec(
        title="Daily repo brief",
        goal="Send top repositories of the day",
        schedule="cron:0 10 * * *",
        template="repo_brief",
        required_capabilities=("research", "channel_delivery"),
    )

    assert spec.schema_version == "routine-spec/v1"
    assert spec.id.startswith("routine-")
    assert spec.id.endswith(spec.id[-8:])
    assert spec.approval_status == RoutineApprovalStatus.DRAFT
    assert spec.delivery_channel == "console"
    assert spec.required_capabilities == ("research", "channel_delivery")
    assert spec.run_count == 0
    assert not spec.is_approved_for_execution


def test_routine_spec_requires_public_contract_fields() -> None:
    with pytest.raises(ValidationError):
        RoutineSpec(id="routine-test", title="", goal="g", schedule="daily", template="t")


def test_routine_store_round_trips_spec(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = create_routine_spec(
        title="Daily repo brief",
        goal="Send top repositories of the day",
        schedule="cron:0 10 * * *",
        template="repo_brief",
        source_preferences=("github_trending", "hacker_news"),
        required_capabilities=("research", "wiki_search"),
        approval_status=RoutineApprovalStatus.APPROVED,
        metadata={"created_from": "chat"},
    )

    store.save(spec)
    loaded = store.load(spec.id)

    assert loaded == spec
    assert store.list_all() == [spec]
    assert store.list_by_status(RoutineApprovalStatus.APPROVED) == [spec]
    assert store.list_executable() == [spec]


def test_routine_store_lists_due_interval_routines(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    due = create_routine_spec(
        title="Every minute brief",
        goal="Send a short brief",
        schedule="interval:60",
        template="morning_briefing",
        approval_status=RoutineApprovalStatus.APPROVED,
    )
    draft = create_routine_spec(
        title="Draft brief",
        goal="Not approved yet",
        schedule="interval:60",
        template="morning_briefing",
    )
    recent = create_routine_spec(
        title="Recently run brief",
        goal="Wait for interval",
        schedule="interval:60",
        template="morning_briefing",
        approval_status=RoutineApprovalStatus.SCHEDULED,
    ).record_run(success=True, finished_at=datetime(2026, 5, 8, 10, 0, tzinfo=UTC))
    store.save(due)
    store.save(draft)
    store.save(recent)

    listed = store.list_due(now=datetime(2026, 5, 8, 10, 0, 30, tzinfo=UTC))

    assert listed == [due]


def test_routine_store_lists_due_cron_once_per_minute(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = create_routine_spec(
        title="Ten AM brief",
        goal="Run once at ten",
        schedule="cron:0 10 * * *",
        template="morning_briefing",
        approval_status=RoutineApprovalStatus.APPROVED,
    )
    store.save(spec)
    first_check = _local(2026, 5, 8, 10, 0, 30)

    assert store.list_due(now=first_check) == [spec]

    updated = store.record_run(spec.id, success=True, finished_at=first_check)
    assert updated is not None
    assert store.list_due(now=_local(2026, 5, 8, 10, 0, 59)) == []
    assert store.list_due(now=_local(2026, 5, 8, 10, 1)) == []


def test_routine_store_lists_due_daily_routines_once_per_day(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = create_routine_spec(
        title="Daily brief",
        goal="Run once daily",
        schedule="daily:10:00",
        template="morning_briefing",
        approval_status=RoutineApprovalStatus.APPROVED,
    )
    store.save(spec)
    first_check = _local(2026, 5, 8, 10, 0)

    assert store.list_due(now=first_check) == [spec]

    updated = store.record_run(spec.id, success=True, finished_at=first_check)
    assert updated is not None
    # Same day, still inside the grace window → not due again (already ran today).
    assert store.list_due(now=_local(2026, 5, 8, 10, 3)) == []
    # Next day at the scheduled local time → due again.
    assert store.list_due(now=_local(2026, 5, 9, 10, 0)) == [updated]


def test_routine_daily_schedule_is_local_time_with_grace(tmp_path) -> None:
    """Regression: daily routines fire in LOCAL time (not UTC) within a grace window,
    and a fully-missed window is not caught up (strict). (issue 0030)"""
    store = RoutineStore(tmp_path / "routines.db")
    spec = create_routine_spec(
        title="Nine AM brief",
        goal="Run at nine local",
        schedule="daily:09:00",
        template="morning_briefing",
        approval_status=RoutineApprovalStatus.APPROVED,
    )
    store.save(spec)
    # Just before the scheduled minute → not yet due.
    assert store.list_due(now=_local(2026, 5, 8, 8, 59)) == []
    # At 09:00 local and a few minutes into the grace window → due.
    assert store.list_due(now=_local(2026, 5, 8, 9, 0)) == [spec]
    assert store.list_due(now=_local(2026, 5, 8, 9, 4)) == [spec]
    # Well past the grace window (strict, no catch-up) → not due.
    assert store.list_due(now=_local(2026, 5, 8, 11, 30)) == []


def test_routine_store_records_run_result(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = create_routine_spec(
        title="Morning brief",
        goal="Send briefing",
        schedule="interval:86400",
        template="morning_briefing",
    )
    store.save(spec)

    finished_at = datetime(2026, 5, 8, 10, 30, tzinfo=UTC)
    updated = store.record_run(spec.id, success=False, finished_at=finished_at)

    assert updated is not None
    assert updated.run_count == 1
    assert updated.success_count == 0
    assert updated.failure_count == 1
    assert updated.last_run_at == finished_at
    assert store.load(spec.id) == updated


def test_routine_store_delete_missing_and_existing(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = create_routine_spec(
        title="Weekly cleanup",
        goal="Review stale wiki pages",
        schedule="cron:0 9 * * 1",
        template="wiki_lint",
    )
    store.save(spec)

    assert not store.delete("missing")
    assert store.delete(spec.id)
    assert store.load(spec.id) is None


def test_routine_store_clear_deletes_all_routines(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    first = create_routine_spec(
        title="Morning brief",
        goal="Send a morning brief",
        schedule="daily:08:00",
        template="morning_briefing",
    )
    second = create_routine_spec(
        title="Daily repo brief",
        goal="Send trending repos",
        schedule="daily:09:00",
        template="daily_repo_brief",
    )
    store.save(first)
    store.save(second)

    assert store.clear() == 2

    assert store.list_all() == []
    assert store.clear() == 0


def test_routine_store_persists_approval_request_lifecycle(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = create_routine_spec(
        title="Daily repo brief",
        goal="Send top repos",
        schedule="daily:08:00",
        template="daily_repo_brief",
    )
    store.save(spec)

    request = store.create_approval_request(
        routine_id=spec.id,
        session_id="chat-session",
        prompt="Every morning send repo brief",
        metadata={"source": "chat"},
    )

    assert request.status == RoutineApprovalRequestStatus.PENDING
    assert store.get_pending_approval_request("chat-session") == request
    assert store.load_approval_request(request.id) == request
    assert store.list_approval_requests(status=RoutineApprovalRequestStatus.PENDING) == [request]

    resolved = store.resolve_approval_request(
        request.id,
        RoutineApprovalRequestStatus.APPROVED,
    )

    assert resolved is not None
    assert resolved.status == RoutineApprovalRequestStatus.APPROVED
    assert resolved.resolved_at is not None
    assert store.get_pending_approval_request("chat-session") is None


def test_routine_store_supersedes_stale_pending_approval_requests(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    first = create_routine_spec(
        title="First repo brief",
        goal="Send first brief",
        schedule="daily:08:00",
        template="daily_repo_brief",
    )
    second = create_routine_spec(
        title="Second repo brief",
        goal="Send second brief",
        schedule="daily:09:00",
        template="daily_repo_brief",
    )
    store.save(first)
    store.save(second)
    first_request = store.create_approval_request(
        routine_id=first.id,
        session_id="chat-session",
        prompt="first",
    )

    assert store.supersede_pending_approval_requests("chat-session") == 1
    second_request = store.create_approval_request(
        routine_id=second.id,
        session_id="chat-session",
        prompt="second",
    )

    assert store.get_pending_approval_request("chat-session") == second_request
    stale = store.load_approval_request(first_request.id)
    assert stale is not None
    assert stale.status == RoutineApprovalRequestStatus.SUPERSEDED


def test_list_recent_approved_by_session_filters_status_and_session(
    tmp_path,
) -> None:
    """Returns only SCHEDULED routines whose metadata.session_id matches.
    Phase B post-approval refinement uses this to find the routine the
    user means by "deliver to telegram" after approval."""

    store = RoutineStore(db_path=tmp_path / "routines.db")

    in_session_scheduled = store.save(
        create_routine_spec(
            title="A",
            goal="g",
            schedule="daily:09:00",
            template="t",
            approval_status=RoutineApprovalStatus.SCHEDULED,
            metadata={"session_id": "s1"},
        )
    )
    # Different session, same status — excluded.
    store.save(
        create_routine_spec(
            title="B",
            goal="g",
            schedule="daily:09:00",
            template="t",
            approval_status=RoutineApprovalStatus.SCHEDULED,
            metadata={"session_id": "s2"},
        )
    )
    # Same session, wrong status — excluded.
    store.save(
        create_routine_spec(
            title="C",
            goal="g",
            schedule="daily:09:00",
            template="t",
            approval_status=RoutineApprovalStatus.DRAFT,
            metadata={"session_id": "s1"},
        )
    )

    matches = store.list_recent_approved_by_session("s1")

    assert [m.id for m in matches] == [in_session_scheduled.id]


def test_list_recent_approved_by_session_honors_since_cutoff(tmp_path) -> None:
    store = RoutineStore(db_path=tmp_path / "routines.db")

    old = store.save(
        create_routine_spec(
            title="old",
            goal="g",
            schedule="daily:09:00",
            template="t",
            approval_status=RoutineApprovalStatus.SCHEDULED,
            metadata={"session_id": "s"},
        )
    )
    # Backdate ``old`` via a direct save with an explicit updated_at.
    rewound = old.model_copy(update={"updated_at": datetime(2020, 1, 1, tzinfo=UTC)})
    store.save(rewound)

    matches = store.list_recent_approved_by_session("s", since=datetime(2024, 1, 1, tzinfo=UTC))

    assert matches == []


def test_find_session_duplicate_matches_session_and_template(tmp_path) -> None:
    """A re-stated draft in the same session for the same template is found, so the
    authoring flow can update-not-accumulate (roadmap 'next slice')."""
    store = RoutineStore(tmp_path / "routines.db")
    draft = create_routine_spec(
        title="Morning brief",
        goal="g",
        schedule="8am",
        template="morning_briefing",
        metadata={"session_id": "sess1"},
    )
    store.save(draft)

    found = store.find_session_duplicate("sess1", template="morning_briefing")
    assert found is not None and found.id == draft.id
    # Different session or template -> no match (distinct routines stay separate).
    assert store.find_session_duplicate("other", template="morning_briefing") is None
    assert store.find_session_duplicate("sess1", template="repo_brief") is None


def test_find_session_duplicate_ignores_approved(tmp_path) -> None:
    """A committed (approved) routine is never silently overwritten by re-authoring."""
    store = RoutineStore(tmp_path / "routines.db")
    store.save(
        create_routine_spec(
            title="Morning brief",
            goal="g",
            schedule="8am",
            template="morning_briefing",
            approval_status=RoutineApprovalStatus.APPROVED,
            metadata={"session_id": "sess1"},
        )
    )
    assert store.find_session_duplicate("sess1", template="morning_briefing") is None


# --- Pinned zone (metadata["timezone"]) + DST: the seeded digest runs in IRIS_TZ -----


def _pinned_daily(time_text: str, zone: str = "America/Chicago") -> RoutineSpec:
    return create_routine_spec(
        title="Digest",
        goal="daily digest",
        schedule=f"daily:{time_text}",
        template="morning-briefing",
        approval_status=RoutineApprovalStatus.SCHEDULED,
        metadata={"timezone": zone},
    )


def _utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)  # type: ignore[misc]


def test_pinned_zone_daily_fires_at_local_time_in_summer_and_winter(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = store.save(_pinned_daily("07:00"))

    # CDT (UTC-5): 07:00 local is 12:00 UTC — regardless of the machine's own zone.
    assert store.list_due(now=_utc(2026, 7, 1, 11, 59)) == []
    assert store.list_due(now=_utc(2026, 7, 1, 12, 0, 30)) == [spec]
    # CST (UTC-6): 07:00 local is 13:00 UTC.
    assert store.list_due(now=_utc(2026, 12, 1, 12, 0, 30)) == []
    assert store.list_due(now=_utc(2026, 12, 1, 13, 0, 30)) == [spec]


def test_pinned_zone_daily_fires_once_on_both_dst_transition_days(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = store.save(_pinned_daily("07:00"))

    # 2026-11-01 fall back: 07:00 CST = 13:00 UTC (not 12:00).
    assert store.list_due(now=_utc(2026, 11, 1, 12, 1)) == []
    assert store.list_due(now=_utc(2026, 11, 1, 13, 1)) == [spec]
    ran = store.record_run(spec.id, success=True, finished_at=_utc(2026, 11, 1, 13, 1))
    assert store.list_due(now=_utc(2026, 11, 1, 13, 2)) == []
    # 2027-03-14 spring forward: 07:00 CDT = 12:00 UTC (not 13:00).
    assert store.list_due(now=_utc(2027, 3, 14, 12, 1)) == [ran]
    assert store.list_due(now=_utc(2027, 3, 14, 13, 1)) == []


def test_pinned_zone_time_in_the_spring_gap_still_fires(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = store.save(_pinned_daily("02:30"))

    # 02:30 does not exist on 2027-03-14 in Chicago; it resolves to 03:30 CDT (08:30 UTC).
    assert store.list_due(now=_utc(2027, 3, 14, 8, 31)) == [spec]


def test_pinned_zone_repeated_hour_on_fall_back_fires_once(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = store.save(_pinned_daily("01:30"))

    # 2026-11-01: 01:30 CDT (06:30 UTC) happens, then 01:30 CST (07:30 UTC) again.
    assert store.list_due(now=_utc(2026, 11, 1, 6, 31)) == [spec]
    store.record_run(spec.id, success=True, finished_at=_utc(2026, 11, 1, 6, 31))
    assert store.list_due(now=_utc(2026, 11, 1, 7, 31)) == []


def test_manual_run_earlier_in_the_day_does_not_swallow_the_scheduled_run(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = store.save(_pinned_daily("07:00"))
    # Owner taps "run now" at 06:00 CDT (11:00 UTC).
    ran = store.record_run(spec.id, success=True, finished_at=_utc(2026, 7, 1, 11, 0))

    assert store.list_due(now=_utc(2026, 7, 1, 12, 0, 30)) == [ran]


def test_unknown_pinned_zone_falls_back_to_machine_local(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = store.save(_pinned_daily("09:00", zone="Mars/Olympus_Mons"))

    assert store.list_due(now=_local(2026, 5, 8, 9, 0)) == [spec]
