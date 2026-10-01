"""Tests for routine_tick heartbeat wiring."""

from __future__ import annotations

from types import SimpleNamespace

from iris_harness.runtime.handlers.ticks import build_routine_tick_handler
from iris_harness.services.heartbeat import (
    HeartbeatDefinition,
    HeartbeatRun,
    HeartbeatScheduler,
    HeartbeatStatus,
)
from iris_harness.services.routines import RoutineApprovalStatus, RoutineStore, create_routine_spec


def test_routine_tick_executes_due_routine_template_once_per_interval(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = create_routine_spec(
        title="Skill brief routine",
        goal="Send briefing",
        schedule="interval:60",
        template="skill_brief",
        approval_status=RoutineApprovalStatus.SCHEDULED,
        metadata={"skill_id": "morning-briefing"},
    )
    store.save(spec)
    scheduler = HeartbeatScheduler()
    seen_params: dict[str, object] = {}

    def skill_brief(definition: HeartbeatDefinition) -> HeartbeatRun:
        seen_params.update(definition.params)
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            output="brief sent",
        )

    scheduler.register_handler("skill_brief", skill_brief)
    scheduler.register(
        HeartbeatDefinition(
            name="skill_brief",
            handler="skill_brief",
            schedule="interval:3600",
        )
    )
    runtime = SimpleNamespace(routine_store=store, heartbeats=scheduler)
    handler = build_routine_tick_handler(runtime)
    definition = HeartbeatDefinition(
        name="routine_tick",
        handler="routine_tick",
        schedule="interval:60",
    )

    first = handler(definition)
    second = handler(definition)
    updated = store.load(spec.id)

    assert first.status is HeartbeatStatus.SUCCESS
    assert "due=1" in first.output
    assert "success=1" in first.output
    assert second.status is HeartbeatStatus.SUCCESS
    assert "due=0" in second.output
    assert updated is not None
    assert updated.run_count == 1
    assert updated.success_count == 1
    assert updated.failure_count == 0
    assert seen_params["channel"] == "console"
    assert seen_params["routine_id"] == spec.id
    assert seen_params["skill_id"] == "morning-briefing"


def test_routine_tick_records_failed_unknown_template(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = create_routine_spec(
        title="Unknown routine",
        goal="Call missing template",
        schedule="interval:60",
        template="missing_template",
        approval_status=RoutineApprovalStatus.APPROVED,
    )
    store.save(spec)
    runtime = SimpleNamespace(routine_store=store, heartbeats=HeartbeatScheduler())
    handler = build_routine_tick_handler(runtime)

    run = handler(
        HeartbeatDefinition(
            name="routine_tick",
            handler="routine_tick",
            schedule="interval:60",
        )
    )
    updated = store.load(spec.id)

    assert run.status is HeartbeatStatus.FAILED
    assert "failed=1" in run.output
    assert "missing_template" in run.error
    assert updated is not None
    assert updated.run_count == 1
    assert updated.success_count == 0
    assert updated.failure_count == 1


def test_routine_tick_executes_registered_template_handler_without_definition(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = create_routine_spec(
        title="Skill brief direct",
        goal="Send brief",
        schedule="interval:60",
        template="skill_brief",
        approval_status=RoutineApprovalStatus.APPROVED,
        metadata={"skill_id": "daily-repo-brief"},
    )
    store.save(spec)
    scheduler = HeartbeatScheduler()
    scheduler.register_handler(
        "skill_brief",
        lambda definition: HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            output=f"sent to {definition.params.get('channel')}",
        ),
    )
    runtime = SimpleNamespace(routine_store=store, heartbeats=scheduler)
    handler = build_routine_tick_handler(runtime)

    run = handler(
        HeartbeatDefinition(
            name="routine_tick",
            handler="routine_tick",
            schedule="interval:60",
        )
    )

    assert run.status is HeartbeatStatus.SUCCESS
    assert "success=1" in run.output
    updated = store.load(spec.id)
    assert updated is not None
    assert updated.success_count == 1


def test_routine_tick_aliases_legacy_morning_briefing_template(tmp_path) -> None:
    """Legacy template names route to skill_brief with the mapped skill_id."""
    store = RoutineStore(tmp_path / "routines.db")
    spec = create_routine_spec(
        title="Legacy morning briefing",
        goal="Send briefing",
        schedule="interval:60",
        template="morning_briefing",
        approval_status=RoutineApprovalStatus.SCHEDULED,
    )
    store.save(spec)
    scheduler = HeartbeatScheduler()
    seen: dict[str, object] = {}

    def skill_brief(definition: HeartbeatDefinition) -> HeartbeatRun:
        seen["name"] = definition.name
        seen["handler"] = definition.handler
        seen["skill_id"] = definition.params.get("skill_id")
        return HeartbeatRun(
            name=definition.name,
            status=HeartbeatStatus.SUCCESS,
            output="ok",
        )

    scheduler.register_handler("skill_brief", skill_brief)
    runtime = SimpleNamespace(routine_store=store, heartbeats=scheduler)
    handler = build_routine_tick_handler(runtime)
    run = handler(
        HeartbeatDefinition(name="routine_tick", handler="routine_tick", schedule="interval:60")
    )

    assert run.status is HeartbeatStatus.SUCCESS
    assert seen["handler"] == "skill_brief"
    assert seen["skill_id"] == "morning-briefing"


def test_routine_tick_resolves_canonical_brief_name_via_registry(tmp_path) -> None:
    """A routine whose template is the canonical brief manifest name (what new
    routines persist) must resolve to skill_brief and fire — previously it fell
    through to 'handler not registered' and failed every tick. The routine's
    section selection + content style are forwarded into the render params."""
    store = RoutineStore(tmp_path / "routines.db")
    spec = create_routine_spec(
        title="Canonical brief",
        goal="Send briefing",
        schedule="interval:60",
        template="morning-briefing",
        approval_status=RoutineApprovalStatus.SCHEDULED,
        source_preferences=("reminders", "stocks"),
        metadata={"content_lines_per_item": 3},
    )
    store.save(spec)
    scheduler = HeartbeatScheduler()
    seen: dict[str, object] = {}

    def skill_brief(definition: HeartbeatDefinition) -> HeartbeatRun:
        seen.update(definition.params)
        return HeartbeatRun(name=definition.name, status=HeartbeatStatus.SUCCESS, output="ok")

    scheduler.register_handler("skill_brief", skill_brief)

    # Fake registry that reports a loadable brief named "morning-briefing".
    brief_pkg = SimpleNamespace(
        manifest=SimpleNamespace(kind="brief", name="morning-briefing"),
        is_loadable=True,
    )
    registry = SimpleNamespace(
        list_packages=lambda *, agent_name=None, only_loadable=False: (brief_pkg,)
    )
    runtime = SimpleNamespace(routine_store=store, heartbeats=scheduler, skill_registry=registry)
    handler = build_routine_tick_handler(runtime)
    run = handler(
        HeartbeatDefinition(name="routine_tick", handler="routine_tick", schedule="interval:60")
    )

    assert run.status is HeartbeatStatus.SUCCESS
    assert "success=1" in run.output
    assert seen["skill_id"] == "morning-briefing"
    assert seen["source_preferences"] == ["reminders", "stocks"]
    assert seen["content_lines_per_item"] == 3


def test_routine_tick_aliases_legacy_daily_repo_brief_template(tmp_path) -> None:
    store = RoutineStore(tmp_path / "routines.db")
    spec = create_routine_spec(
        title="Legacy repo brief",
        goal="Send repos",
        schedule="interval:60",
        template="daily_repo_brief",
        approval_status=RoutineApprovalStatus.SCHEDULED,
    )
    store.save(spec)
    scheduler = HeartbeatScheduler()
    seen: dict[str, object] = {}

    def skill_brief(definition: HeartbeatDefinition) -> HeartbeatRun:
        seen["skill_id"] = definition.params.get("skill_id")
        return HeartbeatRun(name=definition.name, status=HeartbeatStatus.SUCCESS, output="ok")

    scheduler.register_handler("skill_brief", skill_brief)
    runtime = SimpleNamespace(routine_store=store, heartbeats=scheduler)
    handler = build_routine_tick_handler(runtime)
    run = handler(
        HeartbeatDefinition(name="routine_tick", handler="routine_tick", schedule="interval:60")
    )
    assert run.status is HeartbeatStatus.SUCCESS
    assert seen["skill_id"] == "daily-repo-brief"
