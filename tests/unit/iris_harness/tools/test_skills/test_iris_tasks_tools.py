"""Tests for the iris-tasks builtin brief tools (Phase 2 Track 2C)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.services.tasks.models import WaitFor
from iris_harness.services.tasks.store import TaskStore


def _load_tools_module():
    """Import the skill's tools.py without going through SkillRegistry."""
    repo_root = Path(__file__).resolve().parents[5]
    module_path = repo_root / "config" / "skills" / "builtin" / "iris-tasks" / "tools.py"
    spec = importlib.util.spec_from_file_location("iris_tasks_tools_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def store_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Wire IRIS_TASKS_DB to a per-test tasks.db and return a TaskStore on it."""
    db_path = tmp_path / "tasks.db"
    monkeypatch.setenv("IRIS_TASKS_DB", str(db_path))
    store = TaskStore(db_path=db_path)
    store.ensure_schema()
    return store


def _wait() -> WaitFor:
    return WaitFor(kind="reply_from", payload={"from": "x@example.com"})


# ─── list_open_tasks ────────────────────────────────────────────────────────


def test_list_open_tasks_excludes_followups(store_factory: TaskStore) -> None:
    mod = _load_tools_module()
    store = store_factory
    store.create(title="Plain A", priority=3)
    store.create(title="Plain B", priority=5)
    store.create(title="Followup", priority=9, wait_for=_wait())

    out = mod.ListOpenTasksTool()._run()
    titles = [row["title"] for row in out]
    assert "Followup" not in titles
    assert titles == ["Plain B", "Plain A"]  # priority desc


def test_list_open_tasks_orders_by_priority_then_due(store_factory: TaskStore) -> None:
    mod = _load_tools_module()
    store = store_factory
    now = datetime.now(UTC)
    store.create(title="Soon, low", priority=1, due_at=now + timedelta(hours=2))
    store.create(title="Later, high", priority=5, due_at=now + timedelta(days=2))
    store.create(title="No due, low", priority=1)

    out = mod.ListOpenTasksTool()._run()
    assert [row["title"] for row in out][0] == "Later, high"


def test_list_open_tasks_respects_limit(store_factory: TaskStore) -> None:
    mod = _load_tools_module()
    store = store_factory
    for i in range(5):
        store.create(title=f"t{i}", priority=i)

    out = mod.ListOpenTasksTool()._run(limit=2)
    assert len(out) == 2


# ─── list_due_today ──────────────────────────────────────────────────────────


def test_list_due_today_includes_only_tasks_due_today_or_earlier(
    store_factory: TaskStore,
) -> None:
    mod = _load_tools_module()
    store = store_factory
    now = datetime.now(UTC)
    earlier_today = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if earlier_today > now:
        earlier_today = earlier_today - timedelta(days=1)
    store.create(title="Due earlier", due_at=earlier_today)
    store.create(title="Due tomorrow", due_at=now + timedelta(days=1))
    store.create(title="No due_at")

    out = mod.ListDueTodayTool()._run()
    titles = [row["title"] for row in out]
    assert "Due earlier" in titles
    assert "Due tomorrow" not in titles
    assert "No due_at" not in titles


def test_list_due_today_returns_empty_when_nothing_due(store_factory: TaskStore) -> None:
    mod = _load_tools_module()
    store = store_factory
    store.create(title="Future", due_at=datetime.now(UTC) + timedelta(days=7))

    out = mod.ListDueTodayTool()._run()
    assert out == []


# ─── list_resolved_followups ────────────────────────────────────────────────


def test_list_resolved_followups_surfaces_only_resolved_open_followups(
    store_factory: TaskStore,
) -> None:
    mod = _load_tools_module()
    store = store_factory
    # 1) Followup with wait resolved → should appear
    t1 = store.create(title="Reply arrived", wait_for=_wait())
    store.resolve_wait(t1.id, by_event="test")
    # 2) Followup without resolution → should not appear
    store.create(title="Still waiting", wait_for=_wait())
    # 3) Plain task → should not appear
    store.create(title="Plain")
    # 4) Resolved followup that user already marked done → should not appear
    t4 = store.create(title="Already closed", wait_for=_wait())
    store.resolve_wait(t4.id, by_event="test")
    store.complete(t4.id)

    out = mod.ListResolvedFollowupsTool()._run()
    titles = [row["title"] for row in out]
    assert titles == ["Reply arrived"]
    assert out[0]["from"] == "x@example.com"


def test_list_resolved_followups_returns_empty_when_none(store_factory: TaskStore) -> None:
    mod = _load_tools_module()
    store = store_factory
    store.create(title="Plain")
    assert mod.ListResolvedFollowupsTool()._run() == []


# ─── the owner-facing fields the digest shows ─────────────────────────────────


@pytest.mark.parametrize(
    ("due", "text"),
    [
        (datetime(2026, 9, 25, 22, 0, tzinfo=UTC), "due today 17:00"),  # 17:00 in Chicago
        (datetime(2026, 9, 26, 4, 59, tzinfo=UTC), "due today"),  # end of day: a date
        (datetime(2026, 9, 26, 5, 0, tzinfo=UTC), "due tomorrow"),  # midnight: a date
        (datetime(2026, 9, 26, 14, 30, tzinfo=UTC), "due tomorrow 09:30"),
        (datetime(2026, 10, 13, 15, 0, tzinfo=UTC), "due Oct 13"),
        (datetime(2027, 1, 3, 15, 0, tzinfo=UTC), "due Jan 3, 2027"),
        (datetime(2026, 9, 20, 15, 0, tzinfo=UTC), "overdue since Sep 20"),
        (None, ""),
    ],
)
def test_a_due_date_reads_as_a_person_would_say_it(
    monkeypatch: pytest.MonkeyPatch, due: datetime | None, text: str
) -> None:
    monkeypatch.setenv("IRIS_TZ", "America/Chicago")
    now = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)  # 09:00 in Chicago
    assert _load_tools_module()._due_when(due, now) == text


def test_rows_carry_no_codes_for_the_digest(store_factory: TaskStore) -> None:
    """The digest's templates use `when` and `flag`; the raw id/priority stay for chat."""
    mod = _load_tools_module()
    store_factory.create(title="Renew car registration", priority=0)
    store_factory.create(title="Call the bank", priority=2)
    rows = {r["title"]: r for r in mod.ListOpenTasksTool()._run()}
    assert rows["Renew car registration"]["flag"] == ""
    assert rows["Call the bank"]["flag"] == " · high"
    assert rows["Call the bank"]["priority"] == "2" and rows["Call the bank"]["task_id"]


# ─── the Today group: today's items only, overdue once, expired never ──────────
# (a real digest's shape: the same stale class-prep task three times, "overdue")

CLASS = (
    "Prep: Ceramics - Evening Wheel Classes (levels 1 - 4 adults) - Evening Wheel "
    "Classes (levels 1 - 4 adults): Stage 3 - Clay Centring - Fall 1: Evening: Stage 3 "
    "- Clay Centring Thu 6:30pm"
)


@pytest.fixture
def today(store_factory: TaskStore, monkeypatch: pytest.MonkeyPatch) -> dict[str, datetime]:
    """A store with one of each: stale prep, genuine overdue, aged-out, today, later."""
    monkeypatch.setenv("IRIS_TZ", "UTC")
    now = datetime.now(UTC)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    store = store_factory
    store.create(
        title=CLASS,
        due_at=start - timedelta(days=3) + timedelta(hours=17),
        source_kind="calendar-prep",
    )
    store.create(title="Renew registration", due_at=start - timedelta(days=2) + timedelta(hours=9))
    store.create(title="Call bank", due_at=start - timedelta(hours=12), priority=2)
    store.create(title="Ancient chore", due_at=start - timedelta(days=30))
    store.create(title="Pay rent", due_at=start + timedelta(hours=23, minutes=58))
    store.create(title="Book dentist", due_at=start + timedelta(days=4))
    store.create(title="Someday idea")
    return {"now": now, "start": start}


def test_due_today_for_the_digest_lists_only_todays_tasks(today: dict[str, datetime]) -> None:
    mod = _load_tools_module()
    titles = [r["title"] for r in mod.ListDueTodayTool()._run(include_overdue=False)]
    assert titles == ["Pay rent"]


def test_due_today_never_lists_expired_tasks(today: dict[str, datetime]) -> None:
    mod = _load_tools_module()
    titles = [r["title"] for r in mod.ListDueTodayTool()._run()]
    assert "Ancient chore" not in titles  # overdue past expiry.task_overdue_days
    assert not any(t.startswith("Prep:") for t in titles)  # the meeting is over
    assert {"Renew registration", "Call bank", "Pay rent"} <= set(titles)


def test_overdue_is_one_short_line(today: dict[str, datetime]) -> None:
    mod = _load_tools_module()
    assert mod.ListOverdueTool()._run() == (
        "## Overdue (2)\n- Renew registration · Call bank"  # oldest first
    )


def test_overdue_is_empty_when_nothing_is(store_factory: TaskStore) -> None:
    mod = _load_tools_module()
    store_factory.create(title="Future", due_at=datetime.now(UTC) + timedelta(days=3))
    assert mod.ListOverdueTool()._run() == ""


def test_overdue_names_at_most_the_limit(store_factory: TaskStore) -> None:
    mod = _load_tools_module()
    yesterday = datetime.now(UTC) - timedelta(days=1)
    for i in range(4):
        store_factory.create(title=f"Chore {i}", due_at=yesterday.replace(hour=0, minute=i))
    line = mod.ListOverdueTool()._run(limit=2)
    assert line.startswith("## Overdue (4)\n- Chore 0 · Chore 1 · +2 more")


def test_open_tasks_for_the_digest_skip_what_is_listed_above(today: dict[str, datetime]) -> None:
    mod = _load_tools_module()
    titles = [r["title"] for r in mod.ListOpenTasksTool()._run(skip_due_by_today=True)]
    assert sorted(titles) == ["Book dentist", "Someday idea"]
    # Without the flag (chat), everything open and unexpired.
    everything = {r["title"] for r in mod.ListOpenTasksTool()._run()}
    assert "Ancient chore" not in everything
    assert not any(t.startswith("Prep:") for t in everything)
    assert "Call bank" in everything


def test_a_digest_row_carries_the_short_title_and_keeps_the_full_one(
    store_factory: TaskStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_TZ", "UTC")
    mod = _load_tools_module()
    store_factory.create(title=CLASS, due_at=datetime.now(UTC) + timedelta(days=2))
    (row,) = mod.ListOpenTasksTool()._run()
    assert len(row["title"]) <= 80 and row["title"].endswith("…")
    assert row["title"].count("Evening Wheel Classes") == 1
    assert row["full_title"] == CLASS
