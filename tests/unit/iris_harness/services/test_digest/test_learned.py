"""The digest footer's ``learned yesterday`` line (loop-proof D17; graph V31, V36).

Pinned: the line is never omitted (``nothing`` on a quiet day), a Focus 👎 recorded
yesterday is named, today's signals wait for tomorrow, and a new signal store joins by
registering a source — the renderer never changes — while a broken source is skipped.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from iris_harness.foundation.settings.store import SettingsStore
from iris_harness.services.digest import learned
from iris_harness.services.digest.learned import (
    learned_sources,
    learned_yesterday_line,
    previous_local_day,
    register_learned_source,
    unregister_learned_source,
)
from iris_harness.services.learning.suppression import (
    EMAIL_FOCUS_SURFACE,
    EMAIL_SEARCH_SUBSYSTEM,
    NOT_USEFUL,
    SOURCE_AUTO,
    USEFUL,
    SurfaceFeedbackStore,
    email_focus_dims,
)

UTC_TZ = ZoneInfo("UTC")


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("IRIS_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
def ledger(data_dir: Path) -> SurfaceFeedbackStore:
    s = SurfaceFeedbackStore(db_path=data_dir / "learning.db")
    s.ensure_schema()
    return s


@pytest.fixture
def restore_registry() -> Iterator[None]:
    before = learned_sources()
    yield
    for name in list(learned_sources()):
        unregister_learned_source(name)
    for name, source in before.items():
        register_learned_source(name, source)


def _tomorrow() -> datetime:
    """A 'now' whose previous local day is today, so a signal recorded now is 'yesterday'."""
    return datetime.now(UTC) + timedelta(days=1)


def _hide(ledger: SurfaceFeedbackStore, sender: str, verdict: str = NOT_USEFUL) -> None:
    ledger.record(
        EMAIL_SEARCH_SUBSYSTEM,
        EMAIL_FOCUS_SURFACE,
        email_focus_dims(sender),
        verdict,
        emit_signal=False,
    )


def test_a_quiet_day_reads_nothing_never_blank() -> None:
    assert learned_yesterday_line(now=_tomorrow(), tz=UTC_TZ) == "learned yesterday: nothing"


def test_a_focus_not_useful_is_named(ledger: SurfaceFeedbackStore) -> None:
    _hide(ledger, "GoldenPi <news@goldenpi.example>")
    assert (
        learned_yesterday_line(now=_tomorrow(), tz=UTC_TZ)
        == "learned yesterday: news@goldenpi.example hidden from Focus"
    )


def test_todays_signal_waits_for_tomorrow(ledger: SurfaceFeedbackStore) -> None:
    _hide(ledger, "news@goldenpi.example")
    assert learned_yesterday_line(now=datetime.now(UTC), tz=UTC_TZ).endswith("nothing")


def test_the_last_verdict_of_the_day_wins(ledger: SurfaceFeedbackStore) -> None:
    _hide(ledger, "news@goldenpi.example")
    _hide(ledger, "news@goldenpi.example", USEFUL)
    assert (
        learned_yesterday_line(now=_tomorrow(), tz=UTC_TZ)
        == "learned yesterday: news@goldenpi.example back in Focus"
    )


def test_system_verdicts_are_not_the_owners(ledger: SurfaceFeedbackStore) -> None:
    ledger.record(
        EMAIL_SEARCH_SUBSYSTEM,
        EMAIL_FOCUS_SURFACE,
        email_focus_dims("x@auto.example"),
        NOT_USEFUL,
        source=SOURCE_AUTO,
        emit_signal=False,
    )
    assert learned_yesterday_line(now=_tomorrow(), tz=UTC_TZ).endswith("nothing")


def test_settings_changes_are_named_per_section(data_dir: Path) -> None:
    store = SettingsStore(data_dir / "settings.db")
    store.set("digest", "news_topics", ["AI", "world", "Chicago"], old=["AI", "world"], actor="t")
    store.set("digest", "time", "06:30", old="07:00", actor="t")
    assert (
        learned_yesterday_line(now=_tomorrow(), tz=UTC_TZ)
        == "learned yesterday: digest settings changed (news topics, time)"
    )


def test_signals_join_with_a_dot(ledger: SurfaceFeedbackStore, data_dir: Path) -> None:
    _hide(ledger, "news@goldenpi.example")
    SettingsStore(data_dir / "settings.db").set("digest", "time", "06:30", old="07:00", actor="t")
    assert learned_yesterday_line(now=_tomorrow(), tz=UTC_TZ) == (
        "learned yesterday: news@goldenpi.example hidden from Focus · "
        "digest settings changed (time)"
    )


@pytest.mark.usefixtures("restore_registry")
def test_a_new_signal_store_registers_without_touching_the_renderer() -> None:
    register_learned_source("reminders", lambda _s, _e: ["AT&T reminders moved to 18:00"])
    assert learned_yesterday_line(now=_tomorrow(), tz=UTC_TZ) == (
        "learned yesterday: AT&T reminders moved to 18:00"
    )


@pytest.mark.usefixtures("restore_registry")
def test_a_broken_source_is_skipped_not_fatal(ledger: SurfaceFeedbackStore) -> None:
    def boom(_start: datetime, _end: datetime) -> list[str]:
        raise RuntimeError("store locked")

    register_learned_source("broken", boom)
    _hide(ledger, "news@goldenpi.example")
    assert (
        learned_yesterday_line(now=_tomorrow(), tz=UTC_TZ)
        == "learned yesterday: news@goldenpi.example hidden from Focus"
    )


def test_core_registers_its_two_stores() -> None:
    assert {"surface_feedback", "settings"} <= set(learned_sources())
    assert learned.PREFIX == "learned yesterday: "


def test_previous_local_day_is_the_owners_calendar_day() -> None:
    chicago = ZoneInfo("America/Chicago")
    # 03:00 UTC on the 25th is still the evening of the 24th in Chicago.
    start, end = previous_local_day(datetime(2026, 9, 25, 3, 0, tzinfo=UTC), chicago)
    assert (start.isoformat(), end.isoformat()) == (
        "2026-09-23T00:00:00-05:00",
        "2026-09-24T00:00:00-05:00",
    )
