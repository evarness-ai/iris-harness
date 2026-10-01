"""Tests for per-routine formatting capture (roadmap "next slice")."""

from __future__ import annotations

from iris_harness.services.routines.authoring import extract_formatting
from iris_harness.services.routines.models import RoutineFormatting


def test_extract_header_footer_tone() -> None:
    out = extract_formatting(
        "every morning send my brief, start with a greeting 'Good morning' and "
        "end with 'Have a great day', keep it brief"
    )
    assert out == {"header": "Good morning", "footer": "Have a great day", "tone": "brief"}


def test_extract_formal_tone_and_quoted_footer() -> None:
    out = extract_formatting('formal tone please, footer: "Regards, IRIS"')
    assert out["tone"] == "formal"
    assert out["footer"] == "Regards, IRIS"


def test_extract_formatting_none_when_absent() -> None:
    assert extract_formatting("every weekday at 8am send reminders and events") == {}


def test_routine_formatting_round_trips_metadata() -> None:
    f = RoutineFormatting(
        section_order=("reminders", "events"),
        header="Good morning",
        footer="Bye",
        content_lines_per_item=2,
        tone="brief",
    )
    md = f.to_metadata()
    assert md == {
        "section_order": ["reminders", "events"],
        "header": "Good morning",
        "footer": "Bye",
        "content_lines_per_item": 2,
        "tone": "brief",
    }
    assert RoutineFormatting.from_metadata(md) == f


def test_routine_formatting_empty_emits_no_keys() -> None:
    f = RoutineFormatting()
    assert f.is_empty()
    assert f.to_metadata() == {}
    # from_metadata tolerates a metadata dict carrying unrelated keys.
    assert RoutineFormatting.from_metadata({"session_id": "s", "original_query": "x"}).is_empty()
