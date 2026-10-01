"""A task's title on one digest line: repeats collapsed, capped (the full title is kept)."""

from __future__ import annotations

import pytest

from iris_harness.services.tasks.brief_view import TITLE_CAP, collapse_repeats, short_title

# A real-world calendar title shape, a recurring-class feed.
CLASS = (
    "Prep: Ceramics - Evening Wheel Classes (levels 1 - 4 adults) - Evening Wheel "
    "Classes (levels 1 - 4 adults): Stage 3 - Clay Centring - Fall 1: Evening: Stage 3 "
    "- Clay Centring Thu 6:30pm"
)


def test_repeated_segments_collapse() -> None:
    assert collapse_repeats(CLASS) == (
        "Prep: Ceramics - Evening Wheel Classes (levels 1 - 4 adults): Stage 3 - "
        "Clay Centring - Fall 1 - Thu 6:30pm"
    )


def test_a_digest_line_is_capped_at_a_word_with_an_ellipsis() -> None:
    line = short_title(CLASS)
    assert len(line) <= TITLE_CAP
    assert line == "Prep: Ceramics - Evening Wheel Classes (levels 1 - 4 adults): Stage 3 - Clay…"


def test_two_stages_stay_told_apart() -> None:
    stage4 = CLASS.replace("Stage 3 - Clay Centring", "Stage 4 - Trimming Techniques")
    assert short_title(stage4) != short_title(CLASS)
    assert "Stage 4" in short_title(stage4)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Renew car registration", "Renew car registration"),
        ("Call bank: Call bank", "Call bank"),
        ("Team sync - Weekly - Team sync", "Team sync - Weekly"),
        ("Q3 review (draft - v2) - Q3 review (draft - v2)", "Q3 review (draft - v2)"),
        ("A - A", "A"),
    ],
)
def test_collapse_is_generic(title: str, expected: str) -> None:
    assert collapse_repeats(title) == expected


def test_a_short_title_is_left_alone() -> None:
    assert short_title("Pay rent") == "Pay rent"
    assert short_title("x" * 200).endswith("…") and len(short_title("x" * 200)) == TITLE_CAP
