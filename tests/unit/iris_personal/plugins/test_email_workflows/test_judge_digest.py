"""The email judge in the morning digest: Needs reply, Judged yesterday, learned."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.services.digest.learned import (
    learned_between,
    register_learned_source,
    unregister_learned_source,
)
from iris_personal.plugins.email_workflows.judge_config import JudgeConfig
from iris_personal.plugins.email_workflows.judge_digest import (
    judged_line,
    learned_phrases,
    learned_source,
    render_needs_reply,
)
from iris_personal.plugins.email_workflows.judge_words import SurfaceWords

from .judge_fixtures import DAY, Inbox, seed_day

WORDS = SurfaceWords.load()
CONFIG = JudgeConfig.load()
NEXT_MORNING = datetime(2026, 9, 27, 7, tzinfo=UTC)


@pytest.fixture
def inbox(tmp_path: Path) -> Inbox:
    box = Inbox(tmp_path)
    seed_day(box, DAY)
    return box


def needs_reply(inbox: Inbox, now: datetime = NEXT_MORNING, days: int = 3) -> str:
    return render_needs_reply(inbox.judgments, inbox.emails, WORDS, days=days, now=now, tz=UTC)


# -- Needs reply ----------------------------------------------------------------------


def test_needs_reply_lists_sender_and_subject_newest_first(inbox: Inbox) -> None:
    inbox.judged(
        "m-contract",
        "Arun Sample <arun@mail.example>",
        "Re: contract",
        "needs_reply",
        at=DAY + timedelta(hours=15),
    )
    inbox.correct("m-case", "needs_reply", source="chat", at=DAY + timedelta(hours=16))
    assert needs_reply(inbox) == (
        "## Needs reply (3)\n"
        "- Arun Sample: Re: contract\n"
        "- Harbor Bank: About your recent inquiry\n"
        "- Petra Sample: dinner Saturday?"
    )


def test_empty_is_the_section_empty_text(tmp_path: Path) -> None:
    assert needs_reply(Inbox(tmp_path)) == "## Needs reply\nNothing waiting on you."


def test_a_reply_in_the_thread_drops_it(inbox: Inbox) -> None:
    inbox.email(
        "m-sent",
        "Owner <owner@example.com>",
        "Re: dinner Saturday?",
        received=DAY + timedelta(hours=18),
        thread="t-m-dinner",
        labels=("SENT",),
    )
    assert needs_reply(inbox) == "## Needs reply\nNothing waiting on you."


def test_mail_in_the_thread_that_is_not_the_owners_reply_keeps_it(inbox: Inbox) -> None:
    # an earlier sent message, and a later one from the other person
    inbox.email(
        "m-before",
        "Owner <owner@example.com>",
        "dinner?",
        received=DAY - timedelta(days=1),
        thread="t-m-dinner",
        labels=("SENT",),
    )
    inbox.email(
        "m-again",
        "Petra Sample <petra@mail.example>",
        "Re: dinner Saturday?",
        received=DAY + timedelta(hours=19),
        thread="t-m-dinner",
    )
    assert "Petra Sample: dinner Saturday?" in needs_reply(inbox)


def test_a_re_bucket_drops_it(inbox: Inbox) -> None:
    inbox.correct("m-dinner", "fyi", source="web", at=DAY + timedelta(hours=20))
    assert needs_reply(inbox).startswith("## Needs reply\n")


def test_it_expires_after_needs_reply_days(inbox: Inbox) -> None:
    # Judged Sat Sep 26: shown through Tue Sep 29, expired from Wed Sep 30 (3 local days).
    assert "Petra" in needs_reply(inbox, now=datetime(2026, 9, 29, 23, tzinfo=UTC))
    assert "Petra" not in needs_reply(inbox, now=datetime(2026, 9, 30, 0, tzinfo=UTC))


def test_at_most_ten_then_a_count(tmp_path: Path) -> None:
    box = Inbox(tmp_path)
    for i in range(12):
        box.judged(
            f"n{i:02d}",
            f"Person {i} <p{i}@mail.example>",
            f"Question {i}",
            "needs_reply",
            at=DAY + timedelta(minutes=i),
        )
    text = needs_reply(box)
    lines = text.splitlines()
    assert lines[0] == "## Needs reply (12)"
    assert lines[1] == "- Person 11: Question 11"
    assert len(lines) == 12 and lines[-1] == "- +2 more"


# -- Judged yesterday -----------------------------------------------------------------


def _yesterday() -> tuple[datetime, datetime]:
    return DAY, DAY + timedelta(days=1)


def test_the_judged_line_counts_by_effective_bucket(inbox: Inbox) -> None:
    inbox.correct("m-case", "needs_reply", source="gmail", at=DAY + timedelta(hours=20))
    inbox.judgments.mark_waiting("gmail:owner@example.com", ["w1", "w2", "w3"])
    start, end = _yesterday()
    assert judged_line(inbox.judgments, CONFIG, WORDS, start=start, end=end) == (
        "Judged yesterday: 5 · 1 bill · 1 event · 2 needs reply · 1 unsure — please teach "
        "me (Action Center) · 3 waiting for the judge (hidden until judged at the next run)"
    )


def test_the_judged_line_leaves_out_zero_unsure_and_waiting(inbox: Inbox) -> None:
    inbox.correct("m-plan", "promo", source="card", at=DAY + timedelta(hours=20))
    start, end = _yesterday()
    assert judged_line(inbox.judgments, CONFIG, WORDS, start=start, end=end) == (
        "Judged yesterday: 5 · 1 bill · 1 event · 1 needs reply · 1 fyi · 1 promo"
    )


def test_no_judging_and_nothing_waiting_is_no_line(inbox: Inbox) -> None:
    start, end = _yesterday()
    later = (start + timedelta(days=5), end + timedelta(days=5))
    assert judged_line(inbox.judgments, CONFIG, WORDS, start=later[0], end=later[1]) == ""


# -- learned yesterday ----------------------------------------------------------------


def test_each_correction_names_where_the_owner_made_it(inbox: Inbox) -> None:
    at = DAY + timedelta(hours=20)
    inbox.correct("m-case", "needs_reply", source="gmail", at=at)
    inbox.correct("m-plan", "bill", source="card", at=at + timedelta(minutes=1))
    inbox.correct("m-dental", "fyi", source="chat", at=at + timedelta(minutes=2))
    inbox.correct("m-bill", "promo", source="web", at=at + timedelta(minutes=3))
    start, end = _yesterday()
    assert learned_phrases(inbox.judgments, inbox.emails, CONFIG, WORDS, start, end) == [
        "Harbor Bank → Needs reply (you relabelled it in Gmail)",
        "Nimbus Utilities → Bill (your answer on the card)",
        "Bright Smile Dental → FYI (you said so in chat)",
        "Northwind Card → Promo (you changed it on the web)",
    ]
    # outside the window: nothing
    assert (
        learned_phrases(inbox.judgments, inbox.emails, CONFIG, WORDS, end, end + end.resolution)
        == []
    )


def test_the_registered_source_reaches_the_footer(inbox: Inbox, tmp_path: Path) -> None:
    inbox.correct("m-case", "needs_reply", source="chat", at=DAY + timedelta(hours=20))
    register_learned_source("email_judgment_corrections", learned_source(inbox.data_dir))
    try:
        assert "Harbor Bank → Needs reply (you said so in chat)" in learned_between(*_yesterday())
    finally:
        unregister_learned_source("email_judgment_corrections")
    assert learned_source(tmp_path / "none")(*_yesterday()) == []
