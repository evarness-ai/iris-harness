"""The ``email_rebucket`` intercept: correcting the judge in chat (loop-proof PR 5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iris_personal.plugins.email_workflows.judge_chat import RebucketTurn, handle_rebucket_turn
from iris_personal.plugins.email_workflows.judge_config import (
    EMAIL_JUDGMENT_CORRECTED,
    JudgeConfig,
)
from iris_personal.plugins.email_workflows.judge_surfaces import build_rebucket_intercept
from iris_personal.plugins.email_workflows.judge_words import SurfaceWords

from .judge_fixtures import DAY, Inbox, seed_day

NOW = DAY + timedelta(hours=20)


@pytest.fixture(autouse=True)
def _labels_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_EMAIL_JUDGE_LABELS", raising=False)


@pytest.fixture
def inbox(tmp_path: Path) -> Inbox:
    box = Inbox(tmp_path)
    seed_day(box, DAY)
    return box


def say(inbox: Inbox, text: str, now: datetime = NOW) -> RebucketTurn | None:
    return handle_rebucket_turn(
        text,
        inbox.judgments,
        inbox.emails,
        config=JudgeConfig.load(),
        words=SurfaceWords.load(),
        now=now,
        tz=UTC,
        emit=inbox.emit,
    )


def test_isnt_fyi_it_needs_a_reply_moves_it_and_says_what_moved(inbox: Inbox) -> None:
    turn = say(inbox, "the Harbor Bank email isn't FYI, it needs a reply")
    assert turn is not None and turn.kind == "changed" and turn.message_id == "m-case"
    assert turn.reply == (
        '✓ "About your recent inquiry" from Harbor Bank is now **Needs reply** (was FYI). '
        "I moved its Gmail label to IRIS/Needs-Reply, and I'll judge mail like it from "
        "them the same way next time."
    )
    row = inbox.judgments.get("m-case")
    assert row is not None and (row.owner_bucket, row.owner_source) == ("needs_reply", "chat")
    ((topic, payload),) = inbox.emitted
    assert topic == EMAIL_JUDGMENT_CORRECTED and payload.source == "chat"
    assert "<" not in turn.reply  # plain text / markdown, never HTML


@pytest.mark.parametrize(
    ("text", "mid", "bucket"),
    [
        ("that dental email is an event, not a bill", "m-dental", "event"),
        ("the Northwind email is FYI", "m-bill", "fyi"),
        ("the nimbus email is a bill", "m-plan", "bill"),
        ("Nimbus is promo", "m-plan", "promo"),
        ("the dinner email needs a reply.", "m-dinner", "needs_reply"),
    ],
)
def test_the_shapes_the_owner_uses(inbox: Inbox, text: str, mid: str, bucket: str) -> None:
    turn = say(inbox, text)
    assert turn is not None, text
    if bucket == "event":
        assert turn.kind == "already"
        assert turn.reply == '"Appointment confirmed" is already **Event**, so nothing to change.'
        assert inbox.emitted == []
    elif bucket == "needs_reply":
        assert turn.kind == "already"
    else:
        assert turn.kind == "changed" and turn.message_id == mid
        assert inbox.judgments.get(mid).effective_bucket == bucket  # type: ignore[union-attr]


def test_promo_reply_and_labels_off_wording(inbox: Inbox, monkeypatch: pytest.MonkeyPatch) -> None:
    promo = say(inbox, "Nimbus is promo")
    assert promo is not None
    assert promo.reply.startswith('✓ "Important information about your account" from Nimbus')
    assert "**Promo** (was Unsure)" in promo.reply and "won't judge mail from them" in promo.reply
    monkeypatch.setenv("IRIS_EMAIL_JUDGE_LABELS", "0")
    turn = say(inbox, "the Northwind email is FYI")
    assert (
        turn is not None and "Gmail label" not in turn.reply and "**FYI** (was Bill)" in turn.reply
    )


def test_several_matches_are_listed_and_nothing_moves(inbox: Inbox) -> None:
    for i in range(4):
        inbox.judged(
            f"m-hb{i}",
            "Harbor Bank <care@harborbank.example>",
            f"Notice {i}",
            "fyi",
            at=DAY - timedelta(days=1, hours=i),
        )
    turn = say(inbox, "the harbor bank email is a bill")
    assert turn is not None and turn.kind == "ambiguous"
    assert turn.reply.startswith("Which email did you mean? ")
    assert turn.reply.count('"') == 6  # three listed, no more
    assert '"About your recent inquiry" (Harbor Bank, Sat Sep 26)' in turn.reply
    assert inbox.emitted == []


def test_is_a_bill_means_the_one_that_is_not_a_bill_yet(inbox: Inbox) -> None:
    for i in range(3):
        inbox.judged(
            f"m-hb{i}",
            "Harbor Bank <care@harborbank.example>",
            f"Statement {i}",
            "bill",
            at=DAY - timedelta(days=1, hours=i),
        )
    turn = say(inbox, "the harbor bank email is a bill")
    assert turn is not None and turn.kind == "changed" and turn.message_id == "m-case"


def test_a_denied_bucket_narrows_the_candidates(inbox: Inbox) -> None:
    inbox.judged("m-hb", "Harbor Bank <care@harborbank.example>", "Statement", "bill", at=DAY)
    turn = say(inbox, "the Harbor Bank email isn't FYI, it needs a reply")
    assert turn is not None and turn.kind == "changed" and turn.message_id == "m-case"


def test_a_bucket_only_denied_asks_which(inbox: Inbox) -> None:
    turn = say(inbox, "the Harbor Bank email isn't FYI")
    assert turn is not None and turn.kind == "which_bucket"
    assert turn.reply == (
        'What should "About your recent inquiry" from Harbor Bank be: '
        "Bill, Event, Needs reply, FYI or Promo?"
    )
    assert inbox.judgments.get("m-case").owner_bucket is None  # type: ignore[union-attr]


def test_no_match_falls_through_unless_it_clearly_meant_a_judged_email(inbox: Inbox) -> None:
    assert say(inbox, "Zephyr is promo") is None
    turn = say(inbox, "the zephyr email is a bill")
    assert turn is not None and turn.kind == "not_found"
    assert "last 14 days" in turn.reply


def test_every_naming_word_must_match_the_same_email(inbox: Inbox) -> None:
    # "harbor" names one email and "smile" another: neither is "the harbor smile email"
    turn = say(inbox, "the harbor smile email is an event")
    assert turn is not None and turn.kind == "not_found"
    assert inbox.emitted == []


def test_only_the_last_fourteen_days_are_searched(inbox: Inbox) -> None:
    assert say(inbox, "Nimbus is promo", now=DAY + timedelta(days=15)) is None


@pytest.mark.parametrize(
    "text",
    [
        "email the dentist",
        "email the dentist that the appointment is an event",
        "is this a bill?",
        "is the nimbus email a bill?",
        "what is my northwind bill",
        "paid discover",
        "the northwind bill is 83.19",
        "my rent is a bill",
        "the meeting is an event",
        "remind me to reply to petra",
        "that is promo",
        "nimbus is a bill",
        "the nimbus email is a bill?",
        "the nimbus email is fine",
        "",
    ],
)
def test_near_misses_are_not_stolen(inbox: Inbox, text: str) -> None:
    assert say(inbox, text) is None, text
    assert inbox.emitted == []


def test_what_did_you_judge_today(inbox: Inbox) -> None:
    turn = say(inbox, "what did you judge today?")
    assert turn is not None and turn.kind == "counts"
    assert turn.reply == (
        "Judged today: 1 Bill · 1 Event · 1 Needs reply · 1 FYI · 1 Unsure. "
        "Inbox → Judged lists them."
    )
    none = say(inbox, "what did you judge today", now=NOW + timedelta(days=2))
    assert none is not None and none.reply == "I haven't judged any email today."


def test_the_intercept_replies_through_the_governed_reply(inbox: Inbox, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("iris_personal.plugins.email_workflows.judge_view.default_emit", inbox.emit)
    replies: list[dict[str, Any]] = []

    def reply(**kw: Any) -> str:
        replies.append(kw)
        return "result"

    services = SimpleNamespace(data_dir=inbox.data_dir, config_dir=None, deterministic_reply=reply)
    handler = build_rebucket_intercept(services)
    # the real clock: judged "today" relative to a fixed DAY would age out, so re-judge now
    inbox.judged(
        "m-now",
        "Lumen Clinic <desk@lumen.example>",
        "Your visit",
        "fyi",
        at=datetime.now(UTC),
    )
    assert handler("the lumen email is an event", session_id="s1") == "result"
    (kw,) = replies
    assert kw["metadata"] == {"email_rebucket": "changed", "message_id": "m-now"}
    assert kw["session_id"] == "s1"
    assert handler("email the dentist", session_id="s1") is None
    # no email.db at all: falls through
    empty = SimpleNamespace(data_dir=inbox.data_dir / "none", config_dir=None)
    assert build_rebucket_intercept(empty)("Nimbus is promo", session_id="s") is None
