"""The digest's Focus section (loop-proof PR 2, D17; graph V31).

Pinned: the owner's categories are the filter and time is the order (no judge, no
ranking), the window is 24 h across every inbox, a 👎'd sender is gone and the next
newest fills its place, and every line carries the exact 👎 link scheme the web
renderer turns into a button.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import pytest

from iris_harness.services.learning.suppression import NOT_USEFUL, SurfaceFeedbackStore
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.feedback_keys import EMAIL_FOCUS_SURFACE, EMAIL_SUBSYSTEM, email_focus_dims
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.digest_focus import (
    focus_header,
    focus_line,
    focus_messages,
    not_useful_link,
    render_focus,
    sender_name,
)

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)
CATEGORIES = ("personal", "family", "finance")


@pytest.fixture
def store(tmp_path: Path) -> EmailStore:
    s = EmailStore(db_path=tmp_path / "email.db")
    s.ensure_schema()
    return s


@pytest.fixture
def suppression(tmp_path: Path) -> SurfaceFeedbackStore:
    s = SurfaceFeedbackStore(db_path=tmp_path / "learning.db")
    s.ensure_schema()
    return s


def _mail(
    store: EmailStore,
    mid: str,
    *,
    sender: str,
    subject: str,
    category: str | None,
    hours_ago: float,
    account: str = "gmail:owner@example.com",
) -> None:
    store.upsert(
        EmailMessage(
            id=mid,
            provider="gmail",
            account_id=account,
            from_address=sender,
            subject=subject,
            received_at=NOW - timedelta(hours=hours_ago),
        )
    )
    if category:
        store.mark_classified(mid, category=category, confidence=0.9)


def _seed(store: EmailStore) -> None:
    _mail(
        store,
        "school",
        sender="Mo Harbor School <office@school.example>",
        subject="Harbor school on Saturday",
        category="email/personal/community",
        hours_ago=1,
    )
    _mail(
        store,
        "rm",
        sender="Northwind Bank <rm@bank.example>",
        subject="Missed call from your Relationship Manager",
        category="email/finance",
        hours_ago=2,
    )
    _mail(
        store,
        "bonds",
        sender="GoldenPi <news@goldenpi.example>",
        subject="Your weekly guide to bond investing",
        category="email/finance/investing",
        hours_ago=3,
    )
    _mail(
        store,
        "promo",
        sender="Shop <deals@shop.example>",
        subject="50% off",
        category="email/promotions",
        hours_ago=0.5,
    )
    _mail(
        store,
        "old",
        sender="Family <mom@family.example>",
        subject="Last week",
        category="email/family",
        hours_ago=30,
    )
    _mail(
        store,
        "other-inbox",
        sender="Aunt <aunt@family.example>",
        subject="Dinner Sunday?",
        category="email/family",
        hours_ago=4,
        account="gmail:second@example.com",
    )


def test_focus_is_the_chosen_categories_in_the_last_24h_newest_first(
    store: EmailStore, suppression: SurfaceFeedbackStore
) -> None:
    _seed(store)
    got = focus_messages(store, suppression, CATEGORIES, limit=5, now=NOW)
    # Promotions is not a focus category; the 30-hour-old mail is out of the window;
    # the second inbox is read too.
    assert [m.id for m in got] == ["school", "rm", "bonds", "other-inbox"]


def test_focus_is_capped_at_the_limit(store: EmailStore, suppression: SurfaceFeedbackStore) -> None:
    _seed(store)
    got = focus_messages(store, suppression, CATEGORIES, limit=2, now=NOW)
    assert [m.id for m in got] == ["school", "rm"]


def test_a_not_useful_sender_is_gone_and_the_next_newest_fills_in(
    store: EmailStore, suppression: SurfaceFeedbackStore
) -> None:
    _seed(store)
    suppression.record(
        EMAIL_SUBSYSTEM,
        EMAIL_FOCUS_SURFACE,
        email_focus_dims("news@goldenpi.example"),
        NOT_USEFUL,
        emit_signal=False,
    )
    got = focus_messages(store, suppression, CATEGORIES, limit=3, now=NOW)
    assert [m.id for m in got] == ["school", "rm", "other-inbox"]


def test_suppression_is_per_sender_not_per_domain(
    store: EmailStore, suppression: SurfaceFeedbackStore
) -> None:
    """A bank's marketing sender and its RM share a domain; hiding one keeps the other."""
    _seed(store)
    _mail(
        store,
        "bank-promo",
        sender="Northwind Offers <offers@bank.example>",
        subject="Travel smarter",
        category="email/finance/loans",
        hours_ago=0.25,
    )
    suppression.record(
        EMAIL_SUBSYSTEM,
        EMAIL_FOCUS_SURFACE,
        email_focus_dims("Northwind Offers <offers@bank.example>"),
        NOT_USEFUL,
        emit_signal=False,
    )
    got = [m.id for m in focus_messages(store, suppression, CATEGORIES, limit=5, now=NOW)]
    assert "bank-promo" not in got and "rm" in got


def test_the_not_useful_link_uses_the_exact_scheme() -> None:
    link = not_useful_link("GoldenPi <News@GoldenPi.example>")
    assert link == f"[👎](iris:not-useful/{quote('news@goldenpi.example', safe='')})"
    assert link == "[👎](iris:not-useful/news%40goldenpi.example)"


def test_a_line_names_sender_subject_top_group_and_ends_with_the_link(
    store: EmailStore, suppression: SurfaceFeedbackStore
) -> None:
    _seed(store)
    bonds = next(
        m
        for m in focus_messages(store, suppression, CATEGORIES, limit=5, now=NOW)
        if m.id == "bonds"
    )
    # Only the category's top group ("finance", not "finance/investing"): short lines.
    assert focus_line(bonds) == (
        "GoldenPi · Your weekly guide to bond investing · finance "
        "[👎](iris:not-useful/news%40goldenpi.example)"
    )


def test_the_header_names_the_categories_and_the_limit() -> None:
    assert focus_header(CATEGORIES, 5) == "Focus — personal · family · finance, newest 5"
    # A category given as a full triage path reads the same.
    assert focus_header(("email/finance",), 3) == "Focus — finance, newest 3"
    # A per-inbox cap below the total is what the heading names.
    assert (
        focus_header(CATEGORIES, 10, 5) == "Focus — personal · family · finance, newest 5 per inbox"
    )
    assert focus_header(CATEGORIES, 5, 5) == "Focus — personal · family · finance, newest 5"


def test_each_inbox_gives_its_newest_per_account_then_the_total_caps(
    store: EmailStore, suppression: SurfaceFeedbackStore
) -> None:
    """One busy inbox cannot crowd out the other: 3 per inbox, 5 in all, newest first."""
    for i in range(6):  # the busy inbox: the six newest mails overall
        _mail(
            store,
            f"busy{i}",
            sender=f"Busy {i} <b{i}@busy.example>",
            subject=f"busy {i}",
            category="email/personal",
            hours_ago=1 + i * 0.1,
        )
    for i in range(4):  # the quiet inbox: all older than every busy one
        _mail(
            store,
            f"quiet{i}",
            sender=f"Quiet {i} <q{i}@quiet.example>",
            subject=f"quiet {i}",
            category="email/family",
            hours_ago=5 + i,
            account="gmail:family@example.com",
        )

    got = focus_messages(store, suppression, CATEGORIES, limit=5, per_account=3, now=NOW)

    assert [m.id for m in got] == ["busy0", "busy1", "busy2", "quiet0", "quiet1"]
    # No per-inbox cap: the newest overall, which is the busy inbox only.
    uncapped = focus_messages(store, suppression, CATEGORIES, limit=5, now=NOW)
    assert [m.id for m in uncapped] == [f"busy{i}" for i in range(5)]


def test_a_hidden_sender_does_not_use_up_its_inbox_share(
    store: EmailStore, suppression: SurfaceFeedbackStore
) -> None:
    for i in range(3):
        _mail(
            store,
            f"m{i}",
            sender=f"S{i} <s{i}@x.example>",
            subject=f"s {i}",
            category="email/personal",
            hours_ago=1 + i,
        )
    suppression.record(
        EMAIL_SUBSYSTEM,
        EMAIL_FOCUS_SURFACE,
        email_focus_dims("s0@x.example"),
        NOT_USEFUL,
        emit_signal=False,
    )
    got = focus_messages(store, suppression, CATEGORIES, limit=10, per_account=2, now=NOW)
    assert [m.id for m in got] == ["m1", "m2"]


def test_render_is_a_headed_bullet_list(
    store: EmailStore, suppression: SurfaceFeedbackStore
) -> None:
    _seed(store)
    text = render_focus(store, suppression, CATEGORIES, limit=5, now=NOW)
    lines = text.splitlines()
    assert lines[0] == "## Focus — personal · family · finance, newest 5"
    assert len(lines) == 5 and all(line.startswith("- ") for line in lines[1:])
    assert all("(iris:not-useful/" in line for line in lines[1:])


def test_render_says_so_when_nothing_is_in_focus(
    store: EmailStore, suppression: SurfaceFeedbackStore
) -> None:
    text = render_focus(store, suppression, CATEGORIES, limit=5, now=NOW)
    assert text.startswith("## Focus — personal")
    assert "Nothing in these categories in the last 24 h." in text


def test_render_without_categories_points_to_settings(
    store: EmailStore, suppression: SurfaceFeedbackStore
) -> None:
    _seed(store)
    text = render_focus(store, suppression, (), limit=5, now=NOW)
    assert text.startswith("## Focus\n")
    assert "Settings → Digest" in text


@pytest.mark.parametrize(
    ("raw", "name"),
    [
        ("GoldenPi <news@goldenpi.example>", "GoldenPi"),
        ('"Fabrikam Bank" <x@fabrikam.example>', "Fabrikam Bank"),
        ("<x@fabrikam.example>", "x@fabrikam.example"),
        ("plain@example.com", "plain@example.com"),
    ],
)
def test_sender_name(raw: str, name: str) -> None:
    assert sender_name(raw) == name
