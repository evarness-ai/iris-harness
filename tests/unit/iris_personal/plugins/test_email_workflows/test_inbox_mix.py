"""The digest's Inbox summary: one line per mailbox, last 24 h by triage category.

Pinned: every mailbox gets its own line (the owner has two), the groups are the
top-level segment of the stored triage paths (no vocabulary in code), largest first,
unclassified mail closes the line as ``unsorted N`` only when there is some, and mail
older than 24 h is not counted.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.inbox_mix import (
    account_label,
    inbox_mix,
    mix_line,
    top_group,
)

NOW = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> EmailStore:
    s = EmailStore(db_path=tmp_path / "email.db")
    s.ensure_schema()
    return s


def _add(
    store: EmailStore, mid: str, account: str, category: str | None, *, hours_ago: float = 1
) -> None:
    store.upsert(
        EmailMessage(
            id=mid,
            provider="gmail",
            account_id=account,
            from_address="x@example.com",
            subject=mid,
            received_at=NOW - timedelta(hours=hours_ago),
        )
    )
    if category:
        store.mark_classified(mid, category=category, confidence=0.9)


def test_top_group_is_the_segment_under_the_email_root() -> None:
    assert top_group("email/finance/cards") == "finance"
    assert top_group("email/updates") == "updates"
    assert top_group("receipts/amazon") == "receipts"
    assert top_group("email") is None
    assert top_group(None) is None
    assert top_group("") is None


def test_account_label_drops_the_provider_prefix() -> None:
    assert account_label("gmail:owner@example.com") == "owner@example.com"
    assert account_label("owner@example.com") == "owner@example.com"


def test_mix_line_orders_largest_first_and_closes_with_unsorted() -> None:
    counts: dict[str | None, int] = {
        "email/promo": 9,
        "email/updates/github": 7,
        "email/updates/bank": 5,
        "email/finance/cards": 6,
        None: 3,
    }
    assert mix_line(counts) == (
        "30 in the last 24 h — 12 updates · 9 promo · 6 finance · unsorted 3"
    )


def test_mix_line_hides_unsorted_when_zero_and_breaks_ties_by_name() -> None:
    assert mix_line({"email/social": 2, "email/news": 2}) == (
        "4 in the last 24 h — 2 news · 2 social"
    )
    assert mix_line({}) == "nothing in the last 24 h"


def test_inbox_mix_one_row_per_mailbox_last_24h_only(store: EmailStore) -> None:
    a, b = "gmail:owner@example.com", "gmail:other@example.com"
    _add(store, "a1", a, "email/updates/github")
    _add(store, "a2", a, "email/updates/bank")
    _add(store, "a3", a, "email/finance/cards")
    _add(store, "a4", a, None)
    _add(store, "a-old", a, "email/finance/cards", hours_ago=30)
    _add(store, "b1", b, "email/promo", hours_ago=2)
    _add(store, "b-old", b, "email/promo", hours_ago=48)

    rows = inbox_mix(store, now=NOW)

    assert [(r["account"], r["summary"]) for r in rows] == [
        ("owner@example.com", "4 in the last 24 h — 2 updates · 1 finance · unsorted 1"),
        ("other@example.com", "1 in the last 24 h — 1 promo"),
    ]
    assert rows[0]["account_id"] == a
    assert rows[0]["total"] == "4"


def test_inbox_mix_keeps_a_quiet_mailbox_and_filters_by_account(store: EmailStore) -> None:
    a, b = "gmail:owner@example.com", "gmail:other@example.com"
    _add(store, "a1", a, "email/updates")
    _add(store, "b-old", b, "email/promo", hours_ago=48)

    rows = inbox_mix(store, now=NOW)
    assert {r["account"]: r["summary"] for r in rows}["other@example.com"] == (
        "nothing in the last 24 h"
    )
    only_b = inbox_mix(store, now=NOW, account_id=b)
    assert [r["account_id"] for r in only_b] == [b]
    assert inbox_mix(store, now=NOW, account_id="gmail:ghost@example.com") == []
