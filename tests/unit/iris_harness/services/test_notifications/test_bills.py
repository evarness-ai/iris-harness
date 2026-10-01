"""How a bill's reminder reads and what it offers (loop-proof PR 4, graph §5).

The owner-signed prototype's words, per step, from the row's ``meta`` strings; the
buttons each step carries on Telegram (every callback within Telegram's 64 bytes),
on the push notification and in the API.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from iris_harness.services.notifications.bills import (
    bill_actions,
    bill_key,
    bill_keyboard,
    bill_push_actions,
    bill_step,
    bill_wording,
    is_ask,
    paid_by,
    surface_label,
)
from iris_harness.services.notifications.models import Reminder

META = {
    "entity": "Discover",
    "amount": "$35.00",
    "statement": "$1,284.50",
    "due": "2025-10-13",
    "at": "09:00",
}
RID = "0f8fad5b-d9cb-469f-a165-70867728950e"


def _row(step: str | None, **extra: object) -> Reminder:
    meta = {**META, "step": step} if step else {}
    fields: dict[str, object] = {
        "id": RID,
        "target_kind": "bill",
        "target_id": "due-1",
        "remind_at": datetime(2025, 10, 10, 14, tzinfo=UTC),
        "meta": meta,
        "note": "note",
    }
    fields.update(extra)
    return Reminder.model_validate(fields)


@pytest.mark.parametrize(
    ("step", "headline", "line"),
    [
        ("t3d", "💳 Discover — $35.00 min due Mon Oct 13", "in 3 days · statement $1,284.50"),
        ("dayof", "💳 Discover — $35.00 min due today", "Mon Oct 13 · statement $1,284.50"),
        (
            "ask1",
            "❓ Did you pay Discover $35.00?",
            "It was due Mon Oct 13 · no payment email seen",
        ),
        (
            "ask3",
            "❓ Did you pay Discover $35.00?",
            "It was due Mon Oct 13 · no payment email seen · "
            "last ask — then it stays in the digest",
        ),
        (
            "paid",
            "✅ Discover marked paid",
            "Seen: a payment email. No more reminders for this bill.",
        ),
    ],
)
def test_the_prototype_words(step: str, headline: str, line: str) -> None:
    assert bill_wording(_row(step)) == (headline, line)


def test_changed_and_no_statement_and_no_amount() -> None:
    changed = _row("changed").model_copy(
        update={"meta": {**META, "step": "changed", "was": "$60.00"}}
    )
    assert bill_wording(changed) == (
        "💳 Discover — amount changed: $35.00 min due Mon Oct 13",
        "was $60.00 · statement $1,284.50",
    )
    plain = _row("t3d").model_copy(update={"meta": {**META, "step": "t3d", "statement": ""}})
    assert bill_wording(plain) == ("💳 Discover — $35.00 due Mon Oct 13", "in 3 days")
    bare = _row("dayof").model_copy(update={"meta": {**META, "step": "dayof", "amount": ""}})
    assert bill_wording(bare)[0] == "💳 Discover — payment due today"


def test_a_row_without_bill_strings_reads_as_its_note() -> None:
    assert bill_wording(_row(None)) == ("note", "")


def test_the_step_comes_from_meta_or_the_dedupe_key() -> None:
    assert bill_step(_row("ask2")) == "ask2"
    assert bill_step(_row(None, dedupe_key=bill_key("due-1", "dayof"))) == "dayof"
    assert bill_step(_row(None)) is None
    assert bill_step(_row("t3d", target_kind="task")) is None
    assert bill_key("d", "changed", "snap") == "bill:d:changed:snap"
    assert is_ask(_row("ask1")) and not is_ask(_row("dayof"))


@pytest.mark.parametrize(
    ("step", "buttons"),
    [
        ("t3d", [("✅ Paid", f"/paid {RID}"), ("⏰ Tomorrow", f"/snooze {RID} tomorrow_9am")]),
        ("dayof", [("✅ Paid", f"/paid {RID}"), ("⏰ 1 hour", f"/snooze {RID} 1h")]),
        ("ask2", [("✅ Paid", f"/paid {RID}"), ("Not yet", f"/notyet {RID}")]),
        ("changed", [("✅ Paid", f"/paid {RID}"), ("⏰ Tomorrow", f"/snooze {RID} tomorrow_9am")]),
    ],
)
def test_telegram_buttons_per_step(step: str, buttons: list[tuple[str, str]]) -> None:
    [row] = bill_keyboard(_row(step))
    assert [(b["text"], b["callback_data"]) for b in row] == buttons
    assert all(len(b["callback_data"].encode()) <= 64 for b in row)


def test_the_paid_confirmation_has_no_buttons_anywhere() -> None:
    row = _row("paid")
    assert bill_keyboard(row) == [] and bill_push_actions(row) == [] and bill_actions(row) == []


def test_push_and_api_actions() -> None:
    assert [a["action"] for a in bill_push_actions(_row("ask1"))] == ["paid", "not_yet"]
    assert [a["title"] for a in bill_push_actions(_row("ask1"))] == ["Paid", "Not yet"]
    assert [a["action"] for a in bill_push_actions(_row("dayof"))] == ["paid", "snooze_1h"]
    assert [a["action"] for a in bill_push_actions(_row("t3d"))] == ["paid", "snooze_tomorrow"]
    assert bill_actions(_row("ask3")) == ["paid", "not_yet", "1h", "tomorrow_9am"]
    assert bill_actions(_row("t3d")) == ["paid", "1h", "tomorrow_9am"]


def test_paid_by_names_who_closed_it() -> None:
    assert paid_by(_row("t3d", status="done", closed_reason="done: telegram")) == "Telegram"
    assert (
        paid_by(_row("t3d", status="expired", closed_reason="closed: paid (payment_email)"))
        == "a payment email"
    )
    assert paid_by(_row("t3d", status="expired", closed_reason="not yet: push")) is None
    assert paid_by(_row("t3d")) is None
    assert surface_label("some_thing") == "some thing"


# ── the digest's Bills line, in the pushes' words (PR 4 demo, 2026-09-25) ─────


@pytest.mark.parametrize(
    ("today", "line"),
    [
        ("2025-10-10", "$35.00 min due Mon Oct 13 (in 3 days) · statement $1,284.50"),
        ("2025-10-12", "$35.00 min due Mon Oct 13 (tomorrow) · statement $1,284.50"),
        ("2025-10-13", "$35.00 min due Mon Oct 13 (today) · statement $1,284.50"),
        ("2025-10-14", "$35.00 min — due Mon Oct 13, not marked paid"),
        ("2025-10-20", "$35.00 min — due Mon Oct 13, not marked paid"),
    ],
)
def test_digest_wording(today: str, line: str) -> None:
    from datetime import date

    from iris_harness.services.notifications.bills import digest_wording

    assert digest_wording(META, today=date.fromisoformat(today)) == line


def test_digest_wording_without_a_statement_or_an_amount() -> None:
    from datetime import date

    from iris_harness.services.notifications.bills import digest_wording

    plain = {**META, "statement": ""}
    assert digest_wording(plain, today=date(2025, 10, 10)) == "$35.00 due Mon Oct 13 (in 3 days)"
    assert (
        digest_wording(plain, today=date(2025, 10, 14))
        == "$35.00 — due Mon Oct 13, not marked paid"
    )
    unknown = {**META, "amount": "", "statement": ""}
    assert (
        digest_wording(unknown, today=date(2025, 10, 14))
        == "payment — due Mon Oct 13, not marked paid"
    )
