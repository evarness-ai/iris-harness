""" "Delete my promo emails for this week", through the email tool set the loop gets.

The 2026-09-22 bug: "promo" matched no category (every path starts ``email/``), so
the model asked the owner to pick emails it could not show them. Here the plain
word reaches the promotions, including one triage refiled, and the window comes from
the owner's own "this week" when the model leaves ``since_days`` out.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from iris_personal.email.agent_tools import build_email_tools
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.providers import clear_mail_providers, register_mail_provider
from iris_personal.email.store import EmailStore

ACCT = "gmail:owner@gmail.com"
PROMO = ("CATEGORY_PROMOTIONS",)


def _msg(mid: str, days_ago: int, labels: tuple[str, ...] = ()) -> EmailMessage:
    return EmailMessage(
        id=mid,
        provider="gmail",  # type: ignore[arg-type]
        account_id=ACCT,
        from_address="Shop <hi@shop.com>",
        subject=f"Sale {mid}",
        received_at=datetime.now(UTC) - timedelta(days=days_ago, hours=1),
        labels=labels,
    )


class _Gmail:
    name = "gmail"

    def __init__(self) -> None:
        self.trashed: list[str] = []

    def category_labels(self) -> dict[str, str]:
        return {"CATEGORY_PROMOTIONS": "email/promotions"}

    def trash_messages(self, account_id: str, ids: Any) -> list[str]:
        self.trashed.extend(ids)
        return list(ids)


@pytest.fixture()
def gmail(tmp_path: Path) -> Any:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert_many(
        [
            _msg("promo-new", 1, PROMO),
            _msg("promo-refiled", 2, PROMO),
            _msg("promo-old", 20, PROMO),
            _msg("receipt", 1),
        ]
    )
    store.mark_classified("promo-new", category="email/promotions", confidence=0.5)
    store.mark_classified("promo-refiled", category="email/shopping/apparel/gap", confidence=0.9)
    store.mark_classified("promo-old", category="email/promotions", confidence=0.5)
    store.mark_classified("receipt", category="email/finance/receipts", confidence=0.9)
    fake = _Gmail()
    clear_mail_providers()
    register_mail_provider(fake)  # type: ignore[arg-type]
    yield fake
    clear_mail_providers()


def _tools(data_dir: Path, asked: str) -> dict[str, Any]:
    return {
        t.name: t
        for t in build_email_tools(data_dir=data_dir, llm_call=None, current_query=lambda: asked)
    }


def test_the_plain_word_lists_every_promotion(gmail: Any, tmp_path: Path) -> None:
    out = _tools(tmp_path, "show my promo emails")["list_by_category"].call({"category": "promo"})

    assert out.startswith("3 email(s) in 'promo' (email/promotions)")
    assert "promo-refiled" in out and "receipt" not in out


def test_this_week_trashes_this_weeks_promotions_only(gmail: Any, tmp_path: Path) -> None:
    tools = _tools(tmp_path, "can you delete my promo emails for this week")

    out = tools["trash_category"].call({"category": "promo"})

    assert sorted(gmail.trashed) == ["promo-new", "promo-refiled"]
    assert "from the last 7 days" in out


def test_search_inbox_takes_the_plain_word_too(gmail: Any, tmp_path: Path) -> None:
    out = _tools(tmp_path, "any sale in my promo emails")["search_inbox"].call(
        {"query": "sale", "category": "promo"}
    )
    assert "promo-refiled" in out and "receipt" not in out
