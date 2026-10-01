"""A category filter nothing can satisfy must not read as "you have no such mail".

Cloud trial, 2026-09-20. Asked "do I have any bills from Anthropic in my inbox?", the
model called ``search_inbox`` with ``category='bills'``. The VM's mail is uncategorised —
triage needs the ML extra the server image leaves out — so the filter matched nothing and
the reply was "I couldn't find any emails related to Anthropic", while the receipt
("Your receipt from Anthropic, PBC") sat in the store.

The filter is the model's guess; the mail is the fact. When the filter yields nothing,
the search drops it, searches the whole inbox, and says that is what it did.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from iris_personal.email.agent_tools import build_email_tools
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(
        EmailMessage(
            id="1",
            provider="gmail",
            account_id="gmail:a@b.c",
            thread_id="1",
            from_address='"Anthropic, PBC" <invoice+statements@mail.anthropic.com>',
            subject="Your receipt from Anthropic, PBC #2208-9031-8878",
            snippet="Receipt for your subscription",
            received_at=datetime(2026, 9, 19, tzinfo=timezone.utc),
        )
    )
    return tmp_path


def _tools(data_dir: Path) -> dict[str, Any]:
    return {t.name: t for t in build_email_tools(data_dir=data_dir, llm_call=None)}


def _search(data_dir: Path, **args: Any) -> str:
    return str(_tools(data_dir)["search_inbox"].call({"query": "Anthropic", **args}))


def test_an_unsatisfiable_category_filter_is_dropped(data_dir: Path) -> None:
    answer = _search(data_dir, category="bills")

    assert "Anthropic" in answer
    # It says the filter matched no category and that the whole inbox was searched,
    # not that "categorisation has not run" (it may well have: 2026-09-22).
    assert "No category matches 'bills'" in answer
    assert "searched the whole inbox" in answer


def test_a_category_with_no_matching_mail_is_dropped_and_says_so(data_dir: Path) -> None:
    store = EmailStore(db_path=data_dir / "email.db")
    store.mark_classified("1", category="email/finance/receipts", confidence=0.9)
    store.upsert(
        EmailMessage(
            id="2",
            provider="gmail",
            account_id="gmail:a@b.c",
            thread_id="2",
            from_address="bank@example.com",
            subject="Statement ready",
            received_at=datetime(2026, 9, 19, tzinfo=timezone.utc),
        )
    )
    store.mark_classified("2", category="email/bills/bank", confidence=0.9)

    answer = _search(data_dir, category="bills")

    assert "Your receipt from Anthropic" in answer
    assert "No 'bills' email matched, so I searched the whole inbox." in answer


def test_the_same_search_without_a_category_still_works(data_dir: Path) -> None:
    answer = _search(data_dir)

    assert "Anthropic" in answer
    assert "categorisation has not run" not in answer


def test_a_genuine_miss_still_reports_nothing(data_dir: Path) -> None:
    """Dropping the filter must not invent matches for mail that is not there."""
    answer = str(
        _tools(data_dir)["search_inbox"].call({"query": "Volkswagen", "category": "bills"})
    )

    assert "Anthropic" not in answer
