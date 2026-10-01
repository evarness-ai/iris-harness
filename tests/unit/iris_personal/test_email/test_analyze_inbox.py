"""``analyze_inbox``: a which / how-many question reads EVERY matching email, grouped.

The 2026-09-15 session asked "can you give me the list credit card accounts that I have
so far?" and got a dump of dues subjects from a keyword shortcut. The loop had no way
to do better: ``search_inbox`` returns the top 15 hits, which answers a lookup, not an
analysis. This tool groups all matches by sender so the model can name each card once.
Synthetic senders only.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_personal.email.agent_tools import build_email_tools
from iris_personal.email.analysis import (
    MAX_SUBJECTS,
    group_by_sender,
    render_groups,
    sender_key,
)
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore, SearchHit

ACCT = "gmail:user@example.com"
_NOW = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)


def _hit(
    i: int, sender: str, subject: str, *, days_ago: int = 0, domain: str | None = None
) -> SearchHit:
    return SearchHit(
        id=f"m{i}",
        subject=subject,
        from_address=sender,
        from_domain=domain,
        received_at=_NOW - timedelta(days=days_ago),
        classified_category=None,
        snippet_highlighted="",
        rank=0.0,
    )


def test_sender_key_reads_the_domain_and_display_name() -> None:
    assert sender_key("Alpha Card <alerts@alphabank.example>", None) == (
        "alphabank.example",
        "Alpha Card",
    )
    assert sender_key('"Beta | Gamma" <notify@gamma.example>', "gamma.example") == (
        "gamma.example",
        "Beta | Gamma",
    )


def test_hits_collapse_to_one_group_per_sender_with_counts_and_span() -> None:
    hits = [
        _hit(
            1,
            "Alpha Card <a@alphabank.example>",
            "Alpha Credit Card Statement for 2026-08",
            days_ago=30,
        ),
        _hit(2, "Alpha Card <a@alphabank.example>", "Alpha Credit Card Statement for 2026-09"),
        _hit(3, "Beta Card <b@betacard.example>", "Your Beta card payment is due"),
    ]
    groups = group_by_sender(hits)

    assert [(g.key, g.count) for g in groups] == [("alphabank.example", 2), ("betacard.example", 1)]
    alpha = groups[0]
    assert alpha.first == _NOW - timedelta(days=30) and alpha.last == _NOW
    # Monthly statements differ only in their dates: one example subject, not two.
    assert alpha.subjects == ["Alpha Credit Card Statement for 2026-09"]


def test_subjects_are_capped_and_account_numbers_masked() -> None:
    hits = [
        _hit(i, "Alpha <a@alphabank.example>", f"Notice kind {chr(65 + i)} for card 123456789012")
        for i in range(6)
    ]
    (group,) = group_by_sender(hits)
    assert len(group.subjects) == MAX_SUBJECTS
    assert all("123456789012" not in s and "••9012" in s for s in group.subjects)


def test_render_tells_the_model_to_name_each_item_once() -> None:
    text = render_groups(
        "credit card",
        group_by_sender(
            [_hit(1, "Alpha Card <a@alphabank.example>", "Alpha Credit Card Statement")]
        ),
    )
    assert text.startswith("Analysed ALL 1 email(s) matching 'credit card' from 1 sender(s)")
    assert "1. Alpha Card (alphabank.example): 1 email(s)" in text
    assert "ONCE" in text and "Do not list individual emails" in text


def test_render_with_no_matches_says_so() -> None:
    assert "nothing to analyse" in render_groups("credit card", [])


# ─── the tool, over a real store ─────────────────────────────────────────────


def _msg(i: int, sender: str, subject: str, days_ago: int = 0) -> EmailMessage:
    return EmailMessage(
        id=f"m{i}",
        provider="gmail",  # type: ignore[arg-type]
        account_id=ACCT,
        thread_id=None,
        from_address=sender,
        to=("user@example.com",),
        subject=subject,
        received_at=datetime.now(UTC) - timedelta(days=days_ago),
        snippet="",
    )


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    # More matches than search_inbox's 15-hit cap, so a top-N answer would miss senders.
    for i in range(20):
        store.upsert(
            _msg(i, "Alpha Card <a@alphabank.example>", f"Alpha Credit Card Statement {i}", i)
        )
    store.upsert(
        _msg(100, "Beta Card <b@betacard.example>", "Your Beta credit card payment is due", 40)
    )
    store.upsert(
        _msg(
            101, "Gamma Rewards <g@gamma.example>", "Gamma Rewards credit card statement ready", 50
        )
    )
    store.upsert(_msg(102, "News <n@news.example>", "Weekly AI newsletter", 1))
    return tmp_path


def _tool(data_dir: Path, turn: str):  # type: ignore[no-untyped-def]
    tools = build_email_tools(data_dir=data_dir, llm_call=None, current_query=lambda: turn)
    return next(t for t in tools if t.name == "analyze_inbox")


def test_every_sender_behind_the_topic_is_reported(data_dir: Path) -> None:
    turn = "can you give me the list credit card accounts that I have so far?"
    out = _tool(data_dir, turn).call({"query": "credit card"})

    assert "Analysed ALL 22 email(s)" in out
    for sender in ("alphabank.example", "betacard.example", "gamma.example"):
        assert sender in out
    assert "news.example" not in out


def test_an_invented_topic_is_refused(data_dir: Path) -> None:
    out = _tool(data_dir, "which credit cards do I have?").call({"query": "insurance"})
    assert out.startswith("Error:") and "not in the user's question" in out


def test_a_missing_query_is_an_error_observation(data_dir: Path) -> None:
    assert _tool(data_dir, "which cards").call({}).startswith("Error: analyze_inbox needs")


def test_a_short_user_word_does_not_ground_an_invented_topic() -> None:
    from iris_personal.email.agent_tools import _query_grounded

    # "i" used to prefix-match "insurance", so every invented topic passed.
    assert _query_grounded("insurance", "which credit cards do I have?") is False
    assert _query_grounded("credit card", "which credit cards do I have?") is True
    assert _query_grounded("finances", "show my finance emails") is True
