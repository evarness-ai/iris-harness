"""An email still waiting for the judge is invisible to all of IRIS (PR 5, stream M).

Real ``JudgmentStore`` rows over a real ``email.db``: one judged email, one waiting,
one never queued (promo). The core store's hold constants must name this plugin's
table and waiting status, and the email_workflows readers that run their own SQL must
see the judged and promo mail only.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import HELD_STATUS, HELD_TABLE, EmailStore, visible_condition
from iris_personal.plugins.email_workflows.discovery import load_corpus
from iris_personal.plugins.email_workflows.judgments import WAITING, JudgmentStore

ACCOUNT = "gmail:owner@example.com"
JUDGED, WAITING_ID, PROMO = "m-judged", "m-waiting", "m-promo"


def _msg(mid: str, minutes_ago: int) -> EmailMessage:
    return EmailMessage(
        id=mid,
        provider="gmail",
        account_id=ACCOUNT,
        from_address=f"{mid}@shop.example.com",
        subject=f"Order update {mid}",
        snippet="your order",
        received_at=datetime.now(UTC) - timedelta(minutes=minutes_ago),
    )


@pytest.fixture
def db(tmp_path: Path) -> Path:
    path = tmp_path / "email.db"
    store = EmailStore(db_path=path)
    store.ensure_schema()
    store.upsert_many([_msg(JUDGED, 3), _msg(WAITING_ID, 2), _msg(PROMO, 1)])
    judgments = JudgmentStore(db_path=path)
    judgments.ensure_schema()
    judgments.mark_waiting(ACCOUNT, [JUDGED, WAITING_ID])
    judgments.record(JUDGED, ACCOUNT, bucket="fyi", confidence=0.9)
    return path


def test_core_hold_constants_name_the_judgments_table() -> None:
    assert HELD_TABLE == "email_judgments"
    assert HELD_STATUS == WAITING


def test_store_reads_hide_waiting_and_judge_reads_it(db: Path) -> None:
    store = EmailStore(db_path=db)
    assert {m.id for m in store.list_recent(ACCOUNT)} == {JUDGED, PROMO}
    assert store.get(WAITING_ID) is None
    # The judge's own read of the email it is about to judge.
    waiting = store.get(WAITING_ID, include_held=True)
    assert waiting is not None and waiting.id == WAITING_ID


def test_judging_releases_the_email(db: Path) -> None:
    JudgmentStore(db_path=db).record(WAITING_ID, ACCOUNT, bucket="act", confidence=0.8)
    store = EmailStore(db_path=db)
    assert store.get(WAITING_ID) is not None
    assert {m.id for m in store.list_recent(ACCOUNT)} == {JUDGED, WAITING_ID, PROMO}


def test_discovery_corpus_skips_waiting(db: Path) -> None:
    assert {r.id for r in load_corpus(db, ACCOUNT)} == {JUDGED, PROMO}


def test_hold_filter_probes_the_real_status_index(db: Path) -> None:
    with sqlite3.connect(db) as conn:
        plan = " | ".join(
            str(r[3])
            for r in conn.execute(
                "EXPLAIN QUERY PLAN SELECT * FROM emails WHERE account_id = ? AND "  # noqa: S608
                + visible_condition(conn)
                + " ORDER BY received_at DESC LIMIT 50",
                (ACCOUNT,),
            )
        )
    assert "idx_email_judgments_status" in plan, plan
