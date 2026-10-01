"""The "What is this email?" card: Unsure emails only (loop-proof PR 5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_harness.services.tasks import Task, TaskStore
from iris_harness.services.tasks.pending_actions import invoke_and_reconcile, reconcile
from iris_personal.plugins.email_workflows.judge_cards import (
    TARGET_PREFIX,
    EmailJudgeActionProvider,
    days_expired,
    refresh_judge_cards,
)
from iris_personal.plugins.email_workflows.judge_config import EMAIL_JUDGMENT_CORRECTED

from .judge_fixtures import DAY, Inbox, seed_day

UTC_TZ = UTC


@pytest.fixture
def inbox(tmp_path: Path) -> Inbox:
    box = Inbox(tmp_path)
    seed_day(box, DAY)
    return box


@pytest.fixture
def tasks(tmp_path: Path) -> TaskStore:
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    return store


def _provider(inbox: Inbox, now: datetime) -> EmailJudgeActionProvider:
    return EmailJudgeActionProvider(inbox.data_dir, now=lambda: now, tz=UTC_TZ, emit=inbox.emit)


def _open_cards(tasks: TaskStore) -> list[Task]:
    return [t for t in tasks.list(source_kind="email", has_action=True) if t.status == "open"]


def test_only_the_unsure_email_gets_a_card_with_its_facts_and_five_answers(
    inbox: Inbox, tasks: TaskStore
) -> None:
    summary = reconcile(_provider(inbox, DAY + timedelta(hours=20)), tasks)
    assert summary.raised == 1
    (card,) = _open_cards(tasks)
    assert card.title == "What is this email?"
    assert card.description == 'Nimbus Utilities · "Important information about your account"'
    action = card.action
    assert action is not None and action.safe and action.kind == "execute"
    assert action.target_id == f"{TARGET_PREFIX}m-plan"
    assert [(c.value, c.label) for c in action.choices] == [
        ("bill", "Bill"),
        ("event", "Event"),
        ("needs_reply", "Needs reply"),
        ("fyi", "FYI"),
        ("promo", "Promo — hide it"),
    ]
    assert action.card is not None
    facts = {f.label: f.value for f in action.card.facts}
    assert facts == {
        "From": "Nimbus Utilities",
        "Subject": "Important information about your account",
        "Says": "Sample snippet for Important information about your account.",
        "IRIS's guess": "Unsure (confidence 0.52)",
    }
    assert action.card.tag == "unsure email"
    assert "moves its Gmail label" in action.card.note


def test_an_answer_is_a_card_correction_and_the_card_closes(
    inbox: Inbox, tasks: TaskStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRIS_EMAIL_JUDGE_LABELS", raising=False)
    provider = _provider(inbox, DAY + timedelta(hours=20))
    reconcile(provider, tasks)
    (card,) = _open_cards(tasks)

    note = invoke_and_reconcile(provider, card, tasks, choice="needs_reply")

    assert note == "✓ Needs reply — label moved to IRIS/Needs-Reply."
    row = inbox.judgments.get("m-plan")
    assert row is not None
    assert (row.owner_bucket, row.owner_source) == ("needs_reply", "card")
    ((topic, payload),) = inbox.emitted
    assert topic == EMAIL_JUDGMENT_CORRECTED
    assert (payload.bucket, payload.previous, payload.source) == ("needs_reply", "unsure", "card")
    assert _open_cards(tasks) == []
    assert tasks.get(card.id).status == "done"  # type: ignore[union-attr]


def test_promo_answer_and_labels_off_say_so(
    inbox: Inbox, tasks: TaskStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = _provider(inbox, DAY + timedelta(hours=20))
    reconcile(provider, tasks)
    (card,) = _open_cards(tasks)
    note = invoke_and_reconcile(provider, card, tasks, choice="promo")
    assert note.startswith("✓ Promo")
    assert inbox.judgments.get("m-plan").owner_bucket == "promo"  # type: ignore[union-attr]

    inbox.judged("m-two", "Quiet Co <hi@quiet.example>", "Hello", "unsure", at=DAY)
    reconcile(provider, tasks)
    (second,) = _open_cards(tasks)
    monkeypatch.setenv("IRIS_EMAIL_JUDGE_LABELS", "0")
    assert invoke_and_reconcile(provider, second, tasks, choice="fyi") == "✓ FYI."


def test_a_wrong_answer_is_refused_and_changes_nothing(inbox: Inbox, tasks: TaskStore) -> None:
    provider = _provider(inbox, DAY + timedelta(hours=20))
    reconcile(provider, tasks)
    (card,) = _open_cards(tasks)
    with pytest.raises(ValueError):
        invoke_and_reconcile(provider, card, tasks, choice="unsure")
    assert inbox.judgments.get("m-plan").owner_bucket is None  # type: ignore[union-attr]
    assert inbox.emitted == []


def test_a_gmail_relabel_elsewhere_closes_the_card_on_refresh(
    inbox: Inbox, tasks: TaskStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_TZ", "UTC")
    reconcile(_provider(inbox, DAY + timedelta(hours=20)), tasks)
    assert len(_open_cards(tasks)) == 1
    inbox.correct("m-plan", "bill", source="gmail", at=DAY + timedelta(hours=21))
    summary = refresh_judge_cards(inbox.data_dir)
    assert summary is not None and summary.resolved == 1
    assert _open_cards(tasks) == []


def test_an_unanswered_card_expires_after_unsure_card_days(inbox: Inbox, tasks: TaskStore) -> None:
    # Judged Sat Sep 26 (UTC); 7 local days → open through Sat Oct 3, gone from Oct 4.
    reconcile(_provider(inbox, datetime(2026, 10, 3, 23, 59, tzinfo=UTC)), tasks)
    assert len(_open_cards(tasks)) == 1
    summary = reconcile(_provider(inbox, datetime(2026, 10, 4, 0, 0, tzinfo=UTC)), tasks)
    assert summary.resolved == 1 and summary.open_total == 0
    # nothing deleted: the row is still Unsure, for the web list
    assert inbox.judgments.get("m-plan").effective_bucket == "unsure"  # type: ignore[union-attr]


def test_days_expired_counts_local_days() -> None:
    start = datetime(2026, 9, 26, 23, 30, tzinfo=UTC)
    assert not days_expired(start, 0, datetime(2026, 9, 26, 23, 59, tzinfo=UTC), UTC)
    assert days_expired(start, 0, datetime(2026, 9, 27, 0, 0, tzinfo=UTC), UTC)


def test_no_email_db_means_no_cards(tmp_path: Path, tasks: TaskStore) -> None:
    provider = EmailJudgeActionProvider(tmp_path / "nothing-here", tz=UTC)
    assert provider.desired_actions() == []
    assert refresh_judge_cards(tmp_path / "nothing-here") is None


def test_the_card_shows_the_models_own_guess_when_it_was_not_sure_enough(
    inbox: Inbox, tasks: TaskStore
) -> None:
    import sqlite3

    with sqlite3.connect(inbox.judgments.db_path) as conn:
        conn.execute(
            "UPDATE email_judgments SET fields = json_set(fields, '$.guess', 'fyi') "
            "WHERE message_id = 'm-plan'"
        )
    reconcile(_provider(inbox, DAY + timedelta(hours=20)), tasks)
    (card,) = _open_cards(tasks)
    assert card.action is not None and card.action.card is not None
    facts = {f.label: f.value for f in card.action.card.facts}
    assert facts["IRIS's guess"] == "FYI (confidence 0.52)"
