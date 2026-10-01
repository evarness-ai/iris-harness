"""The judgments store and the judge's YAML (loop-proof PR 5 contract)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.judge_config import JudgeConfig
from iris_personal.plugins.email_workflows.judgments import (
    JUDGED,
    PROMO,
    WAITING,
    JudgmentStore,
)


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "email.db"


@pytest.fixture
def store(db: Path) -> JudgmentStore:
    EmailStore(db_path=db).ensure_schema()
    s = JudgmentStore(db_path=db)
    s.ensure_schema()
    return s


def _email(db: Path, mid: str, sender: str, subject: str) -> None:
    EmailStore(db_path=db).upsert(
        EmailMessage(
            id=mid,
            provider="gmail",
            account_id="gmail:owner@example.com",
            thread_id=f"t-{mid}",
            from_address=sender,
            subject=subject,
            snippet="",
            received_at=datetime(2026, 9, 26, 12, tzinfo=UTC),
        )
    )


def test_waiting_rows_drain_oldest_first_and_are_not_duplicated(store: JudgmentStore) -> None:
    assert store.mark_waiting("gmail:a", ["m1", "m2"]) == 2
    assert store.mark_waiting("gmail:a", ["m2", "m3"]) == 1
    assert [j.message_id for j in store.waiting()] == ["m1", "m2", "m3"]
    assert store.count_waiting() == 3
    assert store.missing(["m1", "m9"]) == ["m9"]


def test_record_then_correct_keeps_both_and_effective_is_the_owners(store: JudgmentStore) -> None:
    store.mark_waiting("gmail:a", ["m1"])
    j = store.record("m1", "gmail:a", bucket="fyi", confidence=0.81, fields={"x": 1}, model="q")
    assert (j.status, j.bucket, j.effective_bucket, j.fields) == (JUDGED, "fyi", "fyi", {"x": 1})
    c = store.correct("m1", "needs_reply", source="gmail")
    assert c is not None and c.changed and c.previous == "fyi"
    assert (c.judgment.bucket, c.judgment.effective_bucket) == ("fyi", "needs_reply")
    # the same bucket again changes nothing
    again = store.correct("m1", "needs_reply", source="chat")
    assert again is not None and not again.changed
    # a later re-judge keeps the owner's word
    store.record("m1", "gmail:a", bucket="fyi", confidence=0.9)
    assert store.get("m1").effective_bucket == "needs_reply"  # type: ignore[union-attr]
    assert store.correct("nope", "fyi", source="card") is None
    with pytest.raises(ValueError):
        store.correct("m1", "fyi", source="telepathy")


def test_labels_due_tracks_the_effective_bucket(store: JudgmentStore) -> None:
    store.record("m1", "gmail:a", bucket="bill", confidence=0.9)
    store.record("m2", "gmail:b", bucket="fyi", confidence=0.9)
    assert {j.message_id for j in store.labels_due()} == {"m1", "m2"}
    assert [j.message_id for j in store.labels_due("gmail:a")] == ["m1"]
    store.mark_labelled(["m1", "m2"], None)
    store.mark_labelled(["m1"], "bill")
    store.mark_labelled(["m2"], "fyi")
    assert store.labels_due() == []
    store.correct("m1", "event", source="card")
    assert [j.message_id for j in store.labels_due()] == ["m1"]
    # promo: the label comes off once, then nothing is due
    store.correct("m2", PROMO, source="chat")
    assert {j.message_id for j in store.labels_due()} == {"m1", "m2"}
    store.mark_labelled(["m2"], None)
    assert [j.message_id for j in store.labels_due()] == ["m1"]


def test_between_and_recent(store: JudgmentStore) -> None:
    store.record("m1", "gmail:a", bucket="bill", confidence=0.9)
    store.record("m2", "gmail:a", bucket="needs_reply", confidence=0.9)
    store.correct("m1", "fyi", source="web")
    now = datetime.now(UTC)
    day = (now - timedelta(hours=1), now + timedelta(hours=1))
    assert len(store.judged_between(*day)) == 2
    assert [j.message_id for j in store.corrected_between(*day)] == ["m1"]
    assert [j.message_id for j in store.recent(bucket="fyi")] == ["m1"]
    assert {j.message_id for j in store.recent()} == {"m1", "m2"}


def test_sender_hints_and_promo_senders_read_the_owners_corrections(
    db: Path, store: JudgmentStore
) -> None:
    _email(db, "m1", "Capital One <help@capitalone.example>", "About your recent inquiry")
    _email(db, "m2", "deals@bigbox.example", "Flash sale")
    store.record("m1", "gmail:a", bucket="fyi", confidence=0.8)
    store.record("m2", "gmail:a", bucket="fyi", confidence=0.8)
    stored_from = EmailStore(db_path=db).get("m1").from_address  # type: ignore[union-attr]
    assert store.owner_buckets_for_sender(stored_from) == []
    store.correct("m1", "needs_reply", source="gmail")
    store.correct("m2", PROMO, source="card")
    assert store.owner_buckets_for_sender(stored_from.upper()) == [
        ("About your recent inquiry", "needs_reply")
    ]
    assert store.is_promo_sender("deals@bigbox.example")
    assert not store.is_promo_sender(stored_from)


def test_the_shipped_yaml_loads_and_an_override_replaces_a_key(tmp_path: Path) -> None:
    cfg = JudgeConfig.load()
    assert cfg.keys == ("bill", "event", "needs_reply", "fyi", "unsure")
    assert cfg.labels["needs_reply"] == "IRIS/Needs-Reply"
    assert "CATEGORY_PROMOTIONS" in cfg.skip_labels and "CATEGORY_UPDATES" not in cfg.skip_labels
    assert cfg.unsure_below == 0.7 and cfg.body_chars == 4000
    assert cfg.name("promo") == "Promo" and cfg.name("needs_reply") == "Needs reply"
    for key in ("{today}", "{bucket_definitions}", "{hints}", "{triage_hint}"):
        assert key in cfg.prompt
    (tmp_path / "email").mkdir()
    (tmp_path / "email" / "judge.yaml").write_text("unsure_below: 0.5\n", encoding="utf-8")
    assert JudgeConfig.load(tmp_path).unsure_below == 0.5


def test_waiting_is_the_default_status(store: JudgmentStore) -> None:
    store.mark_waiting("gmail:a", ["m1"])
    assert store.get("m1").status == WAITING  # type: ignore[union-attr]


def test_apply_correction_checks_the_bucket_and_emits_only_on_change(
    store: JudgmentStore,
) -> None:
    from iris_personal.plugins.email_workflows.judge_config import (
        EMAIL_JUDGMENT_CORRECTED,
        JudgmentCorrectedPayload,
    )
    from iris_personal.plugins.email_workflows.judge_corrections import (
        UnknownBucket,
        apply_correction,
    )

    cfg = JudgeConfig.load()
    store.record("m1", "gmail:a", bucket="fyi", confidence=0.8)
    seen: list[tuple[str, object]] = []
    emit = lambda topic, payload: seen.append((topic, payload))  # noqa: E731
    c = apply_correction(store, cfg, "m1", "needs_reply", source="chat", emit=emit)
    assert c is not None and c.changed
    assert seen == [
        (
            EMAIL_JUDGMENT_CORRECTED,
            JudgmentCorrectedPayload("m1", "gmail:a", "needs_reply", "fyi", "chat"),
        )
    ]
    apply_correction(store, cfg, "m1", "needs_reply", source="card", emit=emit)
    assert len(seen) == 1
    apply_correction(store, cfg, "m1", PROMO, source="card", emit=emit)
    assert len(seen) == 2
    with pytest.raises(UnknownBucket):
        apply_correction(store, cfg, "m1", "spam", source="chat", emit=emit)
    assert apply_correction(store, cfg, "nope", "fyi", source="chat", emit=emit) is None
