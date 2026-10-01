"""Tests for the email-followup module (Phase 2 Track 2B)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_harness.foundation.eventbus import EventBus
from iris_harness.services.tasks.store import TaskStore
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.events import EMAIL_NEW_ARRIVED, EmailNewArrivedPayload
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.followup import (
    FollowupDecision,
    FollowupDetector,
    _handle_email_new_arrived,
    detect_and_persist,
    followup_key,
    subscribe_email_followup,
)

ACCOUNT = "gmail:user@gmail.com"
THREAD = "thread-001"


def _make_email(
    *,
    id: str = "msg-1",
    thread_id: str | None = THREAD,
    subject: str = "Quick question for you",
    snippet: str = "Hi, when are you free this week to chat?",
    from_address: str = "colleague@example.com",
    received_at: datetime | None = None,
    labels: tuple[str, ...] = (),
    classified_category: str | None = None,
) -> EmailMessage:
    return EmailMessage(
        id=id,
        provider="gmail",
        account_id=ACCOUNT,
        thread_id=thread_id,
        from_address=from_address,
        subject=subject,
        snippet=snippet,
        received_at=received_at or datetime(2026, 6, 1, tzinfo=UTC),
        labels=labels,
        classified_category=classified_category,
    )


@dataclass
class _StubClient:
    """Mimics LlamaServerClient.complete_json without making HTTP calls."""

    response: str
    last_user: str = ""
    last_system: str = ""

    def complete_json(self, system: str, user: str) -> str:
        self.last_system = system
        self.last_user = user
        return self.response


# ─── FollowupDetector ────────────────────────────────────────────────────────


def test_detector_parses_positive_verdict() -> None:
    client = _StubClient(
        response=json.dumps(
            {"needs_reply": True, "rationale": "asks a direct question", "confidence": 0.9}
        )
    )
    detector = FollowupDetector(client=client)
    decision = detector.detect(_make_email())
    assert decision == FollowupDecision(
        needs_reply=True, rationale="asks a direct question", confidence=0.9
    )


def test_detector_parses_negative_verdict() -> None:
    client = _StubClient(
        response=json.dumps(
            {"needs_reply": False, "rationale": "automated receipt", "confidence": 0.95}
        )
    )
    detector = FollowupDetector(client=client)
    decision = detector.detect(_make_email(subject="Your receipt"))
    assert decision.needs_reply is False
    assert "receipt" in decision.rationale


def test_detector_clamps_confidence_to_unit_interval() -> None:
    client = _StubClient(
        response=json.dumps({"needs_reply": True, "rationale": "x", "confidence": 1.7})
    )
    detector = FollowupDetector(client=client)
    decision = detector.detect(_make_email())
    assert decision.confidence == 1.0


def test_detector_soft_fails_on_invalid_json() -> None:
    client = _StubClient(response="not json {")
    detector = FollowupDetector(client=client)
    decision = detector.detect(_make_email())
    assert decision.needs_reply is False
    assert "json-decode-error" in decision.rationale


def test_detector_soft_fails_on_client_exception() -> None:
    class _Boom:
        def complete_json(self, system: str, user: str) -> str:
            raise RuntimeError("llama-server down")

    detector = FollowupDetector(client=_Boom())
    decision = detector.detect(_make_email())
    assert decision.needs_reply is False
    assert "llm-error" in decision.rationale


# ─── detect_and_persist ──────────────────────────────────────────────────────


def _yes_detector(confidence: float = 0.9, rationale: str = "needs reply") -> FollowupDetector:
    return FollowupDetector(
        client=_StubClient(
            response=json.dumps(
                {"needs_reply": True, "rationale": rationale, "confidence": confidence}
            )
        )
    )


def _no_detector() -> FollowupDetector:
    return FollowupDetector(
        client=_StubClient(
            response=json.dumps(
                {"needs_reply": False, "rationale": "automated", "confidence": 0.95}
            )
        )
    )


def test_detect_and_persist_creates_followup_task(tmp_path: Path) -> None:
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    email = _make_email()

    outcome = detect_and_persist(
        email,
        category_path="email/personal/work",
        detector=_yes_detector(rationale="asks a question"),
        task_store=store,
    )
    assert outcome.action == "created"
    assert outcome.task_id is not None

    task = store.get(outcome.task_id)
    assert task is not None
    assert task.source_kind == "email"
    assert task.wait_for is not None
    assert task.wait_for.kind == "reply_from"
    assert task.wait_for.payload["thread_id"] == THREAD
    assert task.wait_for.payload["account_id"] == ACCOUNT
    assert task.wait_for.payload["from"] == "colleague@example.com"
    assert task.dedup_key == followup_key("gmail", THREAD)
    assert task.description == "asks a question"


def test_detect_and_persist_skips_when_no_thread(tmp_path: Path) -> None:
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    email = _make_email(thread_id=None)

    outcome = detect_and_persist(
        email, category_path=None, detector=_yes_detector(), task_store=store
    )
    assert outcome.action == "skipped:no-thread"
    assert store.list(status=None) == []


@pytest.mark.parametrize(
    "label",
    ["CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL", "CATEGORY_UPDATES", "CATEGORY_FORUMS"],
)
def test_detect_and_persist_skips_provider_bulk_mail(tmp_path: Path, label: str) -> None:
    """Gmail's bulk-tab labels short-circuit before the Tier 3 call.

    Regression for issue 0028: a personalized vendor blast
    ("Rahul, Can AI build you a profitable trading bot?") tagged
    CATEGORY_PROMOTIONS must never become a "reply" followup, even
    though the LLM (given a benign topical category) would say yes.
    """
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    email = _make_email(
        subject="Rahul, Can AI build you a profitable trading bot in 2026?",
        from_address="Anita Rao <anita@quant-academy.example>",
        labels=(label, "UNREAD", "INBOX"),
    )

    outcome = detect_and_persist(
        email,
        category_path="email/learning/finance/quant-trading",
        detector=_yes_detector(),  # LLM says reply-needed; guard overrides
        task_store=store,
    )
    assert outcome.action == "skipped:bulk-category"
    assert label in outcome.detail
    assert store.list(status=None) == []


def test_detect_and_persist_skips_non_actionable_category(tmp_path: Path) -> None:
    """A confident classification into a non-actionable root short-circuits.

    The Quant Academy blast was filed under email/learning/... — not a
    person-to-person reply obligation — so it must skip before the LLM.
    """
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    email = _make_email(classified_category="email/learning/finance/quant-trading")

    outcome = detect_and_persist(
        email,
        category_path=None,  # falls back to email.classified_category
        detector=_yes_detector(),
        task_store=store,
    )
    assert outcome.action == "skipped:non-actionable-category"
    assert "learning" in outcome.detail
    assert store.list(status=None) == []


def test_detect_and_persist_allows_actionable_category(tmp_path: Path) -> None:
    """A personal/work/finance root still flows through to the LLM verdict."""
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    email = _make_email(classified_category="email/work/projects/project-iris")

    outcome = detect_and_persist(
        email, category_path=None, detector=_yes_detector(), task_store=store
    )
    assert outcome.action == "created"


def test_detect_and_persist_respects_suppression_feedback(tmp_path: Path) -> None:
    """A prior user 'not useful' on this domain+category suppresses new followups."""
    from iris_harness.services.learning.suppression import NOT_USEFUL, SurfaceFeedbackStore

    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    feedback = SurfaceFeedbackStore(db_path=tmp_path / "learning.db")
    feedback.ensure_schema()

    email = _make_email(
        from_address="newsletter@acme.com",
        classified_category="email/work/vendor/acme",  # actionable root, so allowlist passes
    )
    # User previously marked this sender not useful.
    feedback.record(
        "email",
        "followup",
        {
            "account": ACCOUNT,
            "from_domain": "acme.com",
        },
        NOT_USEFUL,
    )

    outcome = detect_and_persist(
        email,
        category_path=None,
        detector=_yes_detector(),
        task_store=store,
        feedback_store=feedback,
    )
    assert outcome.action == "skipped:suppressed-by-feedback"
    assert store.list(status=None) == []


def test_detect_and_persist_skips_when_negative_verdict(tmp_path: Path) -> None:
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    email = _make_email(subject="Your receipt")

    outcome = detect_and_persist(
        email, category_path=None, detector=_no_detector(), task_store=store
    )
    assert outcome.action == "skipped:no-reply-needed"
    assert store.list(status=None) == []


def test_detect_and_persist_skips_below_confidence_threshold(tmp_path: Path) -> None:
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    email = _make_email()

    outcome = detect_and_persist(
        email,
        category_path=None,
        detector=_yes_detector(confidence=0.4),
        task_store=store,
        confidence_threshold=0.6,
    )
    assert outcome.action == "skipped:low-confidence"
    assert store.list(status=None) == []


def test_detect_and_persist_is_idempotent_per_thread(tmp_path: Path) -> None:
    """Second call for the same thread does not create a duplicate task."""
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    email = _make_email()

    first = detect_and_persist(
        email, category_path=None, detector=_yes_detector(), task_store=store
    )
    second = detect_and_persist(
        email, category_path=None, detector=_yes_detector(), task_store=store
    )
    assert first.action == "created"
    assert second.action == "skipped:already-exists"
    assert second.task_id == first.task_id
    assert len(store.list(status=None)) == 1


# ─── auto-resolution subscriber ──────────────────────────────────────────────


@pytest.fixture
def wired(tmp_path: Path):
    """Return (email_store, task_store, bus, seed_followup_for_thread)."""
    estore = EmailStore(db_path=tmp_path / "email.db")
    estore.ensure_schema()
    tstore = TaskStore(db_path=tmp_path / "tasks.db")
    tstore.ensure_schema()
    bus = EventBus()

    def _seed_followup_for_thread(thread_id: str, sender: str = "x@example.com") -> str:
        outcome = detect_and_persist(
            _make_email(id=f"seed-{thread_id}", thread_id=thread_id, from_address=sender),
            category_path=None,
            detector=_yes_detector(),
            task_store=tstore,
        )
        assert outcome.task_id
        return outcome.task_id

    subscribe_email_followup(bus, email_store=estore, task_store=tstore)
    return estore, tstore, bus, _seed_followup_for_thread


def test_resolver_marks_wait_resolved_on_thread_match(wired) -> None:
    estore, tstore, bus, seed = wired
    task_id = seed(THREAD)

    new_message = _make_email(id="reply-1", thread_id=THREAD, from_address="colleague@example.com")
    estore.upsert(new_message)

    bus.emit_sync(
        EMAIL_NEW_ARRIVED,
        EmailNewArrivedPayload(
            account_id=ACCOUNT,
            new_message_ids=(new_message.id,),
            count=1,
            fell_back_to_cold_start=False,
        ),
    )

    task = tstore.get(task_id)
    assert task is not None
    assert task.wait_for_resolved_at is not None
    # ADR-0014 #10 — wait resolution does NOT auto-complete the task.
    assert task.status == "open"


def test_resolver_no_op_when_thread_has_no_followup(wired) -> None:
    estore, tstore, bus, _seed = wired
    new_message = _make_email(id="unrelated", thread_id="other-thread")
    estore.upsert(new_message)

    bus.emit_sync(
        EMAIL_NEW_ARRIVED,
        EmailNewArrivedPayload(
            account_id=ACCOUNT,
            new_message_ids=(new_message.id,),
            count=1,
            fell_back_to_cold_start=False,
        ),
    )
    # No tasks created; nothing to resolve. The subscriber is silent.
    assert tstore.list(status=None) == []


def test_resolver_no_op_when_followup_already_resolved(wired) -> None:
    estore, tstore, bus, seed = wired
    task_id = seed(THREAD)
    # Pre-resolve the followup.
    tstore.resolve_wait(task_id, by_event="manual")
    pre = tstore.get(task_id)
    assert pre and pre.wait_for_resolved_at is not None
    first_resolved_at = pre.wait_for_resolved_at

    new_message = _make_email(id="reply-2", thread_id=THREAD)
    estore.upsert(new_message)

    bus.emit_sync(
        EMAIL_NEW_ARRIVED,
        EmailNewArrivedPayload(
            account_id=ACCOUNT,
            new_message_ids=(new_message.id,),
            count=1,
            fell_back_to_cold_start=False,
        ),
    )

    post = tstore.get(task_id)
    assert post is not None
    assert post.wait_for_resolved_at == first_resolved_at  # untouched


def test_resolver_no_op_when_task_is_done(wired) -> None:
    estore, tstore, bus, seed = wired
    task_id = seed(THREAD)
    tstore.complete(task_id)  # user marked done before reply arrived

    new_message = _make_email(id="reply-3", thread_id=THREAD)
    estore.upsert(new_message)

    bus.emit_sync(
        EMAIL_NEW_ARRIVED,
        EmailNewArrivedPayload(
            account_id=ACCOUNT,
            new_message_ids=(new_message.id,),
            count=1,
            fell_back_to_cold_start=False,
        ),
    )

    post = tstore.get(task_id)
    assert post is not None
    assert post.status == "done"
    assert post.wait_for_resolved_at is None  # not auto-resolved on done


def test_resolver_handles_missing_message_gracefully(wired) -> None:
    """A ghost id in new_message_ids should soft-fail, not crash."""
    estore, tstore, bus, seed = wired
    seed(THREAD)

    # The id is not in EmailStore — simulates race with delete or backfill.
    bus.emit_sync(
        EMAIL_NEW_ARRIVED,
        EmailNewArrivedPayload(
            account_id=ACCOUNT,
            new_message_ids=("ghost-id-not-in-store",),
            count=1,
            fell_back_to_cold_start=False,
        ),
    )
    # No exception; nothing changed.


def test_handle_email_new_arrived_ignores_wrong_payload_type(tmp_path: Path) -> None:
    estore = EmailStore(db_path=tmp_path / "email.db")
    estore.ensure_schema()
    tstore = TaskStore(db_path=tmp_path / "tasks.db")
    tstore.ensure_schema()
    _handle_email_new_arrived({"not": "the right type"}, email_store=estore, task_store=tstore)


def test_detect_and_persist_ignores_a_vendor_category(tmp_path: Path) -> None:
    """A vendor path (the mailbox tab) is not a triage verdict: it must not drive the
    non-actionable gate the way an IRIS classification does."""
    store = TaskStore(db_path=tmp_path / "tasks.db")
    store.ensure_schema()
    email = _make_email(classified_category="email/promotions").model_copy(
        update={"classified_source": "vendor"}
    )

    outcome = detect_and_persist(
        email, category_path=None, detector=_yes_detector(), task_store=store
    )
    assert outcome.action == "created"

    iris = _make_email(id="msg-2", thread_id="thread-002", classified_category="email/promotions")
    iris = iris.model_copy(update={"classified_source": "iris"})
    outcome = detect_and_persist(
        iris, category_path=None, detector=_yes_detector(), task_store=store
    )
    assert outcome.action == "skipped:non-actionable-category"


# ─── the dedup key (from services/tasks/test_dedup_keys.py, core/SDK plan PR 2) ──────


def test_followup_key_shape() -> None:
    assert followup_key("gmail", "thread-abc-123") == "followup:gmail:thread-abc-123"


def test_followup_key_is_deterministic() -> None:
    """Same inputs MUST produce identical keys across calls (used for dedup)."""
    assert followup_key("gmail", "t1") == followup_key("gmail", "t1")


def test_followup_key_is_case_sensitive() -> None:
    """The generator does NOT normalize case — callers must canonicalize first."""
    assert followup_key("gmail", "Thread") != followup_key("gmail", "thread")
