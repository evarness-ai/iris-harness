"""Tests for the generic surface-feedback + suppression spine (issue 0028)."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.services.learning.suppression import (
    NOT_USEFUL,
    SOURCE_AUTO,
    USEFUL,
    SurfaceFeedbackStore,
    decode_ref,
    encode_ref,
)


@pytest.fixture
def store(tmp_path: Path) -> SurfaceFeedbackStore:
    s = SurfaceFeedbackStore(db_path=tmp_path / "learning.db")
    s.ensure_schema()
    return s


# ─── ref round-trip ──────────────────────────────────────────────────────────


def test_ref_round_trips() -> None:
    ref = encode_ref(
        "email", "followup", {"account": "gmail:u", "from_domain": "quant-academy.example"}
    )
    sub, kind, dims = decode_ref(ref)
    assert ref.startswith("fb:")
    assert sub == "email"
    assert kind == "followup"
    assert dims == {"account": "gmail:u", "from_domain": "quant-academy.example"}


def test_ref_canonicalizes_dims() -> None:
    # Case/whitespace differences collapse so the same item yields the same ref.
    a = encode_ref("email", "followup", {"From_Domain": " Quant-Academy.Example "})
    b = encode_ref("email", "followup", {"from_domain": "quant-academy.example"})
    assert a == b


def test_decode_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        decode_ref("not-a-ref")
    with pytest.raises(ValueError):
        decode_ref("fb:%%%not-base64%%%")


# ─── suppression semantics ───────────────────────────────────────────────────


def test_unseen_key_is_not_suppressed(store: SurfaceFeedbackStore) -> None:
    assert store.should_suppress("email", "followup", {"from_domain": "x.com"}) is False


def test_single_not_useful_suppresses(store: SurfaceFeedbackStore) -> None:
    dims = {
        "account": "gmail:u",
        "from_domain": "quant-academy.example",
        "category_root": "learning",
    }
    store.record("email", "followup", dims, NOT_USEFUL)
    assert store.should_suppress("email", "followup", dims) is True


def test_useful_lifts_suppression(store: SurfaceFeedbackStore) -> None:
    dims = {"from_domain": "boss.com"}
    store.record("email", "followup", dims, NOT_USEFUL)
    assert store.should_suppress("email", "followup", dims) is True
    store.record("email", "followup", dims, USEFUL)
    # net = 1 - 1 = 0 < min_fp(1) → no longer suppressed
    assert store.should_suppress("email", "followup", dims) is False


def test_suppression_is_key_scoped(store: SurfaceFeedbackStore) -> None:
    store.record("email", "followup", {"from_domain": "spam.com"}, NOT_USEFUL)
    # A different domain is unaffected.
    assert store.should_suppress("email", "followup", {"from_domain": "friend.com"}) is False
    # A different subsystem with identical dims is unaffected.
    assert store.should_suppress("finance", "bill_due", {"from_domain": "spam.com"}) is False


def test_min_fp_threshold(store: SurfaceFeedbackStore) -> None:
    dims = {"category": "groceries", "currency": "inr"}
    store.record("finance", "bill_due", dims, NOT_USEFUL, source=SOURCE_AUTO)
    # One auto signal is below a stricter threshold...
    assert store.should_suppress("finance", "bill_due", dims, min_fp=2) is False
    store.record("finance", "bill_due", dims, NOT_USEFUL, source=SOURCE_AUTO)
    assert store.should_suppress("finance", "bill_due", dims, min_fp=2) is True


def test_stat_reports_counts(store: SurfaceFeedbackStore) -> None:
    dims = {"target": "gmail", "kind": "credential"}
    store.record("system", "health_alert", dims, NOT_USEFUL)
    store.record("system", "health_alert", dims, NOT_USEFUL)
    store.record("system", "health_alert", dims, USEFUL)
    stat = store.stat("system", "health_alert", dims)
    assert (stat.not_useful, stat.useful, stat.net) == (2, 1, 1)


def test_record_ignores_unknown_verdict(store: SurfaceFeedbackStore) -> None:
    dims = {"from_domain": "x.com"}
    store.record("email", "followup", dims, "maybe")  # not in VERDICTS
    assert store.should_suppress("email", "followup", dims) is False


def test_layer2_signal_is_mirrored(store: SurfaceFeedbackStore) -> None:
    from iris_harness.services.learning.store import LearningMetricsStore

    dims = {"from_domain": "quant-academy.example"}
    store.record("email", "followup", dims, NOT_USEFUL, session_id="sess-1")

    lms = LearningMetricsStore(db_path=store.db_path)
    lms.ensure_schema()
    signals = lms.list_user_behavior_signals(kind="surface_feedback_not_useful")
    assert len(signals) == 1
    assert signals[0].subject == "email/followup"


# ─── ledger-wide summary (context-health, ADR-0081) ──────────────────────────


def test_summary_counts_active_suppressions(store: SurfaceFeedbackStore) -> None:
    # email/followup: net +1 -> active. finance/bill: net 0 (one each) -> not active.
    store.record("email", "followup", {"from_domain": "ads.example"}, NOT_USEFUL)
    store.record("finance", "bill", {"label": "card"}, NOT_USEFUL)
    store.record("finance", "bill", {"label": "card"}, USEFUL)
    summary = store.summary()

    assert summary.total_feedback == 3
    assert summary.active_suppressions == 1
    assert summary.by_subsystem == {"email": 1}


def test_summary_empty_store_is_zeros(store: SurfaceFeedbackStore) -> None:
    summary = store.summary()
    assert summary.total_feedback == 0
    assert summary.active_suppressions == 0
    assert summary.by_subsystem == {}


def test_summary_as_dict_shape(store: SurfaceFeedbackStore) -> None:
    store.record("system", "health_alert", {"kind": "gmail"}, NOT_USEFUL)
    d = store.summary().as_dict()
    assert d == {
        "total_feedback": 1,
        "active_suppressions": 1,
        "by_subsystem": {"system": 1},
    }


# ─── the digest's Focus key + the day's verdicts (loop-proof D17) ────────────


def test_focus_key_is_the_bare_sender_address() -> None:
    from iris_harness.services.learning.suppression import email_focus_dims

    assert email_focus_dims("GoldenPi <News@GoldenPi.example>") == {
        "sender": "news@goldenpi.example"
    }
    assert email_focus_dims(" news@goldenpi.example ") == {"sender": "news@goldenpi.example"}


def test_feedback_between_reads_one_window_oldest_first(store: SurfaceFeedbackStore) -> None:
    from datetime import UTC, datetime, timedelta

    store.record("email", "focus", {"sender": "a@x.example"}, NOT_USEFUL, emit_signal=False)
    store.record("email", "focus", {"sender": "b@x.example"}, USEFUL, emit_signal=False)
    store.record(
        "email",
        "focus",
        {"sender": "c@x.example"},
        NOT_USEFUL,
        source=SOURCE_AUTO,
        emit_signal=False,
    )
    now = datetime.now(UTC)
    got = store.feedback_between(now - timedelta(hours=1), now + timedelta(hours=1))
    assert [(e.dims["sender"], e.verdict) for e in got] == [
        ("a@x.example", NOT_USEFUL),
        ("b@x.example", USEFUL),
    ]
    everything = store.feedback_between(
        now - timedelta(hours=1), now + timedelta(hours=1), source=None
    )
    assert len(everything) == 3
    assert store.feedback_between(now + timedelta(hours=1), now + timedelta(hours=2)) == []
