"""Tests for the email subsystem's event topics + payloads."""

from __future__ import annotations

import dataclasses

import pytest

from iris_personal.email.events import (
    EMAIL_CLASSIFIED,
    EMAIL_NEW_ARRIVED,
    EMAIL_SWEPT,
    EmailClassifiedPayload,
    EmailNewArrivedPayload,
)


def test_topic_names_match_convention() -> None:
    """Topics follow ``<domain>.<verb>`` per ADR-0013 + iris_personal.email.events docstring."""
    assert EMAIL_NEW_ARRIVED == "email.new_arrived"
    assert EMAIL_CLASSIFIED == "email.classified"
    assert EMAIL_SWEPT == "email.swept"


def test_email_classified_payload_minimal() -> None:
    p = EmailClassifiedPayload(
        id="msg-1",
        account_id="gmail:user@gmail.com",
        category_path="email/shopping/apparel/outlet-brand",
        confidence=0.87,
    )
    assert p.classifier == "tier3-local-knn"  # default per Phase 1


def test_email_classified_payload_is_frozen() -> None:
    p = EmailClassifiedPayload(
        id="msg-1",
        account_id="gmail:user@gmail.com",
        category_path="email/social/facebook/updates",
        confidence=0.7,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.confidence = 0.2  # type: ignore[misc]


def test_email_new_arrived_payload_still_works() -> None:
    """Regression: the previously-shipped event should keep its shape."""
    p = EmailNewArrivedPayload(
        account_id="gmail:user@gmail.com",
        new_message_ids=("a", "b"),
        count=2,
        fell_back_to_cold_start=False,
    )
    assert p.count == 2
