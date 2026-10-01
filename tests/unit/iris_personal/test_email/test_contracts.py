"""Tests for the ``EmailMessage`` provider-agnostic contract."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from iris_personal.email.contracts import (
    CategoryProposal,
    CategoryRepresentative,
    EmailAttachment,
    EmailMessage,
)


def _now() -> datetime:
    return datetime.now(UTC)


def test_minimal_message_valid() -> None:
    m = EmailMessage(
        id="msg-1",
        provider="gmail",
        account_id="gmail:user@gmail.com",
        from_address="a@b.com",
        received_at=_now(),
    )
    assert m.subject == ""
    assert m.to == ()
    assert m.body_text is None
    assert m.body_html is None
    assert m.labels == ()
    assert m.attachments == ()


def test_id_required_nonempty() -> None:
    with pytest.raises(ValidationError):
        EmailMessage(
            id="",
            provider="gmail",
            account_id="gmail:x@y.com",
            from_address="a@b.com",
            received_at=_now(),
        )


def test_from_address_minimum_length() -> None:
    """Bare "@" or shorter is invalid (we require at least "a@b" = 3 chars)."""
    with pytest.raises(ValidationError):
        EmailMessage(
            id="msg-1",
            provider="gmail",
            account_id="gmail:x@y.com",
            from_address="@",  # 1 char
            received_at=_now(),
        )


def test_the_provider_set_is_open_to_a_plugin_but_never_blank() -> None:
    """A third-party mail provider records its mail under its own name (the registry's
    key, ``email.provider_api``); a closed list of in-tree names could not admit it."""
    message = EmailMessage(
        id="msg-1",
        provider="pigeon",
        account_id="pigeon:x@y.com",
        from_address="a@b.com",
        received_at=_now(),
    )
    assert message.provider == "pigeon"
    with pytest.raises(ValidationError):
        EmailMessage(
            id="msg-1",
            provider="  ",
            account_id="pigeon:x@y.com",
            from_address="a@b.com",
            received_at=_now(),
        )


def test_attachment_validation() -> None:
    """size_bytes must be >= 0; required fields must be non-empty."""
    with pytest.raises(ValidationError):
        EmailAttachment(
            filename="x.pdf",
            mime_type="application/pdf",
            size_bytes=-1,  # negative
            attachment_id="att-1",
        )
    with pytest.raises(ValidationError):
        EmailAttachment(
            filename="",
            mime_type="application/pdf",
            size_bytes=10,
            attachment_id="att-1",
        )


def test_full_message_roundtrip_via_pydantic() -> None:
    """Construct → dump → reconstruct yields an equal object."""
    att = EmailAttachment(
        filename="statement.pdf",
        mime_type="application/pdf",
        size_bytes=42000,
        attachment_id="att-x",
    )
    m = EmailMessage(
        id="msg-1",
        provider="gmail",
        account_id="gmail:user@gmail.com",
        thread_id="thr-1",
        from_address="Bob <bob@chase.com>",
        from_domain="chase.com",
        to=("user@gmail.com",),
        cc=("manager@example.com",),
        subject="May statement",
        received_at=datetime(2026, 5, 25, 10, 0, tzinfo=UTC),
        snippet="Your May statement is ready...",
        body_text="(long body)",
        body_html="<p>(long body)</p>",
        labels=("INBOX", "STATEMENT"),
        attachments=(att,),
        headers_subset={"Message-ID": "<a@b>"},
    )
    dumped = m.model_dump()
    reborn = EmailMessage(**dumped)
    assert reborn == m


def test_immutability() -> None:
    """Pydantic v2 frozen config makes the model immutable."""
    m = EmailMessage(
        id="msg-1",
        provider="gmail",
        account_id="gmail:user@gmail.com",
        from_address="a@b.com",
        received_at=_now(),
    )
    with pytest.raises(ValidationError):
        m.subject = "new subject"  # type: ignore[misc]


# ─── CategoryProposal (Track 1E.2) ──────────────────────────────────────────


def _proposal(**overrides) -> CategoryProposal:  # type: ignore[no-untyped-def]
    defaults: dict = {"cluster_id": 0, "size": 10, "cohesion": 0.8}
    defaults.update(overrides)
    return CategoryProposal(**defaults)


def test_proposal_minimal_valid() -> None:
    p = _proposal()
    assert p.cluster_id == 0
    assert p.representatives == ()
    assert p.member_ids == ()
    assert p.proposed_root is None
    assert p.proposed_branch is None
    assert p.proposed_leaf is None
    assert p.naming_rationale is None


def test_proposal_cohesion_range() -> None:
    """cohesion must be ∈ [0, 1]."""
    with pytest.raises(ValidationError):
        _proposal(cohesion=-0.01)
    with pytest.raises(ValidationError):
        _proposal(cohesion=1.5)


def test_proposal_size_positive() -> None:
    with pytest.raises(ValidationError):
        _proposal(size=0)


def test_proposal_with_full_payload() -> None:
    rep = CategoryRepresentative(id="msg-1", subject="Hello", from_address="a@b.com", snippet="...")
    p = _proposal(
        size=2,
        top_domains=(("b.com", 2),),
        representatives=(rep,),
        member_ids=("msg-1", "msg-2"),
        proposed_root="shopping",
        proposed_branch="apparel",
        proposed_leaf="outlet-brand",
        naming_rationale="dominated by gap.com",
    )
    assert p.representatives[0].id == "msg-1"
    assert p.top_domains == (("b.com", 2),)


def test_proposal_immutable() -> None:
    p = _proposal()
    with pytest.raises(ValidationError):
        p.cluster_id = 7  # type: ignore[misc]


def test_proposal_roundtrip() -> None:
    p = _proposal(
        member_ids=("a", "b", "c"),
        proposed_root="finance",
        proposed_branch="investing",
        proposed_leaf="indian-bonds",
    )
    dumped = p.model_dump()
    reborn = CategoryProposal(**dumped)
    assert reborn == p
