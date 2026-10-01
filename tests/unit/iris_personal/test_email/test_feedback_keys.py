"""Email's "not useful" keys, and the core recorders' copy of them, agree.

The keys are email's (``iris_personal.email.feedback_keys``, email slice step 4). Until
PR 7 of the core/SDK boundary plan moves the recorders to email, three core paths still
record a verdict with ``iris_harness.services.learning.suppression``'s copy: the
``record_feedback`` tool, ``iris feedback`` and the digest's ``/not-useful`` route. A
verdict recorded there must hide what email reads, so any drift between the copies is a
silently broken "not useful" button -- this file is what stops it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.services.learning import suppression as core
from iris_harness.services.learning.suppression import NOT_USEFUL, SurfaceFeedbackStore
from iris_personal.email import feedback_keys as email

SENDERS = [
    "news@goldenpi.example",
    "GoldenPi <News@GoldenPi.example>",
    " offers@bank.example ",
    '"Bank, Offers" <offers@bank.example>',
    "Name < spaced@x.example >",
    "vendor.example",
    "no-at-sign",
    "",
]


def test_the_surfaces_and_the_subsystem_are_the_same_words() -> None:
    assert email.EMAIL_SUBSYSTEM == core.EMAIL_SEARCH_SUBSYSTEM
    assert email.EMAIL_SEARCH_SURFACE == core.EMAIL_SEARCH_SURFACE
    assert email.EMAIL_FOLLOWUP_SURFACE == core.EMAIL_FOLLOWUP_SURFACE
    assert email.EMAIL_FOCUS_SURFACE == core.EMAIL_FOCUS_SURFACE


@pytest.mark.parametrize("sender", SENDERS)
def test_the_focus_key_is_the_same(sender: str) -> None:
    assert email.email_focus_dims(sender) == core.email_focus_dims(sender)


@pytest.mark.parametrize("sender", SENDERS)
def test_the_followup_key_is_the_same(sender: str) -> None:
    assert email.email_followup_dims_from("gmail:me", sender) == core.email_followup_dims_from(
        "gmail:me", sender
    )


@pytest.mark.parametrize("domain", ["Vendor.Example", " vendor.example ", "", "x"])
def test_the_search_key_is_the_same(domain: str) -> None:
    assert email.email_search_dims(domain) == core.email_search_dims(domain)


def test_a_verdict_the_core_records_hides_what_email_reads(tmp_path: Path) -> None:
    """The three recorders, end to end against the store email's readers consult."""
    store = SurfaceFeedbackStore(db_path=tmp_path / "feedback.db")
    store.ensure_schema()
    # `/not-useful` on a Focus line, `record_feedback` on a search sender and on a followup.
    store.record(
        core.EMAIL_SEARCH_SUBSYSTEM,
        core.EMAIL_FOCUS_SURFACE,
        core.email_focus_dims("GoldenPi <news@goldenpi.example>"),
        NOT_USEFUL,
        emit_signal=False,
    )
    store.record(
        core.EMAIL_SEARCH_SUBSYSTEM,
        core.EMAIL_SEARCH_SURFACE,
        core.email_search_dims_from_sender("Promo <deals@vendor.example>"),
        NOT_USEFUL,
        emit_signal=False,
    )
    store.record(
        core.EMAIL_SEARCH_SUBSYSTEM,
        core.EMAIL_FOLLOWUP_SURFACE,
        core.email_followup_dims_from("gmail:me", "Quant <no-reply@quant.example>"),
        NOT_USEFUL,
        emit_signal=False,
    )

    assert store.should_suppress(
        email.EMAIL_SUBSYSTEM,
        email.EMAIL_FOCUS_SURFACE,
        email.email_focus_dims("news@goldenpi.example"),
    )
    assert store.should_suppress(
        email.EMAIL_SUBSYSTEM, email.EMAIL_SEARCH_SURFACE, email.email_search_dims("vendor.example")
    )
    assert store.should_suppress(
        email.EMAIL_SUBSYSTEM,
        email.EMAIL_FOLLOWUP_SURFACE,
        email.email_followup_dims_from("gmail:me", "other@quant.example"),
    )
