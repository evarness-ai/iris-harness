"""Made-up judged emails for the correction-surface tests (loop-proof PR 5).

No real senders or addresses: every name and domain here is invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.judgments import JudgmentStore

ACCOUNT = "gmail:owner@example.com"


@dataclass
class Inbox:
    """``email.db`` in a temp dir, with helpers to add a judged email."""

    data_dir: Path
    emitted: list[tuple[str, Any]] = field(default_factory=list)

    @property
    def db(self) -> Path:
        return self.data_dir / "email.db"

    @property
    def emails(self) -> EmailStore:
        store = EmailStore(db_path=self.db)
        store.ensure_schema()
        return store

    @property
    def judgments(self) -> JudgmentStore:
        store = JudgmentStore(db_path=self.db)
        store.ensure_schema()
        return store

    def emit(self, topic: str, payload: Any) -> None:
        self.emitted.append((topic, payload))

    def email(
        self,
        mid: str,
        sender: str,
        subject: str,
        *,
        received: datetime,
        snippet: str = "",
        thread: str | None = None,
        labels: tuple[str, ...] = ("INBOX",),
    ) -> None:
        self.emails.upsert(
            EmailMessage(
                id=mid,
                provider="gmail",
                account_id=ACCOUNT,
                thread_id=thread or f"t-{mid}",
                from_address=sender,
                subject=subject,
                snippet=snippet,
                received_at=received,
                labels=labels,
            )
        )

    def judged(
        self,
        mid: str,
        sender: str,
        subject: str,
        bucket: str,
        *,
        at: datetime,
        confidence: float = 0.9,
        snippet: str = "",
        fields: dict[str, Any] | None = None,
        thread: str | None = None,
    ) -> None:
        """An email received and judged at ``at`` (the store's clock is patched)."""
        self.email(mid, sender, subject, received=at, snippet=snippet, thread=thread)
        with patched_clock(at):
            self.judgments.record(
                mid, ACCOUNT, bucket=bucket, confidence=confidence, fields=fields or {}
            )

    def correct(self, mid: str, bucket: str, *, source: str, at: datetime) -> None:
        with patched_clock(at):
            self.judgments.correct(mid, bucket, source=source)


class patched_clock:  # used like a function
    """Pin the judgments store's ``_now`` to ``at``."""

    def __init__(self, at: datetime) -> None:
        self._patch = patch(
            "iris_personal.plugins.email_workflows.judgments._now",
            lambda: at.astimezone(UTC).isoformat(),
        )

    def __enter__(self) -> None:
        self._patch.start()

    def __exit__(self, *exc: object) -> None:
        self._patch.stop()


def seed_day(inbox: Inbox, day: datetime) -> list[str]:
    """A small mixed inbox judged on ``day`` (made-up senders)."""
    rows = [
        (
            "m-bill",
            "Northwind Card <statements@northwind.example>",
            "Your statement is ready",
            "bill",
        ),
        (
            "m-dental",
            "Bright Smile Dental <front@brightsmile.example>",
            "Appointment confirmed",
            "event",
        ),
        ("m-dinner", "Petra Sample <petra@mail.example>", "dinner Saturday?", "needs_reply"),
        ("m-case", "Harbor Bank <care@harborbank.example>", "About your recent inquiry", "fyi"),
        (
            "m-plan",
            "Nimbus Utilities <notices@nimbus.example>",
            "Important information about your account",
            "unsure",
        ),
    ]
    out: list[str] = []
    for i, (mid, sender, subject, bucket) in enumerate(rows):
        inbox.judged(
            mid,
            sender,
            subject,
            bucket,
            at=day.replace(hour=9 + i),
            confidence=0.52 if bucket == "unsure" else 0.95,
            snippet=f"Sample snippet for {subject}.",
        )
        out.append(mid)
    return out


DAY = datetime(2026, 9, 26, tzinfo=UTC)
