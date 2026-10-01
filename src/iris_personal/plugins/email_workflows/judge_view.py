"""Judged emails as the surfaces show them: a judgment row joined with its email.

The card, the chat intercept, the web API and the digest all read the same thing — a
row of ``email_judgments`` with the sender, subject and date of its email in
``email.db`` — and all correct through :func:`.judge_corrections.apply_correction`,
emitting ``email.judgment_corrected`` on the process bus (where the label step
listens). This module is that shared read and that emit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from iris_personal.email.contracts import EmailMessage
from iris_personal.email.store import EmailStore

from .digest_focus import sender_name
from .judge_config import JudgeConfig
from .judge_words import SurfaceWords, bucket_name
from .judgments import Judgment, JudgmentStore


def default_emit(topic: str, payload: Any) -> None:
    """Emit on the process bus, where the sweep and the label step live."""
    from iris_harness.sdk.events import get_default_bus

    get_default_bus().emit_sync(topic, payload)


def open_stores(data_dir: Path, *, create: bool = True) -> tuple[JudgmentStore, EmailStore] | None:
    """The judgments and emails stores in ``data_dir/email.db``; ``None`` when the file
    does not exist and ``create`` is false (nothing judged yet)."""
    db = Path(data_dir) / "email.db"
    if not create and not db.exists():
        return None
    emails = EmailStore(db_path=db)
    emails.ensure_schema()
    judgments = JudgmentStore(db_path=db)
    judgments.ensure_schema()
    return judgments, emails


@dataclass(frozen=True)
class JudgedEmail:
    """One judgment with its email (``email`` is None if the email left ``emails``)."""

    judgment: Judgment
    email: EmailMessage | None

    @property
    def message_id(self) -> str:
        return self.judgment.message_id

    @property
    def sender(self) -> str:
        return sender_name(self.email.from_address) if self.email else ""

    @property
    def subject(self) -> str:
        subject = self.email.subject if self.email else ""
        return " ".join((subject or "").split()) or "(no subject)"

    @property
    def received_at(self) -> datetime | None:
        return self.email.received_at if self.email else None


def with_emails(emails: EmailStore, judgments: list[Judgment]) -> list[JudgedEmail]:
    return [JudgedEmail(judgment=j, email=emails.get(j.message_id)) for j in judgments]


def judged_emails(
    judgments: JudgmentStore,
    emails: EmailStore,
    *,
    bucket: str | None = None,
    limit: int = 50,
    since: datetime | None = None,
) -> list[JudgedEmail]:
    """Judged rows newest first (by effective bucket when given), with their emails."""
    return with_emails(emails, judgments.recent(bucket=bucket, limit=limit, since=since))


def row_view(item: JudgedEmail, config: JudgeConfig, words: SurfaceWords) -> dict[str, Any]:
    """One row of ``GET /api/v1/email/judgments``."""
    j = item.judgment
    email = item.email
    effective = j.effective_bucket
    return {
        "message_id": j.message_id,
        "account_id": j.account_id,
        "sender": item.sender,
        "from_address": email.from_address if email else "",
        "subject": item.subject,
        "snippet": email.snippet if email else "",
        "received_at": email.received_at.isoformat() if email else None,
        "bucket": effective,
        "bucket_name": bucket_name(config, words, effective),
        "judge_bucket": j.bucket,
        "owner_bucket": j.owner_bucket,
        "owner_source": j.owner_source,
        "confidence": j.confidence,
        "figures": dict(j.fields),
        "judged_at": j.judged_at,
        "corrected_at": j.corrected_at,
    }


__all__ = [
    "JudgedEmail",
    "default_emit",
    "judged_emails",
    "open_stores",
    "row_view",
    "with_emails",
]
