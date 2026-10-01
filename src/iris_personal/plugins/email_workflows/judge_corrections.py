"""The owner corrects a judgment — the one path every surface uses.

A Gmail relabel (``judge_labels``), the Action Center card, chat and the web list all
call :func:`apply_correction`: it checks the bucket against ``judge.yaml`` (plus
``promo``), writes ``owner_bucket`` on the row and, when something changed, emits
``email.judgment_corrected``. The label step moves the Gmail label from the row
(``labels_due``); the digest footer and the judge's sender hints read the row.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .judge_config import (
    EMAIL_JUDGMENT_CORRECTED,
    JudgeConfig,
    JudgmentCorrectedPayload,
)
from .judgments import PROMO, Correction, JudgmentStore

Emit = Callable[[str, Any], None]


class UnknownBucket(ValueError):
    """The bucket is not one of judge.yaml's, nor ``promo``."""


def valid_buckets(config: JudgeConfig) -> tuple[str, ...]:
    """Every bucket an owner may pick: the judge's, plus ``promo``."""
    return (*config.keys, PROMO)


def apply_correction(
    store: JudgmentStore,
    config: JudgeConfig,
    message_id: str,
    bucket: str,
    *,
    source: str,
    emit: Emit | None = None,
) -> Correction | None:
    """Set the owner's bucket for one email. ``None`` when the email was never judged
    or queued. Emits ``email.judgment_corrected`` only when the bucket changed."""
    if bucket not in valid_buckets(config):
        raise UnknownBucket(bucket)
    correction = store.correct(message_id, bucket, source=source)
    if correction is not None and correction.changed and emit is not None:
        emit(
            EMAIL_JUDGMENT_CORRECTED,
            JudgmentCorrectedPayload(
                message_id=message_id,
                account_id=correction.judgment.account_id,
                bucket=bucket,
                previous=correction.previous,
                source=source,
            ),
        )
    return correction
