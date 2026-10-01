"""Event topics + typed payloads for the email subsystem.

Per ADR-0013, subsystem-private topics live with their producer rather
than in ``iris_harness.foundation.eventbus.topics``. The email subsystem produces these;
downstream subscribers (triage, brief, calendar-sync) consume them via
the runtime event bus.

Topic naming convention: ``<domain>.<verb>``.
"""

from __future__ import annotations

from dataclasses import dataclass

EMAIL_NEW_ARRIVED = "email.new_arrived"
EMAIL_CLASSIFIED = "email.classified"
EMAIL_LABELS_CHANGED = "email.labels_changed"
# The sweep stored new mail (loop-proof PR 5). Not "new mail" for the rest of IRIS yet:
# the email judge's queue takes it, releases what it will not judge at once, and
# releases the rest as ``email.new_arrived`` once judged. Same payload type.
EMAIL_SWEPT = "email.swept"


@dataclass(frozen=True)
class EmailNewArrivedPayload:
    """``email.swept``: emitted by the email-sweep heartbeat after a per-account fetch
    that returned at least one new message. ``email.new_arrived``: emitted when those
    messages are released to the rest of IRIS (skipped by the judge, or judged).

    Subscribers receive the list of new message ids so they can pull each
    from ``EmailStore`` for further processing without a separate
    ``last_processed_at`` cursor.
    """

    account_id: str
    new_message_ids: tuple[str, ...]
    count: int
    fell_back_to_cold_start: bool


@dataclass(frozen=True)
class EmailClassifiedPayload:
    """Emitted by the email-triage classifier after one message gets a
    category path written to ``emails.classified_category``.

    Subscribers (wiki ingestion in Track 1K, future router rules, etc.)
    receive the message id + the path + confidence so they can act
    without re-querying ``email.db``.

    ``classifier`` identifies which classifier produced the result —
    Phase 1 ships ``'tier3-local-knn'`` (hybrid kNN-then-LLM per
    ADR-0021); future tracks may add ``'self-learning-correction'``
    when Track 1J writes back user corrections.
    """

    id: str
    account_id: str
    category_path: str
    confidence: float
    classifier: str = "tier3-local-knn"


@dataclass(frozen=True)
class EmailLabelsChangedPayload:
    """Emitted by the email-sweep heartbeat when a fetch read back label changes on
    mail already in the store (Gmail: history ``labelAdded`` / ``labelRemoved``).

    ``changes`` is ``(message id, its provider labels now)`` per message; the store's
    ``labels`` column already holds the same. The email judge reads its IRIS/* labels
    back from here as the owner's corrections (loop-proof PR 5). IRIS's own label
    writes show up too; the consumer tells them apart.
    """

    account_id: str
    changes: tuple[tuple[str, tuple[str, ...]], ...]
