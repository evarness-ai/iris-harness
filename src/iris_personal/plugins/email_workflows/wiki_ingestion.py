"""Email → wiki ingestion bridge (Track 1K / ADR-0025).

Subscribes to ``email.classified`` and emits ``WikiIngestEvent`` on
``WIKI_INGEST_REQUESTED``. Two-hop architecture per ADR-0025 §2 so future
domains (finance, calendar) reuse the wiki-side consumer without changes.

This is the **email-side** half only. The wiki-side consumer factored out to
``iris_harness.memory.knowledge.event_subscribers`` at OSS plan M3.2 — the factoring
this module's v1 note already planned for Phase 3 — because it takes a
``WikiIngestEvent`` from any producer and belongs to the wiki, while translating
``email.classified`` into one is an email workflow and lives here in the plugin.
The topic constant lives with the consumer that owns the contract.
"""

from __future__ import annotations

import logging
from typing import Any

from iris_harness.sdk.events import EventBus, get_default_bus
from iris_harness.sdk.memory import WIKI_INGEST_REQUESTED, WikiIngestEvent
from iris_personal.email.events import EMAIL_CLASSIFIED, EmailClassifiedPayload
from iris_personal.email.store import EmailStore

logger = logging.getLogger(__name__)

# Tag emitted on WikiIngestEvent.source_agent so the wiki's audit log
# can distinguish email-driven ingestions from other sources (lessons,
# manual, future finance).
EMAIL_SOURCE_AGENT = "email-triage"


# ─── Email-side subscriber: payload → WikiIngestEvent ────────────────────────


def _build_ingest_event(
    payload: EmailClassifiedPayload, *, email_store: EmailStore
) -> WikiIngestEvent | None:
    """Project an ``EmailClassifiedPayload`` into a ``WikiIngestEvent``.

    Pulls the email envelope from email.db (subject + snippet +
    from_domain) since the classification event only carries
    category metadata, not content. Returns None if the message has
    aged out of email.db (defensive — should not happen on the
    auto-fire path but the backfill CLI can race with deletes).
    """
    message = email_store.get(payload.id)
    if message is None:
        logger.warning("wiki-ingestion: ghost id %s on email.classified", payload.id)
        return None

    content = (message.subject or "") + "\n\n" + (message.snippet or "")

    # Hints per ADR-0025 §4 — the from-domain + the 3-level category
    # hierarchy are pre-known facts; the extractor weights hints at
    # 0.95 confidence.
    hints: list[str] = []
    if message.from_domain:
        hints.append(message.from_domain)
    parts = payload.category_path.split("/")
    # path is "<type>/<root>/<branch>/<leaf>"; skip the type prefix
    hints.extend(p for p in parts[1:] if p)

    metadata: dict[str, Any] = {
        "category_path": payload.category_path,
        "confidence": payload.confidence,
        "classifier": payload.classifier,
        "account_id": payload.account_id,
        "from_address": message.from_address,
        "received_at": message.received_at.isoformat(),
    }

    return WikiIngestEvent(
        source_agent=EMAIL_SOURCE_AGENT,
        source_id=f"email/{payload.account_id}/{payload.id}",
        content=content,
        entities_hint=hints,
        metadata=metadata,
    )


def _handle_email_classified(payload: Any) -> None:
    """Subscriber: convert ``EmailClassifiedPayload`` →
    ``WikiIngestEvent`` and re-publish on the same bus.

    Ignores non-EmailClassifiedPayload dispatches with a warn log so
    a misrouted event can't crash the runtime.
    """
    if not isinstance(payload, EmailClassifiedPayload):
        logger.warning(
            "wiki-ingestion: unexpected payload %r on %s",
            type(payload),
            EMAIL_CLASSIFIED,
        )
        return

    store = EmailStore()
    store.ensure_schema()
    event = _build_ingest_event(payload, email_store=store)
    if event is None:
        return

    get_default_bus().emit_sync(WIKI_INGEST_REQUESTED, event)


def subscribe_email_classified_to_wiki(bus: EventBus | None = None) -> None:
    """Wire the email-side translator. Emits ``WikiIngestEvent`` on
    ``WIKI_INGEST_REQUESTED`` for every ``email.classified``."""
    target = bus if bus is not None else get_default_bus()
    target.on(EMAIL_CLASSIFIED, _handle_email_classified)
    logger.info("wiki-ingestion (email-side) subscribed to %s", EMAIL_CLASSIFIED)
