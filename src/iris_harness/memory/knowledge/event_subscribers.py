"""Wiki-side consumer for the ingestion bridge (ADR-0025 §2, OSS plan M3.2).

``wiki_ingestion.py`` shipped both halves of the bridge in one module and said so:

    Both subscriber halves live in this module for v1; the wiki-side consumer
    factors out to ``iris_harness.memory.knowledge.event_subscribers`` when finance
    ingestion lands in Phase 3.

M3 forces that factoring a phase early, for the reason ADR-0025 gave for the two-hop
design in the first place. The **email-side** translator (``email.classified`` →
``WikiIngestEvent``) is an email workflow and moved to the email-workflows plugin. The
**wiki-side** consumer is not email-anything: it takes a ``WikiIngestEvent`` from any
producer and hands it to ``WikiEngine.ingest``. ``build_runtime`` wires it next to the
``WikiEngine`` it wraps, so the wiki still ingests whether or not that plugin — or the
finance one after it — is mounted.

The topic lives here with the consumer, since this side owns the contract that every
producing domain publishes into.
"""

from __future__ import annotations

import logging
from typing import Any

from iris_harness.foundation.eventbus import EventBus, get_default_bus
from iris_harness.memory.knowledge.models import WikiIngestEvent

logger = logging.getLogger(__name__)

WIKI_INGEST_REQUESTED = "wiki.ingest_requested"


def make_wiki_ingest_consumer(wiki_engine: Any) -> Any:
    """Return a handler that calls ``wiki_engine.ingest(event)``.

    ``wiki_engine`` is typed ``Any`` to keep this module import-light at load time.
    """

    def _handle(event: Any) -> None:
        if not isinstance(event, WikiIngestEvent):
            logger.warning(
                "wiki-ingestion (consumer): unexpected payload %r on %s",
                type(event),
                WIKI_INGEST_REQUESTED,
            )
            return
        # Engine-level switch, checked here too so a disabled wiki logs nothing per
        # email instead of an "ingest: pages=0" line for every classified message.
        if not getattr(wiki_engine, "ingest_enabled", True):
            logger.debug("wiki ingest consumer idle: ingest off (source=%s)", event.source_id)
            return
        try:
            pages = wiki_engine.ingest(event)
        except Exception:  # per-event soft-fail
            logger.exception(
                "wiki ingest failed for %s (source_agent=%s)",
                event.source_id,
                event.source_agent,
            )
            return
        logger.info("wiki ingest: source=%s pages=%d", event.source_id, len(pages))

    return _handle


def subscribe_wiki_ingest_consumer(wiki_engine: Any, bus: EventBus | None = None) -> None:
    """Wire the wiki-side consumer. Calls ``wiki_engine.ingest`` for every
    ``WikiIngestEvent`` published on ``WIKI_INGEST_REQUESTED``.

    Defaults to the **process-global** bus, because the producing chains are
    process-global: ``iris email reingest-wiki`` drives this path with no runtime
    built at all (see the bus-scope note in the plugin contract).
    """
    target = bus if bus is not None else get_default_bus()
    target.on(WIKI_INGEST_REQUESTED, make_wiki_ingest_consumer(wiki_engine))
    logger.info("wiki-ingestion (consumer) subscribed to %s", WIKI_INGEST_REQUESTED)
