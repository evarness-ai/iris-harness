"""Tests for the email→wiki ingestion bridge (Track 1K / ADR-0025).

The email-side translator is exercised here. The wiki-side consumer
(WikiEngine.ingest) is the well-tested existing component — we just
verify the bridge emits the right WikiIngestEvent shape.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from iris_harness.foundation.eventbus import EventBus
from iris_harness.memory.knowledge.event_subscribers import WIKI_INGEST_REQUESTED
from iris_harness.memory.knowledge.models import WikiIngestEvent
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.events import EMAIL_CLASSIFIED, EmailClassifiedPayload
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.wiki_ingestion import (
    EMAIL_SOURCE_AGENT,
    _build_ingest_event,
    subscribe_email_classified_to_wiki,
)

ACCOUNT = "gmail:user@gmail.com"


def _make_email(
    *,
    id: str = "msg-1",
    subject: str = "Northwind: Your statement is ready",
    snippet: str = "Your May statement for account ending 4321 is available",
    from_address: str = "bank@northwindbank.test",
    from_domain: str = "northwindbank.test",
) -> EmailMessage:
    return EmailMessage(
        id=id,
        provider="gmail",
        account_id=ACCOUNT,
        from_address=from_address,
        from_domain=from_domain,
        subject=subject,
        snippet=snippet,
        received_at=datetime(2026, 5, 25, 12, 0, tzinfo=UTC),
    )


# ─── _build_ingest_event ────────────────────────────────────────────────────


def test_build_ingest_event_composes_content_subject_and_snippet(tmp_path: Path) -> None:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(_make_email())

    payload = EmailClassifiedPayload(
        id="msg-1",
        account_id=ACCOUNT,
        category_path="email/finance/banking/northwind-savings",
        confidence=0.85,
    )
    event = _build_ingest_event(payload, email_store=store)
    assert event is not None
    assert event.source_agent == EMAIL_SOURCE_AGENT
    assert event.source_id == f"email/{ACCOUNT}/msg-1"
    assert event.content.startswith("Northwind: Your statement is ready")
    assert "May statement" in event.content


def test_build_ingest_event_hints_carry_domain_and_category(tmp_path: Path) -> None:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(_make_email())

    payload = EmailClassifiedPayload(
        id="msg-1",
        account_id=ACCOUNT,
        category_path="email/finance/banking/northwind-savings",
        confidence=0.85,
    )
    event = _build_ingest_event(payload, email_store=store)
    assert event is not None
    # Hints per ADR-0025 §4: from_domain + root + branch + leaf (NOT the type prefix)
    assert event.entities_hint == ["northwindbank.test", "finance", "banking", "northwind-savings"]


def test_build_ingest_event_metadata_round_trips_classification_facts(tmp_path: Path) -> None:
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(_make_email())

    payload = EmailClassifiedPayload(
        id="msg-1",
        account_id=ACCOUNT,
        category_path="email/finance/banking/northwind-savings",
        confidence=0.85,
        classifier="pure-knn",
    )
    event = _build_ingest_event(payload, email_store=store)
    assert event is not None
    meta = event.metadata
    assert meta["category_path"] == "email/finance/banking/northwind-savings"
    assert meta["confidence"] == 0.85
    assert meta["classifier"] == "pure-knn"
    assert meta["account_id"] == ACCOUNT
    assert meta["from_address"] == "bank@northwindbank.test"
    assert isinstance(meta["received_at"], str)


def test_build_ingest_event_returns_none_for_ghost_message(tmp_path: Path) -> None:
    """email.classified arrives but the message has aged out of email.db."""
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    # No upsert — store is empty
    payload = EmailClassifiedPayload(
        id="missing",
        account_id=ACCOUNT,
        category_path="email/finance/banking/northwind-savings",
        confidence=0.85,
    )
    event = _build_ingest_event(payload, email_store=store)
    assert event is None


# ─── subscribe_email_classified_to_wiki ─────────────────────────────────────


def test_subscribe_emits_wiki_ingest_on_email_classified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end on the bus: email.classified → WIKI_INGEST_REQUESTED."""
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(_make_email())

    bus = EventBus()
    captured: list = []
    bus.on(WIKI_INGEST_REQUESTED, captured.append)

    # The subscriber resolves get_default_bus() for emitting the
    # downstream event; patch it to use our test bus.
    import iris_personal.plugins.email_workflows.wiki_ingestion as wiki_ingestion

    monkeypatch.setattr(wiki_ingestion, "get_default_bus", lambda: bus)
    # Patch EmailStore() construction inside the subscriber to use our path.
    monkeypatch.setattr(
        wiki_ingestion,
        "EmailStore",
        lambda *a, **kw: store,
    )

    subscribe_email_classified_to_wiki(bus=bus)

    bus.emit_sync(
        EMAIL_CLASSIFIED,
        EmailClassifiedPayload(
            id="msg-1",
            account_id=ACCOUNT,
            category_path="email/finance/banking/northwind-savings",
            confidence=0.85,
        ),
    )

    assert len(captured) == 1
    event = captured[0]
    assert isinstance(event, WikiIngestEvent)
    assert event.source_id == f"email/{ACCOUNT}/msg-1"
    assert event.source_agent == EMAIL_SOURCE_AGENT


def test_subscriber_ignores_unrelated_payloads(tmp_path: Path) -> None:
    """A non-EmailClassifiedPayload on the topic must not crash."""
    bus = EventBus()
    subscribe_email_classified_to_wiki(bus=bus)
    # Should not raise
    bus.emit_sync(EMAIL_CLASSIFIED, "not-a-payload")
    bus.emit_sync(EMAIL_CLASSIFIED, {"id": "x"})


# ─── Wiki-side consumer ─────────────────────────────────────────────────────


class _StubWikiEngine:
    """Captures every WikiIngestEvent passed to .ingest()."""

    def __init__(
        self, *, raise_on_ingest: Exception | None = None, ingest_enabled: bool = True
    ) -> None:
        self.calls: list = []
        self._raise = raise_on_ingest
        self.ingest_enabled = ingest_enabled

    def ingest(self, event):  # type: ignore[no-untyped-def]
        self.calls.append(event)
        if self._raise is not None:
            raise self._raise
        # Mimic real engine return shape: list of page placeholders
        return ["page"]


def test_wiki_consumer_calls_ingest_on_event() -> None:
    from iris_harness.memory.knowledge.event_subscribers import (
        WIKI_INGEST_REQUESTED,
        subscribe_wiki_ingest_consumer,
    )

    bus = EventBus()
    fake_wiki = _StubWikiEngine()
    subscribe_wiki_ingest_consumer(fake_wiki, bus=bus)

    event = WikiIngestEvent(
        source_agent=EMAIL_SOURCE_AGENT,
        source_id=f"email/{ACCOUNT}/m-1",
        content="Northwind: statement\n\nYour May statement is ready",
        entities_hint=["northwindbank.test", "finance", "banking", "northwind-savings"],
    )
    bus.emit_sync(WIKI_INGEST_REQUESTED, event)
    assert len(fake_wiki.calls) == 1
    assert fake_wiki.calls[0] is event


def test_wiki_consumer_soft_fails_on_ingest_exception() -> None:
    """A raising WikiEngine.ingest must not propagate — keeps the bus healthy."""
    from iris_harness.memory.knowledge.event_subscribers import (
        WIKI_INGEST_REQUESTED,
        subscribe_wiki_ingest_consumer,
    )

    bus = EventBus()
    fake_wiki = _StubWikiEngine(raise_on_ingest=RuntimeError("disk full"))
    subscribe_wiki_ingest_consumer(fake_wiki, bus=bus)

    event = WikiIngestEvent(source_agent="x", source_id="y", content="z" * 60)
    # Must not raise — the consumer absorbs exceptions
    bus.emit_sync(WIKI_INGEST_REQUESTED, event)
    assert len(fake_wiki.calls) == 1


def test_wiki_consumer_ignores_unrelated_payloads() -> None:
    from iris_harness.memory.knowledge.event_subscribers import (
        WIKI_INGEST_REQUESTED,
        subscribe_wiki_ingest_consumer,
    )

    bus = EventBus()
    fake_wiki = _StubWikiEngine()
    subscribe_wiki_ingest_consumer(fake_wiki, bus=bus)
    bus.emit_sync(WIKI_INGEST_REQUESTED, "not-an-event")
    bus.emit_sync(WIKI_INGEST_REQUESTED, {"id": "x"})
    assert fake_wiki.calls == []


def test_end_to_end_email_classified_to_wiki_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full bus path: email.classified → WIKI_INGEST_REQUESTED → WikiEngine.ingest."""
    import iris_personal.plugins.email_workflows.wiki_ingestion as wiki_ingestion_mod
    from iris_harness.memory.knowledge.event_subscribers import subscribe_wiki_ingest_consumer
    from iris_personal.plugins.email_workflows.wiki_ingestion import (
        subscribe_email_classified_to_wiki,
    )

    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(_make_email())

    bus = EventBus()
    monkeypatch.setattr(wiki_ingestion_mod, "get_default_bus", lambda: bus)
    monkeypatch.setattr(wiki_ingestion_mod, "EmailStore", lambda *a, **kw: store)

    fake_wiki = _StubWikiEngine()
    subscribe_email_classified_to_wiki(bus=bus)
    subscribe_wiki_ingest_consumer(fake_wiki, bus=bus)

    bus.emit_sync(
        EMAIL_CLASSIFIED,
        EmailClassifiedPayload(
            id="msg-1",
            account_id=ACCOUNT,
            category_path="email/finance/banking/northwind-savings",
            confidence=0.85,
        ),
    )

    # WikiEngine.ingest was called with the right event
    assert len(fake_wiki.calls) == 1
    event = fake_wiki.calls[0]
    assert isinstance(event, WikiIngestEvent)
    assert event.source_id == f"email/{ACCOUNT}/msg-1"
    assert "northwindbank.test" in event.entities_hint


def test_subscriber_skips_ghost_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the message isn't in email.db, no WikiIngestEvent is emitted."""
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()

    bus = EventBus()
    captured: list = []
    bus.on(WIKI_INGEST_REQUESTED, captured.append)

    import iris_personal.plugins.email_workflows.wiki_ingestion as wiki_ingestion

    monkeypatch.setattr(wiki_ingestion, "get_default_bus", lambda: bus)
    monkeypatch.setattr(wiki_ingestion, "EmailStore", lambda *a, **kw: store)

    subscribe_email_classified_to_wiki(bus=bus)

    bus.emit_sync(
        EMAIL_CLASSIFIED,
        EmailClassifiedPayload(
            id="ghost",
            account_id=ACCOUNT,
            category_path="email/finance/banking/northwind-savings",
            confidence=0.85,
        ),
    )
    assert captured == []


def test_wiki_consumer_skips_when_ingest_is_disabled() -> None:
    """A disabled wiki must not be called once per classified email (PR: off-switch)."""
    from iris_harness.memory.knowledge.event_subscribers import (
        WIKI_INGEST_REQUESTED,
        subscribe_wiki_ingest_consumer,
    )

    bus = EventBus()
    fake_wiki = _StubWikiEngine(ingest_enabled=False)
    subscribe_wiki_ingest_consumer(fake_wiki, bus=bus)

    bus.emit_sync(
        WIKI_INGEST_REQUESTED,
        WikiIngestEvent(source_agent="email-triage", source_id="email/acct/m-9", content="z" * 60),
    )

    assert fake_wiki.calls == []
