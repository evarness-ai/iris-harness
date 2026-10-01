"""EmailSemanticIndex (ADR-0071 slice 1).

A deterministic bag-of-words ``embed_fn`` is injected so retrieval is exercised
against a real local Chroma store WITHOUT loading MiniLM. The index stores only
vectors + ids + metadata — these tests assert ranking-by-meaning, the account /
since filters, idempotent upsert, and graceful no-op when unavailable.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iris_personal.email.contracts import EmailMessage
from iris_personal.email.semantic_index import (
    EmailSemanticIndex,
    backfill_semantic_index,
    hybrid_search,
    rrf_fuse,
)
from iris_personal.email.store import EmailStore

_VOCAB = ["payment", "due", "statement", "ai", "article", "amazon", "invoice", "travel"]


def _fake_embed(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for text in texts:
        low = text.lower()
        v = [float(low.count(w)) for w in _VOCAB]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


def _msg(
    msg_id: str, subject: str, snippet: str, *, account: str = "gmail:a", when=None
) -> EmailMessage:
    return EmailMessage(
        id=msg_id,
        provider="gmail",  # type: ignore[arg-type]
        account_id=account,
        thread_id=None,
        from_address="sender@example.com",
        subject=subject,
        snippet=snippet,
        received_at=when or datetime.now(UTC),
    )


@pytest.fixture
def index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EmailSemanticIndex:
    # conftest sets IRIS_TEST_NULL_EMBEDDINGS=1 to skip the MiniLM model; the
    # injected fake embedder needs no model, so clear it to exercise real Chroma.
    monkeypatch.delenv("IRIS_TEST_NULL_EMBEDDINGS", raising=False)
    return EmailSemanticIndex(persist_dir=tmp_path / "email_semantic", embed_fn=_fake_embed)


def test_search_ranks_by_meaning(index: EmailSemanticIndex) -> None:
    index.index_messages(
        [
            _msg("m1", "Your credit card payment is due", "minimum payment due soon"),
            _msg("m2", "Latest AI article", "a great article on transformers"),
            _msg("m3", "Amazon invoice", "your invoice is attached"),
        ]
    )
    assert index.count() == 3

    # "pending payment" never appears verbatim, but lands nearest the bill email.
    top = index.search("pending payment", k=3)
    assert top and top[0][0] == "m1"
    assert index.search("ai article", k=3)[0][0] == "m2"


def test_account_filter(index: EmailSemanticIndex) -> None:
    index.index_messages(
        [
            _msg("a1", "payment due", "payment", account="gmail:in"),
            _msg("b1", "payment due", "payment", account="gmail:us"),
        ]
    )
    ids = [mid for mid, _ in index.search("payment", k=10, account_id="gmail:us")]
    assert ids == ["b1"]


def test_since_filter(index: EmailSemanticIndex) -> None:
    now = datetime.now(UTC)
    index.index_messages(
        [
            _msg("old", "payment due", "payment", when=now - timedelta(days=400)),
            _msg("new", "payment due", "payment", when=now - timedelta(days=1)),
        ]
    )
    ids = [mid for mid, _ in index.search("payment", k=10, since=now - timedelta(days=30))]
    assert ids == ["new"]


def test_upsert_is_idempotent(index: EmailSemanticIndex) -> None:
    msgs = [_msg("m1", "payment due", "payment")]
    index.index_messages(msgs)
    index.index_messages(msgs)  # same id again
    assert index.count() == 1


def test_backfill_over_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_TEST_NULL_EMBEDDINGS", raising=False)
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    store.upsert(_msg("m1", "Your statement is due", "payment due", account="gmail:a"))
    store.upsert(_msg("m2", "Travel plans", "your travel itinerary", account="gmail:a"))
    index = EmailSemanticIndex(persist_dir=tmp_path / "vec", embed_fn=_fake_embed)

    n = backfill_semantic_index(email_store=store, index=index)
    assert n == 2 and index.count() == 2
    assert index.search("amount due", k=2)[0][0] == "m1"


# ── Hybrid retrieval (ADR-0071 slice 2) ─────────────────────────────────────


def test_rrf_fuse_rewards_agreement_and_rank() -> None:
    # "b" is rank-1 in both lists → tops the fused ranking; "a" is in both too, so it
    # beats "c"/"d" which each appear in only one list.
    fused = rrf_fuse([["b", "a", "c"], ["b", "a", "d"]])
    order = [doc for doc, _ in fused]
    assert order[0] == "b"
    assert set(order) == {"a", "b", "c", "d"}  # union, deduped
    assert order.index("a") < order.index("c")
    assert order.index("a") < order.index("d")


def test_rrf_fuse_handles_empty_legs() -> None:
    assert rrf_fuse([[], ["x", "y"]])[0][0] == "x"
    assert rrf_fuse([[], []]) == []


def test_hybrid_search_unions_lexical_and_semantic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IRIS_TEST_NULL_EMBEDDINGS", raising=False)
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    # m1: lexical match for "payment" (FTS5). m2: NOT lexical for "payment", but
    # semantically near (shares the "due/statement" vector space via the fake embedder).
    store.upsert(_msg("m1", "Your payment confirmation", "payment received", account="gmail:a"))
    store.upsert(_msg("m2", "Statement is due", "amount due this cycle", account="gmail:a"))
    index = EmailSemanticIndex(persist_dir=tmp_path / "vec", embed_fn=_fake_embed)
    backfill_semantic_index(email_store=store, index=index)

    ids = hybrid_search("payment due", email_store=store, semantic_index=index, k=10)
    # Lexical leg finds m1 ("payment"); semantic leg finds m2 ("due"/"statement").
    assert set(ids) == {"m1", "m2"}


def test_hybrid_search_empty_both_legs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRIS_TEST_NULL_EMBEDDINGS", raising=False)
    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    index = EmailSemanticIndex(persist_dir=tmp_path / "vec", embed_fn=_fake_embed)
    assert hybrid_search("anything", email_store=store, semantic_index=index, k=5) == []


def test_unavailable_index_is_a_safe_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # IRIS_TEST_NULL_EMBEDDINGS skips Chroma entirely → every method no-ops.
    monkeypatch.setenv("IRIS_TEST_NULL_EMBEDDINGS", "1")
    idx = EmailSemanticIndex(persist_dir=tmp_path / "vec", embed_fn=_fake_embed)
    assert not idx.is_ready
    assert idx.index_messages([_msg("m1", "payment", "due")]) == 0
    assert idx.search("payment") == []
    assert idx.count() == 0


# ── ADR-0121 PR 4: on by default, scoped by sender domain, refreshed in batches ──


def _from(msg_id: str, sender: str, subject: str, *, days_ago: int = 0) -> EmailMessage:
    domain = sender.split("@", 1)[1]
    return EmailMessage(
        id=msg_id,
        provider="gmail",  # type: ignore[arg-type]
        account_id="gmail:a",
        thread_id=None,
        from_address=sender,
        from_domain=domain,
        subject=subject,
        snippet=subject,
        received_at=datetime.now(UTC) - timedelta(days=days_ago),
    )


def test_a_search_can_be_scoped_to_an_institutions_domains(index: EmailSemanticIndex) -> None:
    index.index_messages(
        [
            _from("d1", "discover@services.discover.com", "statement payment due"),
            _from("c1", "alerts@chase.com", "statement payment due"),
            _from("s1", "statements@alerts.woodgrovebank.test", "statement"),
        ]
    )
    # "discover.com" matches a subdomain sender; the Chase mail with the same words does not.
    assert [i for i, _ in index.search("statement payment", domains=["discover.com"])] == ["d1"]
    assert [i for i, _ in index.search("statement", domains=["woodgrovebank.test"])] == ["s1"]
    assert {i for i, _ in index.search("statement payment")} == {"d1", "c1", "s1"}


def test_refresh_fills_in_missing_and_old_version_vectors_in_batches(
    tmp_path: Path, index: EmailSemanticIndex
) -> None:
    from iris_personal.email import semantic_index as si

    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    for i in range(5):
        store.upsert(_from(f"m{i}", "alerts@chase.com", "statement", days_ago=i))
    # m0 was indexed before domains were stored (version 1): the refresh redoes it.
    index._col.upsert(ids=["m0"], embeddings=_fake_embed(["statement"]), metadatas=[{"ts": 0.0}])

    first = si.refresh_semantic_index(email_store=store, index=index, max_per_run=3)
    assert (first.indexed, first.left) == (3, 2)
    second = si.refresh_semantic_index(email_store=store, index=index, max_per_run=3)
    assert (second.indexed, second.left) == (2, 0)
    assert si.refresh_semantic_index(email_store=store, index=index).indexed == 0
    assert index.stale_ids([f"m{i}" for i in range(5)]) == []


@pytest.mark.parametrize(
    ("value", "on"), [(None, True), ("1", True), ("0", False), ("false", False), ("no", False)]
)
def test_semantic_search_is_on_unless_turned_off(
    monkeypatch: pytest.MonkeyPatch, value: str | None, on: bool
) -> None:
    from iris_personal.email.semantic_index import semantic_search_enabled

    if value is None:
        monkeypatch.delenv("IRIS_EMAIL_SEMANTIC_SEARCH", raising=False)
    else:
        monkeypatch.setenv("IRIS_EMAIL_SEMANTIC_SEARCH", value)
    assert semantic_search_enabled() is on


def test_the_refresh_heartbeat_skips_when_off_and_fails_soft_without_a_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from iris_harness.services.heartbeat.models import HeartbeatDefinition, HeartbeatStatus
    from iris_personal.plugins.email_workflows.semantic_heartbeat import (
        build_semantic_index_handler,
    )

    handler = build_semantic_index_handler(
        SimpleNamespace(services=SimpleNamespace(data_dir=tmp_path))
    )
    definition = HeartbeatDefinition(
        name="email_semantic_index", handler="email_semantic_index", schedule="interval:1800"
    )
    monkeypatch.setenv("IRIS_EMAIL_SEMANTIC_SEARCH", "0")
    assert handler(definition).status == HeartbeatStatus.SKIPPED

    monkeypatch.setenv("IRIS_EMAIL_SEMANTIC_SEARCH", "1")
    monkeypatch.setenv("IRIS_TEST_NULL_EMBEDDINGS", "1")  # no model: the index is not ready
    run = handler(definition)
    assert run.status == HeartbeatStatus.FAILED and "unavailable" in (run.error or "")


def test_the_refresh_stops_for_a_conversation(tmp_path: Path, index: EmailSemanticIndex) -> None:
    from iris_harness.foundation.activity import chat_turn
    from iris_personal.email import semantic_index as si

    store = EmailStore(db_path=tmp_path / "email.db")
    store.ensure_schema()
    for i in range(3):
        store.upsert(_from(f"m{i}", "alerts@chase.com", "statement", days_ago=i))
    with chat_turn():
        run = si.refresh_semantic_index(email_store=store, index=index)
    assert (run.indexed, run.left, run.paused) == (0, 3, True)
