"""Tests for the email-triage classifier (Track 1G / ADR-0021)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from iris_harness.foundation.eventbus import EventBus
from iris_personal.email.category_store import Category, CategoryStore
from iris_personal.email.contracts import EmailMessage
from iris_personal.email.events import EMAIL_CLASSIFIED, EmailClassifiedPayload
from iris_personal.email.store import EmailStore
from iris_personal.plugins.email_workflows.triage import (
    CLASSIFIER_TAG,
    FALLBACK_CONFIDENCE,
    LLM_PICKER_RELIABILITY_PRIOR,
    EmailTriageClassifier,
    build_centroids,
)

# ─── Fixtures ───────────────────────────────────────────────────────────────


ACCOUNT = "gmail:user@gmail.com"
ACCOUNT_SLUG = "gmail-user-at-gmail.com"


def _now() -> datetime:
    return datetime.now(UTC)


def _make_email(
    id: str = "msg-1",
    *,
    account_id: str = ACCOUNT,
    from_address: str = "Bob <bob@gap.com>",
    from_domain: str | None = "gap.com",
    subject: str = "60% off summer sale",
    snippet: str = "...",
) -> EmailMessage:
    return EmailMessage(
        id=id,
        provider="gmail",
        account_id=account_id,
        from_address=from_address,
        from_domain=from_domain,
        subject=subject,
        received_at=_now(),
        snippet=snippet,
    )


def _make_proposal(
    cluster_id: int,
    root: str,
    branch: str,
    leaf: str,
    *,
    cohesion: float = 0.85,
    rep_domain: str = "gap.com",
) -> dict:
    return {
        "cluster_id": cluster_id,
        "size": 10,
        "cohesion": cohesion,
        "top_domains": [[rep_domain, 10]],
        "representatives": [
            {
                "id": f"c{cluster_id}-rep-{i}",
                "subject": f"subject {cluster_id} {i}",
                "from": f"sender@{rep_domain}",
                "snippet": f"snip {cluster_id} {i}",
            }
            for i in range(3)
        ],
        "member_ids": [f"c{cluster_id}-m-{i}" for i in range(10)],
        "proposed_root": root,
        "proposed_branch": branch,
        "proposed_leaf": leaf,
        "naming_rationale": "stubbed",
    }


def _seed_workspace(workspace: Path, proposals: list[dict]) -> Path:
    """Write proposals.jsonl into the conventional workspace path."""
    path = workspace / "email" / ACCOUNT_SLUG / "proposals.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in proposals:
            f.write(json.dumps(p) + "\n")
    return path


def _seed_categories(db_path: Path, paths_with_cohesion: list[tuple[str, float]]) -> None:
    store = CategoryStore(db_path=db_path)
    store.ensure_schema()
    for path, cohesion in paths_with_cohesion:
        type_, root, branch, leaf = path.split("/")
        store.upsert_if_new(
            Category(
                path=path,
                type=type_,
                root=root,
                branch=branch,
                leaf=leaf,
                account_id=ACCOUNT,
                cohesion=cohesion,
            )
        )


class _StubNamingClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._responses.pop(0)


def _stub_embedder_factory(domain_to_vec: dict[str, np.ndarray]):  # type: ignore[no-untyped-def]
    """Each text is keyed by which domain substring it mentions."""

    def _stub(texts: list[str], model_name: str = "stub") -> np.ndarray:
        out: list[np.ndarray] = []
        for t in texts:
            chosen = None
            for k, v in domain_to_vec.items():
                if k in t:
                    chosen = v
                    break
            if chosen is None:
                chosen = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            out.append(chosen.astype(np.float32))
        arr = np.stack(out)
        # L2-normalize like the real embedder does
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)

    return _stub


# ─── build_centroids ────────────────────────────────────────────────────────


def test_build_centroids_skips_categories_not_in_active_set(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    _seed_workspace(
        workspace,
        [
            _make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com"),
            _make_proposal(1, "social", "facebook", "updates", rep_domain="facebookmail.com"),
        ],
    )

    from iris_personal.plugins.email_workflows import triage

    embedder = _stub_embedder_factory(
        {
            "gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "facebookmail.com": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        }
    )
    # Inject embedder by patching at module scope
    original = triage.embed_corpus
    triage.embed_corpus = embedder  # type: ignore[assignment]
    try:
        centroids = build_centroids(
            workspace,
            ACCOUNT,
            active_paths={"email/shopping/apparel/gap"},  # only one accepted
        )
    finally:
        triage.embed_corpus = original

    assert len(centroids) == 1
    assert centroids[0].path == "email/shopping/apparel/gap"


def test_build_centroids_missing_jsonl_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="proposals.jsonl missing"):
        build_centroids(tmp_path / "ws", ACCOUNT, active_paths={"email/x/y/z"})


# ─── EmailTriageClassifier.classify ─────────────────────────────────────────


def _make_classifier(
    tmp_path: Path,
    *,
    naming_responses: list[str],
    domain_vecs: dict[str, np.ndarray] | None = None,
) -> tuple[EmailTriageClassifier, EventBus, list]:
    workspace = tmp_path / "ws"
    db = tmp_path / "iris.db"
    email_db = tmp_path / "email.db"

    bus = EventBus()
    captured: list = []
    bus.on(EMAIL_CLASSIFIED, captured.append)

    classifier = EmailTriageClassifier(
        workspace_dir=workspace,
        db_path=db,
        email_db_path=email_db,
        naming_client=_StubNamingClient(naming_responses),
        bus=bus,
        embedder=_stub_embedder_factory(
            domain_vecs
            or {
                "gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "facebookmail.com": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "shopmart.com": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            }
        ),
    )
    return classifier, bus, captured


def test_classify_with_llm_happy_path(tmp_path: Path) -> None:
    """classify_with_llm is the explicit hybrid path used by the batch CLI."""
    _seed_workspace(
        tmp_path / "ws",
        [
            _make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com"),
            _make_proposal(1, "social", "facebook", "updates", rep_domain="facebookmail.com"),
            _make_proposal(2, "shopping", "apparel", "shopmart", rep_domain="shopmart.com"),
        ],
    )
    _seed_categories(
        tmp_path / "iris.db",
        [
            ("email/shopping/apparel/gap", 0.85),
            ("email/social/facebook/updates", 0.70),
            ("email/shopping/apparel/shopmart", 0.78),
        ],
    )

    classifier, _, _ = _make_classifier(
        tmp_path,
        naming_responses=[json.dumps({"chosen_path": "email/shopping/apparel/gap"})],
    )
    email = _make_email(from_address="Bob <bob@gap.com>", from_domain="gap.com")
    result = classifier.classify_with_llm(email)

    assert result.category_path == "email/shopping/apparel/gap"
    # cohesion 0.85 × prior 0.9
    assert result.confidence == pytest.approx(0.85 * LLM_PICKER_RELIABILITY_PRIOR)
    assert result.error is None
    assert result.classifier == "tier3-local-knn"


def test_classify_returns_top_3_to_picker(tmp_path: Path) -> None:
    """The LLM must see exactly 3 candidates regardless of category count."""
    _seed_workspace(
        tmp_path / "ws",
        [
            _make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com"),
            _make_proposal(1, "social", "facebook", "updates", rep_domain="facebookmail.com"),
            _make_proposal(2, "shopping", "apparel", "shopmart", rep_domain="shopmart.com"),
            _make_proposal(3, "finance", "investing", "bonds", rep_domain="bonds.example.com"),
        ],
    )
    _seed_categories(
        tmp_path / "iris.db",
        [
            ("email/shopping/apparel/gap", 0.85),
            ("email/social/facebook/updates", 0.70),
            ("email/shopping/apparel/shopmart", 0.78),
            ("email/finance/investing/bonds", 0.82),
        ],
    )

    naming_client = _StubNamingClient([json.dumps({"chosen_path": "email/shopping/apparel/gap"})])
    classifier = EmailTriageClassifier(
        workspace_dir=tmp_path / "ws",
        db_path=tmp_path / "iris.db",
        email_db_path=tmp_path / "email.db",
        naming_client=naming_client,
        embedder=_stub_embedder_factory(
            {
                "gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "facebookmail.com": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "shopmart.com": np.array([0.9, 0.1, 0.0], dtype=np.float32),
                "bonds.example.com": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            }
        ),
    )
    classifier.classify_with_llm(
        _make_email(from_address="Bob <bob@gap.com>", from_domain="gap.com")
    )

    assert len(naming_client.calls) == 1
    user_prompt = naming_client.calls[0][1]
    # Three numbered candidates appear; the fourth (bonds) is filtered out
    assert "1." in user_prompt and "2." in user_prompt and "3." in user_prompt
    assert "4." not in user_prompt


def test_classify_falls_back_when_picker_returns_off_menu(tmp_path: Path) -> None:
    """If the LLM picks a path not in the candidate set → fallback to kNN top-1."""
    _seed_workspace(
        tmp_path / "ws",
        [_make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com")],
    )
    _seed_categories(tmp_path / "iris.db", [("email/shopping/apparel/gap", 0.85)])

    classifier, _, _ = _make_classifier(
        tmp_path,
        naming_responses=[json.dumps({"chosen_path": "email/somewhere/else/entirely"})],
    )
    result = classifier.classify_with_llm(
        _make_email(from_address="x@gap.com", from_domain="gap.com")
    )

    assert result.category_path == "email/shopping/apparel/gap"  # kNN top-1
    assert result.confidence == FALLBACK_CONFIDENCE


def test_classify_soft_fails_when_categories_missing(tmp_path: Path) -> None:
    """No categories for the account → soft-fail with descriptive error."""
    # Don't seed categories or workspace
    classifier, _, _ = _make_classifier(tmp_path, naming_responses=[])
    result = classifier.classify(_make_email())
    assert result.category_path is None
    assert result.confidence is None
    assert result.error is not None


def test_classify_soft_fails_on_llm_error(tmp_path: Path) -> None:
    """A raising LLM client → soft-fail, no crash."""
    _seed_workspace(
        tmp_path / "ws",
        [_make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com")],
    )
    _seed_categories(tmp_path / "iris.db", [("email/shopping/apparel/gap", 0.85)])

    class _BrokenClient:
        def complete_json(self, system: str, user: str) -> str:
            raise RuntimeError("llama-server unreachable")

    classifier = EmailTriageClassifier(
        workspace_dir=tmp_path / "ws",
        db_path=tmp_path / "iris.db",
        email_db_path=tmp_path / "email.db",
        naming_client=_BrokenClient(),
        embedder=_stub_embedder_factory({"gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32)}),
    )
    result = classifier.classify_with_llm(
        _make_email(from_address="x@gap.com", from_domain="gap.com")
    )
    assert result.category_path is None
    assert "llama-server unreachable" in (result.error or "")


# ─── classify_and_persist + event emission ──────────────────────────────────


def test_classify_and_persist_writes_to_email_db_and_emits_event(tmp_path: Path) -> None:
    _seed_workspace(
        tmp_path / "ws",
        [_make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com")],
    )
    _seed_categories(tmp_path / "iris.db", [("email/shopping/apparel/gap", 0.85)])

    classifier, bus, captured = _make_classifier(
        tmp_path,
        naming_responses=[json.dumps({"chosen_path": "email/shopping/apparel/gap"})],
    )

    # Seed the email in email.db so mark_classified can find it
    email_store = EmailStore(db_path=tmp_path / "email.db")
    email_store.ensure_schema()
    email = _make_email(from_address="Bob <bob@gap.com>", from_domain="gap.com")
    email_store.upsert(email)

    # use_llm=True routes through classify_with_llm so the existing
    # LLM-flavored event payload is the one asserted below.
    result = classifier.classify_and_persist(email, store=email_store, use_llm=True)
    assert result.category_path == "email/shopping/apparel/gap"

    # Email db row now has classification stamped
    refetched = email_store.get(email.id)
    assert refetched is not None
    # We need to inspect raw SQL since EmailMessage doesn't expose classified_* yet
    import sqlite3

    with sqlite3.connect(email_store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT classified_category, classified_confidence, classified_at, triage_state "
            "FROM emails WHERE id = ?",
            (email.id,),
        ).fetchone()
    assert row["classified_category"] == "email/shopping/apparel/gap"
    assert row["classified_confidence"] is not None
    assert row["classified_at"] is not None
    assert row["triage_state"] == "classified"  # ADR-0022 state

    # email.classified event fired exactly once with correct payload
    assert len(captured) == 1
    payload = captured[0]
    assert isinstance(payload, EmailClassifiedPayload)
    assert payload.id == email.id
    assert payload.category_path == "email/shopping/apparel/gap"
    assert payload.classifier == CLASSIFIER_TAG


def test_classify_and_persist_skips_persist_on_classification_error(tmp_path: Path) -> None:
    """If classify() returns no path (soft-fail), don't write to DB or emit event."""
    classifier, _, captured = _make_classifier(tmp_path, naming_responses=[])
    email = _make_email()
    result = classifier.classify_and_persist(email)
    assert result.category_path is None
    assert captured == []


def test_classify_unclassified_processes_batch(tmp_path: Path) -> None:
    _seed_workspace(
        tmp_path / "ws",
        [_make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com")],
    )
    _seed_categories(tmp_path / "iris.db", [("email/shopping/apparel/gap", 0.85)])

    email_store = EmailStore(db_path=tmp_path / "email.db")
    email_store.ensure_schema()
    emails = [
        _make_email(id=f"m-{i}", from_address=f"u{i}@gap.com", from_domain="gap.com")
        for i in range(3)
    ]
    for e in emails:
        email_store.upsert(e)

    classifier = EmailTriageClassifier(
        workspace_dir=tmp_path / "ws",
        db_path=tmp_path / "iris.db",
        email_db_path=tmp_path / "email.db",
        naming_client=_StubNamingClient(
            [json.dumps({"chosen_path": "email/shopping/apparel/gap"})] * 3
        ),
        bus=EventBus(),
        embedder=_stub_embedder_factory({"gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32)}),
    )
    results = classifier.classify_unclassified(ACCOUNT, limit=10, store=email_store)
    assert len(results) == 3
    assert all(r.category_path == "email/shopping/apparel/gap" for r in results)
    # Re-running should now find 0 unclassified
    again = classifier.classify_unclassified(ACCOUNT, limit=10, store=email_store)
    assert again == []


# ─── Lazy singleton + event subscription ────────────────────────────────────


def test_lazy_singleton_caches_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    from iris_personal.plugins.email_workflows import triage

    triage.reset_lazy_classifier()

    construction_count = [0]

    real_factory = triage.build_email_triage_classifier

    def counting_factory() -> EmailTriageClassifier:
        construction_count[0] += 1
        return real_factory()

    monkeypatch.setattr(triage, "build_email_triage_classifier", counting_factory)

    triage._get_lazy_classifier()
    triage._get_lazy_classifier()
    triage._get_lazy_classifier()
    # Only one construction across three calls
    assert construction_count[0] == 1
    triage.reset_lazy_classifier()


def test_subscribe_email_triage_wires_handler() -> None:
    """subscribe_email_triage should add a listener that fires on event dispatch."""
    from iris_personal.email.events import EMAIL_NEW_ARRIVED, EmailNewArrivedPayload
    from iris_personal.plugins.email_workflows import triage

    triage.reset_lazy_classifier()

    bus = EventBus()
    triage.subscribe_email_triage(bus=bus)

    # Stub the lazy classifier so we can observe the dispatch path
    handler_calls: list[str] = []

    class _StubClassifier:
        email_db_path = Path("/tmp/does-not-matter.db")

        def classify_and_persist(self, message, *, store):  # type: ignore[no-untyped-def]
            handler_calls.append(message.id)

    # Force the stub into the slot; reset afterwards
    triage._LAZY_CLASSIFIER = _StubClassifier()  # type: ignore[assignment]

    # Stub EmailStore.get to return synthetic messages
    class _StubStore:
        def ensure_schema(self) -> None:
            pass

        def get(self, msg_id: str):  # type: ignore[no-untyped-def]
            return _make_email(id=msg_id) if msg_id != "ghost" else None

    import iris_personal.plugins.email_workflows.triage as triage_mod

    original_store = triage_mod.EmailStore
    triage_mod.EmailStore = lambda *a, **kw: _StubStore()  # type: ignore[assignment]
    try:
        bus.emit_sync(
            EMAIL_NEW_ARRIVED,
            EmailNewArrivedPayload(
                account_id=ACCOUNT,
                new_message_ids=("m-1", "ghost", "m-2"),
                count=3,
                fell_back_to_cold_start=False,
            ),
        )
    finally:
        triage_mod.EmailStore = original_store  # type: ignore[assignment]
        triage.reset_lazy_classifier()

    # Ghost id was skipped; m-1 and m-2 were classified
    assert handler_calls == ["m-1", "m-2"]


def test_subscribe_email_triage_ignores_unrelated_payloads() -> None:
    """A bare string or unrelated dataclass on the topic must not crash."""
    from iris_personal.email.events import EMAIL_NEW_ARRIVED
    from iris_personal.plugins.email_workflows import triage

    triage.reset_lazy_classifier()

    bus = EventBus()
    triage.subscribe_email_triage(bus=bus)
    # Force LAZY to a sentinel so we'd notice if it ran
    triage._LAZY_CLASSIFIER = "should-not-be-called"  # type: ignore[assignment]
    try:
        bus.emit_sync(EMAIL_NEW_ARRIVED, "not-a-payload")
        bus.emit_sync(EMAIL_NEW_ARRIVED, {"id": "x"})
        # No exception is the assertion
    finally:
        triage.reset_lazy_classifier()


def test_subscriber_dispatch_makes_zero_llm_calls(tmp_path: Path) -> None:
    """ADR-0022 guarantee: an email.new_arrived dispatch goes through the
    pure-kNN path. The LLM client is never touched by the subscriber."""
    from iris_personal.email.events import EMAIL_NEW_ARRIVED, EmailNewArrivedPayload
    from iris_personal.plugins.email_workflows import triage

    triage.reset_lazy_classifier()

    _seed_workspace(
        tmp_path / "ws",
        [_make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com")],
    )
    _seed_categories(tmp_path / "iris.db", [("email/shopping/apparel/gap", 0.85)])

    email_store = EmailStore(db_path=tmp_path / "email.db")
    email_store.ensure_schema()
    email = _make_email(id="new-1", from_address="x@gap.com", from_domain="gap.com")
    email_store.upsert(email)

    # Stub LAZY classifier with a real one that has a tripwire LLM client
    naming_client = _StubNamingClient([])  # zero responses prepared — any call would error
    triage._LAZY_CLASSIFIER = EmailTriageClassifier(
        workspace_dir=tmp_path / "ws",
        db_path=tmp_path / "iris.db",
        email_db_path=tmp_path / "email.db",
        naming_client=naming_client,
        bus=EventBus(),
        embedder=_stub_embedder_factory({"gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32)}),
    )

    bus = EventBus()
    triage.subscribe_email_triage(bus=bus)
    try:
        bus.emit_sync(
            EMAIL_NEW_ARRIVED,
            EmailNewArrivedPayload(
                account_id=ACCOUNT,
                new_message_ids=("new-1",),
                count=1,
                fell_back_to_cold_start=False,
            ),
        )
    finally:
        triage.reset_lazy_classifier()

    # The classifier's naming_client must not have been touched
    assert naming_client.calls == []
    # And the row got classified via pure-kNN
    import sqlite3

    with sqlite3.connect(email_store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT classified_category, triage_state FROM emails WHERE id = ?",
            (email.id,),
        ).fetchone()
    assert row["classified_category"] == "email/shopping/apparel/gap"
    assert row["triage_state"] == "classified"


def test_load_categories_is_idempotent(tmp_path: Path) -> None:
    _seed_workspace(
        tmp_path / "ws",
        [_make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com")],
    )
    _seed_categories(tmp_path / "iris.db", [("email/shopping/apparel/gap", 0.85)])

    embed_call_count = [0]

    def counting_embedder(texts: list[str], model_name: str = "stub") -> np.ndarray:
        embed_call_count[0] += 1
        n = len(texts)
        v = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n, 1))
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(norms, 1e-12)

    classifier = EmailTriageClassifier(
        workspace_dir=tmp_path / "ws",
        db_path=tmp_path / "iris.db",
        email_db_path=tmp_path / "email.db",
        naming_client=_StubNamingClient([]),
        embedder=counting_embedder,
    )
    classifier.load_categories(ACCOUNT)
    first_count = embed_call_count[0]
    classifier.load_categories(ACCOUNT)
    # Second call cached → no additional embedder invocations
    assert embed_call_count[0] == first_count


# ─── Pure-kNN classifier + confidence gate (ADR-0022) ───────────────────────


def test_classify_pure_knn_gates_classify_when_confident(tmp_path: Path) -> None:
    """High top-1 similarity AND a clear margin → classified with confidence
    equal to the top-1 cosine."""
    _seed_workspace(
        tmp_path / "ws",
        [
            _make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com"),
            _make_proposal(1, "social", "facebook", "updates", rep_domain="facebookmail.com"),
        ],
    )
    _seed_categories(
        tmp_path / "iris.db",
        [
            ("email/shopping/apparel/gap", 0.85),
            ("email/social/facebook/updates", 0.70),
        ],
    )

    classifier, _, _ = _make_classifier(tmp_path, naming_responses=[])  # no LLM stubs needed
    # gap.com email lines up exactly with gap centroid → cos_top1 ≈ 1.0,
    # margin huge.
    email = _make_email(from_address="bob@gap.com", from_domain="gap.com")
    result = classifier.classify_pure_knn(email)

    assert result.category_path == "email/shopping/apparel/gap"
    assert result.confidence is not None and result.confidence >= 0.9
    assert result.queued is False
    assert result.classifier == "pure-knn"


def test_classify_pure_knn_queues_when_ambiguous(tmp_path: Path) -> None:
    """Two centroids almost equally close → margin too small → queued."""
    from iris_personal.plugins.email_workflows.triage import KNN_GATE_MIN_MARGIN

    _seed_workspace(
        tmp_path / "ws",
        [
            _make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com"),
            _make_proposal(1, "news", "tech", "shopmart-news", rep_domain="shopmart.com"),
        ],
    )
    _seed_categories(
        tmp_path / "iris.db",
        [
            ("email/shopping/apparel/gap", 0.85),
            ("email/news/tech/shopmart-news", 0.85),
        ],
    )

    # The email vector is exactly between the two centroids → margin ≈ 0
    ambiguous_vec = np.array([0.7071, 0.7071, 0.0], dtype=np.float32)
    classifier = EmailTriageClassifier(
        workspace_dir=tmp_path / "ws",
        db_path=tmp_path / "iris.db",
        email_db_path=tmp_path / "email.db",
        naming_client=_StubNamingClient([]),
        embedder=_stub_embedder_factory(
            {
                "gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "shopmart.com": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                # Sentinel string for the incoming email's embedding text
                "AMBIG_EMAIL": ambiguous_vec,
            }
        ),
    )

    email = _make_email(
        from_address="z@AMBIG_EMAIL.example",
        from_domain="AMBIG_EMAIL.example",
        subject="generic apparel email",
    )
    result = classifier.classify_pure_knn(email)
    assert result.queued is True
    assert result.category_path is None
    assert result.confidence is None
    assert "margin=" in (result.error or "")
    # Sanity: the recorded margin is below the gate
    assert "margin=0" in (result.error or "")
    del KNN_GATE_MIN_MARGIN  # silence unused import


def test_classify_pure_knn_queues_when_top1_below_threshold(tmp_path: Path) -> None:
    """Even with a clear margin, low absolute similarity → queued."""
    _seed_workspace(
        tmp_path / "ws",
        [_make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com")],
    )
    _seed_categories(tmp_path / "iris.db", [("email/shopping/apparel/gap", 0.85)])

    # Email vector is mostly orthogonal to the gap centroid → cos_top1 < 0.7
    weak_vec = np.array([0.3, 0.3, 0.9], dtype=np.float32)
    classifier = EmailTriageClassifier(
        workspace_dir=tmp_path / "ws",
        db_path=tmp_path / "iris.db",
        email_db_path=tmp_path / "email.db",
        naming_client=_StubNamingClient([]),
        embedder=_stub_embedder_factory(
            {
                "gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "WEAK_EMAIL": weak_vec,
            }
        ),
    )
    email = _make_email(from_address="z@WEAK_EMAIL.example", from_domain="WEAK_EMAIL.example")
    result = classifier.classify_pure_knn(email)
    assert result.queued is True
    assert "cos1=" in (result.error or "")


def test_classify_default_uses_pure_knn(tmp_path: Path) -> None:
    """classify() defaults to pure-kNN per ADR-0022."""
    _seed_workspace(
        tmp_path / "ws",
        [_make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com")],
    )
    _seed_categories(tmp_path / "iris.db", [("email/shopping/apparel/gap", 0.85)])

    naming_client = _StubNamingClient([])  # no responses prepared
    classifier = EmailTriageClassifier(
        workspace_dir=tmp_path / "ws",
        db_path=tmp_path / "iris.db",
        email_db_path=tmp_path / "email.db",
        naming_client=naming_client,
        embedder=_stub_embedder_factory({"gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32)}),
    )
    email = _make_email(from_address="bob@gap.com", from_domain="gap.com")
    classifier.classify(email)
    # NO LLM call happened
    assert naming_client.calls == []


def test_classify_and_persist_queued_marks_pending_review(tmp_path: Path) -> None:
    """A queued result → mark_pending_review, NO classified_category set,
    NO event emitted."""
    _seed_workspace(
        tmp_path / "ws",
        [
            _make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com"),
            _make_proposal(1, "news", "tech", "shopmart-news", rep_domain="shopmart.com"),
        ],
    )
    _seed_categories(
        tmp_path / "iris.db",
        [
            ("email/shopping/apparel/gap", 0.85),
            ("email/news/tech/shopmart-news", 0.85),
        ],
    )

    email_store = EmailStore(db_path=tmp_path / "email.db")
    email_store.ensure_schema()
    email = _make_email(from_address="x@AMBIG.example", from_domain="AMBIG.example")
    email_store.upsert(email)

    bus = EventBus()
    captured: list = []
    bus.on(EMAIL_CLASSIFIED, captured.append)

    classifier = EmailTriageClassifier(
        workspace_dir=tmp_path / "ws",
        db_path=tmp_path / "iris.db",
        email_db_path=tmp_path / "email.db",
        naming_client=_StubNamingClient([]),
        bus=bus,
        embedder=_stub_embedder_factory(
            {
                "gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "shopmart.com": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "AMBIG.example": np.array([0.7071, 0.7071, 0.0], dtype=np.float32),
            }
        ),
    )
    result = classifier.classify_and_persist(email, store=email_store)
    assert result.queued is True

    # Row is queued, not classified
    import sqlite3

    with sqlite3.connect(email_store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT classified_category, triage_state FROM emails WHERE id = ?",
            (email.id,),
        ).fetchone()
    assert row["classified_category"] is None
    assert row["triage_state"] == "pending_review"

    # No event emitted
    assert captured == []


def test_classify_pending_review_batch_drains_queue(tmp_path: Path) -> None:
    """triage-batch path runs LLM picker over the pending queue and
    classifies them, removing them from the queue."""
    _seed_workspace(
        tmp_path / "ws",
        [
            _make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com"),
            _make_proposal(1, "shopping", "apparel", "shopmart", rep_domain="shopmart.com"),
        ],
    )
    _seed_categories(
        tmp_path / "iris.db",
        [
            ("email/shopping/apparel/gap", 0.85),
            ("email/shopping/apparel/shopmart", 0.85),
        ],
    )

    email_store = EmailStore(db_path=tmp_path / "email.db")
    email_store.ensure_schema()
    # Two queued emails
    for i in range(2):
        e = _make_email(
            id=f"queued-{i}",
            from_address=f"x{i}@AMBIG.example",
            from_domain="AMBIG.example",
        )
        email_store.upsert(e)
        email_store.mark_pending_review(e.id)

    classifier = EmailTriageClassifier(
        workspace_dir=tmp_path / "ws",
        db_path=tmp_path / "iris.db",
        email_db_path=tmp_path / "email.db",
        naming_client=_StubNamingClient(
            [json.dumps({"chosen_path": "email/shopping/apparel/gap"})] * 2
        ),
        bus=EventBus(),
        embedder=_stub_embedder_factory(
            {
                "gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "shopmart.com": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "AMBIG.example": np.array([0.7071, 0.7071, 0.0], dtype=np.float32),
            }
        ),
    )
    results = classifier.classify_pending_review_batch(ACCOUNT, store=email_store)
    assert len(results) == 2
    assert all(r.category_path == "email/shopping/apparel/gap" for r in results)

    # Queue is now empty
    assert email_store.list_pending_review(ACCOUNT) == []


def test_classify_pure_knn_same_root_ambiguity_still_classifies(tmp_path: Path) -> None:
    """ADR-0022 amendment (Phase 3 category expansion): ambiguity between
    sibling leaves of the SAME root must not abstain — within-root ties are
    harmless for filing. Measured live: leaf-level margins crushed coverage
    46% -> 20% when the taxonomy grew 47 -> 83 categories."""
    _seed_workspace(
        tmp_path / "ws",
        [
            _make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com"),
            _make_proposal(1, "shopping", "apparel", "shopmart", rep_domain="shopmart.com"),
        ],
    )
    _seed_categories(
        tmp_path / "iris.db",
        [
            ("email/shopping/apparel/gap", 0.85),
            ("email/shopping/apparel/shopmart", 0.85),
        ],
    )

    classifier = EmailTriageClassifier(
        workspace_dir=tmp_path / "ws",
        db_path=tmp_path / "iris.db",
        email_db_path=tmp_path / "email.db",
        embedder=_stub_embedder_factory(
            {
                "gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "shopmart.com": np.array([0.0, 1.0, 0.0], dtype=np.float32),
                "AMBIG.example": np.array([0.7071, 0.7071, 0.0], dtype=np.float32),
            }
        ),
    )
    email = _make_email(from_address="x@AMBIG.example", from_domain="AMBIG.example")
    result = classifier.classify_pure_knn(email)

    assert result.queued is False
    assert result.category_path in (
        "email/shopping/apparel/gap",
        "email/shopping/apparel/shopmart",
    )


def test_correction_exemplar_makes_one_off_sender_classifiable(tmp_path: Path) -> None:
    """ADR-0024 extension: a single recategorize correction becomes an
    exemplar centroid, so the corrected sender's future mail classifies —
    the only lever that reaches unclustered one-off mail."""
    import json as _json
    import sqlite3

    _seed_workspace(
        tmp_path / "ws",
        [_make_proposal(0, "shopping", "apparel", "gap", rep_domain="gap.com")],
    )
    _seed_categories(
        tmp_path / "iris.db",
        [
            ("email/shopping/apparel/gap", 0.85),
            ("email/news/tech/superhuman", 0.85),
        ],
    )

    email_store = EmailStore(db_path=tmp_path / "email.db")
    email_store.ensure_schema()
    corrected = _make_email(
        "corrected-1",
        from_address="news@superhuman.example",
        from_domain="superhuman.example",
    )
    email_store.upsert(corrected)

    with sqlite3.connect(tmp_path / "iris.db") as conn:
        conn.execute(
            "INSERT INTO categories_history "
            "(category_path, op, old_path, new_path, payload, edited_at, source) "
            "VALUES (?, 'update', ?, ?, ?, datetime('now'), "
            "'user-classification-correction')",
            (
                "email/news/tech/superhuman",
                "email/shopping/apparel/gap",
                "email/news/tech/superhuman",
                _json.dumps(
                    {
                        "message_id": "corrected-1",
                        "account_id": ACCOUNT,
                        "new_path": "email/news/tech/superhuman",
                    }
                ),
            ),
        )
        conn.commit()

    classifier = EmailTriageClassifier(
        workspace_dir=tmp_path / "ws",
        db_path=tmp_path / "iris.db",
        email_db_path=tmp_path / "email.db",
        embedder=_stub_embedder_factory(
            {
                "gap.com": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "superhuman.example": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            }
        ),
    )

    centroids = classifier.load_categories(ACCOUNT)
    assert any(c.path == "email/news/tech/superhuman" for c in centroids)

    # A NEW message from the corrected sender now classifies to the
    # corrected path (before the exemplar it had no nearby centroid).
    fresh = _make_email(
        "fresh-1",
        from_address="digest@superhuman.example",
        from_domain="superhuman.example",
    )
    result = classifier.classify_pure_knn(fresh)
    assert result.queued is False
    assert result.category_path == "email/news/tech/superhuman"


def test_correction_exemplar_outranks_stale_cluster_categories(tmp_path: Path) -> None:
    """ADR-0024 extension, precedence rule: a correction is a sender-level
    routing fact. Even when stale cluster categories for the same sender
    exist (and would tie across roots, forcing abstention), the domain-
    matched exemplar wins. Live case: joinsuperhuman.ai had near-identical
    tools/ and learning/ cluster centroids; the user corrected to news/."""
    import json as _json
    import sqlite3

    _seed_workspace(
        tmp_path / "ws",
        [
            _make_proposal(
                0, "tools", "ai-assistants", "superhuman", rep_domain="superhuman.example"
            ),
            _make_proposal(1, "learning", "ai", "superhuman", rep_domain="superhuman.example"),
        ],
    )
    _seed_categories(
        tmp_path / "iris.db",
        [
            ("email/tools/ai-assistants/superhuman", 0.85),
            ("email/learning/ai/superhuman", 0.85),
            ("email/news/tech/superhuman", 0.85),
        ],
    )

    email_store = EmailStore(db_path=tmp_path / "email.db")
    email_store.ensure_schema()
    corrected = _make_email(
        "corrected-1",
        from_address="news@superhuman.example",
        from_domain="superhuman.example",
    )
    email_store.upsert(corrected)
    with sqlite3.connect(tmp_path / "iris.db") as conn:
        conn.execute(
            "INSERT INTO categories_history "
            "(category_path, op, old_path, new_path, payload, edited_at, source) "
            "VALUES (?, 'update', ?, ?, ?, datetime('now'), "
            "'user-classification-correction')",
            (
                "email/news/tech/superhuman",
                "email/tools/ai-assistants/superhuman",
                "email/news/tech/superhuman",
                _json.dumps(
                    {
                        "message_id": "corrected-1",
                        "account_id": ACCOUNT,
                        "new_path": "email/news/tech/superhuman",
                    }
                ),
            ),
        )
        conn.commit()

    classifier = EmailTriageClassifier(
        workspace_dir=tmp_path / "ws",
        db_path=tmp_path / "iris.db",
        email_db_path=tmp_path / "email.db",
        embedder=_stub_embedder_factory(
            {"superhuman.example": np.array([0.0, 0.0, 1.0], dtype=np.float32)}
        ),
    )

    fresh = _make_email(
        "fresh-1",
        from_address="digest@superhuman.example",
        from_domain="superhuman.example",
    )
    result = classifier.classify_pure_knn(fresh)

    # Without precedence the two stale same-vector cluster centroids in
    # different roots would zero the cross-root margin and abstain.
    assert result.queued is False
    assert result.category_path == "email/news/tech/superhuman"
