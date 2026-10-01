"""Unit tests for the corpus-discovery pure functions (Track 1E.2)."""

from __future__ import annotations

import json

import numpy as np

from iris_personal.email.contracts import CategoryProposal, CategoryRepresentative
from iris_personal.plugins.email_workflows.discovery import (
    ALLOWED_ROOTS,
    CorpusRow,
    build_proposals,
    cluster_corpus,
    compute_cohesion,
    name_clusters,
)

# ─── Helpers ────────────────────────────────────────────────────────────────


def _three_clusters(n_per: int = 30) -> np.ndarray:
    """Three well-separated unit-norm clusters in 4D — for HDBSCAN sanity."""
    rng = np.random.default_rng(seed=42)
    centers = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    parts = []
    for c in centers:
        # tight cluster: small noise around the centroid
        noise = rng.normal(0, 0.02, size=(n_per, 4)).astype(np.float32)
        pts = c + noise
        # L2-normalize so embeddings @ embeddings.T = cosine sim
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        parts.append(pts)
    return np.vstack(parts)


# ─── compute_cohesion ───────────────────────────────────────────────────────


def test_compute_cohesion_perfect_cluster_is_one() -> None:
    """All identical unit vectors → cohesion = 1."""
    vec = np.array([[1.0, 0.0, 0.0, 0.0]] * 10, dtype=np.float32)
    mask = np.ones(10, dtype=bool)
    assert compute_cohesion(vec, mask) == 1.0


def test_compute_cohesion_orthogonal_pair_is_low() -> None:
    """Two orthogonal points → cohesion << 1."""
    vec = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mask = np.array([True, True])
    cohesion = compute_cohesion(vec, mask)
    # Centroid = (.5, .5), unit = (.707, .707), dot with each = .707
    assert 0.6 < cohesion < 0.8


def test_compute_cohesion_singleton_is_one() -> None:
    vec = np.array([[1.0, 0.0]], dtype=np.float32)
    mask = np.array([True])
    assert compute_cohesion(vec, mask) == 1.0


def test_compute_cohesion_empty_mask_is_zero() -> None:
    vec = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    mask = np.array([False, False])
    assert compute_cohesion(vec, mask) == 0.0


def test_compute_cohesion_clamped_to_contract_range() -> None:
    """Float drift can't push cohesion outside [0, 1]."""
    vec = np.array([[1.0, 0.0]] * 5, dtype=np.float32)
    mask = np.ones(5, dtype=bool)
    val = compute_cohesion(vec, mask)
    assert 0.0 <= val <= 1.0


# ─── cluster_corpus ─────────────────────────────────────────────────────────


def test_cluster_corpus_finds_three_well_separated_clusters() -> None:
    embeddings = _three_clusters(n_per=30)
    best, sweep = cluster_corpus(embeddings)
    # On obviously-3-cluster data the picker should land on n_clusters=3.
    assert best.n_clusters == 3, f"expected 3, got {best.n_clusters} (sweep: {len(sweep)} runs)"
    # Each true cluster has 30 points → median size near 30.
    assert best.median_cluster_size >= 25


def test_cluster_corpus_sweep_runs_all_combinations() -> None:
    embeddings = _three_clusters(n_per=30)
    _, sweep = cluster_corpus(embeddings)
    # 5 mcs × 3 ms = 15, minus combinations where ms > mcs (only mcs=5, ms=10).
    assert len(sweep) == 14


# ─── build_proposals ────────────────────────────────────────────────────────


def _rows_for_three_clusters(n_per: int = 30) -> list[CorpusRow]:
    rows = []
    for cid, domain in enumerate(["a.com", "b.com", "c.com"]):
        for i in range(n_per):
            rows.append(
                CorpusRow(
                    id=f"c{cid}-msg-{i}",
                    subject=f"subject for cluster {cid} item {i}",
                    from_address=f"sender@{domain}",
                    from_domain=domain,
                    snippet=f"snippet {cid} {i}",
                )
            )
    return rows


def test_build_proposals_skips_noise() -> None:
    rows = _rows_for_three_clusters(3)
    embeddings = np.array(
        [[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3 + [[1.0, 1.0]] * 3,
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    # cluster_id -1 is noise; should be dropped
    labels = np.array([0, 0, 0, 1, 1, 1, -1, -1, -1])
    proposals = build_proposals(rows, embeddings, labels)
    assert len(proposals) == 2
    assert all(p.cluster_id in (0, 1) for p in proposals)


def test_build_proposals_sets_cohesion_and_member_ids() -> None:
    rows = _rows_for_three_clusters(3)
    embeddings = np.array(
        [[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3 + [[1.0, 1.0]] * 3,
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    proposals = build_proposals(rows, embeddings, labels)
    # Every proposal should carry full member ids and a real cohesion value.
    assert len(proposals) == 3
    for p in proposals:
        assert 0.0 <= p.cohesion <= 1.0
        assert len(p.member_ids) == p.size
        assert len(p.representatives) <= p.size


def test_build_proposals_sorted_by_size_desc() -> None:
    rows = _rows_for_three_clusters(10)
    embeddings = np.array(
        [[1.0, 0.0]] * 10 + [[0.0, 1.0]] * 5 + [[1.0, 1.0]] * 3,
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    labels = np.array([0] * 10 + [1] * 5 + [2] * 3)
    proposals = build_proposals(rows, embeddings, labels)
    sizes = [p.size for p in proposals]
    assert sizes == sorted(sizes, reverse=True)


def test_build_proposals_top_domains_aggregated() -> None:
    rows = [
        CorpusRow(
            id=f"m-{i}",
            subject="s",
            from_address="a@a.com" if i < 5 else "b@b.com",
            from_domain="a.com" if i < 5 else "b.com",
            snippet="",
        )
        for i in range(7)
    ]
    embeddings = np.array([[1.0, 0.0]] * 7, dtype=np.float32)
    labels = np.array([0] * 7)
    proposals = build_proposals(rows, embeddings, labels)
    assert len(proposals) == 1
    domains = dict(proposals[0].top_domains)
    assert domains["a.com"] == 5
    assert domains["b.com"] == 2


# ─── name_clusters ──────────────────────────────────────────────────────────


class _StubNamingClient:
    """Test stub matching the NamingClient protocol."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._responses.pop(0)


def _stub_proposal(cluster_id: int = 0, size: int = 10, cohesion: float = 0.8) -> CategoryProposal:
    return CategoryProposal(
        cluster_id=cluster_id,
        size=size,
        cohesion=cohesion,
        top_domains=(("gap.com", size),),
        representatives=(
            CategoryRepresentative(
                id=f"m-{cluster_id}",
                subject="60% off summer steals",
                from_address="gap@email.gap.com",
                snippet="...",
            ),
        ),
        member_ids=tuple(f"m-{cluster_id}-{i}" for i in range(size)),
    )


def test_name_clusters_happy_path() -> None:
    client = _StubNamingClient(
        responses=[
            json.dumps(
                {
                    "root": "shopping",
                    "branch": "apparel",
                    "leaf": "outlet-brand",
                    "rationale": "dominated by gap.com",
                }
            )
        ]
    )
    result = name_clusters([_stub_proposal()], client)
    assert len(result) == 1
    assert result[0].proposed_root == "shopping"
    assert result[0].proposed_branch == "apparel"
    assert result[0].proposed_leaf == "outlet-brand"
    assert result[0].naming_rationale == "dominated by gap.com"


def test_name_clusters_normalizes_whitespace_and_casing() -> None:
    """Model can return  "Shopping" / "Apparel" — we lowercase + dash."""
    client = _StubNamingClient(
        responses=[
            json.dumps(
                {
                    "root": "Shopping",
                    "branch": "APPAREL  brand",
                    "leaf": "Outlet Brand!",
                }
            )
        ]
    )
    result = name_clusters([_stub_proposal()], client)
    assert result[0].proposed_root == "shopping"
    assert result[0].proposed_branch == "apparel-brand"
    assert result[0].proposed_leaf == "outlet-brand"


def test_name_clusters_coerces_non_allowed_root() -> None:
    client = _StubNamingClient(
        responses=[json.dumps({"root": "retail", "branch": "apparel", "leaf": "outlet-brand"})]
    )
    result = name_clusters([_stub_proposal()], client)
    # "retail" is not in ALLOWED_ROOTS → coerced to "other"
    assert "retail" not in ALLOWED_ROOTS
    assert result[0].proposed_root == "other"
    # branch + leaf still come through (the cluster isn't wasted)
    assert result[0].proposed_branch == "apparel"
    assert result[0].proposed_leaf == "outlet-brand"


def test_name_clusters_records_error_in_rationale_on_failure() -> None:
    """A garbage response shouldn't crash the pass; rationale carries
    the error."""
    client = _StubNamingClient(responses=["this is not json at all"])
    result = name_clusters([_stub_proposal()], client)
    assert result[0].proposed_root is None
    assert result[0].proposed_branch is None
    assert result[0].proposed_leaf is None
    assert result[0].naming_rationale is not None
    assert result[0].naming_rationale.startswith("error:")


def test_name_clusters_processes_multiple_clusters() -> None:
    client = _StubNamingClient(
        responses=[
            json.dumps({"root": "shopping", "branch": "a", "leaf": "x"}),
            json.dumps({"root": "finance", "branch": "b", "leaf": "y"}),
        ]
    )
    result = name_clusters(
        [_stub_proposal(cluster_id=0), _stub_proposal(cluster_id=1)],
        client,
    )
    assert [p.proposed_root for p in result] == ["shopping", "finance"]
    assert len(client.calls) == 2


def test_name_clusters_does_not_mutate_input() -> None:
    """CategoryProposal is frozen; the namer must use model_copy."""
    client = _StubNamingClient(
        responses=[json.dumps({"root": "shopping", "branch": "a", "leaf": "b"})]
    )
    original = _stub_proposal()
    result = name_clusters([original], client)
    assert original.proposed_root is None
    assert result[0].proposed_root == "shopping"


# ─── bootstrap_categories (orchestrator) ────────────────────────────────────


def _seed_email_db(db_path, rows: list[dict]) -> None:  # type: ignore[no-untyped-def]
    """Minimal email.db seeder for orchestrator tests."""
    from datetime import UTC, datetime

    from iris_personal.email.contracts import EmailMessage
    from iris_personal.email.store import EmailStore

    store = EmailStore(db_path=db_path)
    store.ensure_schema()
    messages = [
        EmailMessage(
            id=r["id"],
            provider="gmail",
            account_id=r["account_id"],
            from_address=r["from_address"],
            from_domain=r["from_domain"],
            subject=r.get("subject", ""),
            snippet=r.get("snippet", ""),
            received_at=datetime.now(UTC),
        )
        for r in rows
    ]
    store.upsert_many(messages)


def test_bootstrap_categories_raises_when_no_emails(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import pytest

    from iris_personal.plugins.email_workflows import discovery

    db = tmp_path / "email.db"
    # Build empty schema
    _seed_email_db(db, [])
    # Don't let it try to fetch
    with pytest.raises(ValueError, match="no emails"):
        discovery.bootstrap_categories(
            "gmail:nobody@gmail.com",
            db_path=db,
            min_corpus=10,
            fetch_if_low=False,
            naming_client=None,
        )


def test_bootstrap_categories_runs_end_to_end_with_stubs(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Stubbed embedder + stubbed namer + real HDBSCAN over 30 rows."""
    from iris_personal.plugins.email_workflows import discovery

    account = "gmail:user@gmail.com"
    # Seed 30 rows across 3 sender domains — three clean clusters.
    rows = []
    for cid, domain in enumerate(["gap.com", "shopmart.com", "facebook.com"]):
        for i in range(10):
            rows.append(
                {
                    "id": f"c{cid}-{i}",
                    "account_id": account,
                    "from_address": f"sender@{domain}",
                    "from_domain": domain,
                    "subject": f"subject {cid} {i}",
                    "snippet": f"snippet {cid} {i}",
                }
            )
    db = tmp_path / "email.db"
    _seed_email_db(db, rows)

    # Stub embed_corpus to return three obvious clusters in 2D.
    def fake_embed(texts: list[str], model_name: str):  # type: ignore[no-untyped-def]
        import numpy as np

        vecs = []
        for t in texts:
            if "gap.com" in t:
                v = np.array([1.0, 0.0], dtype=np.float32)
            elif "shopmart.com" in t:
                v = np.array([0.0, 1.0], dtype=np.float32)
            else:
                v = np.array([0.7071, 0.7071], dtype=np.float32)
            vecs.append(v)
        out = np.stack(vecs)
        return out

    monkeypatch.setattr(discovery, "embed_corpus", fake_embed)

    client = _StubNamingClient(
        responses=[
            json.dumps({"root": "shopping", "branch": "apparel", "leaf": "gap"}),
            json.dumps({"root": "shopping", "branch": "apparel", "leaf": "shopmart"}),
            json.dumps({"root": "social", "branch": "facebook", "leaf": "updates"}),
        ]
    )

    proposals = discovery.bootstrap_categories(
        account,
        db_path=db,
        min_corpus=10,  # corpus has 30 rows, no fetch needed
        fetch_if_low=False,
        naming_client=client,
    )
    assert len(proposals) == 3
    roots = {p.proposed_root for p in proposals}
    assert roots == {"shopping", "social"}
    # Every proposal got named (no errors).
    assert all(
        p.naming_rationale is None or not p.naming_rationale.startswith("error:") for p in proposals
    )


def test_bootstrap_categories_no_naming_client_returns_unnamed(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    """fetch_if_low=False + no naming_client → orchestrator still
    returns proposals, just without names."""
    from iris_personal.plugins.email_workflows import discovery

    account = "gmail:user@gmail.com"
    rows = []
    for cid, domain in enumerate(["gap.com", "shopmart.com"]):
        for i in range(8):
            rows.append(
                {
                    "id": f"c{cid}-{i}",
                    "account_id": account,
                    "from_address": f"sender@{domain}",
                    "from_domain": domain,
                    "subject": "s",
                    "snippet": "snip",
                }
            )
    db = tmp_path / "email.db"
    _seed_email_db(db, rows)

    def fake_embed(texts, model_name):  # type: ignore[no-untyped-def]
        import numpy as np

        return np.stack(
            [
                (
                    np.array([1.0, 0.0], dtype=np.float32)
                    if "gap.com" in t
                    else np.array([0.0, 1.0], dtype=np.float32)
                )
                for t in texts
            ]
        )

    monkeypatch.setattr(discovery, "embed_corpus", fake_embed)

    proposals = discovery.bootstrap_categories(
        account,
        db_path=db,
        min_corpus=5,
        fetch_if_low=False,
        naming_client=None,
    )
    # Clusters formed (small corpus, may be 1-2 depending on HDBSCAN — we just
    # require the call to succeed and return unnamed proposals.
    for p in proposals:
        assert p.proposed_root is None
        assert p.proposed_branch is None
        assert p.proposed_leaf is None


# ─── Embedder cache ─────────────────────────────────────────────────────────
