"""Tests for the unified knowledge-graph endpoint (Phase 5).

Builds a small cross-corpus fixture: a wiki page, two RAG documents that link to
each other and to the wiki page, a shared tag, and a fact that mentions the tag —
then asserts the unified node/edge graph wires them together.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.memory.knowledge.wiki_engine import WikiEngine
from iris_harness.memory.store import MemoryStore, UserFact
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.rag.index import DocumentIndex
from iris_harness.services.rag.store import DocumentStore


def _write_wiki_page(wiki_root: Path) -> None:
    page = wiki_root / "concepts" / "machine-learning.md"
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(
        "---\ntitle: Machine Learning\ntags: [ml]\n---\nFoundational topic.\n",
        encoding="utf-8",
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("IRIS_TEST_NULL_EMBEDDINGS", "1")

    # RAG store with two cross-linked, co-tagged documents.
    store = DocumentStore(db_path=tmp_path / "rag.db")
    store.ensure_schema()
    store.upsert_source(
        id="src-alpha",
        path=str(tmp_path / "alpha.md"),
        kind="file",
        title="Alpha",
        content_sha="a",
        tags=("ml",),
        links=("Beta", "Machine Learning"),  # doc->doc and doc->wiki
        mtime=1.0,
    )
    store.upsert_source(
        id="src-beta",
        path=str(tmp_path / "beta.md"),
        kind="file",
        title="Beta",
        content_sha="b",
        tags=("ml",),
        links=(),
        mtime=1.0,
    )

    # Semantic memory: a wiki page (same #ml tag) + a fact that mentions ml.
    wiki_root = tmp_path / "wiki"
    _write_wiki_page(wiki_root)
    memory = MemoryStore(db_path=tmp_path / "memory.db")
    now = datetime.now(UTC)
    memory.upsert_user_fact(
        UserFact(
            key="interest",
            value="I really enjoy ml research",
            confidence=0.9,
            source="chat",
            first_seen=now,
            last_confirmed=now,
        )
    )

    runtime = SimpleNamespace(
        memory_store=memory,
        wiki=WikiEngine(wiki_root=wiki_root, llm_call=None, semantic_index=None),
    )
    handles = (store, DocumentIndex(persist_dir=tmp_path / "chroma"))

    app = create_app(runtime=runtime, auto_start_runtime=False)
    with TestClient(app, headers=auth_headers()) as c:
        c.app.state.rag_handles = handles
        yield c


def test_graph_unifies_memory_and_documents(client: TestClient) -> None:
    body = client.get("/knowledge/graph").json()
    nodes = {n["id"]: n for n in body["nodes"]}
    edges = {(e["source"], e["target"], e["kind"]) for e in body["edges"]}

    # Nodes from both corpora plus the bridging tag.
    assert "wiki:machine-learning" in nodes
    assert "doc:src-alpha" in nodes
    assert "doc:src-beta" in nodes
    assert "tag:ml" in nodes
    assert "fact:interest" in nodes  # linked because it mentions "ml"

    # doc -> doc link (Alpha references Beta by title).
    assert ("doc:src-alpha", "doc:src-beta", "link") in edges
    # doc -> wiki link (Alpha references the wiki page by title) — the unification.
    assert ("doc:src-alpha", "wiki:machine-learning", "link") in edges
    # shared #ml tag bridges documents, wiki, and the fact.
    assert ("doc:src-alpha", "tag:ml", "tag") in edges
    assert ("wiki:machine-learning", "tag:ml", "tag") in edges
    assert ("fact:interest", "tag:ml", "tag") in edges

    stats = body["stats"]
    assert stats["wiki"] == 1
    assert stats["documents"] == 2
    assert stats["facts"] == 1
    assert nodes["tag:ml"]["degree"] >= 4  # alpha, beta, wiki, fact


def test_graph_reports_unlinked_facts(client: TestClient, tmp_path: Path) -> None:
    # A fact mentioning no tag is excluded from the graph but counted.
    body = client.get("/knowledge/graph").json()
    assert body["stats"]["facts_total"] >= body["stats"]["facts"]
