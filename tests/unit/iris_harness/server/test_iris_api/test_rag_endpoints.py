"""Tests for the RAG document upload / list / search endpoints (Phase 4).

Core-only: no file domain is registered, so the routes list RAG's own sources
(``StoreDocumentCatalog``) — the public install's shape. The catalog-backed listing a
file domain registers is pinned in the file_organizer plugin's tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.server.iris_api import memory_routes
from iris_harness.server.iris_api.main import create_app
from iris_harness.services.rag.documents import (
    register_document_catalog,
    registered_document_catalog,
)
from iris_harness.services.rag.index import DocumentIndex
from iris_harness.services.rag.ingest import _source_id
from iris_harness.services.rag.ingest_gate import execute_rag_ingest, propose_rag_ingest
from iris_harness.services.rag.ingest_source import (
    RemovedDocument,
    current_ingest_source,
    register_ingest_source,
)
from iris_harness.services.rag.store import DocumentStore


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # Keyword fallback instead of a real ChromaDB embedding model.
    monkeypatch.setenv("IRIS_TEST_NULL_EMBEDDINGS", "1")
    monkeypatch.setattr(memory_routes, "RAG_UPLOAD_DIR", tmp_path / "uploads")

    # A plugin set up by an earlier test may have filled the process-wide seams.
    ingest, documents = current_ingest_source(), registered_document_catalog()
    register_ingest_source(None)
    register_document_catalog(None)

    store = DocumentStore(db_path=tmp_path / "rag.db")
    store.ensure_schema()
    handles = (store, DocumentIndex(persist_dir=tmp_path / "chroma"))

    # A stub runtime: with none, the lifespan builds a real one, whose plugins would
    # fill the seams this test cleared.
    app = create_app(runtime=SimpleNamespace(data_dir=tmp_path), auto_start_runtime=False)
    try:
        with TestClient(app, headers=auth_headers()) as c:
            # Inject tmp-path handles so the endpoints never touch real data/ stores.
            c.app.state.rag_handles = handles
            yield c
    finally:
        register_ingest_source(ingest)
        register_document_catalog(documents)


def test_upload_ingests_and_lists_the_document(client: TestClient) -> None:
    resp = client.post(
        "/rag/upload",
        files={
            "file": (
                "notes.md",
                b"# Meeting\n\nThe quick brown fox jumps over the lazy dog.",
                "text/markdown",
            )
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["filename"] == "notes.md"
    assert body["sources_added"] == 1
    assert body["chunks_indexed"] >= 1
    assert body["document"] is not None
    assert body["document"]["filename"] == "notes.md"
    assert body["document"]["kind"] == "file"
    assert body["document"]["file_id"].startswith("rag_")
    assert body["document"]["byte_size"] > 0

    # Now it shows up in the listing, from RAG's own sources.
    listed = client.get("/rag/documents").json()
    assert listed["count"] == 1
    assert listed["documents"][0]["filename"] == "notes.md"


def test_search_finds_uploaded_content(client: TestClient) -> None:
    client.post(
        "/rag/upload",
        files={
            "file": (
                "doc.md",
                b"Penguins are flightless birds native to Antarctica.",
                "text/markdown",
            )
        },
    )
    resp = client.post("/rag/search", json={"query": "flightless penguins", "limit": 5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] >= 1
    assert any("penguin" in r["text"].lower() for r in body["results"])
    assert body["results"][0]["citation"]


def test_unsupported_type_is_rejected(client: TestClient) -> None:
    resp = client.post(
        "/rag/upload",
        files={"file": ("malware.exe", b"\x00\x01binary", "application/octet-stream")},
    )
    assert resp.status_code == 415


def test_empty_file_is_rejected(client: TestClient) -> None:
    resp = client.post("/rag/upload", files={"file": ("empty.txt", b"", "text/plain")})
    assert resp.status_code == 400


def test_delete_removes_uploaded_doc_and_file(client: TestClient) -> None:
    up = client.post(
        "/rag/upload",
        files={"file": ("temp.md", b"Disposable note about otters.", "text/markdown")},
    ).json()
    file_id = up["document"]["file_id"]
    stored = Path(up["document"]["storage_path"])
    assert stored.exists()  # IRIS created it under the upload dir

    resp = client.delete(f"/rag/documents/{file_id}")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["deleted"] is True
    assert body["de_indexed"] is True
    assert body["file_removed"] is True  # owned upload -> bytes removed (ADR-0067)
    assert not stored.exists()

    # Gone from the listing.
    assert client.get("/rag/documents").json()["count"] == 0


def test_delete_unknown_doc_is_404(client: TestClient) -> None:
    assert client.delete("/rag/documents/rag_nonexistent").status_code == 404


# ---- re-upload of a gated document keeps it classified (FMX8) -------------------------


def _gated_upload(client: TestClient, tmp_path: Path) -> tuple[DocumentStore, str]:
    """An upload-dir document the ingest gate stamped personal; returns (store, source id)."""
    dest = tmp_path / "uploads" / "contact.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("# Contact\n\nReach me at jane@example.com about the herons.")
    store, _index = client.app.state.rag_handles
    execute_rag_ingest(propose_rag_ingest(dest), store=store, index=None)
    return store, _source_id(dest.resolve())


def _labels(store: DocumentStore, sid: str) -> set[str | None]:
    chunks = [store.get_chunk(f"{sid}:{i}") for i in range(store.count_chunks(sid))]
    return {c.classification for c in chunks if c is not None}


def test_reupload_of_a_gated_document_keeps_its_classification(
    client: TestClient, tmp_path: Path
) -> None:
    store, sid = _gated_upload(client, tmp_path)
    assert _labels(store, sid) == {"personal"}

    body = b"# Contact\n\nNow reach me at jane.doe@example.com about the herons."
    resp = client.post("/rag/upload", files={"file": ("contact.md", body, "text/markdown")})

    assert resp.status_code == 200, resp.text
    assert resp.json()["sources_updated"] == 1
    assert _labels(store, sid) == {"personal"}


def test_reupload_of_a_gated_document_as_secret_is_removed(
    client: TestClient, tmp_path: Path
) -> None:
    store, sid = _gated_upload(client, tmp_path)

    body = b"# Contact\n\naws_key=AKIAIOSFODNN7EXAMPLE\n"
    resp = client.post("/rag/upload", files={"file": ("contact.md", body, "text/markdown")})

    assert resp.status_code == 200, resp.text
    assert resp.json()["sources_denied"] == 1
    assert resp.json()["document"] is None
    assert store.get_source(sid) is None and store.count_chunks(sid) == 0
    assert not (tmp_path / "uploads" / "contact.md").exists()  # the secret copy is gone


def test_a_first_upload_is_classified(client: TestClient, tmp_path: Path) -> None:
    body = b"# Contact\n\nReach me at jane@example.com about the herons."
    resp = client.post("/rag/upload", files={"file": ("contact.md", body, "text/markdown")})

    assert resp.status_code == 200, resp.text
    assert resp.json()["sources_added"] == 1
    assert resp.json()["document"]["classification"] == "personal"
    store, _index = client.app.state.rag_handles
    assert _labels(store, _source_id((tmp_path / "uploads" / "contact.md").resolve())) == {
        "personal"
    }


def test_a_secret_first_upload_is_refused_and_not_kept(client: TestClient, tmp_path: Path) -> None:
    body = b"# Keys\n\naws_key=AKIAIOSFODNN7EXAMPLE\n"
    resp = client.post("/rag/upload", files={"file": ("creds.md", body, "text/markdown")})

    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload["sources_denied"] == 1 and payload["sources_added"] == 0
    assert payload["document"] is None
    assert "refused: classified secret" in payload["summary"]
    assert not (tmp_path / "uploads" / "creds.md").exists()
    assert client.get("/rag/documents").json()["count"] == 0


# ---- the size cap and the event loop (security review, 2026-09-26) -------------------
# The route read the whole body before checking the 25 MB cap, then wrote and indexed
# the file on the event loop, stalling every other request while a big PDF indexed.


def _record_reads(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record how many bytes each UploadFile.read returned."""
    from starlette.datastructures import UploadFile

    returned: list[int] = []
    real_read = UploadFile.read

    async def read(self: UploadFile, size: int = -1) -> bytes:
        data = await real_read(self, size)
        returned.append(len(data))
        return data

    monkeypatch.setattr(UploadFile, "read", read)
    return returned


def test_an_oversized_upload_is_refused_while_reading(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(memory_routes, "RAG_MAX_UPLOAD_BYTES", 16)
    monkeypatch.setattr(memory_routes, "RAG_UPLOAD_CHUNK_BYTES", 4)
    reads = _record_reads(monkeypatch)

    resp = client.post("/rag/upload", files={"file": ("big.txt", b"x" * 100, "text/plain")})

    assert resp.status_code == 413
    assert "limit" in resp.json()["detail"]
    # It stopped one chunk past the cap instead of taking all 100 bytes in.
    assert sum(reads) <= 16 + 4
    assert not (tmp_path / "uploads" / "big.txt").exists()


def test_a_file_exactly_at_the_cap_is_accepted(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_routes, "RAG_MAX_UPLOAD_BYTES", 16)
    monkeypatch.setattr(memory_routes, "RAG_UPLOAD_CHUNK_BYTES", 4)

    body = b"otters and heron"
    assert len(body) == 16
    resp = client.post("/rag/upload", files={"file": ("fits.txt", body, "text/plain")})

    assert resp.status_code == 200, resp.text


def test_a_declared_oversized_body_is_refused_before_it_is_parsed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_routes, "RAG_MAX_UPLOAD_BYTES", 16)
    monkeypatch.setattr(memory_routes, "RAG_UPLOAD_FORM_OVERHEAD_BYTES", 1024)
    reads = _record_reads(monkeypatch)

    resp = client.post("/rag/upload", files={"file": ("big.txt", b"x" * 4096, "text/plain")})

    assert resp.status_code == 413
    assert reads == []  # the route never ran


def test_ingest_runs_off_the_event_loop(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    seen: list[bool] = []
    real_ingest = memory_routes.ingest_path

    def ingest(*args: Any, **kwargs: Any) -> Any:
        try:
            asyncio.get_running_loop()
            seen.append(True)  # on the loop: every other request waits
        except RuntimeError:
            seen.append(False)  # a worker thread
        return real_ingest(*args, **kwargs)

    monkeypatch.setattr(memory_routes, "ingest_path", ingest)

    resp = client.post(
        "/rag/upload", files={"file": ("loop.md", b"Notes about herons.", "text/markdown")}
    )

    assert resp.status_code == 200, resp.text
    assert seen == [False]


# ---- the file domain is told when RAG drops a document (issue 101) --------------------


class _RemovalRecorder:
    def __init__(self) -> None:
        self.removed: list[RemovedDocument] = []

    def known_file(self, path: Path) -> None:
        return None

    def record_indexed(self, doc: object) -> None:
        pass

    def record_removed(self, doc: RemovedDocument) -> None:
        self.removed.append(doc)


def test_a_secret_reupload_is_reported_to_the_source(client: TestClient, tmp_path: Path) -> None:
    _gated_upload(client, tmp_path)
    rec = _RemovalRecorder()
    register_ingest_source(rec)  # the fixture restores the previous source

    body = b"# Contact\n\naws_key=AKIAIOSFODNN7EXAMPLE\n"
    resp = client.post("/rag/upload", files={"file": ("contact.md", body, "text/markdown")})

    assert resp.json()["sources_denied"] == 1
    assert [d.reason for d in rec.removed] == ["denied"]


def test_delete_is_reported_to_the_source(client: TestClient) -> None:
    up = client.post(
        "/rag/upload", files={"file": ("temp.md", b"Disposable otters.", "text/markdown")}
    ).json()
    rec = _RemovalRecorder()
    register_ingest_source(rec)

    assert client.delete(f"/rag/documents/{up['document']['file_id']}").status_code == 200

    assert [(d.path.name, d.reason) for d in rec.removed] == [("temp.md", "removed")]
