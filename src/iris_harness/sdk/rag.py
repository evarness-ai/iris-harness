"""Documents the owner lets IRIS read: the RAG index and its ingest seams.

Reading: `DocumentStore` holds the ingested documents, `DocumentIndex` their vectors,
and `search_documents` ranks passages for a question. Ingesting is governed: a plugin
asks with `propose_rag_ingest(path)` (it raises `IngestDeniedError` when policy says
no), the owner approves, and `execute_rag_ingest` runs it and returns an
`IngestResult`.

A plugin that owns where documents come from registers an `IngestSource` (the files
it knows, as `KnownFile` / `IndexedDocument`; one that also implements `record_removed` is
told when RAG denies or drops a file, as `RemovedDocument`, whose `reason` is `"denied"` or
`"removed"`) with `register_ingest_source`, and a
`DocumentCatalog` (what the owner sees listed, as `RagDocument`) with
`register_document_catalog`.
"""

from __future__ import annotations

from iris_harness.services.rag.documents import (
    DocumentCatalog,
    RagDocument,
    register_document_catalog,
)
from iris_harness.services.rag.index import DocumentIndex
from iris_harness.services.rag.ingest_gate import (
    IngestDeniedError,
    IngestProposal,
    execute_rag_ingest,
    propose_rag_ingest,
)
from iris_harness.services.rag.ingest_source import (
    IndexedDocument,
    IngestSource,
    KnownFile,
    RemovalAwareIngestSource,
    RemovedDocument,
    register_ingest_source,
)
from iris_harness.services.rag.models import IngestResult
from iris_harness.services.rag.retrieve import search_documents
from iris_harness.services.rag.store import DocumentStore

__all__ = [
    "DocumentCatalog",
    "DocumentIndex",
    "DocumentStore",
    "IndexedDocument",
    "IngestDeniedError",
    "IngestProposal",
    "IngestResult",
    "IngestSource",
    "KnownFile",
    "RagDocument",
    "RemovalAwareIngestSource",
    "RemovedDocument",
    "execute_rag_ingest",
    "propose_rag_ingest",
    "register_document_catalog",
    "register_ingest_source",
    "search_documents",
]
