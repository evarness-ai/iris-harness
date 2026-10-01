"""The RAG document list the API serves — the seam, its registry, and the core default.

``GET /rag/documents``, ``POST /rag/upload`` and ``DELETE /rag/documents/{file_id}``
answer "which documents has RAG indexed?". A file domain that tracks the user's files
already knows that, with more than RAG does (where the bytes live, their size, their
classification), so it can register a :class:`DocumentCatalog` and the routes list
from it. With none registered — the core-only install, since the file domain leaves
for ``src/iris_personal`` (OSS plan M6, decision 2) — :class:`StoreDocumentCatalog`
lists RAG's own sources, so the routes still work and the core imports no domain.

Same shape as :mod:`iris_harness.services.rag.ingest_source`: a process-wide registry a
plugin fills at ``setup()``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, runtime_checkable

from iris_harness.foundation.process_state import track_globals

from .store import DocumentStore

#: A document's id is RAG's source id under this prefix, on every install, so an id
#: the API handed out stays valid whichever catalog answers.
FILE_ID_PREFIX = "rag_"


@dataclass(frozen=True)
class RagDocument:
    """One indexed document, as the API shows it."""

    file_id: str
    filename: str
    kind: str
    classification: str | None
    byte_size: int | None
    location_tier: str | None
    storage_path: str
    created_at: str
    updated_at: str

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class DocumentCatalog(Protocol):
    """Where the API reads RAG's documents from."""

    def list_documents(self) -> list[RagDocument]:
        """Every document RAG holds, as this catalog knows it."""

    def get_document(self, file_id: str) -> RagDocument | None:
        """The document with ``file_id``, or ``None``."""

    def forget_document(self, file_id: str) -> None:
        """Record that the document was removed from RAG. Never raises."""


class StoreDocumentCatalog:
    """The core default: RAG's own sources, with what the file on disk says."""

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    def list_documents(self) -> list[RagDocument]:
        return [self._document(s) for s in self._store.list_sources()]

    def get_document(self, file_id: str) -> RagDocument | None:
        if not file_id.startswith(FILE_ID_PREFIX):
            return None
        source = self._store.get_source(file_id.removeprefix(FILE_ID_PREFIX))
        return self._document(source) if source is not None else None

    def forget_document(self, file_id: str) -> None:
        """Nothing to do: removing the source from the store is the forgetting."""

    @staticmethod
    def _document(source: Any) -> RagDocument:
        path = Path(source.path)
        try:
            byte_size: int | None = path.stat().st_size
        except OSError:
            byte_size = None
        return RagDocument(
            file_id=FILE_ID_PREFIX + source.id,
            filename=path.name,
            kind=str(source.kind),
            classification=None,  # the core keeps it per chunk, not per document
            byte_size=byte_size,
            location_tier=None,
            storage_path=source.path,
            created_at=source.added_at.isoformat(),
            updated_at=source.last_synced_at.isoformat(),
        )


_lock = Lock()
_catalog: DocumentCatalog | None = None


def register_document_catalog(catalog: DocumentCatalog | None) -> None:
    """Install the process-wide document catalog (``None`` clears it)."""
    global _catalog  # one process-wide registry, like ingest_source
    with _lock:
        _catalog = catalog


def registered_document_catalog() -> DocumentCatalog | None:
    """The registered catalog, or ``None`` on a core-only install."""
    with _lock:
        return _catalog


def document_catalog(store: DocumentStore) -> DocumentCatalog:
    """The registered catalog, else RAG's own sources in ``store``."""
    registered = registered_document_catalog()
    return registered if registered is not None else StoreDocumentCatalog(store)


__all__ = [
    "FILE_ID_PREFIX",
    "DocumentCatalog",
    "RagDocument",
    "StoreDocumentCatalog",
    "document_catalog",
    "register_document_catalog",
    "registered_document_catalog",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_catalog")
