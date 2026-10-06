"""What a file domain contributes to a RAG ingest — the seam, and its registry.

RAG reads documents *in place*: it never owns the bytes, and it does not know what
else the user's files are. A file domain does. So two facts cross this seam, and
nothing else:

- :meth:`IngestSource.known_file` — what the domain already knows about a path
  (its classification, its owner-facing document type), which the ingest gate
  ratchets against its own fresh content scan.
- :meth:`IngestSource.record_indexed` — that RAG has indexed a file, so the domain
  can record it wherever it tracks the user's files (FMX4, ADR-0066).

Neither is required. With no source registered, ingestion still classifies content
(``governance.file_scan``) and still refuses secrets; it simply records nothing
elsewhere. That is the core-only install, and it is a supported shape: the file
domain leaves for ``src/iris_personal`` (OSS plan M6, decision 2), so the core may
not import it.

The registry is the same shape as ``email.providers``, ``tasks.pending_actions`` and
``health.service``: a core registry a plugin plugs into at ``setup()``, plus at CLI
registration time, because ``iris docs`` runs with no runtime built.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol, runtime_checkable

from iris_harness.foundation.process_state import track_globals


@dataclass(frozen=True)
class KnownFile:
    """What the file domain already knows about a path."""

    classification: str  # secret | personal | internal | public
    document_type: str | None = None  # owner-facing facet, e.g. "bank_statement"


@dataclass(frozen=True)
class IndexedDocument:
    """One file RAG has just indexed, as the file domain needs to see it."""

    path: Path
    source_id: str  # RAG's id for this source, stable across re-ingests
    content_sha: str  # RAG's hash of the content it indexed
    kind: str  # the source kind: file | folder | obsidian | ...
    classification: str  # the label RAG stamped at ingest (sensitivity.classify)
    byte_size: int
    mtime: float
    head: bytes  # first bytes of the file, for media-type sniffing


@runtime_checkable
class IngestSource(Protocol):
    """The file domain's side of a RAG ingest. Both methods are advisory."""

    def known_file(self, path: Path) -> KnownFile | None:
        """What is already known about ``path``, or ``None`` if nothing is."""

    def record_indexed(self, doc: IndexedDocument) -> None:
        """Record that RAG indexed ``doc``. Never raises into the ingest."""


_lock = Lock()
_source: IngestSource | None = None


def register_ingest_source(source: IngestSource | None) -> None:
    """Install the process-wide ingest source (``None`` clears it)."""
    global _source
    with _lock:
        _source = source


def current_ingest_source() -> IngestSource | None:
    """The registered ingest source, or ``None`` on a core-only install."""
    with _lock:
        return _source


__all__ = [
    "IndexedDocument",
    "IngestSource",
    "KnownFile",
    "current_ingest_source",
    "register_ingest_source",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_source")
