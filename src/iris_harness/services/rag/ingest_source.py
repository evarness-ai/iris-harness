"""What a file domain contributes to a RAG ingest — the seam, and its registry.

RAG reads documents *in place*: it never owns the bytes, and it does not know what
else the user's files are. A file domain does. So two facts cross this seam, and
nothing else:

- :meth:`IngestSource.known_file` — what the domain already knows about a path
  (its classification, its owner-facing document type), which the ingest gate
  ratchets against its own fresh content scan.
- :meth:`IngestSource.record_indexed` — that RAG has indexed a file, so the domain
  can record it wherever it tracks the user's files (FMX4, ADR-0066).
- :meth:`RemovalAwareIngestSource.record_removed` — that RAG no longer holds a file
  (ingest denied it, or the owner removed it), so the domain does not keep a stale
  "indexed" record or a less-sensitive label for it. Optional on top of ``IngestSource``:
  a source written before this method existed keeps working and is simply not told.

None is required. With no source registered, ingestion still classifies content
(``governance.file_scan``) and still refuses secrets; it simply records nothing
elsewhere. That is the core-only install, and it is a supported shape: the file
domain leaves for ``src/iris_personal`` (OSS plan M6, decision 2), so the core may
not import it.

The registry is the same shape as ``email.providers``, ``tasks.pending_actions`` and
``health.service``: a core registry a plugin plugs into at ``setup()``, plus at CLI
registration time, because ``iris docs`` runs with no runtime built.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal, Protocol, runtime_checkable

from iris_harness.foundation.process_state import track_globals

logger = logging.getLogger(__name__)


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


#: Why RAG stopped holding a file. A closed set on purpose: widening it later is
#: compatible, narrowing it would break every source that matched on a value.
RemovalReason = Literal["denied", "removed"]


@dataclass(frozen=True)
class RemovedDocument:
    """One file RAG no longer holds, as the file domain needs to see it."""

    path: Path
    source_id: str  # RAG's id for this source, the one ``IndexedDocument`` carried
    reason: RemovalReason  # "denied" (ingest refused it) | "removed" (the owner removed it)
    classification: str | None = None  # the label that caused a denial ("secret"), else None


@runtime_checkable
class RemovalAwareIngestSource(Protocol):
    """An :class:`IngestSource` that is also told when RAG stops holding a file."""

    def record_removed(self, doc: RemovedDocument) -> None:
        """Record that RAG no longer holds ``doc``. Never raises into the caller."""


def report_removed(
    source: IngestSource | None,
    path: Path,
    *,
    source_id: str,
    reason: RemovalReason,
    classification: str | None = None,
) -> None:
    """Tell the file domain RAG no longer holds ``path``. Advisory: never raises.

    The one call every path that takes a file out of RAG makes (ingest denial, ``iris docs
    remove``, the API delete), so the domain's catalog cannot keep saying "indexed". A
    ``None`` source, or one without ``record_removed``, is a no-op.
    """
    if not isinstance(source, RemovalAwareIngestSource):
        return
    try:
        source.record_removed(
            RemovedDocument(
                path=path, source_id=source_id, reason=reason, classification=classification
            )
        )
    except Exception:  # an ingest source never breaks RAG
        logger.exception("rag: ingest source rejected removal of %s", path)


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
    "RemovalAwareIngestSource",
    "RemovalReason",
    "RemovedDocument",
    "current_ingest_source",
    "register_ingest_source",
    "report_removed",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_source")
