"""Document-RAG domain models (RAG R0).

The document layer indexes the user's *existing* sources (markdown folders,
Obsidian vaults, PDFs later) for retrieval — it does not own them. Every
retrieved chunk carries a ``Citation`` back to the source file so answers can
attribute their evidence (the NotebookLM-style requirement).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

SourceKind = Literal["file", "folder", "obsidian", "pdf", "image"]


@dataclass(frozen=True)
class DocumentSource:
    """A registered source the user pointed IRIS at (the file, not a copy)."""

    id: str
    path: str
    kind: SourceKind
    title: str
    content_sha: str
    added_at: datetime
    last_synced_at: datetime
    tags: tuple[str, ...] = field(default_factory=tuple)
    links: tuple[str, ...] = field(default_factory=tuple)  # outgoing [[wikilinks]]
    mtime: float = 0.0  # filesystem mtime at last sync (incremental skip)


@dataclass(frozen=True)
class DocumentChunk:
    """One retrievable slice of a source, with where-it-came-from metadata."""

    id: str  # f"{source_id}:{chunk_index}"
    source_id: str
    source_path: str
    title: str  # nearest heading (or the file title)
    chunk_index: int
    text: str
    page: int | None = None  # 1-based page number for PDF sources
    # Sensitivity of the source document at ingest time (FMX8). None = legacy
    # chunk ingested before classification threading; downstream egress gating
    # treats it as unclassified.
    classification: str | None = None


@dataclass(frozen=True)
class Citation:
    source_path: str
    title: str
    chunk_index: int
    page: int | None = None

    def label(self) -> str:
        """Human-readable attribution, e.g. ``Project X (notes/x.md#2)`` or,
        for a PDF, ``Report (report.pdf p.3)``."""
        if self.page is not None:
            return f"{self.title} ({self.source_path} p.{self.page})"
        return f"{self.title} ({self.source_path}#{self.chunk_index})"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    citation: Citation
    # Carried through from the chunk so egress gating can see how sensitive a
    # retrieved passage is (FMX8). None = ingested before classification.
    classification: str | None = None


@dataclass(frozen=True)
class IngestResult:
    sources_added: int = 0
    sources_updated: int = 0
    sources_skipped: int = 0
    chunks_indexed: int = 0
    paths: tuple[str, ...] = field(default_factory=tuple)
    # Classified sources whose edited content now classifies secret: not indexed, and
    # their earlier chunks removed (secret content never enters RAG).
    sources_denied: int = 0

    def summary(self) -> str:
        text = (
            f"{self.sources_added} added, {self.sources_updated} updated, "
            f"{self.sources_skipped} unchanged; {self.chunks_indexed} chunk(s) indexed"
        )
        if self.sources_denied:
            text += (
                f"; {self.sources_denied} removed: now classified secret "
                "(secret documents never enter RAG; use the vault)"
            )
        return text


__all__ = [
    "SourceKind",
    "DocumentSource",
    "DocumentChunk",
    "Citation",
    "RetrievedChunk",
    "IngestResult",
]
