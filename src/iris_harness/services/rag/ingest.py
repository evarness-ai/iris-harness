"""Document ingestion (RAG R0/R1/R2/R4) — index markdown, PDFs, images, docx.

Connector model: IRIS reads the user's existing files and indexes them; it
never copies or owns them. Idempotent by raw-bytes hash (so an unchanged PDF/
image is skipped without re-extracting or re-running OCR), with an mtime
fast-path that skips unchanged files without reading them at all.

Per type:
- markdown/text → front-matter-aware (title/tags/[[wikilinks]]), one unit;
- PDF → text per page via pypdfium2; an image-only page is OCR'd (opt-in) so
  scanned statements/notes are searchable; chunks cite their page;
- image → OCR'd to text (opt-in); skipped with a log when OCR is unavailable;
- docx → full text via python-docx (opt-in), one unit; skipped with a log when
  python-docx is unavailable.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from iris_harness.services.rag.chunker import chunk_markdown
from iris_harness.services.rag.index import DocumentIndex
from iris_harness.services.rag.ingest_source import IndexedDocument, IngestSource, report_removed
from iris_harness.services.rag.loaders import extract_docx_text, extract_pdf_pages, render_pdf_page
from iris_harness.services.rag.models import DocumentChunk, IngestResult, SourceKind
from iris_harness.services.rag.obsidian import ParsedNote, context_line, parse_note
from iris_harness.services.rag.ocr import OcrUnavailable, ocr_available, ocr_image
from iris_harness.services.rag.sensitivity import classify, ratchet
from iris_harness.services.rag.store import DocumentStore

logger = logging.getLogger(__name__)


class SecretIngestError(RuntimeError):
    """A secret-classified document was handed to the RAG ingester (FMX8).

    Defense in depth: the ingest gate (``iris_harness.services.rag.ingest_gate``) should have
    denied the document long before this point — secret content never enters
    the retrievable index; the vault is the right home for it.
    """


TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text"}
PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
DOCX_SUFFIXES = {".docx"}
_ALL_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES | IMAGE_SUFFIXES | DOCX_SUFFIXES


def _source_id(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]  # noqa: S324 — id, not security


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _report_indexed(
    source: IngestSource,
    file: Path,
    raw: bytes,
    *,
    kind: SourceKind,
    content_sha: str,
    mtime: float,
    classification: str,
) -> None:
    """Tell the file domain that RAG indexed ``file``. Never breaks ingestion.

    ``classification`` is the label RAG stamped (``sensitivity.classify``), computed in
    the core, so it is the same on every install — with or without a file domain
    mounted (OSS plan M6.1b).
    """
    try:
        source.record_indexed(
            IndexedDocument(
                path=file,
                source_id=_source_id(file),
                content_sha=content_sha,
                kind=str(kind),
                classification=classification,
                byte_size=len(raw),
                mtime=mtime,
                head=raw[:64],
            )
        )
    except Exception:  # an ingest source never breaks RAG ingest
        logger.exception("rag: ingest source rejected %s", file)


def _iter_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() in _ALL_SUFFIXES:
            yield root
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _ALL_SUFFIXES:
            yield p


@dataclass
class _Loaded:
    """Extracted, type-agnostic content ready to chunk."""

    title: str
    tags: tuple[str, ...]
    links: tuple[str, ...]
    units: list[tuple[int | None, str]]  # (page_number_or_None, text)


def _pdf_units(file: Path) -> list[tuple[int | None, str]]:
    units: list[tuple[int | None, str]] = []
    can_ocr = ocr_available()
    for i, page_text in enumerate(extract_pdf_pages(file)):
        text = (page_text or "").strip()
        if not text and can_ocr:  # image-only page → OCR the rendered page
            try:
                text = ocr_image(render_pdf_page(file, i)).strip()
            except OcrUnavailable:
                text = ""
        if text:
            units.append((i + 1, text))
    return units


def _load(file: Path, raw: bytes) -> _Loaded | None:
    """Extract a file's content. None = nothing indexable (e.g. image w/o OCR)."""
    suffix = file.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        note = parse_note(raw.decode("utf-8", errors="replace"), file_title=file.stem)
        return _Loaded(note.title, note.tags, note.links, [(None, note.body)])
    if suffix in PDF_SUFFIXES:
        units = _pdf_units(file)
        return _Loaded(file.stem, (), (), units) if units else None
    if suffix in DOCX_SUFFIXES:
        try:
            text = extract_docx_text(file).strip()
        except RuntimeError:  # python-docx not installed — skip, never crash
            logger.info(
                "rag: skipping docx %s — python-docx unavailable (pip install python-docx)",
                file.name,
            )
            return None
        return _Loaded(file.stem, (), (), [(None, text)]) if text else None
    if suffix in IMAGE_SUFFIXES:
        if not ocr_available():
            logger.info(
                "rag: skipping image %s — OCR unavailable (pip install pytesseract pillow + tesseract)",
                file.name,
            )
            return None
        try:
            text = ocr_image(file).strip()
        except OcrUnavailable:
            return None
        return _Loaded(file.stem, (), (), [(None, text)]) if text else None
    return None


def _default_kind(file: Path) -> SourceKind:
    suffix = file.suffix.lower()
    if suffix in PDF_SUFFIXES:
        return "pdf"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return "file"


def deny_secret_source(
    file: Path,
    *,
    store: DocumentStore,
    index: DocumentIndex | None,
    source: IngestSource | None,
    classification: str,
) -> None:
    """The one denial: ``file`` classifies secret, so RAG must not hold any of it.

    Removes any earlier chunks and tells the file domain, even with no earlier RAG
    record: its own catalog may still say "indexed" (or carry a lower label) for what
    RAG now refuses. Used by ``ingest_path`` and by the ingest gate's own denials, so
    a denial means the same thing whichever of them reaches it.
    """
    logger.warning(
        "rag: %s classifies secret; not indexed (secret documents never enter RAG)", file
    )
    sid = _source_id(file)
    if store.get_source(sid) is not None:
        store.delete_source(sid)
        if index is not None:
            index.delete_source(sid)
    report_removed(source, file, source_id=sid, reason="denied", classification=classification)


def ingest_path(
    path: str | Path,
    *,
    store: DocumentStore,
    index: DocumentIndex | None = None,
    kind: SourceKind = "file",
    source: IngestSource | None = None,
    classification: str | None = None,
) -> IngestResult:
    """Ingest one file or every supported file under a folder. Idempotent.

    When ``source`` is given, every indexed document is reported to it, so the file
    domain can record what RAG holds (FMX4, ADR-0066) — one unified file catalog
    alongside finance docs, media and photos. Without one, ingestion is unchanged.

    Every file indexed is classified (FMX8) by the gate's rule, ``sensitivity.classify``:
    what the file domain knows, ratcheted with a fresh scan of its bytes and of the text
    extracted from them. That label is ratcheted again with the label the source already
    carries (a re-ingest) and with ``classification`` (what the ingest gate approved),
    then recorded on the source and stamped on each chunk, so retrieval carries it to
    egress gating. A label stays or rises, never falls, the same ratchet the gate applies
    whenever two sources disagree; lowering one takes removing the source and ingesting
    it again.

    A file that classifies ``secret`` is refused: it is not indexed, any earlier chunks
    of it are removed, and it is counted in ``sources_denied``. Passing
    ``classification="secret"`` raises :class:`SecretIngestError` (the ingest gate should
    have denied it earlier; use the vault).

    A source with no label (indexed before classification existed) is never skipped as
    unchanged: it is re-ingested, and so classified, on its next sync.
    """
    if classification == "secret":
        raise SecretIngestError(
            "refusing to ingest secret-classified content into RAG: "
            "secret documents never enter the retrievable index; use the vault."
        )
    root = Path(path).expanduser().resolve()
    added = updated = skipped = denied = chunks_total = 0
    touched: list[str] = []

    for file in _iter_files(root):
        sid = _source_id(file)
        existing = store.get_source(sid)
        try:
            mtime = file.stat().st_mtime
        except OSError:
            continue
        # A source indexed before classification existed is re-ingested, never skipped.
        labelled = existing is not None and existing.classification is not None
        # Fast-path: unchanged mtime on an already-indexed file → skip unread.
        if labelled and existing is not None and existing.mtime and existing.mtime == mtime:
            skipped += 1
            continue
        try:
            raw = file.read_bytes()
        except OSError:
            continue
        sha = _sha_bytes(raw)
        if labelled and existing is not None and existing.content_sha == sha:
            # Content identical (mtime touched only) — refresh mtime, no reindex.
            store.upsert_source(
                id=sid,
                path=str(file),
                kind=existing.kind,
                title=existing.title,
                content_sha=sha,
                tags=existing.tags,
                links=existing.links,
                mtime=mtime,
                classification=existing.classification,
            )
            skipped += 1
            continue

        loaded = _load(file, raw)
        if loaded is None:  # unreadable / image without OCR / empty scan
            skipped += 1
            continue

        scanned, _ = classify(file, source, raw=raw, texts=tuple(t for _, t in loaded.units))
        file_classification = ratchet(
            existing.classification if existing is not None else None, scanned, classification
        )
        if file_classification == "secret":
            deny_secret_source(
                file,
                store=store,
                index=index,
                source=source,
                classification=file_classification,
            )
            denied += 1
            continue

        file_kind = kind if file.suffix.lower() in TEXT_SUFFIXES else _default_kind(file)
        store.upsert_source(
            id=sid,
            path=str(file),
            kind=file_kind,
            title=loaded.title,
            content_sha=sha,
            tags=loaded.tags,
            links=loaded.links,
            mtime=mtime,
            classification=file_classification,
        )
        # tags/links context line → embedded into chunk 0 (markdown vault graph).
        ctx = (
            context_line(ParsedNote(loaded.title, "", loaded.tags, loaded.links))
            if (loaded.tags or loaded.links)
            else ""
        )
        doc_chunks: list[DocumentChunk] = []
        for page, text in loaded.units:
            for c in chunk_markdown(text, file_title=loaded.title):
                idx = len(doc_chunks)
                doc_chunks.append(
                    DocumentChunk(
                        id=f"{sid}:{idx}",
                        source_id=sid,
                        source_path=str(file),
                        title=c.title,
                        chunk_index=idx,
                        text=f"{ctx}\n\n{c.text}" if ctx and idx == 0 else c.text,
                        page=page,
                        classification=file_classification,
                    )
                )
        store.replace_chunks(sid, doc_chunks)
        if index is not None:
            index.delete_source(sid)
            index.index_chunks(doc_chunks)
        if source is not None:
            _report_indexed(
                source,
                file,
                raw,
                kind=file_kind,
                content_sha=sha,
                mtime=mtime,
                classification=file_classification,
            )
        chunks_total += len(doc_chunks)
        touched.append(str(file))
        if existing is None:
            added += 1
        else:
            updated += 1

    return IngestResult(
        sources_added=added,
        sources_updated=updated,
        sources_skipped=skipped,
        chunks_indexed=chunks_total,
        paths=tuple(touched),
        sources_denied=denied,
    )


def sync_all(
    *,
    store: DocumentStore,
    index: DocumentIndex | None = None,
    source: IngestSource | None = None,
) -> IngestResult:
    """Re-ingest every registered source path (picks up external edits).

    Each file is classified again when its content changed (or when it has no label
    yet); see ``ingest_path``.
    """
    added = updated = skipped = denied = chunks_total = 0
    touched: list[str] = []
    for registered in store.list_sources():
        r = ingest_path(
            registered.path, store=store, index=index, kind=registered.kind, source=source
        )
        added += r.sources_added
        updated += r.sources_updated
        skipped += r.sources_skipped
        chunks_total += r.chunks_indexed
        denied += r.sources_denied
        touched.extend(r.paths)
    return IngestResult(
        sources_added=added,
        sources_updated=updated,
        sources_skipped=skipped,
        chunks_indexed=chunks_total,
        paths=tuple(touched),
        sources_denied=denied,
    )


class EmbedderConflict(RuntimeError):
    """The persisted collection was made under another embedding model (``--reset-collection``)."""


class EmptyStoreRefused(RuntimeError):
    """A rebuild would wipe a populated index because the store has no chunks."""


class IndexRebuildIncomplete(RuntimeError):
    """The collection was reset but the rebuild failed part-way: the index is partial."""


def reindex_all(*, store: DocumentStore, index: DocumentIndex, force: bool = False) -> int:
    """Rebuild the vector index from the store's chunks; return how many were indexed.

    The repair for a lost, corrupted or drifted ``chroma_docs``: the store holds the
    canonical chunk text (and each chunk's classification), the index only mirrors it.
    ``sync_all`` cannot do this job, because it skips every file whose mtime or content
    hash is unchanged and so never re-populates an empty index. Reads no source file.
    Raises ``RuntimeError`` when the index is unavailable (nothing to rebuild into).

    Safety rule: a rebuild makes the index equal to the store, so an empty store (a wrong
    ``IRIS_DATA_DIR``, a fresh or lost ``rag.db``) would delete every vector. When the
    store has zero chunks but the index does not, raises ``EmptyStoreRefused`` unless
    ``force=True``. The rule lives here so every caller (CLI, SDK, API) gets it.

    Race: a chunk a running server ingests while the rebuild runs can be absent from the
    snapshot and so be dropped from the index until the next ``sync``; see
    ``docs/architecture/memory-subsystem.md``.
    """
    if not index.is_ready:
        if index.embedder_conflict:
            raise EmbedderConflict(
                "the document index is unavailable: it was built with a different "
                "embedding model than the one now configured. Your documents, labels and "
                "sources are safe in rag.db. Run `iris docs reindex --reset-collection` "
                "to delete the vector index and rebuild it from them"
            )
        raise RuntimeError("document index unavailable; cannot rebuild it")
    if not force and store.count_all_chunks() == 0 and index.count() > 0:
        raise EmptyStoreRefused(
            f"the chunk store is empty but the index holds {index.count()} entries; a rebuild "
            "would delete them all. Check IRIS_DATA_DIR points at the right rag.db, or "
            "re-run with --force to empty the index deliberately"
        )
    return index.rebuild(store.iter_chunks())


def reset_and_reindex(*, store: DocumentStore, index: DocumentIndex, force: bool = False) -> int:
    """Delete the docs collection and rebuild it from the store; return chunks indexed.

    The explicit repair for an embedding-model change, which leaves the persisted
    collection unopenable (``EmbedderConflict`` from ``reindex_all``). Only the
    ``iris_documents`` collection is deleted; ``rag.db`` (every source and every label)
    is read, never written, and no source file is read. ``store.iter_chunks()`` carries each
    chunk's label, so the rebuilt index mirrors them exactly. Never calls the file-domain
    ingest seam. Idempotent: running it again rebuilds the same contents.

    Same safety rule as ``reindex_all``: a store with zero chunks would leave an empty
    index in place of a populated (or unreadable) one, so it raises ``EmptyStoreRefused``
    unless ``force=True``; the check runs before anything is deleted.

    Invalidates other handles on the collection (a running server's); see
    ``DocumentIndex.reset_collection``. A failure after the delete raises
    ``IndexRebuildIncomplete`` (the index is partial until the command is rerun).
    """
    if not force and store.count_all_chunks() == 0 and (index.count() > 0 or not index.is_ready):
        raise EmptyStoreRefused(
            "the chunk store is empty, so resetting the vector index would leave it empty. "
            "Check IRIS_DATA_DIR points at the right rag.db, or re-run with --force to "
            "empty the index deliberately"
        )
    index.reset_collection()
    try:
        count = index.rebuild(store.iter_chunks())
    except Exception as exc:
        raise IndexRebuildIncomplete(
            f"the rebuild failed ({exc}); the vector index is incomplete and rag.db is "
            "untouched. Rerun `iris docs reindex --reset-collection`: it is idempotent"
        ) from exc
    logger.info("document index reset and rebuilt from the store (chunks=%d)", count)
    return count


__all__ = [
    "SecretIngestError",
    "reset_and_reindex",
    "EmbedderConflict",
    "ingest_path",
    "sync_all",
    "reindex_all",
    "EmptyStoreRefused",
    "IndexRebuildIncomplete",
    "TEXT_SUFFIXES",
    "PDF_SUFFIXES",
    "IMAGE_SUFFIXES",
    "DOCX_SUFFIXES",
]
