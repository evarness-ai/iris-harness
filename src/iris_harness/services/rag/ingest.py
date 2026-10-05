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

from iris_harness.kernel.governance.file_scan import scan_text
from iris_harness.services.rag.chunker import chunk_markdown
from iris_harness.services.rag.index import DocumentIndex
from iris_harness.services.rag.ingest_source import IndexedDocument, IngestSource
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
) -> None:
    """Tell the file domain that RAG indexed ``file``. Never breaks ingestion.

    The classification is scanned here, in the core, so it is the same scan on every
    install — with or without a file domain mounted (OSS plan M6.1b).
    """
    try:
        try:
            classification = scan_text(
                raw[:200_000].decode("utf-8", errors="ignore")
            ).classification
        except Exception as exc:  # classification is advisory, never fatal
            # Fail closed: a file nobody could scan is treated as personal, the rank the
            # ingest gate already gives an unknown label, never as public.
            logger.warning(
                "rag: classification scan failed for %s (%s); recording it as personal",
                file,
                type(exc).__name__,
                exc_info=True,
            )
            classification = "personal"
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

    ``classification`` (FMX8) is stamped on every chunk stored/indexed by this
    call so retrieval can carry sensitivity through to egress gating. Passing
    ``"secret"`` raises :class:`SecretIngestError` — secret documents never
    enter RAG (the ingest gate should have denied them earlier; use the vault).
    ``None`` (the default) leaves a first ingest unclassified, as before.

    A source that already carries a classification stamp keeps being classified: when
    its content changed, the new bytes are classified by the gate's rule
    (``sensitivity.classify``: what the file domain knows, ratcheted with a fresh scan)
    and ratcheted with the stamp it carries and with ``classification``. The label can
    stay or rise, never fall, the same ratchet the gate applies whenever two sources
    disagree; lowering one takes removing the source and passing the gate again. If the
    result is ``secret`` the new content is not indexed and the source's earlier chunks
    are removed (counted in ``sources_denied``).
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
        # Fast-path: unchanged mtime on an already-indexed file → skip unread.
        if existing is not None and existing.mtime and existing.mtime == mtime:
            skipped += 1
            continue
        try:
            raw = file.read_bytes()
        except OSError:
            continue
        sha = _sha_bytes(raw)
        if existing is not None and existing.content_sha == sha:
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
            )
            skipped += 1
            continue

        # A classified source stays classified: its new content is classified again.
        file_classification = classification
        prior = store.chunk_classifications(sid) if existing is not None else set()
        if prior:
            scanned, _ = classify(file, source, raw=raw)
            file_classification = ratchet(*prior, scanned, classification)
            if file_classification == "secret":
                logger.warning(
                    "rag: %s now classifies secret; removed from the index "
                    "(secret documents never enter RAG)",
                    file,
                )
                store.delete_source(sid)
                if index is not None:
                    index.delete_source(sid)
                denied += 1
                continue

        loaded = _load(file, raw)
        if loaded is None:  # unreadable / image without OCR / empty scan
            skipped += 1
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
            _report_indexed(source, file, raw, kind=file_kind, content_sha=sha, mtime=mtime)
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

    An edited source that carries a classification stamp is classified again by
    ``ingest_path``; see there.
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


__all__ = [
    "SecretIngestError",
    "ingest_path",
    "sync_all",
    "TEXT_SUFFIXES",
    "PDF_SUFFIXES",
    "IMAGE_SUFFIXES",
    "DOCX_SUFFIXES",
]
