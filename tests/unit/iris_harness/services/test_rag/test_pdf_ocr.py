"""Tests for PDF + image/OCR ingestion (RAG R2).

PDF text + OCR are injected/monkeypatched (the real backends — pypdfium2 /
tesseract — are exercised by the skipif-guarded live test and verified
manually), mirroring how Finance tests stub PDF extraction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.services.rag import ingest as ingest_mod
from iris_harness.services.rag.ingest import ingest_path
from iris_harness.services.rag.ocr import ocr_available
from iris_harness.services.rag.qa import answer_question
from iris_harness.services.rag.retrieve import search_documents
from iris_harness.services.rag.store import DocumentStore


@pytest.fixture
def store(tmp_path: Path) -> DocumentStore:
    s = DocumentStore(db_path=tmp_path / "rag.db")
    s.ensure_schema()
    return s


# ─── PDF (text extraction monkeypatched) ────────────────────────────────


def test_pdf_chunks_carry_page_citations(
    store: DocumentStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake bytes")
    monkeypatch.setattr(
        ingest_mod,
        "extract_pdf_pages",
        lambda path: ["Revenue grew in Q1.", "Risks are detailed on page two."],
    )
    r = ingest_path(pdf, store=store, index=None)
    assert r.sources_added == 1 and r.chunks_indexed == 2
    assert store.list_sources()[0].kind == "pdf"

    hits = search_documents("risks detailed", store=store, index=None)
    assert hits and hits[0].citation.page == 2
    assert "report.pdf p.2" in hits[0].citation.label()


def test_pdf_empty_page_ocr_fallback(
    store: DocumentStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(ingest_mod, "extract_pdf_pages", lambda path: ["", ""])  # image-only
    monkeypatch.setattr(ingest_mod, "ocr_available", lambda: True)
    monkeypatch.setattr(ingest_mod, "render_pdf_page", lambda path, i: object())
    monkeypatch.setattr(ingest_mod, "ocr_image", lambda img: f"ocr text page {id(img) % 2}")
    r = ingest_path(pdf, store=store, index=None)
    assert r.chunks_indexed == 2  # both image-only pages recovered via OCR


def test_scanned_pdf_skipped_without_ocr(
    store: DocumentStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(ingest_mod, "extract_pdf_pages", lambda path: ["", ""])
    monkeypatch.setattr(ingest_mod, "ocr_available", lambda: False)
    r = ingest_path(pdf, store=store, index=None)
    assert r.sources_added == 0 and r.sources_skipped == 1  # nothing indexable, no crash


# ─── images (OCR) ────────────────────────────────────────────────────────


def test_image_ingested_when_ocr_available(
    store: DocumentStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    img = tmp_path / "whiteboard.png"
    img.write_bytes(b"\x89PNG fake")
    monkeypatch.setattr(ingest_mod, "ocr_available", lambda: True)
    monkeypatch.setattr(ingest_mod, "ocr_image", lambda p: "Action item: ship the RAG agent.")
    r = ingest_path(img, store=store, index=None)
    assert r.sources_added == 1 and store.list_sources()[0].kind == "image"
    ans = answer_question("what is the action item?", store=store, index=None)
    assert "ship the RAG agent" in ans.answer


def test_image_skipped_without_ocr(
    store: DocumentStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8 fake jpeg")
    monkeypatch.setattr(ingest_mod, "ocr_available", lambda: False)
    r = ingest_path(img, store=store, index=None)
    assert r.sources_added == 0 and r.sources_skipped == 1


# ─── real OCR (only where tesseract is installed) ────────────────────────


@pytest.mark.skipif(not ocr_available(), reason="tesseract/pytesseract not installed")
def test_real_ocr_roundtrip(store: DocumentStore, tmp_path: Path) -> None:
    from PIL import Image, ImageDraw

    img_path = tmp_path / "note.png"
    image = Image.new("RGB", (520, 90), "white")
    ImageDraw.Draw(image).text((10, 30), "INVOICE TOTAL 4242 USD", fill="black")
    image.save(img_path)

    r = ingest_path(img_path, store=store, index=None)
    assert r.sources_added == 1
    hits = search_documents("invoice total", store=store, index=None)
    assert hits and "4242" in hits[0].text
