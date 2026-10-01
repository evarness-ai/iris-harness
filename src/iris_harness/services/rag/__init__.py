"""Document RAG — index the user's existing docs (markdown/Obsidian/PDF) for
cited retrieval, without owning them (connector model).

R0 ships the foundation: a `DocumentStore` registry + markdown chunker +
`DocumentIndex` (a dedicated ChromaDB collection) + cited `search_documents`.
Obsidian vault (R1), PDF (R2), and grounded Q&A + MCP exposure (R3) layer on
top. Distinct from the auto-synthesised knowledge wiki: documents are
user-provided *source material*.
"""

from .index import DocumentIndex
from .ingest import (
    DOCX_SUFFIXES,
    IMAGE_SUFFIXES,
    PDF_SUFFIXES,
    TEXT_SUFFIXES,
    ingest_path,
    sync_all,
)
from .loaders import extract_docx_text, extract_pdf_pages
from .models import Citation, DocumentChunk, DocumentSource, IngestResult, RetrievedChunk
from .obsidian import ParsedNote, parse_note
from .ocr import OcrUnavailable, ocr_available, ocr_image
from .qa import GroundedAnswer, answer_question, default_llm_call
from .retrieve import search_documents
from .store import DocumentStore

__all__ = [
    "DocumentIndex",
    "DocumentStore",
    "DocumentChunk",
    "DocumentSource",
    "Citation",
    "RetrievedChunk",
    "IngestResult",
    "ParsedNote",
    "parse_note",
    "ingest_path",
    "sync_all",
    "search_documents",
    "GroundedAnswer",
    "answer_question",
    "default_llm_call",
    "extract_pdf_pages",
    "extract_docx_text",
    "ocr_available",
    "ocr_image",
    "OcrUnavailable",
    "TEXT_SUFFIXES",
    "PDF_SUFFIXES",
    "IMAGE_SUFFIXES",
    "DOCX_SUFFIXES",
]
