"""Optional OCR for images + scanned PDF pages (RAG R2).

OCR is **opt-in**, like Finance's pymupdf4llm backend: the engine (Tesseract,
Apache-2.0) is a *system* binary and ``pytesseract``/``Pillow`` are not bundled
dependencies of an Apache-2.0 project, so this module lazy-imports them and
degrades gracefully when they're absent — ingestion of typed PDFs/markdown is
unaffected. OCR runs locally; images never leave the box.

Install to enable: ``pip install pytesseract pillow`` + the tesseract binary
(``brew install tesseract`` / ``apt-get install tesseract-ocr``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class OcrUnavailable(RuntimeError):
    """Raised when OCR is requested but the toolchain isn't installed."""


def ocr_available() -> bool:
    """True iff pytesseract + Pillow import and the tesseract binary responds."""
    try:
        import pytesseract  # type: ignore[import-untyped]
        from PIL import Image  # noqa: F401

        pytesseract.get_tesseract_version()
        return True
    except Exception:  # noqa: BLE001 — silent-ok: a capability probe; False IS the answer
        return False


def ocr_image(image: str | Path | Any) -> str:
    """OCR a file path or a PIL image to text. Raises OcrUnavailable if absent."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised by ocr_available gate
        raise OcrUnavailable(
            "OCR needs `pip install pytesseract pillow` + the tesseract binary"
        ) from exc

    try:
        img = Image.open(image) if isinstance(image, (str, Path)) else image
        return str(pytesseract.image_to_string(img) or "").strip()
    except Exception as exc:  # bad/corrupt image, or tesseract missing
        raise OcrUnavailable(f"OCR failed: {exc}") from exc


__all__ = ["OcrUnavailable", "ocr_available", "ocr_image"]
