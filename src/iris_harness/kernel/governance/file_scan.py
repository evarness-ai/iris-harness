"""Secret / sensitivity detection for file content (FM0).

Wraps the governance credential + PII regex packs (:class:`DataClassifier`) so a
file's content is classified exactly like every other piece of text IRIS handles —
``secret`` (API keys, private keys, tokens), ``personal`` (PII), ``internal``, or
``public``. Confidential content is then handled local-only (ADR-0002): nothing
classified ``secret`` reaches a cloud model.

This lived in ``filemanager/secrets.py`` until M6.1b. It is the kernel's, not the
file domain's: it wraps a kernel classifier, and RAG ingest classifies content on a
core-only install where no file domain is mounted (OSS plan M6, decision 2). The
file domain still calls it — a domain reaching down into the kernel is the allowed
direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .plugins.classifier import DataClassifier

_SCAN_LIMIT = 200_000  # bytes read for classification (heads off huge files)
_classifier = DataClassifier()


@dataclass(frozen=True)
class FileScan:
    classification: str  # secret | personal | internal | public
    matched: tuple[str, ...]

    @property
    def is_sensitive(self) -> bool:
        return self.classification in ("secret", "personal")

    @property
    def is_secret(self) -> bool:
        return self.classification == "secret"


def scan_text(text: str) -> FileScan:
    result = _classifier.classify(text)
    return FileScan(classification=result.classification, matched=tuple(result.matched_patterns))


def scan_file(path: str | Path, *, limit: int = _SCAN_LIMIT) -> FileScan:
    """Classify a file by its (head of) content. Binary/unreadable → public."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read(limit)
        text = raw.decode("utf-8", errors="ignore")
    except OSError:
        return FileScan(classification="public", matched=())
    return scan_text(text)


__all__ = ["FileScan", "scan_text", "scan_file"]
