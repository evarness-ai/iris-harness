"""How RAG classifies a document: one rule for every ingest (FMX8).

Every document that enters the retrievable index is classified by this rule, whichever
surface brought it in (the ingest gate, ``iris docs add`` / ``sync``, ``POST /rag/upload``):

- the classification is the most sensitive of what the file domain already knows about
  the path (:meth:`IngestSource.known_file`) and a fresh scan of the content
  (``governance.file_scan``); a missing file domain can only make it stricter;
- a label only ratchets up when sources disagree, and an unknown label ranks as
  ``personal`` (never ratchet an odd label down); a scan that fails counts as
  ``personal`` (fail closed, never public);
- ``secret`` never enters the index.

``ingest_gate`` applies it before the owner approves an ingest; ``ingest`` applies it to
every file it indexes, with the label the source already carries as one more source to
ratchet against.
"""

from __future__ import annotations

import logging
from pathlib import Path

from iris_harness.kernel.governance.file_scan import scan_file, scan_text
from iris_harness.services.rag.ingest_source import IngestSource, KnownFile

logger = logging.getLogger(__name__)

# Sensitivity ratchet: classification can only go UP when sources disagree.
# Unknown labels rank as "personal" (defensive: never ratchet an odd label down).
_CLASSIFICATION_ORDER = {"public": 0, "internal": 1, "personal": 2, "secret": 3}
_UNKNOWN_RANK = _CLASSIFICATION_ORDER["personal"]
_SCAN_HEAD = 200_000  # bytes (or characters) scanned, the head ``scan_file`` reads
_SCAN_FAILED = "personal"  # the rank an unknown label gets


def rank(classification: str) -> int:
    return _CLASSIFICATION_ORDER.get(classification, _UNKNOWN_RANK)


def ratchet(*classifications: str | None) -> str:
    """Return the most sensitive of the given classifications (missing -> public)."""
    present = [c for c in classifications if c]
    if not present:
        return "public"
    return max(present, key=rank)


def known_file(source: IngestSource | None, resolved: Path) -> KnownFile | None:
    """What the file domain knows about ``resolved``; advisory, so never fatal."""
    if source is None:
        return None
    try:
        return source.known_file(resolved)
    except Exception as exc:  # the lookup is advisory, never fatal
        logger.warning(
            "rag: ingest source lookup failed (%s); classifying from the content scan alone",
            type(exc).__name__,
            exc_info=True,
        )
        return None


def _scan(resolved: Path, raw: bytes | None, texts: tuple[str, ...]) -> str:
    """Scan the content; a scan that fails is recorded as personal, never public."""
    try:
        if raw is None:
            found = [scan_file(resolved).classification]
        else:
            head = raw[:_SCAN_HEAD].decode("utf-8", errors="ignore")
            found = [scan_text(head).classification]
        found.extend(scan_text(t[:_SCAN_HEAD]).classification for t in texts if t)
    except Exception as exc:  # a broken scanner fails closed, not open
        logger.warning(
            "rag: classification scan failed for %s (%s); recording it as personal",
            resolved,
            type(exc).__name__,
            exc_info=True,
        )
        return _SCAN_FAILED
    return ratchet(*found)


def classify(
    resolved: Path,
    source: IngestSource | None,
    *,
    raw: bytes | None = None,
    texts: tuple[str, ...] = (),
) -> tuple[str, KnownFile | None]:
    """Classification = ratchet(what the file domain knows, fresh content scan).

    ``raw`` scans those exact bytes (the ones about to be indexed) instead of re-reading
    the file. ``texts`` adds the text extracted from them (PDF pages, OCR, docx), so a
    PDF or scan is classified by what it says, not only by its encoded bytes.
    """
    row = known_file(source, resolved)
    scanned = _scan(resolved, raw, texts)
    return ratchet(row.classification if row else None, scanned), row


__all__ = ["classify", "known_file", "rank", "ratchet"]
