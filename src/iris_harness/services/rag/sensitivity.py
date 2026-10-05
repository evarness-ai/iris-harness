"""How RAG classifies a document: the rule the ingest gate applies (FMX8).

One rule, used at every point where content enters the index:

- the classification is the most sensitive of what the file domain already knows about
  the path (:meth:`IngestSource.known_file`) and a fresh scan of the content
  (``governance.file_scan``); a missing file domain can only make it stricter;
- a label only ratchets up when sources disagree, and an unknown label ranks as
  ``personal`` (never ratchet an odd label down);
- ``secret`` never enters the index.

``ingest_gate`` applies it before the owner approves a first ingest; ``ingest`` applies it
again when a document the gate stamped is re-ingested after an edit, with the stamp it
already carries as one more source to ratchet against.
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
_SCAN_HEAD = 200_000  # bytes scanned, the head ``file_scan.scan_file`` reads


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


def classify(
    resolved: Path, source: IngestSource | None, *, raw: bytes | None = None
) -> tuple[str, KnownFile | None]:
    """Classification = ratchet(what the file domain knows, fresh content scan).

    ``raw`` scans those exact bytes (the ones about to be indexed) instead of re-reading
    the file, so the label belongs to the content that is stored.
    """
    row = known_file(source, resolved)
    if raw is None:
        scanned = scan_file(resolved).classification
    else:
        scanned = scan_text(raw[:_SCAN_HEAD].decode("utf-8", errors="ignore")).classification
    return ratchet(row.classification if row else None, scanned), row


__all__ = ["classify", "known_file", "rank", "ratchet"]
