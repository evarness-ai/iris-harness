"""RAG ingest gate — the approval seam in front of document ingestion (FMX8).

Nothing enters the retrievable index without passing through here:

- :func:`propose_rag_ingest` classifies the file (what the file domain already
  knows + a fresh content scan, ratcheted to the more sensitive of the two,
  so a missing file domain can only make the gate stricter) and returns an
  :class:`IngestProposal` for the human-approval flow. A ``secret`` file never
  produces a proposal — it raises :class:`IngestDeniedError` immediately.
- :func:`execute_rag_ingest` runs after approval. It re-resolves and RE-SCANS
  the file (TOCTOU guard): if the bytes changed since the proposal, or the
  re-scan says secret, it denies instead of ingesting.

Both functions are pure/injectable — the runtime wiring owns the confirmation
stash and the Action Center task; this module owns the policy.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from iris_harness.kernel.governance.file_scan import scan_file
from iris_harness.services.rag.index import DocumentIndex
from iris_harness.services.rag.ingest import ingest_path
from iris_harness.services.rag.ingest_source import IngestSource, KnownFile
from iris_harness.services.rag.models import IngestResult
from iris_harness.services.rag.store import DocumentStore

logger = logging.getLogger(__name__)

_VAULT_REFUSAL_MESSAGE = (
    "secret documents never enter RAG; use the vault "
    "(`iris files vault add`) to store confidential content."
)

# Sensitivity ratchet: classification can only go UP when sources disagree.
# Unknown labels rank as "personal" (defensive: never ratchet an odd label down).
_CLASSIFICATION_ORDER = {"public": 0, "internal": 1, "personal": 2, "secret": 3}
_UNKNOWN_RANK = _CLASSIFICATION_ORDER["personal"]


class IngestDeniedError(RuntimeError):
    """The ingest gate refused a document (secret content, or it changed
    between proposal and approval). The message is user-facing and actionable."""


@dataclass(frozen=True)
class IngestProposal:
    """An approved-pending request to ingest one file into RAG."""

    path: str  # the path as the user gave it
    resolved_path: str  # absolute, symlink-resolved
    classification: str  # ratcheted max(what the domain knows, content scan)
    document_type: str | None  # owner-facing facet from the file domain, if known
    size: int  # bytes at proposal time
    reason: str  # human-readable summary for the approval prompt
    content_sha: str  # sha256 at proposal time (TOCTOU guard)

    def summary(self) -> str:
        return self.reason


def _rank(classification: str) -> int:
    return _CLASSIFICATION_ORDER.get(classification, _UNKNOWN_RANK)


def _ratchet(*classifications: str | None) -> str:
    """Return the most sensitive of the given classifications (missing → public)."""
    present = [c for c in classifications if c]
    if not present:
        return "public"
    return max(present, key=_rank)


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _known(source: IngestSource | None, resolved: Path) -> KnownFile | None:
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


def _classify(resolved: Path, source: IngestSource | None) -> tuple[str, KnownFile | None]:
    """Classification = ratchet(what the file domain knows, fresh content scan)."""
    row = _known(source, resolved)
    scanned = scan_file(resolved).classification
    return _ratchet(row.classification if row else None, scanned), row


def propose_rag_ingest(path: str | Path, source: IngestSource | None = None) -> IngestProposal:
    """Classify ``path`` and build an :class:`IngestProposal` for approval.

    Raises :class:`IngestDeniedError` for secret content (no proposal is ever
    created for a secret file) and for missing / non-file paths.
    """
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise IngestDeniedError(
            f"cannot ingest {resolved}: not an existing file "
            "(point add_to_rag at a single readable file)."
        )
    classification, row = _classify(resolved, source)
    if classification == "secret":
        raise IngestDeniedError(
            f"denied: {resolved.name} is classified secret — {_VAULT_REFUSAL_MESSAGE}"
        )
    size = resolved.stat().st_size
    document_type = row.document_type if row else None
    facet = f" ({document_type})" if document_type else ""
    reason = (
        f"add {resolved.name}{facet} to the document index: "
        f"{classification} content, {size} bytes."
    )
    return IngestProposal(
        path=str(path),
        resolved_path=str(resolved),
        classification=classification,
        document_type=document_type,
        size=size,
        reason=reason,
        content_sha=_sha_file(resolved),
    )


def execute_rag_ingest(
    proposal: IngestProposal,
    *,
    store: DocumentStore,
    index: DocumentIndex | None = None,
    source: IngestSource | None = None,
) -> IngestResult:
    """Ingest an approved proposal — after re-resolving and RE-SCANNING it.

    TOCTOU guard: the approval was granted for the bytes seen at proposal
    time. If the file vanished, its content changed, or a fresh scan now says
    secret, this raises :class:`IngestDeniedError` instead of ingesting.
    """
    resolved = Path(proposal.resolved_path).expanduser().resolve()
    if not resolved.is_file():
        raise IngestDeniedError(
            f"cannot ingest {resolved}: the file no longer exists — propose again."
        )
    if _sha_file(resolved) != proposal.content_sha:
        raise IngestDeniedError(
            f"denied: {resolved.name} changed since it was approved — "
            "the approval covered different content; propose again."
        )
    classification, _ = _classify(resolved, source)
    classification = _ratchet(classification, proposal.classification)
    if classification == "secret":
        raise IngestDeniedError(
            f"denied: {resolved.name} now scans as secret — {_VAULT_REFUSAL_MESSAGE}"
        )
    return ingest_path(
        resolved,
        store=store,
        index=index,
        source=source,
        classification=classification,
    )


__all__ = [
    "IngestDeniedError",
    "IngestProposal",
    "propose_rag_ingest",
    "execute_rag_ingest",
]
