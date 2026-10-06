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

The classification rule itself lives in ``sensitivity``, shared with ``ingest``, which
applies it to every file it indexes, whichever surface asked (``iris docs``, the upload
route, ``sync_all``): the label this gate approved is a floor that rule ratchets from.

Both functions are pure/injectable — the runtime wiring owns the confirmation
stash and the Action Center task; this module owns the policy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from iris_harness.services.rag.index import DocumentIndex
from iris_harness.services.rag.ingest import ingest_path
from iris_harness.services.rag.ingest_source import IngestSource
from iris_harness.services.rag.models import IngestResult
from iris_harness.services.rag.sensitivity import classify, ratchet
from iris_harness.services.rag.store import DocumentStore

_VAULT_REFUSAL_MESSAGE = (
    "secret documents never enter RAG; use the vault "
    "(`iris files vault add`) to store confidential content."
)


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


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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
    classification, row = classify(resolved, source)
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
    secret, this raises :class:`IngestDeniedError` instead of ingesting. It also raises
    if ingest itself refuses the file after scanning its extracted text.
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
    classification, _ = classify(resolved, source)
    classification = ratchet(classification, proposal.classification)
    if classification == "secret":
        raise IngestDeniedError(
            f"denied: {resolved.name} now scans as secret — {_VAULT_REFUSAL_MESSAGE}"
        )
    result = ingest_path(
        resolved,
        store=store,
        index=index,
        source=source,
        classification=classification,
    )
    if result.sources_denied:
        # The byte scan above passed, but ingest also scans the text it extracts (PDF,
        # OCR, docx) and refused it: surface that as a denial, not a normal result.
        raise IngestDeniedError(
            f"denied: {resolved.name} classifies secret once its text is extracted — "
            f"{_VAULT_REFUSAL_MESSAGE}"
        )
    return result


__all__ = [
    "IngestDeniedError",
    "IngestProposal",
    "propose_rag_ingest",
    "execute_rag_ingest",
]
