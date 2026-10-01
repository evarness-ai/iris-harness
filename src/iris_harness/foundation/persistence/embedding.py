"""One process-wide ChromaDB embedding function (Phase 3 footprint work).

ChromaDB's ``DefaultEmbeddingFunction`` is an ONNX all-MiniLM-L6-v2 model
(~80 MB in RAM). Calling ``get_or_create_collection(name)`` without an explicit
embedding function makes each collection build its own — so the 5 memory
collections + RAG + email index each loaded a separate copy in the iris_api
process. Share a single instance here and pass it to every collection so the
model loads once per process.

Returns ``None`` when embeddings are stubbed (``IRIS_TEST_NULL_EMBEDDINGS``) or
the load fails, so callers omit the embedding function and degrade to keyword
ranking exactly as before.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from iris_harness.foundation.process_state import track_globals

logger = logging.getLogger(__name__)

_shared: Any = None
_load_failed = False
_lock = threading.Lock()


def default_embedding_function() -> Any | None:
    """Return the shared ONNX MiniLM embedding function, or ``None``.

    ``None`` means "no embeddings available" (stubbed in tests, or the ONNX load
    failed); callers should create/query collections without an embedding
    function and fall back to keyword ranking.
    """
    global _shared, _load_failed
    if os.environ.get("IRIS_TEST_NULL_EMBEDDINGS"):
        return None
    if _shared is not None or _load_failed:
        return _shared
    with _lock:
        if _shared is not None or _load_failed:
            return _shared
        try:
            from chromadb.utils import embedding_functions

            _shared = embedding_functions.DefaultEmbeddingFunction()
            logger.info("shared embedding function loaded (ONNX MiniLM, one per process)")
        except Exception:  # degrade to keyword ranking, never crash
            _load_failed = True
            logger.warning("shared embedding function unavailable", exc_info=True)
    return _shared


def collection_kwargs() -> dict[str, Any]:
    """Kwargs for ``get_or_create_collection`` that pin the shared embedder.

    Empty when no embedder is available so the call behaves exactly as the old
    no-argument form (ChromaDB's own default / keyword fallback).
    """
    ef = default_embedding_function()
    return {"embedding_function": ef} if ef is not None else {}


def _reset_for_tests() -> None:
    """Drop the cached instance (used by tests that toggle the null-embed flag)."""
    global _shared, _load_failed
    with _lock:
        _shared = None
        _load_failed = False


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_shared", "_load_failed")
