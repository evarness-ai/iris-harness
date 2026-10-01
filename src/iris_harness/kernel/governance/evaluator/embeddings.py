"""Embedding helper for the evaluator's semantic signals.

Story 12.gov-3.6 (and the future ``goal_drift`` signal) need to compare
short prose snippets (ReAct thoughts, original task statements) by
cosine similarity. This module exposes:

- ``Embedder`` — a minimal protocol so signals stay testable. Tests
  pass a deterministic stub; production wires the default ChromaDB
  ONNX MiniLM-L6 embedder (the same model the memory subsystem
  already uses — ~80 MB one-time download).
- ``DefaultEmbedder`` — lazy wrapper around chroma's default function.
  Construction is cheap; the heavy model load is deferred to the
  first ``__call__`` so importing this module doesn't move the ONNX
  cost into process startup. Under ``IRIS_TEST_NULL_EMBEDDINGS`` it never
  fetches the model: it uses a copy already on disk, and without one it
  embeds nothing (``[]``), which every caller treats as "no embedding".
- ``cosine_similarity`` — numpy-backed helper, returns a float in
  ``[-1.0, 1.0]``. Returns ``0.0`` for zero-norm inputs rather than
  raising — signals that get a zero vector should treat it as
  "uncomparable", not as a crash.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Callable that turns one text into a fixed-dimension vector."""

    def __call__(self, text: str) -> Sequence[float]: ...


class DefaultEmbedder:
    """Lazy ChromaDB-backed embedder.

    Importing this class does NOT trigger the ONNX model download —
    the underlying ``chromadb.utils.embedding_functions.DefaultEmbeddingFunction``
    is constructed on the first ``__call__``. That keeps test runs
    that never invoke the embedder free of the ~80 MB cost.
    """

    def __init__(self) -> None:
        self._fn: Any | None = None

    @property
    def available(self) -> bool:
        """False when a call would embed nothing without trying, so a caller can take
        its no-embeddings path up front instead of reading every score as 0.0.

        That is the test flag's promise, "no model fetch": the ONNX download is a
        network call the suite forbids. A model already on disk is used as before (the
        semantic-routing tests assert on real similarities); without one, nothing is
        embedded.
        """
        if self._fn is not None:
            return True
        return not (os.environ.get("IRIS_TEST_NULL_EMBEDDINGS") and not default_model_on_disk())

    def __call__(self, text: str) -> Sequence[float]:
        if self._fn is None:
            if not self.available:
                return []
            self._fn = _load_default_embedding_function()
        # chromadb's embedding function takes a list of inputs and
        # returns a list of vectors. We pass one and unpack one.
        vectors = self._fn([text])
        if not vectors:
            return []
        return list(vectors[0])


def default_model_path() -> Path:
    """Where chromadb keeps the default ONNX model (all-MiniLM-L6-v2) once fetched."""
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
        ONNXMiniLM_L6_V2,
    )

    return Path(ONNXMiniLM_L6_V2.DOWNLOAD_PATH) / ONNXMiniLM_L6_V2.EXTRACTED_FOLDER_NAME


def default_model_on_disk() -> bool:
    """True when the default ONNX model is on disk, so loading it fetches nothing."""
    try:
        return (default_model_path() / "model.onnx").is_file()
    except Exception:  # noqa: BLE001 — no chromadb, no model
        return False


def _load_default_embedding_function() -> Any:
    # Reuse the shared per-process ONNX embedder (Phase 3 footprint); fall back
    # to a fresh instance if it is unavailable so behavior never regresses.
    from iris_harness.foundation.persistence.embedding import (
        default_embedding_function,
    )

    shared = default_embedding_function()
    if shared is not None:
        return shared
    from chromadb.utils import embedding_functions

    return embedding_functions.DefaultEmbeddingFunction()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in ``[-1.0, 1.0]``; ``0.0`` for zero-norm or mismatched shapes."""
    if len(a) == 0 or len(b) == 0 or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += float(x) * float(y)
        norm_a += float(x) * float(x)
        norm_b += float(y) * float(y)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
