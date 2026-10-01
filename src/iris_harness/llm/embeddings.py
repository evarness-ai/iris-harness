"""Shared sentence-embedding access (OSS plan M3.1).

``embed_corpus`` encodes text into L2-normalised 384-dim vectors. It lived in
``iris_personal.email.discovery`` because email category discovery was its first
caller, but nothing about it is email-shaped: finance category matching and the
email semantic index both reach for the same embedder, and the M3 split would
otherwise have put a shared capability behind a mounted plugin (and created a
plugin -> plugin import, since finance is itself a plugin at M4).

So it sits here, next to the tier router, as the harness's one model-access
point for embeddings — the same call the M2.6 split made for ``ApprovalError``:
what both halves need is defined once, in the core half.

Two backends produce the same vectors (all-MiniLM-L6-v2, 384-dim, L2-normalised):

* ``onnx`` (default) — ChromaDB's ``DefaultEmbeddingFunction``, the ONNX export
  of the same model. It is already resident in every iris-api process because
  the memory subsystem's five collections use it
  (``foundation/persistence/embedding.py``), so email triage costs no extra RAM
  and no extra dependency. Measured 2026-09-22: mean cosine 1.0000 against the
  torch model on the same texts, identical kNN predictions.
* ``sentence-transformers`` — the torch model, the optional ``ml`` extra (the only
  dependent of torch, which pulls ~5 GB of CUDA wheels on Linux and ~900 MiB of
  RSS on x86_64). Kept for hosts that already run it and for models other than
  the default (``model_name`` is honoured only by this backend).

Select with ``IRIS_EMBED_BACKEND``. Both imports stay **lazy** so ``iris --help``
and the test suite never pay the cold-import cost, and a missing extra degrades
with a clear message.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from iris_harness.foundation.process_state import track_globals

logger = logging.getLogger(__name__)

EMBED_MODEL_DEFAULT = "sentence-transformers/all-MiniLM-L6-v2"

EMBED_BACKEND_ENV = "IRIS_EMBED_BACKEND"
EMBED_BACKEND_ONNX = "onnx"
EMBED_BACKEND_ST = "sentence-transformers"
EMBED_BACKEND_DEFAULT = EMBED_BACKEND_ONNX

# Texts per call into the ONNX function. Each call pads its batch, and the buffers
# grow with it. Measured 2026-09-22 on the Mac (arm64, 300 real email texts, model
# loaded; RSS growth / wall time): 1 per call +5 MiB 1.5 s, 4 +30 MiB 1.5 s, 8 +109 MiB
# 1.9 s, ChromaDB's default 32 +549 MiB 2.5 s, none of it given back. x86 Linux kept
# far less (+17 MiB for 100 emails at 32) but ran one-at-a-time about 4x slower than
# batched on 2 vCPU, so 4 keeps some batching for small CPUs without the Mac's spike.
# A memory probe on the small server is where this number gets confirmed.
_ONNX_TEXTS_PER_CALL = 4
# all-MiniLM-L6-v2's width; the only model the onnx backend serves.
_MINILM_DIM = 384

# Module-level model cache. SentenceTransformer construction loads the
# tokenizer + config + model weights; even with disk-cached weights the
# 14 sequential triage embed calls observed in the Track 1G e2e run each
# re-paid this cost. Keep one instance per (model_name) for the process
# lifetime. Tests inject a stub via ``EmailTriageClassifier.embedder``,
# so this cache stays out of their way.
_EMBEDDER_CACHE: dict[str, Any] = {}


def embed_backend() -> str:
    """The configured backend name (``onnx`` unless ``IRIS_EMBED_BACKEND`` says otherwise)."""
    value = os.environ.get(EMBED_BACKEND_ENV, EMBED_BACKEND_DEFAULT).strip().lower()
    if value not in (EMBED_BACKEND_ONNX, EMBED_BACKEND_ST):
        raise RuntimeError(
            f"{EMBED_BACKEND_ENV}={value!r} is not a backend: "
            f"use {EMBED_BACKEND_ONNX!r} or {EMBED_BACKEND_ST!r}"
        )
    return value


def _get_embedder(model_name: str) -> Any:
    cached = _EMBEDDER_CACHE.get(model_name)
    if cached is not None:
        return cached
    try:
        from sentence_transformers import SentenceTransformer  # lazy
    except ImportError as exc:  # optional `ml` extra (torch) not installed
        raise RuntimeError(
            "corpus discovery needs the optional ML stack: run `poetry install -E ml` "
            f"(or leave {EMBED_BACKEND_ENV} at its default, {EMBED_BACKEND_ONNX!r})"
        ) from exc

    logger.info("loading embedder %s", model_name)
    model = SentenceTransformer(model_name)
    _EMBEDDER_CACHE[model_name] = model
    return model


def _embed_onnx(texts: list[str], model_name: str) -> np.ndarray[Any, Any]:
    """Encode with the process-wide ONNX MiniLM the memory subsystem already holds."""
    if model_name != EMBED_MODEL_DEFAULT:
        raise RuntimeError(
            f"the {EMBED_BACKEND_ONNX!r} backend serves only {EMBED_MODEL_DEFAULT!r}; "
            f"set {EMBED_BACKEND_ENV}={EMBED_BACKEND_ST} for {model_name!r}"
        )
    from iris_harness.foundation.persistence.embedding import (  # lazy
        default_embedding_function,
    )

    ef = default_embedding_function()
    if ef is None:
        raise RuntimeError("the shared ONNX embedding function is unavailable in this process")
    if not texts:
        return np.empty((0, _MINILM_DIM), dtype=np.float32)
    chunks = [
        ef(texts[i : i + _ONNX_TEXTS_PER_CALL]) for i in range(0, len(texts), _ONNX_TEXTS_PER_CALL)
    ]
    vecs = np.asarray([v for chunk in chunks for v in chunk], dtype=np.float32).reshape(
        len(texts), -1
    )
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    normalised: np.ndarray[Any, Any] = vecs / np.maximum(norms, np.float32(1e-12))
    return normalised


def embed_corpus(texts: list[str], model_name: str = EMBED_MODEL_DEFAULT) -> np.ndarray[Any, Any]:
    """Encode texts → L2-normalized 384-dim vectors.

    With the default ``onnx`` backend this reuses the one embedding function
    the process already loaded for ChromaDB. With ``sentence-transformers`` it
    lazy-imports the torch model AND caches the singleton per ``model_name``,
    so a single-process triage run paying many ``embed_corpus`` calls (one per
    incoming email) does not re-construct the model each time.
    """
    if embed_backend() == EMBED_BACKEND_ONNX:
        logger.info("embedding %d texts (onnx)", len(texts))
        return _embed_onnx(texts, model_name)
    model = _get_embedder(model_name)
    logger.info("embedding %d texts", len(texts))
    vecs = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def reset_embedder_cache() -> None:
    """Test seam — clear the module-level model cache."""
    _EMBEDDER_CACHE.clear()


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_EMBEDDER_CACHE")
