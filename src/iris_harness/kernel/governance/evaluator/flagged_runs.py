"""Flagged-run thought persistence for evaluator signals.

Signals such as ``loop_detect`` and ``goal_drift`` can optionally persist
the thoughts that triggered a halt / approval request into a ChromaDB
collection for later operator inspection. The writer is intentionally
best-effort: evaluator verdicts must never depend on Chroma availability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from iris_harness.foundation.persistence import data_path
from iris_harness.kernel.governance.hooks.types import DataClassification

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "flagged_run_thoughts"


@runtime_checkable
class FlaggedRunThoughtWriter(Protocol):
    """Best-effort sink for flagged evaluator thoughts."""

    def record(
        self,
        *,
        run_id: str,
        signal: str,
        step_id: int,
        thought: str,
        embedding: list[float],
        classification: DataClassification | None,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...


@dataclass
class ChromaFlaggedRunThoughtWriter:
    """Persist flagged thoughts into a local ChromaDB collection.

    The collection shares the repo's standard ``data/chroma`` location so
    operators only manage one vector-store footprint. Rows are keyed by
    ``run_id:signal:step_id`` to keep writes idempotent across retries.
    """

    persist_dir: Path = field(default_factory=lambda: data_path("chroma"))

    def __post_init__(self) -> None:
        self._ok = False
        self._collection: Any = None
        try:
            import chromadb

            from iris_harness.foundation.persistence.embedding import (
                collection_kwargs,
            )

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.persist_dir))
            # Shared embedding function (Phase 3): one ONNX model per process.
            self._collection = client.get_or_create_collection(
                _COLLECTION_NAME, **collection_kwargs()
            )
            self._ok = True
        except Exception:
            logger.warning(
                "flagged-run thought writer unavailable; evaluator persistence disabled",
                exc_info=True,
            )

    @property
    def is_ready(self) -> bool:
        return self._ok

    def record(
        self,
        *,
        run_id: str,
        signal: str,
        step_id: int,
        thought: str,
        embedding: list[float],
        classification: DataClassification | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self._ok or not thought.strip() or not embedding:
            return

        doc_id = f"{run_id}:{signal}:{step_id}"
        payload: dict[str, Any] = {
            "run_id": run_id,
            "signal": signal,
            "step_id": step_id,
            "classification": classification or "",
            "ts": datetime.now(UTC).isoformat(),
        }
        for key, value in (metadata or {}).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                payload[key] = value

        try:
            self._collection.upsert(
                ids=[doc_id],
                documents=[thought],
                embeddings=[embedding],
                metadatas=[payload],
            )
        except Exception:
            logger.warning("failed to persist flagged evaluator thought", exc_info=True)
