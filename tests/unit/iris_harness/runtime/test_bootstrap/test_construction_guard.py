"""Reliability: build_runtime degrades on a vector-store fault instead of aborting."""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.memory.identity import loader
from iris_harness.runtime import bootstrap, build_runtime


def _dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    config_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(loader, "USER_MD_PATH", data_dir / "USER.md")
    return config_dir, data_dir


def test_chromadb_failure_degrades_to_keyword_recall(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SemanticIndex (ChromaDB) construction fault must NOT abort startup."""

    class _Boom:
        def __init__(self, *a: object, **k: object) -> None:
            raise RuntimeError("chroma is wedged")

    monkeypatch.setattr(bootstrap, "SemanticIndex", _Boom)
    config_dir, data_dir = _dirs(tmp_path, monkeypatch)

    runtime = build_runtime(
        config_dir=config_dir, data_dir=data_dir, use_background_scheduler=False
    )
    # Built successfully, but recall fell back to keyword-only (no index).
    assert runtime is not None
    assert runtime.memory_retriever.index is None
    # The memory store itself is still live (degradation is index-only).
    assert runtime.memory_store is not None


def test_memory_store_fault_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An essential memory-store fault re-raises (clear failure, not silent degrade)."""

    class _BadStore:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def ensure_schema(self) -> None:
            raise RuntimeError("disk full")

    monkeypatch.setattr(bootstrap, "MemoryStore", _BadStore)
    config_dir, data_dir = _dirs(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="disk full"):
        build_runtime(config_dir=config_dir, data_dir=data_dir, use_background_scheduler=False)
