"""IRIS runtime — composition root that wires every subsystem into one harness."""

from __future__ import annotations

from .bootstrap import (
    ChatResult,
    IrisRuntime,
    build_runtime,
)

__all__ = ["ChatResult", "IrisRuntime", "build_runtime"]
