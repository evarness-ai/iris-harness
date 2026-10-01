"""The seam the learning layer uses to get an isolated eval runtime.

Replay pre-flight (ADR-0070) and the routing experiments both need a *real*
runtime to run their workload through — but building one is a composition-root
job, and the learning layer sits well below the composition root. So learning
declares what it needs and the runtime supplies it:
:mod:`iris_harness.runtime.eval_runtime` registers the factory when
``runtime.bootstrap`` is imported, which is any process that has a runtime at
all.

Until then :func:`build_eval_runtime` raises rather than guessing. A pre-flight
that cannot isolate itself must not fall back to the production runtime: that is
exactly the signal pollution the eval sandbox exists to prevent.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from iris_harness.foundation.process_state import track_globals

EvalRuntimeFactory = Callable[..., tuple[Any, Path]]

_factory: EvalRuntimeFactory | None = None


def register_eval_runtime_factory(factory: EvalRuntimeFactory) -> None:
    """Register the callable that builds an isolated eval runtime."""
    global _factory
    _factory = factory


def eval_runtime_factory() -> EvalRuntimeFactory | None:
    """The registered factory, or ``None`` when no runtime layer is loaded."""
    return _factory


def build_eval_runtime(*, config_dir: Path | None = None) -> tuple[Any, Path]:
    """Build an isolated eval runtime and return ``(runtime, scratch_data_dir)``.

    The caller owns cleanup of the returned path. Raises ``RuntimeError`` when no
    factory is registered — see the module docstring for why that is not a
    fallback-to-production situation.
    """
    if _factory is None:
        raise RuntimeError(
            "no eval-runtime factory is registered; import iris_harness.runtime.bootstrap "
            "(or call register_eval_runtime_factory) before running a replay pre-flight"
        )
    return _factory(config_dir=config_dir)


__all__ = [
    "EvalRuntimeFactory",
    "build_eval_runtime",
    "eval_runtime_factory",
    "register_eval_runtime_factory",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_factory")
