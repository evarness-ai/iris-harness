"""The runtime's answer to the learning layer's eval-runtime seam.

Building a runtime is a composition-root job, so the factory lives here and
registers itself with :mod:`iris_harness.services.learning.eval_runtime` at import time.
``runtime.bootstrap`` imports this module, so any process that can build a
runtime can also build an isolated eval one — and a process that cannot gets the
seam's explicit error instead of a silent fall-back to production.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from iris_harness.services.learning.eval_runtime import register_eval_runtime_factory


def build_eval_runtime(*, config_dir: Path | None = None) -> tuple[Any, Path]:
    """Build an isolated runtime for eval and return ``(runtime, scratch_data_dir)``.

    Throwaway temp ``data_dir`` (eval signals never reach the real ``learning.db``)
    and background scheduler off (no heartbeats). Heavy — it builds the full
    runtime — so build once per eval session, run all arms, then discard the
    scratch dir. The caller owns cleanup of the returned path.
    """
    # Imported here, not at module scope: bootstrap imports this module to register
    # the factory, so a top-level import would be circular.
    from iris_harness.runtime.bootstrap import build_runtime

    data_dir = Path(tempfile.mkdtemp(prefix="iris-eval-"))
    runtime = build_runtime(
        config_dir=config_dir,
        data_dir=data_dir,
        use_background_scheduler=False,
    )
    return runtime, data_dir


register_eval_runtime_factory(build_eval_runtime)

__all__ = ["build_eval_runtime"]
