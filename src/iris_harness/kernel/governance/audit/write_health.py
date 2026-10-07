"""How the ledger's own writes are doing, as one dict every surface renders (issue #134).

``iris system status``, ``GET /governance/audit`` and the System Health row all read this, so
no surface can miss a state the others show. Counts only; no row content.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from iris_harness.kernel.governance.audit import sequence, spool


def write_health(db_path: Path) -> dict[str, Any]:
    """``spool_pending`` / ``spool_rejected`` (on disk), and this process's ``writes_spooled``,
    ``writes_lost`` and ``last_error_class``. ``ok`` is false when anything is non-zero."""
    waiting = spool.state(spool.spool_path_for(db_path))
    stats = sequence.stats()
    return {
        "ok": not (waiting.pending or waiting.rejected or stats.lost),
        "spool_pending": waiting.pending,
        "spool_rejected": waiting.rejected,
        "writes_spooled": stats.spooled,
        "writes_lost": stats.lost,
        "last_error_class": stats.last_cause,
    }


__all__ = ["write_health"]
