"""Post-run review: where a finished loop run is handed to a reviewer (design §9.2).

Every governed loop — the harness's own and the ones plugins build, like the email
agent — reports a run that ENDED (not one paused for the owner) here, with the model
route it ran on. The runtime installs the reviewer once (the governance judge,
``runtime/governance_judge.py``); a loop only names its route, so a plugin needs no
handle on the runtime to be reviewed like the core's loop.

Nothing here may slow or break a run: with no reviewer installed a report is a no-op,
and a raising reviewer is logged and swallowed.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from iris_harness.foundation.process_state import track_globals

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompletedRun:
    """A finished loop run. ``steps`` are the loop's steps as dicts (thought, action,
    action_input, observation, …), replayed steps of a resumed run included."""

    run_id: str
    query: str
    steps: tuple[dict[str, Any], ...]
    final_answer: str
    success: bool

    @property
    def used_tools(self) -> bool:
        return any(step.get("action") for step in self.steps)


RunReviewer = Callable[[CompletedRun, str, str], object]
"""``(run, route, agent_type)``: ``route`` is the routing intent whose model ran it."""

_lock = threading.Lock()
_reviewer: RunReviewer | None = None


def install_run_reviewer(reviewer: RunReviewer | None) -> None:
    """Install the process's reviewer, or remove it with ``None``."""
    global _reviewer  # one reviewer per process, like the health watch
    with _lock:
        _reviewer = reviewer


def review_completed_run(run: CompletedRun, *, route: str, agent_type: str) -> None:
    with _lock:
        reviewer = _reviewer
    if reviewer is None:
        return
    try:
        reviewer(run, route, agent_type)
    except Exception:  # a reviewer must never break the run it reviews
        logger.exception("run reviewer failed for run %s", run.run_id)


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_reviewer")
