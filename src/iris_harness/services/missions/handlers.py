"""Generic mission step handlers.

The auto-created missions (see :mod:`iris_harness.services.missions.proposer`) bind to the
``agent_query`` handler: each step runs its ``payload["query"]`` (or the step name)
through the agent and records the output. The actual agent call is injected
(``run_query``) so the missions package stays decoupled from the runtime and the
handler is unit-testable without a model.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from iris_harness.services.missions.engine import MissionHandler
from iris_harness.services.missions.models import Mission, MissionStep, StepStatus

logger = logging.getLogger(__name__)

RunQuery = Callable[[str], str]


def make_agent_query_handler(run_query: RunQuery) -> MissionHandler:
    """Build the ``agent_query`` mission handler from a ``run_query(text) -> str``.

    Per step: run the step's query through the agent, store a capped output, and
    report COMPLETED / FAILED. Never raises — the engine treats a raised handler as a
    hard mission failure, so we convert errors into a FAILED step instead.
    """

    def handler(mission: Mission, step: MissionStep) -> StepStatus:
        query = str(step.payload.get("query") or step.name or "").strip()
        if not query:
            step.error = "empty step query"
            return StepStatus.FAILED
        try:
            step.output = (run_query(query) or "")[:4000]
        except Exception as exc:  # a step error must not crash the engine
            logger.debug("mission step %r failed", step.name, exc_info=True)
            step.error = f"{type(exc).__name__}: {exc}"
            return StepStatus.FAILED
        return StepStatus.COMPLETED

    return handler


__all__ = ["make_agent_query_handler", "RunQuery"]
