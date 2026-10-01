"""Stage 0 — the turn-level input screen: the kernel's ``PRE_TURN`` hooks on the raw message.

Every turn passes here before anything can answer it. The kernel's input screens
used to fire only where text enters a model (``PRE_CLASSIFY``), so a turn a
deterministic handler answered without a model call was never screened at all
(docs/architecture/deterministic-path-parity.md). This stage fires the same
screens once per turn, on what the user said, and records the classification
they inferred for the rest of the turn.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from dataclasses import replace

from iris_harness.agent.response_curator import GOVERNANCE_BLOCKED_TEXT
from iris_harness.foundation.observability.session_log import log_timeline_event, log_user_message
from iris_harness.kernel.governance import HookContext, HookPoint
from iris_harness.kernel.governance.turn_label import lift_turn_label
from iris_harness.runtime.turn.host import TurnHost
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import ChatResult, StreamEvent

logger = logging.getLogger(__name__)


def run(runtime: TurnHost, state: TurnState) -> Iterator[StreamEvent]:
    message, session_id = state.message, state.session_id
    # The user message is logged once, here, whatever answers it — including a
    # message the screen refuses, which the audit trail must still show arrived.
    log_user_message(session_id, text=message)
    log_timeline_event(
        "turn.start",
        phase="turn.start",
        payload={"message_chars": len(message)},
        session_id=session_id,
    )
    yield StreamEvent(
        kind="trace",
        text="turn.start",
        payload={"name": "turn.start", "message_chars": len(message)},
    )
    runtime._span_input(state.span, message)

    kernel = runtime.governance_kernel
    if kernel is None:
        # Governance explicitly disabled by the operator; kernel_from_env logged it
        # at WARNING when the runtime was built.
        return
    decision, screened = kernel.fire_sync(
        HookPoint.PRE_TURN,
        HookContext(
            hook_point=HookPoint.PRE_TURN,
            run_id=uuid.uuid4().hex,
            agent_type="turn",
            payload={"message": message},
            metadata={"channel": state.request.channel},
        ),
    )
    state.classification = screened.classification
    # The turn's label from here on: the harness stamps it on every code call a plugin
    # makes in this turn, and the loop never goes below it (kernel/governance/turn_label.py).
    lift_turn_label(screened.classification)
    yield StreamEvent(
        kind="trace",
        text="screen.end",
        payload={
            "name": "screen.end",
            "outcome": decision.outcome,
            "classification": screened.classification,
        },
    )

    if decision.outcome in ("deny", "require_approval"):
        # A held user message has no run to resume — nothing has started — so an
        # approval verdict here refuses the turn, as it refuses a model call today.
        # The refused text never reaches session memory, where it would become
        # context for the next turn's model.
        logger.warning(
            "turn screen %s session %s: %s", decision.outcome, session_id, decision.reason
        )
        state.result = ChatResult(
            response=GOVERNANCE_BLOCKED_TEXT,
            intent="governance",
            agent_type="governance",
            sources=("governance",),
            has_errors=True,
            error_summary=decision.reason,
            metadata={
                "governance_screen": decision.outcome,
                "classification": screened.classification,
                "session_id": session_id,
            },
        )
        state.screened_out = True
        return

    if decision.outcome == "transform":
        rewritten = screened.payload.get("message")
        if isinstance(rewritten, str) and rewritten != message:
            state.request = replace(state.request, message=rewritten)
