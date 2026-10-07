"""Stage 8a: tell the owner when the answer they are about to read has redacted text in it.

The external-content floor cuts instruction-like spans out of third-party text and leaves a
marker; an answer that repeats such text (a tool answered directly, a model that quoted it)
used to show the bare marker with no word of why (issue #139). One shared stage, so the
generated path, the deterministic path, ``chat`` and ``chat_stream`` all say it the same way,
once per turn: the notice is appended to the final answer after the response checks and
before the turn is recorded, so the transcript holds exactly what the owner saw.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

from iris_harness.kernel.governance.external_content import add_redaction_notice
from iris_harness.runtime.turn.host import TurnHost
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import StreamEvent


def run(runtime: TurnHost, state: TurnState) -> Iterator[StreamEvent]:
    result = state.result
    if result is not None:
        noted = add_redaction_notice(result.response)
        if noted != result.response:
            state.result = replace(
                result, response=noted, metadata={**result.metadata, "redaction_notice": True}
            )
    yield from ()
