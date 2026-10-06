"""The owner is told, once, when text they are reading had a span redacted (issue #139).

The floor cuts instruction-like spans out of third-party text and leaves a marker; an answer
that repeats such text used to show the bare marker with no word of why. One shared turn
stage appends a plain-words notice after the response checks and before the turn is
recorded, so ``chat``, ``chat_stream``, the generated path and the deterministic path all say
it once, and the transcript holds what the owner saw.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from iris_harness.kernel.governance.external_content import (
    MARKER,
    REDACTION_NOTICE,
    add_redaction_notice,
)
from iris_harness.runtime import build_runtime
from iris_harness.runtime.intercepts import InterceptHit, InterceptSpec
from iris_harness.runtime.turn import STAGES, TurnRequest
from iris_harness.runtime.turn.pipeline import OPENER_STAGES, RESUME_STAGES, serves
from iris_harness.runtime.turn.stages import notice
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import ChatResult

ANSWER = f"Your inbox: lunch? {MARKER} and a bill."


def _state(text: str, *, intercepted: bool = False) -> TurnState:
    state = TurnState(request=TurnRequest("m"))
    state.intercepted = intercepted
    state.result = ChatResult(text, "system", "system", ("system",), False, None, {})
    return state


def test_the_notice_is_added_once_and_only_when_there_is_a_marker() -> None:
    noted = add_redaction_notice(ANSWER)
    assert noted.startswith(ANSWER) and noted.count(REDACTION_NOTICE) == 1
    assert add_redaction_notice(noted) == noted  # idempotent: every sink may call it
    assert add_redaction_notice("nothing to say") == "nothing to say"


def test_the_notice_names_no_pattern_and_points_at_where_the_ids_are() -> None:
    assert "iris governance redactions" in REDACTION_NOTICE
    assert "never shows the text" in REDACTION_NOTICE


@pytest.mark.parametrize("intercepted", [False, True], ids=["generated", "deterministic"])
def test_the_stage_appends_the_notice_to_either_kind_of_answer(intercepted: bool) -> None:
    state = _state(ANSWER, intercepted=intercepted)
    list(notice.run(None, state))  # type: ignore[arg-type]
    assert state.result is not None
    assert state.result.response == add_redaction_notice(ANSWER)
    assert state.result.metadata["redaction_notice"] is True


def test_the_stage_leaves_a_clean_answer_alone() -> None:
    state = _state("All quiet.")
    before = state.result
    list(notice.run(None, state))  # type: ignore[arg-type]
    assert state.result is before


def test_the_stage_runs_after_the_checks_and_before_the_record() -> None:
    names = [n for n, _ in STAGES]
    assert names.index("guard") < names.index("notice") < names.index("record")
    assert "notice" in [n for n, _ in RESUME_STAGES]  # a resumed run says it too
    assert "notice" not in [n for n, _ in OPENER_STAGES]  # the welcome holds no third-party text
    for intercepted in (False, True):
        assert serves("notice", _state("x", intercepted=intercepted))


# ---------------------------------------------------------------- both entries, end to end
@pytest.fixture()
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    config_dir = Path(__file__).resolve().parents[5] / "config"
    return build_runtime(
        config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False
    )


def _answer_with(runtime: Any, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    spec = InterceptSpec(name="test_handler", handler="test")
    result = ChatResult(text, "system", "system", ("system",), False, None, {})
    monkeypatch.setattr(
        runtime.intercepts, "dispatch", lambda *_a, **_k: InterceptHit(spec=spec, result=result)
    )


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_an_answer_with_a_redaction_carries_the_notice_once_on_both_entries(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    from iris_harness.foundation.observability.session_log import session_log_path

    _answer_with(runtime, monkeypatch, ANSWER)
    session = f"notice-{entry}"
    if entry == "chat":
        result = runtime.chat("what is in my inbox?", session_id=session)
    else:
        events = list(runtime.chat_stream("what is in my inbox?", session_id=session))
        assert events[-1].kind == "done"
        result = events[-1].result

    assert result.response.count(REDACTION_NOTICE) == 1
    assert result.response.startswith(ANSWER)
    # The transcript holds what the owner saw: the notice was added before the record.
    lines = [json.loads(x) for x in session_log_path(session).read_text().splitlines() if x]
    recorded = [e for e in lines if e["kind"] == "agent_response"]
    assert len(recorded) == 1 and REDACTION_NOTICE in json.dumps(recorded[0])


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_an_answer_without_a_redaction_is_unchanged_on_both_entries(
    runtime: Any, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    _answer_with(runtime, monkeypatch, "All quiet.")
    if entry == "chat":
        result = runtime.chat("what is in my inbox?", session_id=f"quiet-{entry}")
    else:
        events = list(runtime.chat_stream("what is in my inbox?", session_id=f"quiet-{entry}"))
        result = events[-1].result
    assert result.response == "All quiet."
