"""The explicit turn pipeline: one code path, drained by chat(), streamed by chat_stream()."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from iris_harness.runtime import build_runtime
from iris_harness.runtime.turn import OPENER_STAGES, STAGES, TurnRequest, drain, run_turn
from iris_harness.runtime.turn.state import TurnState
from iris_harness.runtime.types import ChatResult, StreamEvent


@pytest.fixture()
def runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    config_dir = Path(__file__).resolve().parents[5] / "config"
    return build_runtime(
        config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False
    )


def test_stage_order_is_the_design() -> None:
    assert [name for name, _ in STAGES] == [
        "screen",
        "intercept",
        "classify",
        "route",
        "plan",
        "execute",
        "curate",
        "guard",
        "notice",
        "record",
    ]


def test_intercept_ends_the_turn_on_both_entry_points(runtime) -> None:  # type: ignore[no-untyped-def]
    events = list(runtime.chat_stream("what time is it?", session_id="tp-stream"))
    kinds = [e.kind for e in events]
    assert kinds[0] == "trace" and events[0].text == "turn.start"
    assert kinds[-1] == "done"
    assert "activity" not in kinds[1:-1] or all(
        e.text != "routing intent" for e in events if e.kind == "activity"
    )  # no classifier ran
    done = events[-1].result
    assert done is not None and done.metadata.get("deterministic_time_date") is True
    sync = runtime.chat("what time is it?", session_id="tp-sync")
    assert isinstance(sync, ChatResult)
    assert sync.response.startswith("Current local time")
    assert sync.metadata.get("deterministic_time_date") is True


def test_drain_returns_done_result_and_raises_on_error() -> None:
    result = ChatResult("r", "system", "system", (), False, None, {})
    assert (
        drain(iter([StreamEvent(kind="token", text="x"), StreamEvent(kind="done", result=result)]))
        is result
    )
    with pytest.raises(RuntimeError, match="boom"):
        drain(iter([StreamEvent(kind="error", error="boom")]))
    with pytest.raises(RuntimeError, match="without a result"):
        drain(iter([]))


def _exploding_stage(_runtime: Any, _state: TurnState) -> Iterator[StreamEvent]:
    raise ValueError("stage exploded")
    yield  # pragma: no cover


def test_on_error_event_vs_raise(runtime) -> None:  # type: ignore[no-untyped-def]
    request = TurnRequest(message="anything", session_id="tp-err")
    stages = (("boom", _exploding_stage),)
    events = list(run_turn(runtime, request, on_error="event", stages=stages))
    assert events[-1].kind == "error" and events[-1].error
    with pytest.raises(ValueError, match="stage exploded"):
        list(run_turn(runtime, request, on_error="raise", stages=stages))


def test_stage_spans_only_on_sync_path(runtime) -> None:  # type: ignore[no-untyped-def]
    from iris_harness.runtime.turn.stages import stage_span

    with stage_span(runtime, TurnState(request=TurnRequest("m"), stage_spans=False), "x") as s:
        assert s is None
    # With no tracer configured the sync path also yields a no-op span object or None;
    # the point is that it does not raise and the context is symmetric.
    with stage_span(runtime, TurnState(request=TurnRequest("m"), stage_spans=True), "x"):
        pass


def test_dead_llm_answers_with_error_text_on_both_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression: with the model endpoint unreachable the streaming ReAct loop used
    to end with metadata only, so the turn answered "" (the sync loop said
    "LLM error: ..."). Both entry points now return the same non-empty text."""
    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    config_dir = Path(__file__).resolve().parents[5] / "config"
    rt = build_runtime(
        config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False
    )
    sync = rt.chat("what can you do for me?", session_id="dead-sync")
    assert sync.response.strip(), "sync path answered empty"
    events = list(rt.chat_stream("what can you do for me?", session_id="dead-stream"))
    done = next(e.result for e in events if e.kind == "done")
    assert done is not None and done.response.strip(), "stream path answered empty"
    # The curator replaces the raw "LLM error: ..." with one of its canned fallback
    # answers (varied on purpose); what matters is that neither path answers "".


def test_curate_hands_the_turn_to_the_escalation_collaborator_on_both_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """M5.7 track C wiring: ``_finalize_chat`` routes the turn through
    ``EscalationActions`` and applies what it decides. The action loop's own tests drive
    the collaborator directly, so without this nothing fails if the call site is dropped
    or its verdict is ignored. One collaborator per runtime — it owns the egress
    classifier cache, so a fresh one per turn would rebuild the classifier every turn."""
    from iris_harness.runtime.escalation_actions import EscalationActions

    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    hosts: list[Any] = []

    def _clarify(self: EscalationActions, **kw: Any) -> Any:
        hosts.append(self)
        return kw["results"], kw["curated"], None, "Which account did you mean?"

    monkeypatch.setattr(EscalationActions, "maybe_escalate", _clarify)
    config_dir = Path(__file__).resolve().parents[5] / "config"
    rt = build_runtime(
        config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False
    )
    sync = rt.chat("what can you do for me?", session_id="esc-sync")
    assert sync.response == "Which account did you mean?"
    events = list(rt.chat_stream("what can you do for me?", session_id="esc-stream"))
    done = next(e.result for e in events if e.kind == "done")
    assert done is not None and done.response == "Which account did you mean?"
    assert len(hosts) == 2 and hosts[0] is hosts[1]


# ── display masking runs on the way out, after `record` ───────────────────────
#
# The regression these pin: a turn that summarised an inbox printed the account
# addresses verbatim to every channel. Masking the *yielded* events (not the state)
# is what keeps the audit ledger reconcilable with what actually happened.

_RAW = "mail from jordan1.kp@example.com today"
_MASKED = "mail from jo***kp@example.com today"


def _answering_stage(_runtime: Any, state: TurnState) -> Iterator[StreamEvent]:
    """Streams the address split across tokens, then sets the turn's result."""
    for chunk in ("mail from jord", "an1.kp@ex", "ample.com today"):
        yield StreamEvent(kind="token", text=chunk)
    state.result = ChatResult(_RAW, "chat", "chat", (), False, None, {})


def test_the_screen_sees_the_mask_and_the_record_sees_the_original(runtime) -> None:  # type: ignore[no-untyped-def]
    seen_by_record: list[str] = []

    def _record_spy(_runtime: Any, state: TurnState) -> Iterator[StreamEvent]:
        assert state.result is not None
        seen_by_record.append(state.result.response)
        return
        yield  # pragma: no cover

    request = TurnRequest(message="how does my day look?", session_id="tp-mask")
    stages = (("answer", _answering_stage), ("record", _record_spy))
    events = list(run_turn(runtime, request, stages=stages))

    # The audit/session-log stage ran on the original text.
    assert seen_by_record == [_RAW]

    # Everything yielded to a channel is masked — the tokens the REPL prints live,
    # and the final result `chat()` returns by draining them.
    streamed = "".join(e.text or "" for e in events if e.kind == "token")
    assert streamed == _MASKED
    assert "jordan1.kp@example.com" not in streamed
    done = events[-1]
    assert done.kind == "done" and done.result is not None
    assert done.result.response == _MASKED
    assert drain(iter(events)).response == _MASKED


def test_masking_off_leaves_both_paths_untouched(  # type: ignore[no-untyped-def]
    runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IRIS_GOVERNANCE_DISPLAY_MASK", "0")
    request = TurnRequest(message="how does my day look?", session_id="tp-mask-off")
    events = list(run_turn(runtime, request, stages=(("answer", _answering_stage),)))
    assert "".join(e.text or "" for e in events if e.kind == "token") == _RAW
    assert events[-1].result is not None and events[-1].result.response == _RAW


# ── a picked shortlist option reaches its owner on both entry points ──────────
#
# The live bug surfaced in the REPL, which streams. The pick must arrive resolved on
# the owner's task whichever entry point served the turn (ADR-0106 ``choice``).


def test_a_shortlist_pick_reaches_the_owner_on_both_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from iris_harness.agent.agent_executor import AgentTask

    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    config_dir = Path(__file__).resolve().parents[5] / "config"
    rt = build_runtime(
        config_dir=config_dir, data_dir=tmp_path / "data", use_background_scheduler=False
    )
    seen: list[AgentTask] = []

    def _email(task: AgentTask) -> tuple[str, dict[str, object]]:
        seen.append(task)
        return f"Reading {(task.selected_choice or {}).get('message_id')}", {}

    def _email_stream(task: AgentTask) -> Iterator[object]:
        text, meta = _email(task)
        yield text
        yield meta

    rt.agent_executor.register("email", _email)
    rt.agent_executor.register_stream("email", _email_stream)
    choices = [{"message_id": "m-first"}, {"message_id": "m-second"}]

    for session_id, run in (
        ("pick-stream", lambda m, s: list(rt.chat_stream(m, session_id=s))),
        ("pick-sync", lambda m, s: rt.chat(m, session_id=s)),
    ):
        rt.continuations.ask(
            session_id,
            "email",
            kind="choice",
            question="Which one should I read?",
            intent="communication",
            payload={"choices": choices},
        )
        seen.clear()
        run("just go with the 1 st one", session_id)
        assert [t.agent_type for t in seen] == ["email"], session_id
        assert seen[0].selected_choice == {"message_id": "m-first"}, session_id
        assert rt.continuations.pending(session_id) is None, session_id


def test_each_stage_runs_under_its_own_agent_name() -> None:
    """The session log credits an LLM call to whoever was running; every stage names itself."""
    from iris_harness.foundation.observability.session_log import _agent_type_var
    from iris_harness.runtime.turn.pipeline import STAGE_AGENTS

    seen: dict[str, str | None] = {}

    def _stage(name: str, *, handled: bool) -> Any:
        def run(_runtime: Any, state: TurnState) -> Iterator[StreamEvent]:
            seen[name] = _agent_type_var.get()
            if name in ("intercept", "open") and handled:
                state.intercepted = True
                state.result = ChatResult("r", "system", "system", (), False, None, {})
            if name == "record" and state.result is None:
                state.result = ChatResult("r", "system", "system", (), False, None, {})
            yield StreamEvent(kind="activity", text=name)

        return run

    # A generated turn runs the model path; a handled turn runs `guard` instead.
    # Between them every stage runs, each under its own name.
    for handled in (False, True):
        stages = tuple((name, _stage(name, handled=handled)) for name, _ in STAGES)
        request = TurnRequest(message="hi", session_id=f"agents-{handled}")
        events = list(run_turn(None, request, stages=stages))  # type: ignore[arg-type]
        assert events[-1].kind == "done"
    # A turn the system opens (ADR-0127) runs its own stages, `open` first.
    stages = tuple((name, _stage(name, handled=True)) for name, _ in OPENER_STAGES)
    request = TurnRequest(message="", session_id="agents-open", opener="welcome")
    events = list(run_turn(None, request, stages=stages))  # type: ignore[arg-type]
    assert events[-1].kind == "done"

    assert seen == STAGE_AGENTS
    assert seen["curate"] == "response_curator" and seen["record"] == "turn_capture"
    assert _agent_type_var.get() is None  # restored after the turn
