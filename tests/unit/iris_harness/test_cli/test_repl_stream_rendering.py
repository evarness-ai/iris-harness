"""Tests for streamed REPL response rendering."""

from __future__ import annotations

from iris_harness.cli import repl


class _StatusStub:
    """Minimal stand-in for Rich status object used by ``_make_token_printer``."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_stream_tokens_are_buffered_and_flushed_as_one_response(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    rendered: list[str] = []
    state: dict[str, object] = {"started": False, "buffer": []}
    status = _StatusStub()

    monkeypatch.setattr(repl, "print_response", lambda text: rendered.append(text))

    on_token = repl._make_token_printer(status, state)
    on_token("1. **microsoft/edit**")
    on_token(" — We all edit.")
    repl._flush_streamed_response(state)

    assert status.stopped
    assert state["started"] is True
    assert rendered == ["1. **microsoft/edit** — We all edit."]


def test_stream_tokens_can_render_live_without_final_flush(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    rendered: list[str] = []
    live: list[str] = []
    state: dict[str, object] = {"started": False, "buffer": []}
    status = _StatusStub()

    monkeypatch.setattr(repl, "print_response", lambda text: rendered.append(text))

    on_token = repl._make_token_printer(status, state, live.append)
    on_token("hello")
    on_token(" there")
    repl._flush_streamed_response(state)

    assert status.stopped
    assert state["started"] is True
    assert state["live_rendered"] is True
    assert live == ["hello", " there"]
    assert rendered == []


def test_done_response_is_rendered_when_no_tokens_stream(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    rendered: list[str] = []
    monkeypatch.setattr(repl, "print_response", lambda text: rendered.append(text))

    did_render = repl._flush_final_response(
        {
            "response": "Drafted routine `Morning briefing`.",
            "intent": "routine_authoring",
        }
    )

    assert did_render is True
    assert rendered == ["Drafted routine `Morning briefing`."]


def test_empty_done_response_is_not_rendered(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    rendered: list[str] = []
    monkeypatch.setattr(repl, "print_response", lambda text: rendered.append(text))

    did_render = repl._flush_final_response({"response": ""})

    assert did_render is False
    assert rendered == []
