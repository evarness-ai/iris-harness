"""A chat turn outlives the connection that started it (2026-09-21).

The owner answered "yes" to "trash the promo emails", switched apps, and the turn
stopped: it ran inside the HTTP response, and iOS dropped the connection. These pin the
runner (the turn finishes with nobody reading; Stop still stops it) and the endpoints
(/chat/stream relays, /chat/cancel stops, /api/chat-status says a turn is running).
"""

from __future__ import annotations

import contextvars
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.runtime.types import ChatResult, StreamEvent
from iris_harness.server.iris_api.detached_turns import DetachedTurns
from iris_harness.server.iris_api.main import create_app

WAIT = 5.0


class _Turn:
    """A turn the test steps by hand: each ``release`` lets one more event out."""

    def __init__(self, n: int = 3) -> None:
        self.n = n
        self.gate = threading.Semaphore(0)
        self.emitted: list[int] = []
        self.finished = threading.Event()
        self.closed = threading.Event()

    def release(self, k: int = 1) -> None:
        for _ in range(k):
            self.gate.release()

    def events(self) -> Iterator[Any]:
        try:
            for i in range(self.n):
                assert self.gate.acquire(timeout=WAIT)
                self.emitted.append(i)
                yield i
            self.finished.set()  # where the pipeline's record stage files the answer
        finally:
            self.closed.set()


def _join(turns: DetachedTurns, session: str) -> None:
    for t in threading.enumerate():
        if t.name == f"iris-turn-{session}":
            t.join(WAIT)


def test_the_relay_yields_the_turns_events_in_order() -> None:
    turns, turn = DetachedTurns(), _Turn()
    turn.release(3)
    running = turns.start("s1", turn.events)
    assert [e for e in turns.relay(running) if e is not None] == [0, 1, 2]
    _join(turns, "s1")
    assert turns.running("s1") is None


def test_the_turn_finishes_after_the_client_goes_away() -> None:
    turns, turn = DetachedTurns(), _Turn()
    running = turns.start("s1", turn.events)
    relay = turns.relay(running, keepalive_s=0.05)
    turn.release()
    assert next(e for e in relay if e is not None) == 0
    relay.close()  # the phone backgrounded the app

    assert turns.running("s1") is running  # still going
    turn.release(2)
    assert turn.finished.wait(WAIT)  # and it got to the end
    _join(turns, "s1")
    assert turns.running("s1") is None


def test_cancel_stops_the_turn_at_its_next_event_and_closes_it() -> None:
    turns, turn = DetachedTurns(), _Turn()
    running = turns.start("s1", turn.events)
    turn.release()
    relay = turns.relay(running)
    assert next(relay) == 0

    assert turns.cancel("s1") is True
    turn.release(2)
    assert turn.closed.wait(WAIT)
    assert not turn.finished.is_set()
    assert turn.emitted == [0, 1]  # the event it was cancelled on is not relayed
    assert list(relay) == []
    assert turns.cancel("s1") is False  # nothing left to stop


def test_a_failing_turn_is_reported_to_the_relay() -> None:
    def boom() -> Iterator[Any]:
        yield "first"
        raise RuntimeError("model down")

    turns = DetachedTurns()
    out = list(turns.relay(turns.start("s1", boom)))
    assert out[0] == "first"
    assert isinstance(out[1], RuntimeError) and str(out[1]) == "model down"


def test_a_quiet_turn_gets_keepalives() -> None:
    turns, turn = DetachedTurns(), _Turn(n=1)
    relay = turns.relay(turns.start("s1", turn.events), keepalive_s=0.01)
    assert next(relay) is None
    turn.release()
    assert [e for e in relay if e is not None] == [0]


_VAR: contextvars.ContextVar[str] = contextvars.ContextVar("_VAR", default="unset")


def test_the_turn_runs_in_the_requests_context() -> None:
    def read() -> Iterator[str]:
        yield _VAR.get()

    token = _VAR.set("the request's")
    try:
        turns = DetachedTurns()
        running = turns.start("s1", read)
    finally:
        _VAR.reset(token)
    assert list(turns.relay(running)) == ["the request's"]


# --- the endpoints -----------------------------------------------------------------


class _Runtime:
    def __init__(self, turn: _Turn) -> None:
        self.turn = turn
        self.sessions = self

    def context_health(self, session_id: str) -> dict[str, Any]:
        return {}

    def chat_stream(self, message: str, **kwargs: Any) -> Iterator[StreamEvent]:
        for i in self.turn.events():
            if i < self.turn.n - 1:
                yield StreamEvent(kind="token", text=f"t{i} ")
        yield StreamEvent(
            kind="done",
            result=ChatResult(
                response="Trashed.",
                intent="communication",
                agent_type="email",
                sources=(),
                has_errors=False,
                error_summary=None,
                metadata={},
            ),
        )


@pytest.fixture()
def api() -> tuple[TestClient, _Turn, Any]:
    app = create_app(auto_start_runtime=False)
    turn = _Turn()
    app.state.runtime = _Runtime(turn)
    app.state.turns = DetachedTurns(keepalive_s=0.05)  # a dropped relay lets go quickly
    return TestClient(app, headers=auth_headers()), turn, app


def test_the_stream_relays_the_turn(api: Any) -> None:
    client, turn, _app = api
    turn.release(3)
    with client.stream("POST", "/chat/stream", json={"message": "hi", "session_id": "w"}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert [e["event"] for e in events] == ["token", "token", "done"]
    assert events[-1]["response"] == "Trashed."


async def _disconnect_after_first_chunk(app: Any, session: str) -> list[bytes]:
    """POST /chat/stream the way a phone does, then go away after the first line.

    Starlette's TestClient buffers a whole response, so it cannot leave early; this
    drives the ASGI app directly and sends ``http.disconnect`` once a chunk arrives.
    """
    import asyncio

    got_chunk = asyncio.Event()
    chunks: list[bytes] = []
    body = json.dumps({"message": "yes", "session_id": session}).encode()
    headers = [(k.lower().encode(), v.encode()) for k, v in auth_headers().items()]
    headers += [(b"content-type", b"application/json"), (b"host", b"testserver")]
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/chat/stream",
        "raw_path": b"/chat/stream",
        "query_string": b"",
        "headers": headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    sent_body = False

    async def receive() -> dict[str, Any]:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        await got_chunk.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            chunks.append(message["body"])
            got_chunk.set()

    await asyncio.wait_for(app(scope, receive, send), WAIT)
    return chunks


@pytest.mark.usefixtures("offline_ollama_inventory")  # /api/chat-status reads the model list
async def test_a_dropped_stream_leaves_the_turn_running_and_status_says_so(api: Any) -> None:
    client, turn, app = api
    turn.release()
    chunks = await _disconnect_after_first_chunk(app, "w")

    assert json.loads(chunks[0])["event"] == "token"
    # The client is gone; the turn is not.
    assert app.state.turns.running("w") is not None
    status = client.get("/api/chat-status", params={"session_id": "w"}).json()
    assert status["turn_in_progress"] is True

    turn.release(2)
    assert turn.finished.wait(WAIT)  # it ran to the end with nobody reading
    _join(app.state.turns, "w")
    status = client.get("/api/chat-status", params={"session_id": "w"}).json()
    assert status["turn_in_progress"] is False


async def test_cancel_stops_it(api: Any) -> None:
    client, turn, app = api
    turn.release()
    await _disconnect_after_first_chunk(app, "w")

    assert client.post("/chat/cancel", json={"session_id": "w"}).json() == {"cancelled": True}
    turn.release(2)
    assert turn.closed.wait(WAIT)
    assert not turn.finished.is_set()
    assert client.post("/chat/cancel", json={"session_id": "w"}).json() == {"cancelled": False}


def test_cancel_is_open_to_a_read_only_console(api: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Chat is ungated; stopping your own turn must be too."""
    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    client, _turn, _app = api
    assert client.post("/chat/cancel", json={"session_id": "nobody"}).status_code == 200
