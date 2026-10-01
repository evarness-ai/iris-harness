"""The chat history lists the owner's conversations, not the harness testing itself.

Found on the owner's box (2026-09-19): after the live playground suites ran, the chat
sidebar listed "my bank is Barclays", "my wife is Petra"… — playground sessions, and a
smoke test's — as past conversations. They are kept out of memory; they were not kept
out of the list the sidebar reads (the session logs).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.foundation.observability import session_log
from iris_harness.server.iris_api.main import create_app


def _session(log_dir: Path, session_id: str, text: str) -> None:
    ts = "2026-09-18T22:00:00.000000+00:00"
    events = [
        {"kind": "user_message", "ts": ts, "session_id": session_id, "text": text},
        {
            "kind": "llm_call",
            "ts": ts,
            "session_id": session_id,
            "agent_type": "intent_router",
            "model": "m",
            "provider": "p",
            "input_messages": [],
            "output": {"text": "x"},
            "tokens": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "duration_ms": 10.0,
        },
        {
            "kind": "agent_response",
            "ts": ts,
            "session_id": session_id,
            "response": "ok",
            "intent": "system",
            "agent_type": "system",
            "has_errors": False,
            "total_tokens": 2,
            "total_duration_ms": 10.0,
        },
    ]
    (log_dir / f"session-{session_id}.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    _session(tmp_path, "web-1a2b3c4d", "what's on my calendar?")
    _session(tmp_path, "playground-tells-a-bank-9f1e", "my bank is Barclays")
    _session(tmp_path, "eval:case-7", "plan my day")
    runtime = SimpleNamespace(memory_store=None, semantic_index=None)
    with TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    ) as c:
        yield c


def test_runs_are_not_the_owner_s_conversations(client: TestClient) -> None:
    ids = [s["session_id"] for s in client.get("/api/sessions").json()]
    assert ids == ["web-1a2b3c4d"]


def test_they_can_still_be_listed_for_debugging(client: TestClient) -> None:
    ids = {s["session_id"] for s in client.get("/api/sessions?include_runs=true").json()}
    assert ids == {"web-1a2b3c4d", "playground-tells-a-bank-9f1e", "eval:case-7"}
