"""The first-chat welcome (ADR-0127): one system-opened turn, once per IRIS_HOME.

Through the real pipeline (``open_turn`` → ``OPENER_STAGES``) and the real API route,
with no model: the first call makes exactly one session with a call trace and audit
rows, a second call changes nothing, the record lives under IRIS_HOME, and replay shows
only what IRIS said.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from iris_harness.foundation.auth import auth_headers
from iris_harness.foundation.observability import trace_builder as tb
from iris_harness.foundation.observability.session_log import session_log_dir
from iris_harness.foundation.paths import audit_db_path
from iris_harness.runtime.intercepts import DEFAULT_OPENERS, load_openers
from iris_harness.runtime.welcome import (
    DEFAULT_WORDING,
    WELCOME_OPENER,
    capability_summaries,
    welcome_marker_path,
    welcome_text,
)

CONFIG_DIR = Path(__file__).resolve().parents[5] / "config"


@pytest.fixture()
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A fresh IRIS_HOME: the welcome has never run here."""
    home = tmp_path / "home"
    monkeypatch.setenv("IRIS_HOME", str(home))
    return home


@pytest.fixture()
def runtime(home: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    from iris_harness.runtime import build_runtime

    monkeypatch.setenv("IRIS_DISABLE_ARBITER", "1")
    monkeypatch.setenv("IRIS_DISABLE_WARMUP", "1")
    return build_runtime(
        config_dir=CONFIG_DIR, data_dir=tmp_path / "data", use_background_scheduler=False
    )


def _session_files() -> list[Path]:
    return sorted(session_log_dir().glob("session-*.jsonl"))


def _audit_rows(session_id: str) -> list[tuple[str, str, str]]:
    with sqlite3.connect(audit_db_path()) as conn:
        return conn.execute(
            "SELECT hook_point, decision, payload_json FROM audit_log "
            "WHERE json_extract(payload_json, '$.session_id') = ?",
            (session_id,),
        ).fetchall()


def test_the_first_call_runs_one_governed_turn_with_no_model(runtime: Any, home: Path) -> None:
    def no_model(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("the welcome must not reach a model")

    runtime.tier_router.get_llm_config = no_model
    outcome = runtime.welcome.ensure(channel="web")

    assert outcome.created is True
    files = _session_files()
    assert [p.name for p in files] == [f"session-{outcome.session_id}.jsonl"]
    kinds = [json.loads(line)["kind"] for line in files[0].read_text().splitlines()]
    assert kinds[0] == "turn_open"
    assert "user_message" not in kinds, "no words are logged in the owner's name"
    assert "llm_call" not in kinds
    assert {"handler.end", "guard.end", "agent_response"} <= set(kinds)

    sessions = tb.list_sessions()
    assert [(s["session_id"], s["turn_count"]) for s in sessions] == [(outcome.session_id, 1)]
    assert sessions[0]["title"] == "First-chat welcome"
    traces = tb.list_traces()
    assert [t["trace_id"] for t in traces] == [outcome.trace_id]

    trace = tb.get_trace(outcome.trace_id)
    assert trace is not None and trace["opened_by_system"] is True
    nodes = {n["id"]: n for n in trace["nodes"]}
    assert nodes["handler"]["label"] == WELCOME_OPENER
    assert nodes["guard"]["status"] == "ok"
    assert [s["type"] for s in trace["steps"]] == ["request", "handler", "guard", "response"]

    # The response check's audit row is there; there was no input, so no input screen.
    rows = _audit_rows(outcome.session_id)
    hooks = {hook for hook, _decision, _payload in rows}
    assert "pre_response" in hooks and "pre_turn" not in hooks
    assert any(
        json.loads(payload).get("deterministic") is True
        for hook, _d, payload in rows
        if hook == "pre_response"
    )


def test_replay_shows_only_what_iris_said(runtime: Any) -> None:
    outcome = runtime.welcome.ensure(channel="web")
    messages = tb.session_messages(outcome.session_id)
    assert [m["role"] for m in messages] == ["assistant"]
    assert messages[0]["text"] == outcome.response
    assert messages[0]["trace_id"] == outcome.trace_id


def test_a_second_call_is_a_no_op(runtime: Any) -> None:
    first = runtime.welcome.ensure(channel="web")
    log = session_log_dir() / f"session-{first.session_id}.jsonl"
    before = log.read_text()
    marker_before = welcome_marker_path().read_text()

    second = runtime.welcome.ensure(channel="telegram")

    assert second.created is False
    assert (second.session_id, second.response, second.trace_id) == (
        first.session_id,
        first.response,
        first.trace_id,
    )
    assert len(_session_files()) == 1
    assert log.read_text() == before
    assert welcome_marker_path().read_text() == marker_before


def test_the_record_lives_under_iris_home(runtime: Any, home: Path) -> None:
    outcome = runtime.welcome.ensure(channel="console")
    marker = home / "welcome.json"
    assert welcome_marker_path() == marker
    recorded = json.loads(marker.read_text())
    assert recorded["session_id"] == outcome.session_id
    assert recorded["channel"] == "console"
    # The session log and the ledger followed IRIS_HOME too.
    assert session_log_dir().is_relative_to(home)
    assert audit_db_path().is_relative_to(home)


def test_a_failed_turn_leaves_the_welcome_due(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime, "open_turn", broken)
    with pytest.raises(RuntimeError):
        runtime.welcome.ensure(channel="web")
    assert not welcome_marker_path().exists()
    monkeypatch.undo()


def test_the_text_lists_what_the_mounted_plugins_say_they_do(runtime: Any) -> None:
    summaries = capability_summaries(runtime.plugin_registry)
    mounted = {r.name: r for r in runtime.plugin_registry.plugins() if r.manifest is not None}
    assert mounted["system"].manifest.summary in summaries
    # A delivery surface has nothing to say to the owner, so it is not listed.
    assert "web_channel" in mounted and not mounted["web_channel"].manifest.summary
    assert len(summaries) == sum(1 for r in mounted.values() if r.manifest.summary)
    outcome = runtime.welcome.ensure(channel="web")
    for line in summaries:
        assert f"- {line}" in outcome.response
    assert "Call trace" in outcome.response


def test_no_mounted_capability_says_so() -> None:
    text = welcome_text([], dict(DEFAULT_WORDING))
    assert text.startswith(DEFAULT_WORDING["no_capabilities"])
    assert DEFAULT_WORDING["invitation"] in text


def test_the_wording_comes_from_config() -> None:
    raw = yaml.safe_load((CONFIG_DIR / "welcome.yaml").read_text())
    assert set(raw) == set(DEFAULT_WORDING)
    assert "{summary}" in raw["capability"]


def test_the_declared_openers_match_the_fallback() -> None:
    assert load_openers(CONFIG_DIR / "intercepts.yaml") == {s.name: s for s in DEFAULT_OPENERS}


@pytest.fixture()
def client(runtime: Any) -> Iterator[TestClient]:
    from iris_harness.server.iris_api.main import create_app

    with TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    ) as c:
        yield c


def test_the_route_runs_it_once_and_then_returns_it(client: TestClient) -> None:
    first = client.post("/chat/welcome", json={"channel": "web"})
    assert first.status_code == 200
    body = first.json()
    assert body["created"] is True and body["trace_id"] == f"{body['session_id']}~0"
    again = client.post("/chat/welcome", json={"channel": "web"})
    assert again.json() == {**body, "created": False}
    assert len(_session_files()) == 1
    # The surfaces read it back like any session.
    sessions = client.get("/api/sessions").json()
    assert [s["session_id"] for s in sessions] == [body["session_id"]]
    assert client.get(f"/api/traces/{body['trace_id']}").status_code == 200


def test_the_route_takes_a_bare_post(client: TestClient) -> None:
    # Every field of the body has a default: no body is the console's welcome.
    bare = client.post("/chat/welcome")
    assert bare.status_code == 200
    assert bare.json()["created"] is True


def test_the_route_is_authenticated_like_chat(runtime: Any) -> None:
    from iris_harness.server.iris_api.main import create_app

    with TestClient(create_app(runtime=runtime, auto_start_runtime=False)) as anonymous:
        assert anonymous.post("/chat/welcome", json={}).status_code == 401
        assert anonymous.post("/chat", json={"message": "hi"}).status_code == 401
    assert not welcome_marker_path().exists()


def test_a_read_only_console_still_gets_its_welcome(
    runtime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iris_harness.server.iris_api.main import create_app

    monkeypatch.delenv("IRIS_WEBUI_ALLOW_WRITES", raising=False)
    with TestClient(
        create_app(runtime=runtime, auto_start_runtime=False), headers=auth_headers()
    ) as c:
        assert c.post("/chat/welcome", json={"channel": "web"}).json()["created"] is True


def test_a_home_that_already_has_conversations_skips_the_welcome(
    runtime: Any, client: TestClient
) -> None:
    """Owner's decision: an install from before the welcome gets none, and says why."""
    runtime.chat("what time is it?", session_id="web-1a2b3c4d", channel="web")
    before = [p.name for p in _session_files()]

    body = client.post("/chat/welcome", json={"channel": "web"}).json()

    assert body["created"] is False and body["skipped"] is True
    assert body["session_id"] == "" and body["response"] == "" and body["trace_id"] == ""
    assert "already had 1 conversation" in body["skip_reason"]
    assert [p.name for p in _session_files()] == before, "no turn ran"
    recorded = json.loads(welcome_marker_path().read_text())
    assert recorded["skipped"] is True and recorded["reason"] == body["skip_reason"]
    # Recorded, so it never runs later either, even once that history is gone.
    for path in _session_files():
        path.unlink()
    again = runtime.welcome.ensure(channel="web")
    assert (again.created, again.skipped) == (False, True)
    assert _session_files() == []


def test_the_harness_testing_itself_is_not_history(runtime: Any) -> None:
    """A playground or eval run is not the owner's conversation (retention.yaml), so a
    home with only those is still fresh and gets its welcome."""
    runtime.chat("what time is it?", session_id="playground-probe-1", channel="web")
    outcome = runtime.welcome.ensure(channel="web")
    assert outcome.created is True and outcome.skipped is False
    assert "skipped" not in json.loads(welcome_marker_path().read_text())
