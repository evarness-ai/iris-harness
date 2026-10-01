"""Unit tests for the router-decision audit logger."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from iris_harness.agent.intent_router import IntentResult
from iris_harness.runtime.router_audit import RouterAuditLogger


def _result(**overrides: object) -> IntentResult:
    base = {
        "intent": "general",
        "agent_type": "system",
        "confidence": 0.85,
        "is_multi_step": False,
        "raw_query": "hello",
        "source": "keyword",
    }
    base.update(overrides)
    return IntentResult(**base)  # type: ignore[arg-type]


def test_record_persists_decision_row(tmp_path: Path) -> None:
    logger = RouterAuditLogger(tmp_path / "audit.db")

    logger.record(
        session_id="session-1",
        message="What's my name?",
        result=_result(intent="profile_query", agent_type="system"),
        router_model="llama3.2:3b",
        channel="console",
    )

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute("SELECT * FROM router_decisions"))

    assert len(rows) == 1
    r = rows[0]
    assert r["session_id"] == "session-1"
    assert r["intent"] == "profile_query"
    assert r["agent_type"] == "system"
    assert r["source"] == "keyword"
    assert r["router_model"] == "llama3.2:3b"
    assert r["channel"] == "console"
    assert r["message_length"] == len("What's my name?")
    assert len(r["message_hash"]) == 16
    # Content must NOT be stored.
    assert "What's my name" not in r["message_hash"]


def test_record_is_append_only(tmp_path: Path) -> None:
    """The schema mirrors the governor audit pattern — UPDATE / DELETE
    must be rejected by triggers."""

    logger = RouterAuditLogger(tmp_path / "audit.db")
    logger.record(
        session_id="s",
        message="hi",
        result=_result(),
    )

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE router_decisions SET intent='hacked'")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM router_decisions")


def test_record_stores_source_distinguishing_keyword_vs_llm(tmp_path: Path) -> None:
    """The whole point of the table — knowing whether the keyword
    classifier or the LLM router decided."""

    logger = RouterAuditLogger(tmp_path / "audit.db")
    logger.record(session_id="s", message="hi", result=_result(source="keyword"))
    logger.record(session_id="s", message="ambiguous", result=_result(source="llm"))
    logger.record(session_id="s", message="nope", result=_result(source="fallback"))

    with sqlite3.connect(tmp_path / "audit.db") as conn:
        rows = list(conn.execute("SELECT source, COUNT(*) FROM router_decisions GROUP BY source"))
    counts = {r[0]: r[1] for r in rows}
    assert counts == {"keyword": 1, "llm": 1, "fallback": 1}


def test_record_swallows_db_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit write failures MUST NOT break the chat turn."""

    logger = RouterAuditLogger(tmp_path / "audit.db")

    def boom(*_a: object, **_kw: object) -> None:
        raise sqlite3.Error("disk full")

    monkeypatch.setattr(sqlite3, "connect", boom)

    # Must not raise.
    logger.record(session_id="s", message="hi", result=_result())


def test_intent_result_source_default_is_empty_for_backward_compat() -> None:
    """Existing classifier code that doesn't set ``source`` still
    works — the default must be the empty string, not error."""

    result = IntentResult(intent="x", agent_type="y", confidence=0.5)
    assert result.source == ""
