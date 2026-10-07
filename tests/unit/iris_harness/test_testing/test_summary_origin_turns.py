"""A summary that absorbed an external turn comes back enveloped, on both chat entries (#145).

Driven through the governed harness (the composition root, a scripted model, a plugin tool that
declares ``content: external``), on ``chat`` and ``chat_stream``. Turn 1 reads a page and answers;
compaction is then forced (the summariser is stubbed to write a paraphrase carrying an instruction,
which is what a model-written summary can do); the stored summary row says ``has_external``; the
NEXT turn's model prompt shows the summary inside the untrusted-content envelope with the phrase
gone. A session that read nothing keeps an unflagged summary, which is not enveloped.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from iris_harness.testing import harness

from .test_turn_origin_turns import (
    _SCRIPT,
    FIRST,
    INJECTED,
    SECOND,
    STORED_ENVELOPE,
    _page_plugin,
    _turn,
)

SUMMARY = f"Weather chat: the owner asked about Oslo. {INJECTED}"


def _stored_flag(h: Any, session_id: str) -> int | None:
    (db,) = list(h.home.rglob("memory.db"))
    with sqlite3.connect(db) as conn:
        row = conn.execute(
            "SELECT has_external FROM conversation_summaries WHERE session_id = ?", (session_id,)
        ).fetchone()
    assert row is not None, "no summary was stored"
    return row[0]


def _force_compaction(h: Any, session_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = h._runtime
    compactor = runtime.compactor
    monkeypatch.setattr(compactor, "token_budget", None)
    monkeypatch.setattr(compactor, "keep_recent", 1)
    monkeypatch.setattr(compactor, "_summarize", lambda turns, previous_summary="": SUMMARY)
    assert runtime.sessions.compact_now(session_id)["compacted"] is True


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_summary_of_an_external_turn_is_flagged_and_comes_back_enveloped(
    entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = f"summary-{entry}"
    with harness(plugins=[_page_plugin()], fake_model=_SCRIPT) as h:
        _turn(entry, h, FIRST, session)  # reads the page: the stored answer is external
        _turn(entry, h, SECOND, session)
        _force_compaction(h, session, monkeypatch)
        flag = _stored_flag(h, session)
        _turn(entry, h, SECOND, session)  # the summary is now part of the next prompt
        prompt = [c for c in h.model_calls() if c.rule == "second turn"][-1].user
    assert flag == 1
    assert STORED_ENVELOPE in prompt and "Weather chat: the owner asked about Oslo." in prompt
    assert "bankrupt" not in prompt  # the instruction in the paraphrase is redacted


@pytest.mark.parametrize("entry", ["chat", "chat_stream"])
def test_a_summary_of_turns_that_read_nothing_is_not_flagged_or_enveloped(
    entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = f"summary-plain-{entry}"
    with harness(plugins=[_page_plugin()], fake_model=_SCRIPT) as h:
        _turn(entry, h, SECOND, session)  # no tool: the loop records the turn as internal
        _turn(entry, h, SECOND, session)
        _turn(entry, h, SECOND, session)
        _force_compaction(h, session, monkeypatch)
        flag = _stored_flag(h, session)
        _turn(entry, h, SECOND, session)
        prompt = [c for c in h.model_calls() if c.rule == "second turn"][-1].user
    assert flag == 0
    assert STORED_ENVELOPE not in prompt
    assert "bankrupt" not in prompt  # still scanned: the paraphrase's instruction is redacted
