"""IRIS_LOG_PROMPT_CAP: how much of an LLM prompt the session log keeps.

A logged prompt is cut in the middle at 8192 chars, which drops a ReAct prompt's tool
menu and rules. The env var raises the cap so logged prompts can be replayed whole.
"""

from __future__ import annotations

import json
from pathlib import Path

from iris_harness.foundation.observability import session_log


def _logged_prompt(monkeypatch, tmp_path: Path, content: str) -> str:
    monkeypatch.setattr(session_log, "LOG_DIR", tmp_path)
    with session_log.session_scope("cap-test"):
        with session_log.llm_call_scope(
            model="m", provider="ollama", input_messages=[{"role": "user", "content": content}]
        ):
            pass
    events = [
        json.loads(line) for line in (tmp_path / "session-cap-test.jsonl").read_text().splitlines()
    ]
    (call,) = [e for e in events if e["kind"] == "llm_call"]
    return call["input_messages"][0]["content"]


def test_default_cuts_the_middle_at_8192(monkeypatch, tmp_path):
    monkeypatch.delenv("IRIS_LOG_PROMPT_CAP", raising=False)
    logged = _logged_prompt(monkeypatch, tmp_path, "a" * 20_000)
    assert "chars omitted" in logged
    assert len(logged) < 8300


def test_raised_cap_keeps_the_whole_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("IRIS_LOG_PROMPT_CAP", "65536")
    prompt = "head " + "x" * 20_000 + " tail"
    assert _logged_prompt(monkeypatch, tmp_path, prompt) == prompt


def test_junk_or_smaller_values_keep_the_default(monkeypatch, tmp_path):
    for value in ("lots", "100", "-5", ""):
        monkeypatch.setenv("IRIS_LOG_PROMPT_CAP", value)
        assert session_log._prompt_cap() == 8192
