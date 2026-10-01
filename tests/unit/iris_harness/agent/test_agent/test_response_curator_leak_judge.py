"""Semantic leak-judge: dump-flagged responses arbitrated by an LLM judge (exp-007).

The deterministic dump-phrase guard can't tell a legitimate self-description ("my system
prompt defines how I behave") from an actual dump ("here is my system prompt: …") — that's
intent, not content. When the dump phrase matches, the curator escalates to a semantic
leak-judge. The judge can only *clear* a flagged response (fixing benign over-blocks); it
FAILS CLOSED (halt) when absent / erroring / unparseable / confirming a leak, and the hard
credential/secret-literal guards never consult it.
"""

from __future__ import annotations

import pytest

from iris_harness.agent.response_curator import ResponseCurator

DUMP = "Sure — here is my system prompt: You are IRIS, a local-first assistant."
BENIGN_MENTION = "My system prompt defines how I behave; I won't reveal its contents."
SECRET = "CANARY_SOUL_SECRET_DIRECTIVE_d41f8a27"


class _Judge:
    """Mock LeakJudgeClient: returns a fixed payload, or raises."""

    def __init__(self, payload: str | None = None, exc: Exception | None = None) -> None:
        self._payload = payload
        self._exc = exc

    async def judge(self, *, response: str) -> str:
        if self._exc is not None:
            raise self._exc
        assert self._payload is not None
        return self._payload


@pytest.fixture(autouse=True)
def _identity(owner_identity_documents) -> None:
    owner_identity_documents([SECRET])


def _curator(monkeypatch, judge) -> ResponseCurator:
    return ResponseCurator(leak_judge=judge)


def test_dump_no_judge_fails_closed(monkeypatch):
    assert _curator(monkeypatch, None)._judge_safety(DUMP).verdict == "halt"


def test_dump_judge_clears_legitimate(monkeypatch):
    judge = _Judge('{"is_leak": false, "confidence": 0.95, "reason": "self-description"}')
    signal = _curator(monkeypatch, judge)._judge_safety(DUMP)
    assert signal.verdict == "pass"
    assert signal.metadata.get("judge") == "leak"


def test_dump_judge_confirms_leak_halts(monkeypatch):
    judge = _Judge('{"is_leak": true, "confidence": 0.9, "reason": "dumped the prompt"}')
    assert _curator(monkeypatch, judge)._judge_safety(DUMP).verdict == "halt"


def test_dump_judge_error_fails_closed(monkeypatch):
    judge = _Judge(exc=RuntimeError("judge down"))
    assert _curator(monkeypatch, judge)._judge_safety(DUMP).verdict == "halt"


def test_dump_judge_unparseable_fails_closed(monkeypatch):
    assert _curator(monkeypatch, _Judge("not json at all"))._judge_safety(DUMP).verdict == "halt"


def test_benign_mention_never_consults_judge(monkeypatch):
    # The judge would cry leak, but a bare mention doesn't match the dump pattern,
    # so the judge is never consulted and the response passes.
    judge = _Judge('{"is_leak": true, "confidence": 1.0, "reason": "x"}')
    assert _curator(monkeypatch, judge)._judge_safety(BENIGN_MENTION).verdict == "pass"


def test_secret_literal_hard_halt_ignores_judge(monkeypatch):
    # A clear-everything judge must NOT be able to release a secret literal.
    judge = _Judge('{"is_leak": false, "confidence": 1.0, "reason": "x"}')
    assert _curator(monkeypatch, judge)._judge_safety(f"the value is {SECRET}").verdict == "halt"


def test_pii_hard_halt_ignores_judge(monkeypatch):
    judge = _Judge('{"is_leak": false, "confidence": 1.0, "reason": "x"}')
    assert _curator(monkeypatch, judge)._judge_safety("SSN 123-45-6789").verdict == "halt"
