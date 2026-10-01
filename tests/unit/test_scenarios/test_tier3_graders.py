"""Pure-logic tests for the Tier-3 quality battery's graders (no model)."""

from __future__ import annotations

from scenarios.tier3_quality import _extract_code, _run_code


def test_extract_code_prefers_last_fenced_block() -> None:
    text = "intro\n```python\nx = 1\n```\nmid\n```python\ndef f():\n    return 2\n```\n"
    assert "def f()" in _extract_code(text)


def test_extract_code_falls_back_to_raw() -> None:
    assert _extract_code("def f():\n    return 1") == "def f():\n    return 1"


def test_run_code_passes_when_asserts_hold() -> None:
    ok, detail = _run_code("def add(a, b):\n    return a + b", "assert add(2, 3) == 5")
    assert ok is True
    assert detail == ""


def test_run_code_fails_when_asserts_break() -> None:
    ok, detail = _run_code("def add(a, b):\n    return a - b", "assert add(2, 3) == 5")
    assert ok is False
    assert detail  # carries the last stderr line


def test_run_code_fails_on_syntax_error() -> None:
    ok, _ = _run_code("def broken(:", "assert True")
    assert ok is False
