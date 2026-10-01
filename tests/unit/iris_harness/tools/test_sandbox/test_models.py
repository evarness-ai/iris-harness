"""Tests for ExecResult shape and helpers."""

from __future__ import annotations

from iris_harness.tools.sandbox.models import ExecResult


def test_ok_true_when_exit_zero_and_not_timed_out() -> None:
    r = ExecResult(stdout="hi", stderr="", exit_code=0, duration_ms=1.0)
    assert r.ok is True


def test_ok_false_when_nonzero_exit() -> None:
    r = ExecResult(stdout="", stderr="boom", exit_code=1, duration_ms=1.0)
    assert r.ok is False


def test_ok_false_when_timed_out() -> None:
    r = ExecResult(stdout="", stderr="", exit_code=0, duration_ms=1.0, timed_out=True)
    assert r.ok is False


def test_summary_includes_exit_and_duration() -> None:
    r = ExecResult(stdout="hello", stderr="", exit_code=0, duration_ms=12.0)
    s = r.summary()
    assert "exit_code=0" in s
    assert "duration=12ms" in s
    assert "hello" in s


def test_summary_truncates_long_output() -> None:
    big = "x" * 10_000
    r = ExecResult(stdout=big, stderr="", exit_code=0, duration_ms=1.0)
    s = r.summary(max_chars=4000)
    assert "truncated" in s
    assert len(s) < 5_000


def test_summary_lists_artifacts() -> None:
    r = ExecResult(
        stdout="",
        stderr="",
        exit_code=0,
        duration_ms=1.0,
        artifacts=("/tmp/a.pdf", "/tmp/b.csv"),
    )
    assert "/tmp/a.pdf" in r.summary()
    assert "/tmp/b.csv" in r.summary()
