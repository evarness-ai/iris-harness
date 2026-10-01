"""The host pressure sample: pmset parsing, and the snapshot it builds.

Moved here with the sampler at M6.2 -- it was `llm/arbiter.py`'s, but reading psutil
and pmset is not LLM logic, and the session log and `iris system status` read the same
sample (OSS plan M6, decision 6).
"""

from __future__ import annotations

import subprocess

from iris_harness.foundation.observability import host_pressure
from iris_harness.foundation.observability.host_pressure import _read_pmset_thermal


def test_pmset_returns_100_when_pmset_missing(monkeypatch) -> None:
    def _raise(*_a, **_kw):
        raise FileNotFoundError("pmset")

    monkeypatch.setattr(host_pressure.subprocess, "run", _raise)
    assert _read_pmset_thermal() == 100


def test_pmset_returns_100_when_no_match(monkeypatch) -> None:
    class _Result:
        stdout = "no relevant output here"

    monkeypatch.setattr(host_pressure.subprocess, "run", lambda *a, **kw: _Result())
    assert _read_pmset_thermal() == 100


def test_pmset_parses_throttled_value(monkeypatch) -> None:
    class _Result:
        stdout = "Currently in use:\n CPU_Speed_Limit \t = 70\n"

    monkeypatch.setattr(host_pressure.subprocess, "run", lambda *a, **kw: _Result())
    assert _read_pmset_thermal() == 70


def test_pmset_returns_100_on_timeout(monkeypatch) -> None:
    def _timeout(*_a, **_kw):
        raise subprocess.TimeoutExpired(cmd="pmset", timeout=2.0)

    monkeypatch.setattr(host_pressure.subprocess, "run", _timeout)
    assert _read_pmset_thermal() == 100
