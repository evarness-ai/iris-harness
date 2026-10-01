"""The playground's runtime builder applies a write-safety floor.

A scenario run must never side-effect the real machine (Apple/Google calendar
write-back) unless the suite explicitly opts back in — the regression here is
the 2026-07-05 campaign writing six real calendar events from an "isolated"
profile.
"""

from __future__ import annotations

import os

import pytest

from iris_harness.playground import service

_KEYS = ("IRIS_DISABLE_EXTERNAL_WRITES", "IRIS_CALENDAR_APPLE_WRITE")


class _StubRuntime:
    def __init__(self, seen: dict[str, str | None]) -> None:
        self._seen = seen

    def startup(self) -> None:
        self._seen.update({k: os.environ.get(k) for k in _KEYS})

    def shutdown(self) -> None:
        pass


def _patch_build(monkeypatch: pytest.MonkeyPatch, seen: dict[str, str | None]) -> None:
    import iris_harness.runtime

    monkeypatch.setattr(iris_harness.runtime, "build_runtime", lambda: _StubRuntime(seen))


def test_safety_floor_applied_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _KEYS:
        monkeypatch.delenv(key, raising=False)
    seen: dict[str, str | None] = {}
    _patch_build(monkeypatch, seen)
    with service._built_runtime({}):
        pass
    assert seen["IRIS_DISABLE_EXTERNAL_WRITES"] == "1"
    assert seen["IRIS_CALENDAR_APPLE_WRITE"] == "0"
    # Restored afterwards — the floor must not leak into the caller's env.
    for key in _KEYS:
        assert key not in os.environ


def test_suite_env_can_opt_back_in(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str | None] = {}
    _patch_build(monkeypatch, seen)
    with service._built_runtime({"IRIS_CALENDAR_APPLE_WRITE": "1"}):
        pass
    assert seen["IRIS_CALENDAR_APPLE_WRITE"] == "1"
    assert seen["IRIS_DISABLE_EXTERNAL_WRITES"] == "1"


def test_prior_env_values_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRIS_CALENDAR_APPLE_WRITE", "1")
    seen: dict[str, str | None] = {}
    _patch_build(monkeypatch, seen)
    with service._built_runtime({}):
        pass
    assert seen["IRIS_CALENDAR_APPLE_WRITE"] == "0"
    assert os.environ["IRIS_CALENDAR_APPLE_WRITE"] == "1"
