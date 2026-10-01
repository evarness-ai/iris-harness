"""Shared fixtures for the IRIS API endpoint tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _allow_webui_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable control writes by default for endpoint tests.

    The IRIS_WEBUI_ALLOW_WRITES gate is a deployment concern; these tests
    exercise the handler logic behind it. test_write_gate.py overrides this
    per-case to assert the gate itself.
    """
    monkeypatch.setenv("IRIS_WEBUI_ALLOW_WRITES", "1")
