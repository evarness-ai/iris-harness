"""Regression: governance kernel is SECURE BY DEFAULT (exp-007 GAP-13).

Previously ``kernel_from_env()`` returned ``None`` unless ``IRIS_GOVERNANCE_ENABLED``
was explicitly truthy, so the shipped default ran the agent with the governance kernel
*off*. The red-team confirmed the kernel was dormant. Now the kernel is built unless
the operator explicitly opts out with a falsy value.
"""

from __future__ import annotations

import pytest

from iris_harness.kernel.governance import kernel_from_env


def test_enabled_when_unset(monkeypatch):
    monkeypatch.delenv("IRIS_GOVERNANCE_ENABLED", raising=False)
    assert kernel_from_env() is not None


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "On"])
def test_enabled_when_truthy(monkeypatch, value):
    monkeypatch.setenv("IRIS_GOVERNANCE_ENABLED", value)
    assert kernel_from_env() is not None


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "FALSE", "Off"])
def test_disabled_only_on_explicit_optout(monkeypatch, value):
    monkeypatch.setenv("IRIS_GOVERNANCE_ENABLED", value)
    assert kernel_from_env() is None


def test_unrecognised_value_defaults_to_enabled(monkeypatch):
    # A typo must fail safe (enabled), not silently disable governance.
    monkeypatch.setenv("IRIS_GOVERNANCE_ENABLED", "enabledd")
    assert kernel_from_env() is not None
