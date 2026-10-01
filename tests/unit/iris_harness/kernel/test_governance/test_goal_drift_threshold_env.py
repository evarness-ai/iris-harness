"""``IRIS_GOVERNANCE_GOAL_DRIFT_MAX_DISTANCE`` — retuning goal_drift without a code edit.

The threshold was a constructor default that no caller passed, so the only way to
retune a signal that halts real turns (session ``web-6d670ccd``) was to edit the
signal. A bad value must degrade to the default, never fail the kernel build.
"""

from __future__ import annotations

import pytest

from iris_harness.kernel.governance.wiring import _goal_drift_overrides

_VAR = "IRIS_GOVERNANCE_GOAL_DRIFT_MAX_DISTANCE"


def test_unset_leaves_the_signal_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_VAR, raising=False)
    assert _goal_drift_overrides() == {}


def test_a_valid_threshold_is_passed_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_VAR, "0.85")
    assert _goal_drift_overrides() == {"max_distance": 0.85}


@pytest.mark.parametrize("raw", ["", "   ", "loose", "0.65x", "-0.1", "2.5"])
def test_unusable_values_fall_back_to_the_default(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unparseable and out-of-range alike: ignored, not raised. A typo in an env var
    must not be able to take the harness down at kernel-build time."""
    monkeypatch.setenv(_VAR, raw)
    assert _goal_drift_overrides() == {}


def test_the_bounds_are_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cosine distance lies in [0, 2] — the signal's own constructor accepts both ends."""
    monkeypatch.setenv(_VAR, "0.0")
    assert _goal_drift_overrides() == {"max_distance": 0.0}
    monkeypatch.setenv(_VAR, "2.0")
    assert _goal_drift_overrides() == {"max_distance": 2.0}
