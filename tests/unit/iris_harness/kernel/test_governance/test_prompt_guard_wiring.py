"""The prompt guards are opt-in; once requested, losing them must be loud.

``scan_tools`` left the threat-detection config when the retrieved-content guard began
reading each tool's own ``content: external`` declaration. The config model refuses
unknown keys, so an operator override that still lists it no longer loads, and wiring
degrades to no guards at all. That degrade is kept (kernel construction must not
break), but it is a warning naming both guards, never a debug line.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from iris_harness.kernel.governance.wiring import _prompt_guards_from_env


def test_an_override_that_no_longer_loads_warns_that_both_guards_are_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = tmp_path / "threat.yaml"
    config.write_text("version: 1\nretrieved:\n  enabled: true\n  scan_tools: [rag_search]\n")
    monkeypatch.setenv("IRIS_GOVERNANCE_PROMPT_GUARD", "1")
    monkeypatch.setenv("IRIS_THREAT_DETECTION_CONFIG", str(config))
    with caplog.at_level(logging.WARNING):
        assert _prompt_guards_from_env() == (None, None)
    assert "BOTH are OFF" in caplog.text


def test_not_requested_is_silent(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("IRIS_GOVERNANCE_PROMPT_GUARD", raising=False)
    with caplog.at_level(logging.WARNING):
        assert _prompt_guards_from_env() == (None, None)
    assert "prompt guards" not in caplog.text


def test_an_override_with_the_removed_fail_mode_key_says_what_to_delete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The user sees the warning, not a traceback: it names the key and says to delete it."""
    config = tmp_path / "threat.yaml"
    config.write_text("version: 1\nfail_mode: closed\n")
    monkeypatch.setenv("IRIS_GOVERNANCE_PROMPT_GUARD", "1")
    monkeypatch.setenv("IRIS_THREAT_DETECTION_CONFIG", str(config))
    with caplog.at_level(logging.WARNING):
        assert _prompt_guards_from_env() == (None, None)
    warning = next(r.getMessage() for r in caplog.records if "BOTH are OFF" in r.getMessage())
    assert "`fail_mode` key was removed" in warning and "delete that line" in warning
