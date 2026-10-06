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

from iris_harness.kernel.governance import kernel_from_env
from iris_harness.kernel.governance.threat import RemovedConfigKeyError
from iris_harness.kernel.governance.wiring import _input_safety_from_env, _prompt_guards_from_env
from iris_harness.runtime.judges import build_curator_output_safety_client


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


def _override_with_fail_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    config = tmp_path / "threat.yaml"
    config.write_text("version: 1\nfail_mode: closed\n")
    monkeypatch.setenv("IRIS_THREAT_DETECTION_CONFIG", str(config))
    return config


def test_an_override_with_the_removed_fail_mode_key_fails_startup_naming_the_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A guard the operator asked for must not turn off because of a line a release removed.

    Other config errors still degrade to "guards off" with a warning (above); a key that a
    release removed fails the build instead, with a message that names the key and the file.
    """
    config = _override_with_fail_mode(monkeypatch, tmp_path)
    monkeypatch.setenv("IRIS_GOVERNANCE_PROMPT_GUARD", "1")
    with pytest.raises(RemovedConfigKeyError, match=r"`fail_mode` key was removed.*delete") as err:
        _prompt_guards_from_env()
    assert str(config) in str(err.value)


def test_the_removed_key_also_fails_the_input_safety_screen_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _override_with_fail_mode(monkeypatch, tmp_path)
    monkeypatch.setenv("IRIS_GOVERNANCE_INPUT_SAFETY", "1")
    with pytest.raises(RemovedConfigKeyError, match="fail_mode"):
        _input_safety_from_env()


def test_the_removed_key_also_fails_the_curator_output_guard_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg_dir = tmp_path / "config"
    (cfg_dir / "governance").mkdir(parents=True)
    (cfg_dir / "governance" / "threat-detection.yaml").write_text("version: 1\nfail_mode: closed\n")
    monkeypatch.setenv("IRIS_CURATOR_OUTPUT_SAFETY", "1")
    with pytest.raises(RemovedConfigKeyError, match="fail_mode"):
        build_curator_output_safety_client(cfg_dir=cfg_dir)


def test_the_removed_key_fails_the_whole_kernel_build_when_a_guard_is_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _override_with_fail_mode(monkeypatch, tmp_path)
    monkeypatch.setenv("IRIS_GOVERNANCE_AUDIT_DB_PATH", str(tmp_path / "a.db"))
    monkeypatch.setenv("IRIS_GOVERNANCE_PROMPT_GUARD", "1")
    with pytest.raises(RemovedConfigKeyError):
        kernel_from_env()


def test_the_removed_key_is_not_read_when_no_guard_is_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Guards that are off never load the file, so the key cannot fail an install that does
    not use them (the packaged config no longer carries it)."""
    _override_with_fail_mode(monkeypatch, tmp_path)
    monkeypatch.delenv("IRIS_GOVERNANCE_PROMPT_GUARD", raising=False)
    monkeypatch.delenv("IRIS_GOVERNANCE_INPUT_SAFETY", raising=False)
    monkeypatch.delenv("IRIS_CURATOR_OUTPUT_SAFETY", raising=False)
    assert _prompt_guards_from_env() == (None, None)
    assert _input_safety_from_env() is None
