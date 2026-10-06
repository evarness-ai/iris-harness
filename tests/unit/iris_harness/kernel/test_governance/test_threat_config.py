"""Tests for the threat-detection config loader (sub-phase 6a.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from iris_harness.kernel.governance.threat.config import ThreatDetectionConfig


def _repo_config() -> Path:
    # tests/unit/test_governance/ -> repo root -> config/governance/...
    return Path(__file__).resolve().parents[5] / "config" / "governance" / "threat-detection.yaml"


def test_packaged_config_loads_and_matches_decisions() -> None:
    cfg = ThreatDetectionConfig.from_yaml(_repo_config())

    assert cfg.enabled is True
    assert not hasattr(cfg, "fail_mode")  # parsed once, read by nothing: removed
    assert cfg.mode == "shadow"  # shadow-first rollout
    # D1
    assert cfg.backend.prompt_guard.provider == "transformers"
    assert cfg.backend.output_guard.model == "llama-guard3:1b"  # D5
    # D2 / D3
    assert cfg.inbound.on_detect == "require_approval"
    assert cfg.retrieved.on_detect == "transform"
    # D4
    assert "privacy" in cfg.output.enforce
    assert "hate" in cfg.output.log_only
    assert set(cfg.output.enforce).isdisjoint(cfg.output.log_only)
    # D5
    assert cfg.budget_for("inbound") == 150
    assert cfg.budget_for("output") == 400


def test_absent_file_degrades_to_disabled(tmp_path: Path) -> None:
    cfg = ThreatDetectionConfig.from_yaml(tmp_path / "does-not-exist.yaml")
    assert cfg.enabled is False
    assert cfg.inbound.enabled is False
    assert cfg.retrieved.enabled is False
    assert cfg.output.enabled is False


def test_defaults_apply_for_minimal_file(tmp_path: Path) -> None:
    p = tmp_path / "min.yaml"
    p.write_text("version: 1\n", encoding="utf-8")
    cfg = ThreatDetectionConfig.from_yaml(p)
    # All sub-blocks fall back to their D-default values.
    assert cfg.enabled is True
    assert cfg.inbound.threshold == pytest.approx(0.8)
    assert cfg.backend.output_guard.endpoint == "http://localhost:11434"


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("version: 1\nbogus_key: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ThreatDetectionConfig.from_yaml(p)


def test_out_of_range_threshold_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"inbound": {"threshold": 1.5}}), encoding="utf-8")
    with pytest.raises(ValueError):
        ThreatDetectionConfig.from_yaml(p)


def test_non_mapping_file_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ThreatDetectionConfig.from_yaml(p)


def test_fail_mode_is_not_a_config_key(tmp_path: Path) -> None:
    """It was parsed and read by nothing, so a "closed" in the file promised what never
    happened. The packaged file no longer carries it, and a file that does fails loudly
    (as ``scan_tools`` does) instead of being accepted and ignored."""
    packaged = yaml.safe_load(_repo_config().read_text(encoding="utf-8"))
    assert "fail_mode" not in packaged
    p = tmp_path / "old.yaml"
    p.write_text("version: 1\nfail_mode: closed\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ThreatDetectionConfig.from_yaml(p)
