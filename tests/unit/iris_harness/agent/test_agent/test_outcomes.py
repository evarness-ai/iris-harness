"""Unit tests for the measured-outcomes core module (§4.2)."""

from __future__ import annotations

from pathlib import Path

from iris_harness.agent.outcomes import (
    CorrectionVerdict,
    OutcomeConfig,
    load_outcome_config,
    looks_like_correction,
    parse_correction_verdict,
)


def test_parse_correction_true() -> None:
    v = parse_correction_verdict('{"is_correction": true, "confidence": 0.9, "reason": "re-ask"}')
    assert v is not None
    assert v.is_correction is True
    assert v.confidence == 0.9


def test_parse_correction_accepts_correction_key_alias() -> None:
    v = parse_correction_verdict('{"correction": false, "confidence": 0.2}')
    assert v is not None
    assert v.is_correction is False


def test_parse_correction_tolerates_prose() -> None:
    v = parse_correction_verdict('Verdict: {"is_correction": true, "confidence": 0.7} done')
    assert v is not None and v.is_correction is True


def test_parse_correction_rejects_missing_field() -> None:
    assert parse_correction_verdict('{"confidence": 0.9}') is None


def test_parse_correction_rejects_non_json() -> None:
    assert parse_correction_verdict("no json here") is None
    assert parse_correction_verdict("") is None


def test_correction_confidence_clamped() -> None:
    v = parse_correction_verdict('{"is_correction": true, "confidence": 9}')
    assert v is not None and v.confidence == 1.0


def test_verdict_to_metadata() -> None:
    meta = CorrectionVerdict(is_correction=True, confidence=0.5, reason="x").to_metadata()
    assert meta == {"is_correction": True, "confidence": 0.5, "reason": "x"}


def test_looks_like_correction_positive() -> None:
    assert looks_like_correction("No, that's wrong") is True
    assert looks_like_correction("actually I meant the other one") is True
    assert looks_like_correction("that's not what I asked") is True


def test_looks_like_correction_negative() -> None:
    assert looks_like_correction("thanks, that's perfect") is False
    assert looks_like_correction("what's the weather tomorrow?") is False
    assert looks_like_correction("") is False


def test_load_outcome_config_defaults_off(tmp_path: Path) -> None:
    cfg = load_outcome_config(tmp_path, env={})
    assert isinstance(cfg, OutcomeConfig)
    assert cfg.user_correction_enabled is False
    assert cfg.correction_confidence_floor == 0.6


def test_load_outcome_config_reads_yaml(tmp_path: Path) -> None:
    (tmp_path / "outcomes.yaml").write_text(
        "user_correction:\n  enabled: true\n  confidence_floor: 0.8\n", encoding="utf-8"
    )
    cfg = load_outcome_config(tmp_path, env={})
    assert cfg.user_correction_enabled is True
    assert cfg.correction_confidence_floor == 0.8


def test_env_flag_force_enables_correction(tmp_path: Path) -> None:
    cfg = load_outcome_config(tmp_path, env={"IRIS_LEARNING_CORRECTION_JUDGE": "1"})
    assert cfg.user_correction_enabled is True


def test_malformed_outcomes_yaml_degrades_off(tmp_path: Path) -> None:
    (tmp_path / "outcomes.yaml").write_text("{bad: yaml: :", encoding="utf-8")
    cfg = load_outcome_config(tmp_path, env={})
    assert cfg.user_correction_enabled is False
