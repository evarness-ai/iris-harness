"""Unit tests for the escalation core module (ADR-0068)."""

from __future__ import annotations

from pathlib import Path

from iris_harness.agent.escalation import (
    EscalationConfig,
    EscalationVerdict,
    egress_eligible,
    load_escalation_config,
    more_restrictive,
    parse_escalation_verdict,
)


def test_parse_valid_verdict() -> None:
    v = parse_escalation_verdict(
        '{"action": "escalate", "diagnosis": "capability_gap",'
        ' "confidence": 0.8, "reason": "too hard for tier 1"}'
    )
    assert v is not None
    assert v.action == "escalate"
    assert v.diagnosis == "capability_gap"
    assert v.confidence == 0.8
    assert v.would_act is True


def test_parse_tolerates_prose_around_json() -> None:
    v = parse_escalation_verdict(
        'Here is my verdict:\n{"action": "accept", "diagnosis": "acceptable",'
        ' "confidence": 0.9}\nThat is all.'
    )
    assert v is not None
    assert v.action == "accept"
    assert v.would_act is False


def test_parse_derives_action_from_diagnosis_when_missing() -> None:
    # No/invalid action → route is derived so it always matches the cause.
    v = parse_escalation_verdict('{"diagnosis": "ambiguity", "confidence": 0.7}')
    assert v is not None
    assert v.action == "clarify"


def test_parse_rejects_unknown_diagnosis() -> None:
    assert parse_escalation_verdict('{"action": "escalate", "diagnosis": "vibes"}') is None


def test_parse_rejects_non_json() -> None:
    assert parse_escalation_verdict("the model is confused") is None
    assert parse_escalation_verdict("") is None


def test_confidence_is_clamped() -> None:
    v = parse_escalation_verdict('{"diagnosis": "acceptable", "confidence": 5}')
    assert v is not None and v.confidence == 1.0


def test_verdict_to_metadata_roundtrips_fields() -> None:
    v = EscalationVerdict(
        action="reroute",
        diagnosis="grounding_gap",
        confidence=0.6,
        reason="needs a search",
        tool_hint="research",
    )
    meta = v.to_metadata()
    assert meta["action"] == "reroute"
    assert meta["tool_hint"] == "research"
    assert meta["target_tier"] is None


def test_load_config_defaults_to_off(tmp_path: Path) -> None:
    cfg = load_escalation_config(tmp_path, env={})
    assert isinstance(cfg, EscalationConfig)
    assert cfg.enabled is False
    assert cfg.mode == "shadow"
    assert cfg.acts is False  # L2 never acts


def test_load_config_reads_yaml(tmp_path: Path) -> None:
    (tmp_path / "escalation.yaml").write_text(
        "enabled: true\nmode: shadow\nmax_escalations: 2\nsample_rate: 0.5\n",
        encoding="utf-8",
    )
    cfg = load_escalation_config(tmp_path, env={})
    assert cfg.enabled is True
    assert cfg.max_escalations == 2
    assert cfg.sample_rate == 0.5


def test_action_confidence_floor_defaults_and_reads_yaml(tmp_path: Path) -> None:
    assert load_escalation_config(tmp_path, env={}).action_confidence_floor == 0.7
    (tmp_path / "escalation.yaml").write_text("action_confidence_floor: 0.85\n", encoding="utf-8")
    assert load_escalation_config(tmp_path, env={}).action_confidence_floor == 0.85


def test_action_confidence_floor_is_clamped(tmp_path: Path) -> None:
    (tmp_path / "escalation.yaml").write_text("action_confidence_floor: 3\n", encoding="utf-8")
    assert load_escalation_config(tmp_path, env={}).action_confidence_floor == 1.0


def test_env_flag_force_enables(tmp_path: Path) -> None:
    # No yaml present, but the env flag opts the judge in.
    cfg = load_escalation_config(tmp_path, env={"IRIS_CURATOR_ESCALATION": "1"})
    assert cfg.enabled is True


def test_malformed_yaml_degrades_to_off(tmp_path: Path) -> None:
    (tmp_path / "escalation.yaml").write_text("{ not: valid: yaml: ::", encoding="utf-8")
    cfg = load_escalation_config(tmp_path, env={})
    assert cfg.enabled is False


def test_acts_only_when_enabled_and_enforce(tmp_path: Path) -> None:
    (tmp_path / "escalation.yaml").write_text("enabled: true\nmode: enforce\n", encoding="utf-8")
    cfg = load_escalation_config(tmp_path, env={})
    assert cfg.mode == "enforce"
    assert cfg.acts is True  # L3: enabled + enforce => the runtime acts


def test_acts_false_in_shadow(tmp_path: Path) -> None:
    (tmp_path / "escalation.yaml").write_text("enabled: true\nmode: shadow\n", encoding="utf-8")
    assert load_escalation_config(tmp_path, env={}).acts is False


def test_acts_false_when_disabled(tmp_path: Path) -> None:
    (tmp_path / "escalation.yaml").write_text("enabled: false\nmode: enforce\n", encoding="utf-8")
    assert load_escalation_config(tmp_path, env={}).acts is False


def test_priors_config_defaults_off(tmp_path: Path) -> None:
    cfg = load_escalation_config(tmp_path, env={})
    assert cfg.priors_enabled is False
    assert cfg.priors_min_samples == 20
    assert cfg.priors_escalate_rate == 0.4


def test_priors_config_reads_yaml_and_env(tmp_path: Path) -> None:
    (tmp_path / "escalation.yaml").write_text(
        "priors:\n  enabled: true\n  min_samples: 50\n  escalate_rate: 0.6\n", encoding="utf-8"
    )
    cfg = load_escalation_config(tmp_path, env={})
    assert cfg.priors_enabled is True
    assert cfg.priors_min_samples == 50
    assert cfg.priors_escalate_rate == 0.6
    # env override force-enables
    cfg2 = load_escalation_config(tmp_path, env={"IRIS_CURATOR_ESCALATION_PRIORS": "1"})
    assert cfg2.priors_enabled is True


# --- cloud escalation egress (D4) ---


def test_egress_eligible_secret_never() -> None:
    assert egress_eligible("secret", frozenset({"secret", "public"})) is False


def test_egress_eligible_requires_allowlist() -> None:
    allowed = frozenset({"public"})
    assert egress_eligible("public", allowed) is True
    assert egress_eligible("internal", allowed) is False
    assert egress_eligible("personal", allowed) is False


def test_more_restrictive_picks_higher_severity() -> None:
    assert more_restrictive("public", "secret") == "secret"
    assert more_restrictive("internal", "public") == "internal"
    assert more_restrictive("personal", "internal") == "personal"
    assert more_restrictive("public", "unknown_label") == "secret"  # unknown -> fail closed


def test_cloud_config_defaults_off(tmp_path: Path) -> None:
    cfg = load_escalation_config(tmp_path, env={})
    assert cfg.allow_cloud is False
    assert cfg.cloud_classifications == frozenset({"public"})


def test_cloud_config_strips_secret_and_reads_env(tmp_path: Path) -> None:
    (tmp_path / "escalation.yaml").write_text(
        "cloud:\n  allow: true\n  classifications: [public, internal, secret]\n", encoding="utf-8"
    )
    cfg = load_escalation_config(tmp_path, env={})
    assert cfg.allow_cloud is True
    assert "secret" not in cfg.cloud_classifications  # secret never eligible, even if listed
    assert cfg.cloud_classifications == frozenset({"public", "internal"})
    cfg2 = load_escalation_config(tmp_path, env={"IRIS_CURATOR_ESCALATION_CLOUD": "1"})
    assert cfg2.allow_cloud is True
