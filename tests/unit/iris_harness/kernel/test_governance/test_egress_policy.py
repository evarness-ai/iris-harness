"""``config/governance/egress.yaml``: the egress gate's class -> tier policy, as config.

The shipped table is the owner's policy (2026-09-30): secret is local only (any local
tier, never tier_3), personal asks before tier_3, internal and public may reach tier_3.
The loader refuses a table that would send a secret off the owner's machines or let it
ask instead of deny, so no config edit can loosen that rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_harness.foundation.paths import repo_root
from iris_harness.kernel.governance import HookContext, HookPoint, LLMTier
from iris_harness.kernel.governance.egress_policy import (
    EgressPolicy,
    egress_policy_path,
    load_egress_policy,
)
from iris_harness.kernel.governance.plugins import EgressGate

_SHIPPED = repo_root() / "config" / "governance" / "egress.yaml"

_TIERS: tuple[LLMTier, ...] = ("tier_1", "tier_2", "tier_3")


def _table(secret_max: str, *, violation: str = "deny") -> str:
    return f"""
classes:
  secret: {{max_tier: {secret_max}, on_violation: {violation}}}
  personal: {{max_tier: tier_2, on_violation: require_approval}}
  internal: {{max_tier: tier_3}}
  public: {{max_tier: tier_3}}
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "egress.yaml"
    path.write_text(text)
    return path


def test_the_shipped_policy_matrix() -> None:
    """secret local only; personal/internal/public exactly as before."""
    policy = load_egress_policy(_SHIPPED)

    allowed = {(c, t): policy.allows(c, t) for c in policy.classes for t in _TIERS}

    assert allowed == {
        ("secret", "tier_1"): True,
        ("secret", "tier_2"): True,
        ("secret", "tier_3"): False,
        ("personal", "tier_1"): True,
        ("personal", "tier_2"): True,
        ("personal", "tier_3"): False,
        ("internal", "tier_1"): True,
        ("internal", "tier_2"): True,
        ("internal", "tier_3"): True,
        ("public", "tier_1"): True,
        ("public", "tier_2"): True,
        ("public", "tier_3"): True,
    }
    assert policy.rule("secret").on_violation == "deny"
    assert policy.rule("personal").on_violation == "require_approval"


def test_secret_to_tier_3_does_not_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="secret is local only"):
        load_egress_policy(_write(tmp_path, _table("tier_3")))


def test_a_secret_violation_that_asks_does_not_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hard deny"):
        load_egress_policy(_write(tmp_path, _table("tier_2", violation="require_approval")))


def test_a_missing_class_does_not_load(tmp_path: Path) -> None:
    text = "classes:\n  secret: {max_tier: tier_2}\n  public: {max_tier: tier_3}\n"
    with pytest.raises(ValueError, match="no rule for internal, personal"):
        load_egress_policy(_write(tmp_path, text))


def test_a_misspelled_key_does_not_load(tmp_path: Path) -> None:
    text = _table("tier_2").replace("public: {max_tier: tier_3}", "public: {max_teir: tier_3}")
    with pytest.raises(ValueError):
        load_egress_policy(_write(tmp_path, text))


def test_no_file_does_not_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unreadable"):
        load_egress_policy(tmp_path / "absent.yaml")


def test_an_owner_config_dir_without_the_file_reads_the_shipped_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment's config dir holds only the files it changes."""
    monkeypatch.setenv("IRIS_CONFIG_DIR", str(tmp_path))

    assert egress_policy_path() == _SHIPPED
    assert load_egress_policy() == load_egress_policy(_SHIPPED)


def test_an_owner_config_dir_file_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "governance").mkdir()
    own = tmp_path / "governance" / "egress.yaml"
    own.write_text(_table("tier_1"))
    monkeypatch.setenv("IRIS_CONFIG_DIR", str(tmp_path))

    assert egress_policy_path() == own
    assert load_egress_policy().rule("secret").max_tier == "tier_1"


async def test_the_gate_decides_by_the_table_it_is_given(tmp_path: Path) -> None:
    """A table may tighten secret to tier_1; the gate then refuses tier_2 (still local)."""
    gate = EgressGate(policy=EgressPolicy.from_yaml(_write(tmp_path, _table("tier_1"))))
    ctx = HookContext(
        hook_point=HookPoint.PRE_LLM_CALL,
        run_id="run-egress-policy",
        agent_type="chat",
        classification="secret",
        tier="tier_2",
        payload={"prompt": "hello"},
    )

    decision = await gate(ctx)

    assert decision.outcome == "deny"
    assert decision.severity == "critical"
    assert decision.reason == "egress_gate: secret data may reach at most tier_1 (target=tier_2)"
