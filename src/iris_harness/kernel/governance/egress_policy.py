"""``config/governance/egress.yaml``: where each data class may be sent, as config.

The egress gate (``plugins/egress_gate.py``) decides every ``PreLLMCall`` by the
prompt's class and the call's target tier. The class -> furthest-tier table was a dict in
the gate; it is the privacy policy, so it lives in config beside the other governance
tables and the gate reads it once, when the kernel is built (design §5.3: nothing
changes after ``init_lock``).

The tiers mean locality (``llm/locality.py``): ``tier_1``/``tier_2`` run on the owner's
machines, ``tier_3`` leaves them. One rule is not the file's to change: ``secret`` never
reaches ``tier_3`` and a violation is always a hard deny, so the loader refuses a table
that loosens it rather than govern by it.

``extra="forbid"`` everywhere, as in ``identity_config.py``: a misspelled key fails at
load. An owner config dir without the file reads the shipped one (as the memory configs
do); no file anywhere, or a malformed one, raises -- the kernel is not built without its
egress policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from iris_harness.foundation.paths import config_path, default_config_dir
from iris_harness.kernel.governance.hooks.types import DataClassification, LLMTier

EGRESS_FILE = ("governance", "egress.yaml")

CLASSES: tuple[DataClassification, ...] = ("public", "internal", "personal", "secret")
TIER_RANK: dict[LLMTier, int] = {"tier_1": 1, "tier_2": 2, "tier_3": 3}

# What a call beyond its class's furthest tier gets.
OnViolation = Literal["deny", "require_approval"]


class ClassRule(BaseModel):
    """One class: the furthest tier it may reach, and what a call beyond it gets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tier: LLMTier
    on_violation: OnViolation = "deny"


class EgressPolicy(BaseModel):
    """The parsed ``egress.yaml``: a rule for every class."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    classes: dict[DataClassification, ClassRule]

    @model_validator(mode="after")
    def _complete_and_safe(self) -> EgressPolicy:
        missing = [c for c in CLASSES if c not in self.classes]
        if missing:
            raise ValueError(f"egress policy: no rule for {', '.join(missing)}")
        secret = self.classes["secret"]
        if secret.max_tier == "tier_3":
            raise ValueError("egress policy: secret is local only; it may never reach tier_3")
        if secret.on_violation != "deny":
            raise ValueError("egress policy: a secret violation is always a hard deny")
        return self

    def rule(self, classification: DataClassification) -> ClassRule:
        return self.classes[classification]

    def allows(self, classification: DataClassification, tier: LLMTier) -> bool:
        """Whether ``classification`` may reach ``tier`` without a violation."""
        return TIER_RANK[tier] <= TIER_RANK[self.classes[classification].max_tier]

    @classmethod
    def from_yaml(cls, path: Path) -> EgressPolicy:
        """Load and validate ``path``; absent or malformed raises ``ValueError``."""
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"egress policy unreadable: {path}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"egress policy must decode to a mapping: {path}")
        return cls.model_validate(raw)


def egress_policy_path() -> Path:
    """The config dir's ``egress.yaml``, else the shipped one."""
    path = config_path(*EGRESS_FILE)
    if path.exists():
        return path
    return default_config_dir().joinpath(*EGRESS_FILE)


def load_egress_policy(path: Path | None = None) -> EgressPolicy:
    """``egress.yaml`` from the resolved config dir (or ``path``)."""
    return EgressPolicy.from_yaml(path or egress_policy_path())


__all__ = [
    "CLASSES",
    "EGRESS_FILE",
    "TIER_RANK",
    "ClassRule",
    "EgressPolicy",
    "OnViolation",
    "egress_policy_path",
    "load_egress_policy",
]
