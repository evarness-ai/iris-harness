"""``config/governance/identity.yaml``: owner-identity policy that is config, not code.

Two blocks (ADR-0125):

- ``ontology_kinds:`` which memory-ontology attributes of the owner's confirmed facts are
  owner identity, and as which kind. It lives here, not in ``config/memory/ontology.yaml``:
  the memris spec refuses unknown keys and memris knows no IRIS vocabulary (amendment 1).
- ``guards:`` the kind x guard action table (amendment 6): for each identity kind, what
  each guard does with an occurrence of it. Typed per guard, so an action a guard cannot
  take fails at load, and validated as a whole (every kind present, ``secret`` never
  unmasked, a first name alone never denying or halting).

``extra="forbid"`` everywhere, as in ``threat/config.py``: a misspelled key fails at load
rather than silently dropping policy. An absent file is an empty policy (no table).
"""

from __future__ import annotations

import threading
from fnmatch import fnmatch
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from iris_harness.foundation.paths import config_path
from iris_harness.foundation.process_state import track_globals
from iris_harness.kernel.governance.owner_identity import KINDS, OWNER_PII_KINDS, IdentityKind

IDENTITY_FILE = ("governance", "identity.yaml")

# The kinds an ontology attribute may map to: owner_identity.DECLARABLE_KINDS, as a type
# (a test keeps the two equal).
DeclarableKind = Literal["name", "email", "phone", "address", "handle", "link"]

# What each guard can do with an occurrence of an owner literal.
EgressAction = Literal["deny", "log", "pass"]  # network tool arguments
AnswerAction = Literal["halt", "mask", "pass"]  # the answer (PRE_RESPONSE)
CapabilityAction = Literal["mask", "pseudonym", "pass"]  # a capability result
Tier3Action = Literal["placeholder", "pass"]  # a prompt to a tier-3 (cloud) model
WebSearchAction = Literal["deny", "mask", "pass"]  # web-search arguments

# The table's columns. ``answer`` is two: an answer to the owner, and to anyone else.
GuardColumn = Literal["egress", "answer_owner", "answer_other", "capability", "tier3", "web_search"]
GUARD_COLUMNS: tuple[GuardColumn, ...] = (
    "egress",
    "answer_owner",
    "answer_other",
    "capability",
    "tier3",
    "web_search",
)
Action = Literal["deny", "halt", "log", "mask", "pseudonym", "placeholder", "pass"]
# Actions that stop the call or the answer.
BLOCKING: frozenset[str] = frozenset({"deny", "halt"})


class KindActions(BaseModel):
    """One row of the table: what every guard does with one kind."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    egress: EgressAction
    answer_owner: AnswerAction
    answer_other: AnswerAction
    capability: CapabilityAction
    tier3: Tier3Action
    web_search: WebSearchAction

    def of(self, column: GuardColumn) -> Action:
        action: Action = getattr(self, column)
        return action


class LogOnlyDestinations(BaseModel):
    """Network tools where an egress ``deny`` of ``kinds`` is only logged.

    Until a per-destination allowlist exists (ADR-0125 decision 4). Never ``secret`` or
    ``link``: egress refuses both on every network tool today, and a table must not be
    able to weaken that.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tools: tuple[str, ...] = Field(default_factory=tuple)
    kinds: tuple[IdentityKind, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _pii_only(self) -> LogOnlyDestinations:
        bad = sorted(set(self.kinds) - set(OWNER_PII_KINDS))
        if bad:
            raise ValueError(f"log_only_destinations cannot relax {', '.join(bad)}")
        return self

    def relaxes(self, kind: IdentityKind, tool: str | None) -> bool:
        return (
            tool is not None
            and kind in self.kinds
            and any(fnmatch(tool, pattern) for pattern in self.tools)
        )


class GuardTable(BaseModel):
    """The kind x guard action table (ADR-0125 amendment 6).

    - ``kinds``: one row per identity kind; every kind must have one.
    - ``first_name_alone``: the row for a ``name`` literal of one word (a first name or a
      nickname on its own). It never denies or halts, anywhere (ADR-0125 decision 4).
    - ``log_only_destinations``: network tools where an egress ``deny`` of some PII kinds
      is only logged, until a per-destination allowlist exists.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kinds: dict[IdentityKind, KindActions]
    first_name_alone: KindActions
    log_only_destinations: LogOnlyDestinations = Field(default_factory=LogOnlyDestinations)

    @model_validator(mode="after")
    def _complete_and_safe(self) -> GuardTable:
        missing = [k for k in KINDS if k not in self.kinds]
        if missing:
            raise ValueError(f"guards: no row for {', '.join(missing)}")
        secret = self.kinds["secret"]
        if secret.capability != "mask":
            # A secret reaches no consumer, granted or not: masked, never a pseudonym.
            raise ValueError("guards: secret must be 'mask' in capability results")
        for kind, row in self.kinds.items():
            if row.capability == "pseudonym" and kind not in OWNER_PII_KINDS:
                raise ValueError(f"guards: {kind} cannot be a pseudonym (it is not grantable)")
            if kind != "secret" and row.answer_owner == "halt":
                # Answers to the owner never halt on the owner's own identity.
                raise ValueError(f"guards: {kind} cannot halt an answer to the owner")
        blocking = [c for c in GUARD_COLUMNS if self.first_name_alone.of(c) in BLOCKING]
        if blocking:
            raise ValueError(
                f"guards: a first name alone never denies or halts ({', '.join(blocking)})"
            )
        return self

    def action(
        self,
        kind: IdentityKind,
        column: GuardColumn,
        *,
        first_name: bool = False,
        destination: str | None = None,
    ) -> Action:
        """The action for an occurrence of ``kind`` at ``column``.

        ``first_name``: the occurrence is a one-word ``name`` literal. ``destination``: the
        network tool, at ``egress``.
        """
        row = self.first_name_alone if first_name and kind == "name" else self.kinds[kind]
        action = row.of(column)
        if (
            column == "egress"
            and action == "deny"
            and self.log_only_destinations.relaxes(kind, destination)
        ):
            return "log"
        return action

    def grantable(self) -> frozenset[IdentityKind]:
        """The kinds a capability consumer sees as pseudonyms, so a manifest may unmask."""
        return frozenset(k for k, row in self.kinds.items() if row.capability == "pseudonym")


class IdentityConfig(BaseModel):
    """The parsed ``identity.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ontology attribute -> identity kind. Keys are checked against the loaded ontology by
    # the composition root's facts source and by a test; the kernel cannot read it.
    ontology_kinds: dict[str, DeclarableKind] = Field(default_factory=dict)
    guards: GuardTable | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> IdentityConfig:
        """Load and validate ``path``; absent is empty, malformed raises ``ValueError``."""
        if not path.exists():
            return cls()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"identity config must decode to a mapping: {path}")
        return cls.model_validate(raw)


def load_identity_config(path: Path | None = None) -> IdentityConfig:
    """``identity.yaml`` from the resolved config dir (or ``path``)."""
    return IdentityConfig.from_yaml(path or config_path(*IDENTITY_FILE))


_lock = threading.Lock()
_cached: tuple[tuple[str, int, int] | None, GuardTable | None] | None = None


def guard_table(path: Path | None = None) -> GuardTable | None:
    """The action table, re-read when the file changes; ``None`` when it has none.

    Raises ``ValueError`` when the file is malformed: a caller that acts on the table
    decides how it fails closed.
    """
    global _cached
    target = path or config_path(*IDENTITY_FILE)
    try:
        st = target.stat()
        stamp: tuple[str, int, int] | None = (str(target), st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    with _lock:
        if _cached is not None and _cached[0] == stamp and stamp is not None:
            return _cached[1]
    table = load_identity_config(target).guards
    with _lock:
        _cached = (stamp, table)
    return table


__all__ = [
    "BLOCKING",
    "GUARD_COLUMNS",
    "IDENTITY_FILE",
    "Action",
    "AnswerAction",
    "CapabilityAction",
    "DeclarableKind",
    "EgressAction",
    "GuardColumn",
    "GuardTable",
    "IdentityConfig",
    "KindActions",
    "LogOnlyDestinations",
    "Tier3Action",
    "WebSearchAction",
    "guard_table",
    "load_identity_config",
]

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_cached")
