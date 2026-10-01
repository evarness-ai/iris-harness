"""Core models for the code-first skills subsystem."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from re import findall
from typing import Annotated, Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, model_validator


def skills_utc_now() -> datetime:
    """Return the current UTC timestamp for skill proposal records."""
    return datetime.now(UTC)


TOOL_ARG_TYPES = ("string", "int", "number", "enum", "bool")
_NUMERIC_TOOL_ARG_TYPES = frozenset({"int", "number"})


class ToolArg(BaseModel):
    """A single argument exposed by a skill tool, declared in manifest YAML.

    Carries the metadata the routine authoring flow (and any other
    LLM-driven UX) needs to: (1) decide whether to ask the user, (2)
    compose the prompt, (3) enumerate options when applicable, and (4)
    validate the parsed answer. Source of truth for required-arg
    detection — preferred over introspecting ``tool.args_schema`` which
    cannot distinguish "has a default" from "do not ask the user".
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    type: Literal["string", "int", "number", "enum", "bool"] = "string"
    options: tuple[str, ...] = Field(default_factory=tuple)
    required: bool = True
    default: Any = None
    examples: tuple[str, ...] = Field(default_factory=tuple)
    prompt: str = ""
    pattern: str | None = Field(default=None)
    min: int | float | None = Field(default=None)
    max: int | float | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_type_constraints(self) -> ToolArg:
        if self.type == "enum" and not self.options:
            raise ValueError(
                f"args[{self.name!r}]: type='enum' requires a non-empty 'options' list"
            )
        if self.options and self.type != "enum":
            raise ValueError(f"args[{self.name!r}]: 'options' is only valid when type='enum'")
        if self.pattern is not None:
            if self.type != "string":
                raise ValueError(f"args[{self.name!r}]: 'pattern' is only valid when type='string'")
            try:
                re.compile(self.pattern)
            except re.error as exc:
                raise ValueError(
                    f"args[{self.name!r}]: 'pattern' is not a valid regex: {exc}"
                ) from exc
        if self.min is not None and self.type not in _NUMERIC_TOOL_ARG_TYPES:
            raise ValueError(
                f"args[{self.name!r}]: 'min' is only valid when type is 'int' or 'number'"
            )
        if self.max is not None and self.type not in _NUMERIC_TOOL_ARG_TYPES:
            raise ValueError(
                f"args[{self.name!r}]: 'max' is only valid when type is 'int' or 'number'"
            )
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f"args[{self.name!r}]: 'min' ({self.min}) must be <= 'max' ({self.max})"
            )
        if self.default is not None:
            self._validate_default()
        return self

    def _validate_default(self) -> None:
        value = self.default
        if self.type == "enum":
            if value not in self.options:
                raise ValueError(
                    f"args[{self.name!r}]: default {value!r} not in options {list(self.options)}"
                )
        elif self.type == "string":
            if not isinstance(value, str):
                raise ValueError(f"args[{self.name!r}]: default must be a string for type='string'")
            if self.pattern is not None and not re.fullmatch(self.pattern, value):
                raise ValueError(
                    f"args[{self.name!r}]: default {value!r} does not match pattern {self.pattern!r}"
                )
        elif self.type == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"args[{self.name!r}]: default must be an int for type='int'")
            if self.min is not None and value < self.min:
                raise ValueError(f"args[{self.name!r}]: default {value} is below min {self.min}")
            if self.max is not None and value > self.max:
                raise ValueError(f"args[{self.name!r}]: default {value} is above max {self.max}")
        elif self.type == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"args[{self.name!r}]: default must be a number for type='number'")
            if self.min is not None and value < self.min:
                raise ValueError(f"args[{self.name!r}]: default {value} is below min {self.min}")
            if self.max is not None and value > self.max:
                raise ValueError(f"args[{self.name!r}]: default {value} is above max {self.max}")
        elif self.type == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"args[{self.name!r}]: default must be a bool for type='bool'")


class SkillToolManifest(BaseModel):
    """Metadata describing a tool exposed by a skill package."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    governor_route: str = Field(..., min_length=1)
    args: tuple[ToolArg, ...] = Field(default_factory=tuple)


class RequiredCredential(BaseModel):
    """One vault handle a skill needs to execute (Phase 2 governance §7.1).

    The handle is a ``vault://...`` reference resolved at PreToolUse by
    the CredentialBroker. ``route`` records the governor route the
    credential is scoped to (informational; matched against the route
    declared on the corresponding tool).
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    handle: str = Field(..., min_length=1)
    route: str | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_handle_prefix(self) -> RequiredCredential:
        if not self.handle.startswith("vault://"):
            raise ValueError(
                f"required_credentials handle must start with 'vault://': got {self.handle!r}"
            )
        return self


class SkillRequirements(BaseModel):
    """Prerequisites that must be satisfied before a skill can load."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    python: str = Field(default=">=3.12", min_length=1)
    packages: tuple[str, ...] = Field(default_factory=tuple)
    env_vars: tuple[str, ...] = Field(default_factory=tuple)
    config_files: tuple[str, ...] = Field(default_factory=tuple)
    agents: tuple[str, ...] = Field(default_factory=tuple)
    required_credentials: tuple[RequiredCredential, ...] = Field(default_factory=tuple)


SLOT_FORMATS = ("bullets", "json", "text")


class BriefLiteralSlot(BaseModel):
    """A slot whose value is a literal format string evaluated at run time."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    kind: Literal["literal"]
    value: str = Field(..., min_length=1)


class BriefToolSlot(BaseModel):
    """A slot filled by invoking a tool exposed by a loaded skill."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    kind: Literal["tool"]
    skill: str = Field(..., min_length=1)
    tool: str = Field(..., min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    format: Literal["bullets", "json", "text"] = "text"
    empty: str = ""
    summary: str = Field(
        default="",
        description=(
            "Optional semantic descriptor used ONLY for section matching during "
            "routine authoring. When set, it replaces the auto-derived snippet "
            "(slot key + item_template + empty), letting an author disambiguate "
            "slots whose names embed close together — e.g. 'stocks' (market "
            "trending) vs 'portfolio' (the user's own holdings). Not rendered."
        ),
    )
    item_template: str | None = Field(
        default=None,
        description="Optional str.format template applied to each dict item when format='bullets'.",
    )
    count_if: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*$",
        description=(
            "Optional item key for a format='bullets' slot: only the dict items whose "
            "value for it is truthy count as the section's items (the digest's push "
            "line, e.g. 'Bills 1'); every item still renders. Unset: every bullet counts."
        ),
    )
    synthesis: bool = Field(
        default=False,
        description=(
            "A synthesis/headline slot that composes other sections (e.g. a "
            "'today's plan' summary). It still renders in the brief, but is NOT "
            "user-selectable as a section and is excluded from section matching "
            "during routine authoring — otherwise its content would collide "
            "with the brief's own name (see ADR-0059)."
        ),
    )


BriefSlot = Annotated[BriefLiteralSlot | BriefToolSlot, Field(discriminator="kind")]

_PLACEHOLDER_RE = r"{{\s*([A-Za-z_][A-Za-z0-9_]*)\s*}}"


class BriefGreeting(BaseModel):
    """One row of a brief's greeting table: from ``starts`` (owner's local HH:MM) on."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    starts: str = Field(..., pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    greeting: str = Field(..., min_length=1)
    daypart: str = ""


def pick_greeting(greetings: tuple[BriefGreeting, ...], hhmm: str) -> BriefGreeting | None:
    """The row whose ``starts`` is the latest at or before ``hhmm``; before the first
    start, the last row (the night wraps past midnight). None for an empty table."""
    rows = sorted(greetings, key=lambda g: g.starts)
    if not rows:
        return None
    earlier = [g for g in rows if g.starts <= hhmm]
    return earlier[-1] if earlier else rows[-1]


class BriefSpec(BaseModel):
    """Declarative brief template — layout, slots, and the skills it may invoke."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    subject: str = Field(..., min_length=1)
    recipient: str = Field(default="user", min_length=1)
    uses: tuple[str, ...] = Field(default_factory=tuple)
    layout: str = Field(..., min_length=1)
    slots: dict[str, BriefSlot] = Field(default_factory=dict)
    # Slots that close the brief (e.g. the digest's "learned yesterday" line):
    # always rendered last, after any "couldn't build" line.
    footer_slots: tuple[str, ...] = Field(default_factory=tuple)
    # Time-of-day greeting (vocabulary, so it lives in the manifest): literal slots and
    # the subject may use {greeting} / {daypart}, picked by the owner's local time.
    greetings: tuple[BriefGreeting, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _validate_slots_and_layout(self) -> BriefSpec:
        placeholders = set(findall(_PLACEHOLDER_RE, self.layout))
        missing = placeholders - set(self.slots)
        if missing:
            raise ValueError(f"layout references unknown slots: {sorted(missing)}")
        unknown_footer = set(self.footer_slots) - set(self.slots)
        if unknown_footer:
            raise ValueError(f"footer_slots names unknown slots: {sorted(unknown_footer)}")
        uses_set = set(self.uses)
        for name, slot in self.slots.items():
            if isinstance(slot, BriefToolSlot) and slot.skill not in uses_set:
                raise ValueError(
                    f"slot {name!r} invokes skill {slot.skill!r} which is not in brief.uses"
                )
        return self


class SkillSource(BaseModel):
    """Distribution origin for a community skill (canonical doc §4.4)."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    type: Literal["git"] = "git"
    url: str = Field(..., min_length=1)
    ref: str | None = Field(default=None, min_length=1)  # tag / branch / commit


class SkillMaintainer(BaseModel):
    """Skill maintainer contact info (canonical doc §4.4)."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: str = Field(..., min_length=1)
    contact: str | None = Field(default=None, min_length=1)


class SkillManifest(BaseModel):
    """Manifest metadata loaded from a code-first skill package."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    author: str = Field(..., min_length=1)
    license: str = Field(..., min_length=1)
    tools: tuple[SkillToolManifest, ...] = Field(default_factory=tuple)
    requires: SkillRequirements = Field(default_factory=SkillRequirements)
    kind: str | None = Field(default=None)
    brief: BriefSpec | None = Field(default=None)

    # ─── Personal-assistant upgrade additions ──────────────────────────────
    default_enabled: bool = Field(default=True)
    """Whether the skill loads automatically on discovery. See ADR-0012."""

    iris_compatibility: str | None = Field(default=None)
    """Semver range against the running IRIS version (e.g. '>=0.7,<2.0').
    Validated for parseability at install time by `iris skill add`. See
    canonical doc §4.4."""

    source: SkillSource | None = Field(default=None)
    """Distribution origin for community skills. None for first-party
    in-tree skills. See canonical doc §4.4 + ADR-0009."""

    maintainers: tuple[SkillMaintainer, ...] = Field(default_factory=tuple)
    """Skill maintainer contacts. Empty for first-party. See canonical doc §4.4."""

    trust_level: Literal["first-party", "community"] = Field(default="first-party")
    """Drives install-time capability-prompt severity; not runtime
    enforcement. See ADR-0009."""

    homepage: str | None = Field(default=None)
    """URL to the skill's homepage/repo for human discovery."""

    @model_validator(mode="after")
    def _validate_kind_brief_consistency(self) -> SkillManifest:
        if self.kind == "brief" and self.brief is None:
            raise ValueError("manifest with kind='brief' must include a 'brief' section")
        if self.kind != "brief" and self.brief is not None:
            raise ValueError("'brief' section is only valid when kind='brief'")
        return self

    @model_validator(mode="after")
    def _validate_community_trust_has_source(self) -> SkillManifest:
        if self.trust_level == "community" and self.source is None:
            raise ValueError("trust_level='community' requires a 'source' block")
        return self


class SkillPackage(BaseModel):
    """Loaded skill package state for runtime discovery and registration."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    manifest: SkillManifest = Field(...)
    skill_dir: Path = Field(...)
    tools_module_path: Path = Field(...)
    tool_classes: tuple[type[BaseTool], ...] = Field(default_factory=tuple)
    agent_context: str | None = Field(default=None)
    missing_prerequisites: tuple[str, ...] = Field(default_factory=tuple)

    @property
    def is_loadable(self) -> bool:
        """Return whether the package can be registered for execution."""
        if self.missing_prerequisites:
            return False
        if self.manifest.kind == "brief":
            return self.manifest.brief is not None
        return bool(self.tool_classes)


class SkillProposal(BaseModel):
    """Manual quarantined skill proposal derived from a coding task."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    proposal_id: str = Field(..., min_length=1)
    task_id: str = Field(..., min_length=1)
    skill_name: str = Field(..., min_length=1)
    skill_slug: str = Field(..., min_length=1)
    scope: str = Field(..., min_length=1)
    project_slug: str | None = Field(default=None)
    source_description: str = Field(..., min_length=1)
    source_changed_files: tuple[str, ...] = Field(default_factory=tuple)
    tool_usage: tuple[str, ...] = Field(default_factory=tuple)
    skill_usage: tuple[str, ...] = Field(default_factory=tuple)
    reward_summary: str | None = Field(default=None)
    proposal_dir: str = Field(..., min_length=1)
    manifest_path: str = Field(..., min_length=1)
    status: str = Field(default="proposed", min_length=1)
    preflight_reason: str | None = Field(default=None)
    source_kind: Literal["crystallized", "sandbox", "manual"] = Field(default="crystallized")
    sandbox_script_path: str | None = Field(default=None)
    run_count: int = Field(default=0, ge=0)
    last_run_at: datetime | None = Field(default=None)
    wiki_page_id: str | None = Field(default=None)
    promotion_threshold: int = Field(default=3, ge=1)
    created_at: datetime = Field(default_factory=skills_utc_now)
