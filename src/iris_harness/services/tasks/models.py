"""Pydantic models for the user-facing tasks subsystem.

See ADR-0005 (Task vocabulary + soft migration) and the canonical doc
§3.4 for the full vocabulary. The shape here intentionally stays
minimal — only what user-domain tasks and goals genuinely need; the
4-layer memory subsystem and the legacy reminders module keep their
own concerns.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from iris_harness.foundation.clock import utc_now

# ``expired``: aged out automatically (a meeting-prep task after its meeting, an
# overdue task after the digest's grace days); terminal like ``done``/``dropped``, but
# the system's, not the owner's, and it keeps its ``dedup_key`` so a producer never
# re-raises it. ``closed_reason`` says why. Nothing is deleted.
TaskStatus = Literal["open", "doing", "done", "dropped", "expired"]
GoalStatus = Literal["active", "paused", "achieved", "dropped"]
SourceKind = Literal[
    "email",
    "manual",
    "finance-bills",
    "finance-statements",
    "calendar-prep",
    "calendar-approval",  # ADR-0076: an R3 calendar invite awaiting approve/reject
    "mission-proposal",  # auto-created mission awaiting approve (run) / reject
    "filemanager-organize",  # FMX5: an organize plan awaiting batch approve/reject
    "filemanager-quarantine",  # FMX6: quarantined files nearing purge (warn before delete)
    "rag-ingest",  # FMX8: a model-proposed RAG ingestion awaiting approve/reject
    "photos-albums",  # FMX7: proposed Apple Photos albums awaiting approve/reject
    "reminder",  # loop-proof D14: a reminder not delivered / not acknowledged (Done / Snooze)
    "routine",
    "other",
]

WaitForKind = Literal["reply_from", "event", "manual"]

# A remediation CTA on a system-raised pending action (ADR-0073). A Task with
# ``action`` set is a "pending action" surfaced in the Action Center; a Task
# without one is an ordinary user todo.
ActionKind = Literal[
    "copy_command",  # display-only: a command the user runs locally (e.g. set a secret)
    "re_extract",  # safe: re-run extraction for the target item
    "register",  # safe: open the registration flow for an unknown institution/account
    "trust_domain",  # safe: trust a discovered finance sender domain (user overlay)
    "ignore_domain",  # safe: suppress future onboarding prompts for a sender domain
    "review",  # safe: open the offending item in the UI for inspection
    "execute",  # safe: the owning provider runs this approved pending action (FMX5)
]


class ActionChoice(BaseModel):
    """One answer on a choice card ("Yes, it's mine" / "Ignore").

    ``needs_option`` means the answer carries the card's option (the account type a
    "Yes" confirms); ``primary`` marks the answer the UI emphasises.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    value: str = Field(..., min_length=1, max_length=40)
    label: str = Field(..., min_length=1, max_length=60)
    primary: bool = False
    needs_option: bool = False


class ActionOptionValue(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    value: str = Field(..., min_length=1, max_length=40)
    label: str = Field(..., min_length=1, max_length=60)


class ActionOptions(BaseModel):
    """A single pick the owner can change before answering (e.g. an account type)."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    name: str = Field(..., min_length=1, max_length=40)
    label: str = Field(..., min_length=1, max_length=60)
    values: tuple[ActionOptionValue, ...] = Field(..., min_length=1)
    default: str | None = None  # None: the owner must pick before a needs_option answer

    @model_validator(mode="after")
    def _default_is_a_value(self) -> ActionOptions:
        if self.default is not None and self.default not in {v.value for v in self.values}:
            raise ValueError(f"default {self.default!r} is not one of the option values")
        return self


class ActionFact(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    label: str = Field(..., min_length=1, max_length=40)
    value: str = Field(..., min_length=1, max_length=200)


class ActionEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    when: str = Field(default="", max_length=40)
    text: str = Field(..., min_length=1, max_length=300)


class ActionCard(BaseModel):
    """What a choice card shows besides its title: facts, the evidence behind the
    question, and a note on what answering does. Rendering is the channel's job."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    tag: str | None = Field(default=None, max_length=40)
    facts: tuple[ActionFact, ...] = Field(default_factory=tuple)
    evidence: tuple[ActionEvidence, ...] = Field(default_factory=tuple)
    evidence_label: str = Field(default="Why IRIS asks", max_length=60)
    note: str = Field(default="", max_length=500)


class TaskAction(BaseModel):
    """Remediation CTA attached to a system-raised pending action (ADR-0073).

    ``safe`` actions (``re_extract``/``register``/``trust_domain``/``ignore_domain``/
    ``review``/``execute``) may be invoked from the Web UI behind the write gate;
    ``copy_command`` is display-only — secrets never cross the web boundary, so the
    user runs ``command`` locally. An ``execute`` action is an explicit human
    approval that lets the owning provider run an approved FileManager mutation
    (organize/rag/photos/quarantine) — the model never reaches this path.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    kind: ActionKind
    label: str = Field(..., min_length=1, max_length=120)
    command: str | None = Field(default=None, min_length=1)
    target_id: str | None = Field(default=None, min_length=1)
    safe: bool = False
    # A choice card (ADR-0121): the owner answers with one of ``choices`` instead of
    # pressing a single button, optionally carrying the value picked in ``options``.
    # Empty ``choices`` is the classic one-button action.
    choices: tuple[ActionChoice, ...] = Field(default_factory=tuple)
    options: ActionOptions | None = None
    card: ActionCard | None = None

    @model_validator(mode="after")
    def _validate_action_shape(self) -> TaskAction:
        if self.kind == "copy_command":
            if not self.command:
                raise ValueError("copy_command action requires a command")
            if self.safe:
                raise ValueError("copy_command actions are display-only and cannot be safe")
        if self.choices and not self.safe:
            raise ValueError("a choice card is answered from the UI, so it must be safe")
        if len({c.value for c in self.choices}) != len(self.choices):
            raise ValueError("choice values must be unique")
        if any(c.needs_option for c in self.choices) and self.options is None:
            raise ValueError("a choice that needs an option requires options")
        return self

    def check_answer(self, choice: str | None, option: str | None) -> None:
        """Raise ValueError unless ``choice``/``option`` answer this action."""
        if not self.choices:
            if choice is not None or option is not None:
                raise ValueError("this action takes no choice")
            return
        picked = next((c for c in self.choices if c.value == choice), None)
        if picked is None:
            allowed = ", ".join(c.value for c in self.choices)
            raise ValueError(f"answer with one of: {allowed}")
        if picked.needs_option:
            assert self.options is not None  # enforced by the validator
            if option is None:
                option = self.options.default
            if option not in {v.value for v in self.options.values}:
                raise ValueError(f"pick a {self.options.label.lower()} first")


class WaitFor(BaseModel):
    """Auto-resolution condition for a follow-up Task.

    A Task with ``wait_for`` set auto-resolves when the matching event
    arrives (e.g., an email reply from the awaited sender); otherwise
    it fires a reminder at ``due_at``.
    """

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    kind: WaitForKind
    payload: dict[str, Any] = Field(default_factory=dict)


class Task(BaseModel):
    """User-facing actionable item with status, optional due date, and
    optional auto-resolution condition. See ADR-0005."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=500)
    description: str = ""
    status: TaskStatus = "open"
    priority: int = 0
    source_kind: SourceKind | None = None
    source_id: str | None = Field(default=None, min_length=1)
    parent_task_id: str | None = Field(default=None, min_length=1)
    dedup_key: str | None = Field(default=None, min_length=1)
    due_at: datetime | None = None
    wait_for: WaitFor | None = None
    wait_for_resolved_at: datetime | None = None
    parent_goal_id: str | None = Field(default=None, min_length=1)
    related_wikilinks: tuple[str, ...] = Field(default_factory=tuple)
    calendar_event_id: str | None = Field(default=None, min_length=1)
    calendar_visible: bool = False
    # Set on system-raised pending actions (ADR-0073); None for user todos.
    action: TaskAction | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    # Why a task was closed without the owner doing it ("expired: meeting ended").
    closed_reason: str | None = None

    @model_validator(mode="after")
    def _validate_completion_consistency(self) -> Task:
        if self.status == "done" and self.completed_at is None:
            raise ValueError("status='done' requires completed_at to be set")
        if self.status != "done" and self.completed_at is not None:
            # 'dropped' is a terminal status too but we track it via status, not timestamp.
            raise ValueError("completed_at may only be set when status='done'")
        return self


class Goal(BaseModel):
    """Long-horizon outcome. Tasks may reference a parent goal via
    ``Task.parent_goal_id``. See ADR-0005."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)

    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=500)
    description: str = ""
    status: GoalStatus = "active"
    target_date: datetime | None = None
    success_criteria: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def _validate_completion_consistency(self) -> Goal:
        if self.status == "achieved" and self.completed_at is None:
            raise ValueError("status='achieved' requires completed_at to be set")
        if self.status != "achieved" and self.completed_at is not None:
            raise ValueError("completed_at may only be set when status='achieved'")
        return self
