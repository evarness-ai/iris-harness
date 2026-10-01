"""The declared intercept dispatch chain (Phase 2, YAML-first).

The chain of deterministic short-circuits that run before the agent loop is
DECLARED in ``config/intercepts.yaml`` and loaded here, so ``chat()`` and
``chat_stream()`` consume one ordered list instead of hand-copying it. A
hardcoded default mirrors the YAML so the runtime still boots if the file is
missing or unreadable (config-with-defaults, like the rest of ``config/``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from iris_harness.foundation.paths import config_path
from iris_harness.runtime.types import ChatResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InterceptHit:
    """The intercept that answered a turn — its spec + the result it produced.

    Lives here rather than on the runtime because it is the chain's *return type*,
    and the turn pipeline's host interface has to name it (``TurnHost``). It was
    ``bootstrap._InterceptHit``, which made the type of a published seam private to
    the module the seam points away from.
    """

    spec: InterceptSpec
    result: ChatResult


@dataclass(frozen=True)
class InterceptSpec:
    """One entry in the intercept chain."""

    name: str
    handler: str
    enabled: bool = True
    passes_channel: bool = False
    # None → the streaming path emits no trace event for this intercept (it
    # still yields the done result). A string is the trace label.
    trace_text: str | None = None
    trace_fields: tuple[str, ...] = field(default_factory=tuple)
    # ADR-0106: this intercept resolves a pending decision, so a bare "yes" is in
    # its vocabulary. Those are the handlers that can steal an answer meant for
    # someone else, so dispatch shields them when the session's open question
    # belongs to a different owner. Purely declarative — the flag says what the
    # intercept IS, the runtime decides what to do about it.
    resolves_confirmation: bool = False
    # Deterministic-path parity, decision B: this handler's answer repeats text someone
    # else wrote (email subjects, senders, statement text), which the harness did not
    # author and cannot vouch for. When the model-based output guard is on, it also runs
    # on this handler's answers; handlers that answer only in IRIS's own words skip it.
    guard_output: bool = False


# Hardcoded fallback — kept in sync with config/intercepts.yaml. Used only when
# the YAML is missing/unreadable so a broken config never bricks chat.
DEFAULT_INTERCEPTS: tuple[InterceptSpec, ...] = (
    InterceptSpec(
        "confirmation",
        "confirmations.handle_confirmation_turn",
        trace_text="confirmation resolved",
        trace_fields=("confirmation",),
        resolves_confirmation=True,
    ),
    InterceptSpec(
        "cleanup_selection",
        "plugin:filemanager",
        trace_text="cleanup selection archived",
        trace_fields=("cleanup_selection", "plan_id", "archived"),
        resolves_confirmation=True,
    ),
    InterceptSpec(
        "categorize_selection",
        "plugin:filemanager",
        trace_text="categorize selection filed",
        trace_fields=("categorize_selection", "plan_id", "filed"),
        resolves_confirmation=True,
    ),
    InterceptSpec(
        "organize_confirmation",
        "plugin:filemanager",
        trace_text="organize plan approved/rejected",
        trace_fields=("organize_confirmation", "plan_id"),
        resolves_confirmation=True,
    ),
    InterceptSpec(
        "filemanager_skill_confirmation",
        "plugin:filemanager",
        trace_text="filemanager action approved/rejected",
        trace_fields=("fm_skill_confirmation", "source_kind"),
        resolves_confirmation=True,
    ),
    InterceptSpec(
        "folder_files",
        "plugin:filemanager",
        trace_text="folder file count answered locally",
        trace_fields=("folder_files_intercept", "folder", "file_count"),
    ),
    InterceptSpec(
        "image_categorize_request",
        "plugin:filemanager",
        passes_channel=True,
        trace_text="image categories proposed",
        trace_fields=("categorize_intercept", "folder", "categories"),
    ),
    InterceptSpec(
        "organize_request",
        "plugin:filemanager",
        passes_channel=True,
        trace_text="organize plan proposed",
        trace_fields=("organize_intercept", "folder", "plan_id"),
    ),
    InterceptSpec(
        "cleanup_request",
        "plugin:filemanager",
        passes_channel=True,
        trace_text="cleanup groups proposed",
        trace_fields=("cleanup_intercept", "folder", "groups"),
    ),
    InterceptSpec(
        "move_request",
        "plugin:filemanager",
        trace_text="move-by-type proposed/moved",
        trace_fields=("move_intercept", "folder", "count"),
    ),
    InterceptSpec(
        "email_rebucket",
        "plugin:email_workflows",
        trace_text="email re-bucketed",
        trace_fields=("email_rebucket", "message_id"),
    ),
    InterceptSpec(
        "reminder_action",
        "plugin:calendar",
        trace_text="reminder answered",
        trace_fields=("reminder_action", "reminder_id"),
    ),
    InterceptSpec(
        "reminder_creation",
        "plugin:calendar",
        trace_text="reminder created",
        trace_fields=("reminder_id",),
    ),
    InterceptSpec(
        "meeting_creation",
        "plugin:calendar",
        trace_text="event created",
        trace_fields=("calendar_event_id",),
    ),
    InterceptSpec("time_date", "plugin:system", trace_text="deterministic time/date response"),
    InterceptSpec(
        "brief_request",
        "plugin:planner",
        trace_text="brief rendered on demand",
        trace_fields=("brief_skill",),
    ),
    InterceptSpec("brief_config", "plugin:planner", trace_text="brief configured"),
    InterceptSpec(
        "portfolio_request",
        "plugin:finance_workflows",
        trace_text="portfolio answered locally",
    ),
    InterceptSpec(
        "bill_paid",
        "plugin:finance_workflows",
        trace_text="bill marked paid",
        trace_fields=("bill_paid", "due_id"),
    ),
    InterceptSpec(
        "account_statement",
        "plugin:finance_workflows",
        trace_text="account statement answered from records",
        trace_fields=("account_statement", "institution", "emails_read"),
    ),
    InterceptSpec("dues_request", "plugin:finance_workflows", trace_text="dues answered locally"),
    InterceptSpec(
        "bill_amount_request",
        "plugin:finance_workflows",
        trace_text="bill amount answered locally",
        trace_fields=("bill_amount_intercept", "matched_issuers"),
    ),
    InterceptSpec(
        "statement_email_details",
        "plugin:finance_workflows",
        trace_text="statement details answered from inbox",
        trace_fields=("search_query",),
    ),
    InterceptSpec(
        "account_confirmation",
        "plugin:finance_workflows",
        trace_text="discovered accounts decided",
        trace_fields=("account_confirmation", "accounts"),
    ),
    InterceptSpec(
        "accounts_request",
        "plugin:finance_workflows",
        trace_text="accounts answered locally",
        trace_fields=("accounts", "filtered_to"),
    ),
    InterceptSpec("routine_management", "routines.handle_routine_management_turn"),
    InterceptSpec(
        "routine_authoring", "routines.handle_routine_authoring_turn", passes_channel=True
    ),
    InterceptSpec(
        "standing_instruction",
        "_handle_standing_instruction_turn",
        trace_text="standing instruction captured",
        trace_fields=("behavior_name",),
    ),
)


def _config_path() -> Path:
    override = os.environ.get("IRIS_INTERCEPTS_CONFIG")
    if override:
        return Path(override).expanduser()
    return config_path("intercepts.yaml")


def _spec_from_dict(raw: dict[str, object]) -> InterceptSpec:
    name = str(raw["name"])
    handler = str(raw.get("handler") or f"_handle_{name}_turn")
    trace_text = raw.get("trace_text")
    fields_raw = raw.get("trace_fields")
    fields = fields_raw if isinstance(fields_raw, list) else []
    return InterceptSpec(
        name=name,
        handler=handler,
        enabled=bool(raw.get("enabled", True)),
        passes_channel=bool(raw.get("passes_channel", False)),
        trace_text=str(trace_text) if trace_text is not None else None,
        trace_fields=tuple(str(f) for f in fields),
        resolves_confirmation=bool(raw.get("resolves_confirmation", False)),
        guard_output=bool(raw.get("guard_output", False)),
    )


def load_intercept_chain(config_path: Path | None = None) -> tuple[InterceptSpec, ...]:
    """Load the enabled intercept chain from YAML, or the hardcoded default.

    Falls back to ``DEFAULT_INTERCEPTS`` (and logs) on any error so a malformed
    or missing config can never disable chat. Disabled entries are dropped.
    """
    path = config_path or _config_path()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return DEFAULT_INTERCEPTS
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("intercepts.yaml unreadable (%s); using built-in default chain", exc)
        return DEFAULT_INTERCEPTS
    entries = (raw or {}).get("intercepts")
    if not isinstance(entries, list) or not entries:
        logger.warning("intercepts.yaml has no 'intercepts' list; using built-in default chain")
        return DEFAULT_INTERCEPTS
    try:
        specs = tuple(_spec_from_dict(e) for e in entries if isinstance(e, dict))
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("intercepts.yaml malformed (%s); using built-in default chain", exc)
        return DEFAULT_INTERCEPTS
    return tuple(s for s in specs if s.enabled)


def resolve_runtime_handler(host: object, handler: str) -> Any:
    """The callable a core ``handler:`` row names on ``host``, or ``None`` if it is missing.

    A plain name is a method on the runtime (``_handle_standing_instruction_turn``). A
    dotted name walks attributes, so a handler that lives on a collaborator the runtime
    holds is named where it lives (``routines.handle_routine_authoring_turn``) rather than
    through a delegating method kept on the runtime for the YAML's sake (OSS plan M5.7
    track C). ``plugin:`` rows never reach this; the chain resolves them first.
    """
    target: Any = host
    for part in handler.split("."):
        target = getattr(target, part, None)
        if target is None:
            return None
    return target
