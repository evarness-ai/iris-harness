"""Routine authoring — the in-chat routine conversation, a core harness capability.

The routine authoring / management / refinement / capability-swap / clarification
handlers were extracted from ``IrisRuntime`` as a mixin in Phase 2 (byte-identical
bodies, inherited). OSS plan M5.7 track C slice 10 turned the mixin into a collaborator:
``RoutineAuthoring(host)``, reached as ``runtime.routines``. Routines are core by the
owner's ruling, so this is a carve, not an extraction.

The per-session conversation state — pending refinement picks, pending capability
swaps, pending capability clarifications, the undo snapshots — moved here from the
runtime's fields; only this module read or wrote it. ``has_active_routine_conversation``
moved with it, and is how the planner plugin's brief intercept steps aside
(``HarnessServices.conversation_in_flight``).

:class:`RoutineHost` declares the six runtime members read, so mypy checks the runtime
still supplies them. The host is read **at call time**, not captured. The intercept chain
reaches the two turn handlers by the dotted handler names in ``config/intercepts.yaml``
(``routines.handle_routine_authoring_turn``). Tests stub
``runtime.routines.routine_authoring_llm_caller`` to stay independent of a local LLM.

The routine-only module-level helpers + constants live here too; three symbols still
used by non-routine bootstrap code are imported there.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, cast

from iris_harness.agent.intent_router import IntentResult
from iris_harness.agent.response_curator import CuratedResponse
from iris_harness.foundation.process_state import track_globals
from iris_harness.runtime.types import ChatResult
from iris_harness.services.routines import (
    LLMCaller,
    RoutineApprovalRequest,
    RoutineApprovalRequestStatus,
    RoutineApprovalStatus,
    RoutineSpec,
    brief_slot_callbacks,
    brief_slot_capabilities,
    brief_tool_slot_keys,
    detect_capability_swap_intent,
    extract_briefing_sections,
    extract_routine_refinements,
    format_brief_sections,
    format_tool_arg_prompt_line,
    normalize_briefing_sections,
    parse_routine_authoring,
    parse_tool_arg_reply,
    resolve_routine_pick,
)
from iris_harness.tools.skills.models import ToolArg
from iris_harness.tools.skills.registry import SkillRegistry

if TYPE_CHECKING:
    from iris_harness.llm.tier_router import TierRouter
    from iris_harness.runtime.session_memory import SessionMemory
    from iris_harness.runtime.turn_capture import TurnCapture
    from iris_harness.services.routines import RoutineStore

logger = logging.getLogger(__name__)


_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


_SEMANTIC_ROUTER_ENABLED_ENV = "IRIS_SKILL_ROUTER_SEMANTIC"


_SEMANTIC_ROUTER_THRESHOLD_ENV = "IRIS_SKILL_ROUTER_THRESHOLD"


_semantic_router_singleton: Any | None = None


_semantic_router_init_failed: bool = False


def _get_semantic_router() -> Any | None:
    """Return a process-wide ``SemanticSkillRouter`` or ``None`` if unavailable.

    Init is lazy: the underlying ChromaDB ONNX model only loads on first
    embed call. Set ``IRIS_SKILL_ROUTER_SEMANTIC=0`` to force the legacy
    keyword scorer (useful for diagnosing routing regressions).

    A router whose embedder cannot embed (no model, and fetching one is not
    allowed) is unavailable too: every caller's no-router path is its
    no-embeddings path (the keyword scorer), and a router that scores
    everything 0.0 would instead filter every skill out. Checked per call,
    because the model can appear on disk after the router is built.
    """
    global _semantic_router_singleton, _semantic_router_init_failed
    if _semantic_router_init_failed:
        return None
    if _semantic_router_singleton is not None:
        return _semantic_router_singleton if _semantic_router_singleton.available else None
    if os.getenv(_SEMANTIC_ROUTER_ENABLED_ENV, "1").strip().lower() in _FALSE_ENV_VALUES:
        _semantic_router_init_failed = True
        return None
    try:
        from iris_harness.kernel.governance.evaluator.embeddings import (
            DefaultEmbedder,
        )
        from iris_harness.tools.skills.semantic_router import (
            DEFAULT_SIMILARITY_THRESHOLD,
            SemanticSkillRouter,
        )

        threshold_raw = os.getenv(_SEMANTIC_ROUTER_THRESHOLD_ENV, "").strip()
        try:
            threshold = float(threshold_raw) if threshold_raw else DEFAULT_SIMILARITY_THRESHOLD
        except ValueError:
            threshold = DEFAULT_SIMILARITY_THRESHOLD
        _semantic_router_singleton = SemanticSkillRouter(
            embedder=DefaultEmbedder(), threshold=threshold
        )
    except Exception:
        logger.exception("semantic skill router init failed; using keyword scorer")
        _semantic_router_init_failed = True
        return None
    return _semantic_router_singleton if _semantic_router_singleton.available else None


_BRIEF_TEMPLATE_ALIASES: dict[str, str] = {
    # Read-side normalization so legacy persisted routines whose ``template``
    # field uses the old underscore-style identifier still resolve to the
    # canonical brief manifest name. New routines persist the manifest name
    # directly.
    "morning_briefing": "morning-briefing",
    "daily_repo_brief": "daily-repo-brief",
}


def _normalize_template_name(template: str) -> str:
    return _BRIEF_TEMPLATE_ALIASES.get(template, template)


def _capability_package_for_template(
    template: str,
    skill_registry: SkillRegistry | None,
) -> Any | None:
    """Return the skill package a routine is bound to, regardless of kind.

    Routines bind to any addressable capability — a brief skill, a
    regular tool skill, or any future kind. The template field stores
    the bound skill's manifest name; legacy underscore-style values
    (``morning_briefing``) are normalized via
    ``_BRIEF_TEMPLATE_ALIASES``.
    """

    if not template or skill_registry is None:
        return None
    name = _normalize_template_name(template)
    try:
        packages = skill_registry.list_packages(only_loadable=True)
    except Exception:
        logger.debug("capability lookup failed for template %s", template, exc_info=True)
        return None
    for package in packages:
        if package.manifest.name == name:
            return package
    return None


def _routine_template_label(
    template: str,
    *,
    brief_package: Any | None = None,
) -> str:
    if brief_package is not None and brief_package.manifest.brief is not None:
        return cast(str, brief_package.manifest.brief.subject)
    if template == "skill_brief":
        return "Skill brief"
    return _normalize_template_name(template).replace("-", " ").replace("_", " ").title()


def _routine_template_summary(
    template: str,
    *,
    brief_package: Any | None = None,
) -> str:
    if brief_package is not None:
        return cast(str, brief_package.manifest.description)
    if template == "skill_brief":
        return "renders a brief skill's layout and dispatches it to the channel"
    return "runs the selected routine template"


def _apply_routine_refinements(
    routine: RoutineSpec,
    refinements: dict[str, object],
) -> RoutineSpec:
    updates: dict[str, object] = {}
    metadata = dict(routine.metadata)
    delivery_channel = refinements.get("delivery_channel")
    if isinstance(delivery_channel, str) and delivery_channel:
        updates["delivery_channel"] = delivery_channel
    for key in ("content_lines_per_item", "content_unit", "content_style"):
        if key in refinements:
            metadata[key] = refinements[key]
    # Phase B slice 3: capability swap. ``template`` is the manifest
    # name of the new skill; the caller has already confirmed with the
    # user before passing this in.
    new_template = refinements.get("template")
    if isinstance(new_template, str) and new_template and new_template != routine.template:
        updates["template"] = new_template
        metadata["bound_skill"] = new_template
        # Tool callbacks / source preferences belong to the old skill;
        # drop them so the next scheduler tick re-derives from the new
        # one (or the next refinement clarification re-asks).
        metadata.pop("tool_callbacks", None)
        metadata.pop("bound_tool", None)
        metadata.pop("pending_tool_args", None)
        metadata.pop("resolved_tool_args", None)
        updates["source_preferences"] = ()
    if metadata != routine.metadata:
        updates["metadata"] = metadata
    if not updates:
        return routine
    updates["updated_at"] = datetime.now(UTC)
    return routine.model_copy(update=updates)


def _content_style_line(metadata: dict[str, object]) -> str:
    style = metadata.get("content_style")
    if isinstance(style, str) and style:
        return f"- Content style: {style}\n"
    return ""


def _routine_refinement_summary(refinements: dict[str, object]) -> str:
    changes: list[str] = []
    delivery_channel = refinements.get("delivery_channel")
    if isinstance(delivery_channel, str) and delivery_channel:
        changes.append(f"delivery to {delivery_channel}")
    content_style = refinements.get("content_style")
    if isinstance(content_style, str) and content_style:
        changes.append(f"content style to {content_style}")
    template = refinements.get("template")
    if isinstance(template, str) and template:
        changes.append(f"bound skill to `{template}`")
    return " and ".join(changes) if changes else "the routine details"


_ROUTINE_ID_RE = re.compile(r"\broutine-[a-z0-9][a-z0-9-]*\b", re.IGNORECASE)


# A permissive "is this turn plausibly about a routine/brief?" check. Used only
# to KEEP the routine-refinement path for a bare tone adjective ("detailed",
# "formal") that would otherwise be too weak to route on its own: such a word
# collides with ordinary language (a self-harm probe reached the routine
# intercept via "detailed"), so a tone-only refinement must not swallow a turn
# unless there's real routine context. Over-matching here is safe — it only
# widens when the helpful "no draft in progress" reply may show.
_ROUTINE_REFERENCE_RE = re.compile(
    r"\b(routine|routines|briefing|brief|digest|summary|schedule|scheduled|"
    r"reminder|daily|weekly|every\s+day|each\s+day|deliver|delivery)\b",
    re.IGNORECASE,
)


_UNDO_RE = re.compile(r"^\s*(undo|revert)\b", re.IGNORECASE)


_SAMPLE_RE = re.compile(
    r"^\s*(sample|preview|show\s+(me\s+)?(a|the)\s+(sample|preview))\b", re.IGNORECASE
)


_REFINEMENT_UNDO_WINDOW = timedelta(minutes=5)


def _extract_routine_id(text: str) -> str:
    match = _ROUTINE_ID_RE.search(text)
    return match.group(0) if match else ""


def _routine_status_filter(lowered: str) -> RoutineApprovalStatus | None:
    for status in RoutineApprovalStatus:
        if re.search(rf"\b{re.escape(status.value)}\b", lowered):
            return status
    return None


def _is_routine_list_request(lowered: str) -> bool:
    has_list_verb = re.search(r"\b(list|show|display|what|which)\b", lowered)
    return bool(has_list_verb and re.search(r"\broutines?\b", lowered)) or "my routines" in lowered


def _is_all_routines_target(lowered: str) -> bool:
    return bool(re.search(r"\b(all|every|each)\s+(?:my\s+)?routines?\b", lowered))


def _routine_management_status(lowered: str) -> RoutineApprovalStatus | None:
    status_commands = {
        "approve": RoutineApprovalStatus.APPROVED,
        "approved": RoutineApprovalStatus.APPROVED,
        "scheduled": RoutineApprovalStatus.SCHEDULED,
        "pause": RoutineApprovalStatus.PAUSED,
        "paused": RoutineApprovalStatus.PAUSED,
        "resume": RoutineApprovalStatus.SCHEDULED,
        "retire": RoutineApprovalStatus.RETIRED,
        "retired": RoutineApprovalStatus.RETIRED,
        "draft": RoutineApprovalStatus.DRAFT,
    }
    for word, status in status_commands.items():
        if re.search(rf"\b{word}\b", lowered):
            return status
    return None


def _extract_routine_schedule_update(text: str) -> str:
    match = re.search(
        r"\bschedule\s+(?:to|as|=)?\s*"
        r"(?P<schedule>daily:\d{1,2}:\d{2}|interval:\d+|cron:[^,.;]+)",
        text,
        re.IGNORECASE,
    )
    return match.group("schedule").strip() if match else ""


def _format_routine_inventory(routines: list[RoutineSpec], *, title: str = "Routines") -> str:
    if not routines:
        return f"No {title.lower()} found."
    lines = [f"{title}: {len(routines)}"]
    for index, routine in enumerate(routines, start=1):
        lines.append(f"{index}. ID: {routine.id}")
        lines.append(f"   Title: {routine.title}")
        lines.append(f"   Schedule: {routine.schedule}")
        lines.append(f"   Status: {routine.approval_status}")
        lines.append(f"   Type: {_routine_template_label(routine.template)}")
        lines.append(f"   Delivery: {routine.delivery_channel}")
    return "\n".join(lines)


class RoutineHost(Protocol):
    """The six runtime members routine authoring reaches.

    ``capture`` is the per-turn learning capture, reached for ``record_signal`` (OSS plan
    M5.7 track C slice 12); ``sessions`` is session memory, reached for ``record_turn``
    (slice 16).
    """

    routine_store: RoutineStore
    skill_registry: SkillRegistry
    tier_router: TierRouter

    capture: TurnCapture
    sessions: SessionMemory

    def preview_routine(self, routine_id: str) -> str | None: ...


class RoutineAuthoring:
    """Routine authoring, management, refinement, capability swaps and clarifications for
    one runtime, with the per-session conversation state they keep. See the module
    docstring."""

    def __init__(self, host: RoutineHost) -> None:
        self._host = host
        # Phase B: per-session pending refinement when the post-approval
        # refinement target was ambiguous. Cleared when the user picks,
        # cancels, or starts a new routine. In-memory only — refinements
        # are conversational and don't need to survive runtime restarts.
        self._pending_refinement_picks: dict[str, dict[str, Any]] = {}
        # Phase B slice 3: per-session pending capability swap awaiting
        # explicit confirmation. Capability changes are large so we never
        # apply them silently — same in-memory lifetime as the pick state.
        self._pending_capability_swaps: dict[str, dict[str, Any]] = {}
        # Phase 3 follow-up: per-session memory of a routine request whose
        # capability couldn't be resolved ("set up a routine every morning at 7"
        # has a schedule but no matchable skill). The next turn naming the
        # capability ("the morning briefing") is combined with the remembered
        # request and re-parsed. Without this the clarification was stateless —
        # the reply resolved nothing and the user had to restate everything.
        self._pending_capability_clarifications: dict[str, dict[str, Any]] = {}
        # Phase B slice 4: snapshot of the pre-refinement routine spec
        # captured immediately before _handle_post_approval_refinement
        # saves the updated spec. ``undo`` within the 5-minute window
        # restores it. One-step undo only — a fresh refinement replaces
        # the snapshot.
        self._recent_refinement_snapshots: dict[str, dict[str, Any]] = {}

    def handle_routine_management_turn(
        self,
        message: str,
        *,
        session_id: str,
        span: Any = None,
    ) -> ChatResult | None:
        text = " ".join(message.strip().split())
        lowered = text.lower()
        if not text or not re.search(r"\broutines?\b", lowered):
            return None

        routine_id = _extract_routine_id(text)
        wants_delete = bool(re.search(r"\b(delete|remove|rm)\b", lowered))
        # NB: bare "set" must not swallow "set up ..." — that's creation
        # phrasing and belongs to the authoring flow (Phase 3 multiturn
        # scenario caught "set up a routine every morning at 7" routing to
        # the update path and dead-ending in missing_update_id).
        wants_update = bool(re.search(r"\b(update|change|set(?!\s+up)|edit)\b", lowered))
        refinements = extract_routine_refinements(text)
        schedule_update = _extract_routine_schedule_update(text)
        status_update = _routine_management_status(lowered)
        all_routines_target = _is_all_routines_target(lowered)
        wants_status_change = bool(
            re.search(r"\b(approve|approved|pause|paused|resume|retire|retired|draft)\b", lowered)
            or re.search(r"^\s*schedule\s+(?:the\s+)?routine\b", lowered)
            or (all_routines_target and re.search(r"\bschedule\b", lowered))
        )
        if (
            status_update is None
            and routine_id
            and not schedule_update
            and re.search(r"\bschedule\b", lowered)
        ):
            status_update = RoutineApprovalStatus.SCHEDULED
        if (
            status_update is None
            and all_routines_target
            and not schedule_update
            and re.search(r"\bschedule\b", lowered)
        ):
            status_update = RoutineApprovalStatus.SCHEDULED

        if _is_routine_list_request(lowered):
            status_filter = _routine_status_filter(lowered)
            routines = (
                self._host.routine_store.list_by_status(status_filter)
                if status_filter is not None
                else self._host.routine_store.list_all()
            )
            title = (
                f"{status_filter.value.title()} routines"
                if status_filter is not None
                else "Routines"
            )
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response=_format_routine_inventory(routines, title=title),
                metadata={
                    "routine_action": "list",
                    "routine_count": len(routines),
                    "routine_ids": [routine.id for routine in routines],
                    "routine_status_filter": status_filter.value if status_filter else "",
                },
                span=span,
            )

        if not routine_id and all_routines_target and (wants_update or wants_status_change):
            routines = self._host.routine_store.list_all()
            if not routines:
                return self._routine_chat_result(
                    message=message,
                    session_id=session_id,
                    response="No routines found to update.",
                    metadata={"routine_action": "updated_many", "routine_count": 0},
                    span=span,
                )
            bulk_changes: list[str] = []
            if refinements:
                bulk_changes.append(_routine_refinement_summary(refinements))
            if schedule_update:
                bulk_changes.append(f"schedule to {schedule_update}")
            if status_update is not None:
                bulk_changes.append(f"status to {status_update.value}")
            if not bulk_changes:
                return self._routine_chat_result(
                    message=message,
                    session_id=session_id,
                    response=(
                        "I found all routines, but I did not find a supported bulk update. "
                        "I can update delivery, content style, explicit schedule values, or status."
                    ),
                    metadata={
                        "routine_action": "unsupported_update",
                        "routine_count": len(routines),
                    },
                    span=span,
                    has_errors=True,
                    error_summary="unsupported routine bulk update",
                )
            saved_routines: list[RoutineSpec] = []
            for routine in routines:
                updated = routine
                if refinements:
                    updated = _apply_routine_refinements(updated, refinements)
                if schedule_update:
                    updated = updated.model_copy(
                        update={"schedule": schedule_update, "updated_at": datetime.now(UTC)}
                    )
                if status_update is not None:
                    updated = updated.with_status(status_update)
                saved_routines.append(self._host.routine_store.save(updated))
            response = (
                f"Updated {' and '.join(bulk_changes)} for {len(saved_routines)} routines.\n\n"
                f"{_format_routine_inventory(saved_routines)}"
            )
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response=response,
                metadata={
                    "routine_action": "updated_many",
                    "routine_count": len(saved_routines),
                    "routine_ids": [routine.id for routine in saved_routines],
                    "delivery_channel": refinements.get("delivery_channel", ""),
                    "content_style": refinements.get("content_style", ""),
                },
                span=span,
            )

        if not routine_id and (wants_update or wants_status_change):
            routines = self._host.routine_store.list_all()
            response = "I need the full routine ID to update a routine."
            if routines:
                response += "\n\n" + _format_routine_inventory(routines)
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response=response,
                metadata={
                    "routine_action": "missing_update_id",
                    "routine_count": len(routines),
                    "routine_ids": [routine.id for routine in routines],
                },
                span=span,
                has_errors=True,
                error_summary="routine update missing id",
            )

        if wants_delete:
            if not routine_id:
                routines = self._host.routine_store.list_all()
                response = "I need the full routine ID to delete a routine."
                if routines:
                    response += "\n\n" + _format_routine_inventory(routines)
                return self._routine_chat_result(
                    message=message,
                    session_id=session_id,
                    response=response,
                    metadata={
                        "routine_action": "missing_delete_id",
                        "routine_count": len(routines),
                        "routine_ids": [routine.id for routine in routines],
                    },
                    span=span,
                    has_errors=True,
                    error_summary="routine delete missing id",
                )
            spec = self._host.routine_store.load(routine_id)
            if spec is None:
                return self._routine_chat_result(
                    message=message,
                    session_id=session_id,
                    response=f"I could not find routine `{routine_id}`. Ask `list routines` for full IDs.",
                    metadata={"routine_action": "not_found", "routine_id": routine_id},
                    span=span,
                    has_errors=True,
                    error_summary="routine not found",
                )
            deleted = self._host.routine_store.delete(routine_id)
            for request in self._host.routine_store.list_approval_requests(
                status=RoutineApprovalRequestStatus.PENDING
            ):
                if request.routine_id == routine_id:
                    self._host.routine_store.resolve_approval_request(
                        request.id,
                        RoutineApprovalRequestStatus.CANCELLED,
                    )
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response=f"Deleted routine `{spec.title}`.\n- ID: {routine_id}",
                metadata={
                    "routine_action": "deleted",
                    "routine_id": routine_id,
                    "deleted": deleted,
                },
                span=span,
            )

        if routine_id and (
            wants_update or status_update is not None or refinements or schedule_update
        ):
            spec = self._host.routine_store.load(routine_id)
            if spec is None:
                return self._routine_chat_result(
                    message=message,
                    session_id=session_id,
                    response=f"I could not find routine `{routine_id}`. Ask `list routines` for full IDs.",
                    metadata={"routine_action": "not_found", "routine_id": routine_id},
                    span=span,
                    has_errors=True,
                    error_summary="routine not found",
                )
            updated = spec
            changes: list[str] = []
            if refinements:
                updated = _apply_routine_refinements(updated, refinements)
                changes.append(_routine_refinement_summary(refinements))
            if schedule_update:
                updated = updated.model_copy(
                    update={"schedule": schedule_update, "updated_at": datetime.now(UTC)}
                )
                changes.append(f"schedule to {schedule_update}")
            if status_update is not None:
                updated = updated.with_status(status_update)
                changes.append(f"status to {status_update.value}")
            if not changes:
                return self._routine_chat_result(
                    message=message,
                    session_id=session_id,
                    response=(
                        f"I found routine `{routine_id}`, but I did not find a supported update. "
                        "I can update delivery, content style, explicit schedule values, or status."
                    ),
                    metadata={"routine_action": "unsupported_update", "routine_id": routine_id},
                    span=span,
                    has_errors=True,
                    error_summary="unsupported routine update",
                )
            saved = self._host.routine_store.save(updated)
            response = (
                f"Updated {' and '.join(changes)} for `{saved.title}`.\n"
                f"- ID: {saved.id}\n"
                f"- Schedule: {saved.schedule}\n"
                f"- Status: {saved.approval_status}\n"
                f"- Delivery: {saved.delivery_channel}\n"
                f"{_content_style_line(saved.metadata)}"
            ).rstrip()
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response=response,
                metadata={
                    "routine_action": "updated",
                    "routine_id": saved.id,
                    "routine_status": str(saved.approval_status),
                    "delivery_channel": saved.delivery_channel,
                    "content_style": saved.metadata.get("content_style", ""),
                },
                span=span,
            )

        return None

    def _handle_routine_sample_request(
        self,
        message: str,
        *,
        session_id: str,
        span: Any = None,
    ) -> ChatResult | None:
        """Render the pending draft routine's real output for verification.

        Returns ``None`` (so the caller continues normal parsing) when there is
        no pending draft to sample. Never persists or changes status — the draft
        stays a draft so previews can't leak into the routine store.
        """
        pending_request = self._host.routine_store.get_pending_approval_request(session_id)
        if pending_request is None:
            return None
        routine = self._host.routine_store.load(pending_request.routine_id)
        if routine is None:
            return None
        try:
            body = self._host.preview_routine(routine.id)
        except ValueError as exc:
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response=(
                    f"I can't preview `{routine.title}` directly ({exc}). "
                    "Reply `approve` to schedule it, or keep refining."
                ),
                metadata={"routine_action": "sample_unavailable", "routine_id": routine.id},
                span=span,
            )
        response = (
            f"Here's a sample of `{routine.title}` (delivery: {routine.delivery_channel}):\n\n"
            f"{body}\n\n"
            "Reply `approve` to schedule it, `cancel` to retire the draft, or keep refining."
        )
        return self._routine_chat_result(
            message=message,
            session_id=session_id,
            response=response,
            metadata={"routine_action": "sample", "routine_id": routine.id},
            span=span,
        )

    def handle_routine_authoring_turn(
        self,
        message: str,
        *,
        session_id: str,
        channel: str = "console",
        span: Any = None,
    ) -> ChatResult | None:
        # Phase B slice 4: an explicit ``undo`` (within 5 min of the
        # last post-approval refinement) restores the prior spec.
        # Handled ahead of pick / swap / refinement detection so a
        # pending mid-flow state doesn't swallow the undo intent.
        # Only in a session that refined a routine: anywhere else "undo that"
        # belongs to whatever the session did last (2026-09-22: it answered "Nothing
        # to undo" to the owner undoing an email trash, which never reached the tool).
        if _UNDO_RE.match(message) and session_id in self._recent_refinement_snapshots:
            return self._handle_refinement_undo(message, session_id=session_id, span=span)

        # Phase 4: a bare ``sample`` / ``preview`` while a draft is pending renders
        # the routine's real output inline for verification — without persisting or
        # changing status. Falls through to normal parsing when nothing is pending.
        if _SAMPLE_RE.match(message):
            sample_result = self._handle_routine_sample_request(
                message, session_id=session_id, span=span
            )
            if sample_result is not None:
                return sample_result

        candidate_packages = self._loaded_capability_packages()
        parsed = parse_routine_authoring(
            message,
            candidate_packages=candidate_packages,
            router=_get_semantic_router(),
            llm_caller=self.routine_authoring_llm_caller(),
            origin_channel=channel,
        )
        pending_request = self._host.routine_store.get_pending_approval_request(session_id)
        pending_clarification = self._pending_routine_clarification(session_id)
        refinements = extract_routine_refinements(message)
        # Phase B: when we last asked "which routine?", consume this
        # turn as the pick reply unless the user clearly moved on
        # (a fresh draft / clarification / approval).
        pending_pick = self._pending_refinement_picks.get(session_id)
        if pending_pick is not None:
            pick_result = self._handle_refinement_pick_reply(
                message,
                session_id=session_id,
                pending_pick=pending_pick,
                parsed_action=parsed.action,
                span=span,
            )
            if pick_result is not None:
                return pick_result
        # Phase B slice 3: when we last proposed a capability swap,
        # consume this turn as the confirmation reply.
        pending_swap = self._pending_capability_swaps.get(session_id)
        if pending_swap is not None:
            swap_result = self._handle_capability_swap_confirmation(
                message,
                session_id=session_id,
                pending=pending_swap,
                parsed_action=parsed.action,
                span=span,
            )
            if swap_result is not None:
                return swap_result
        # Phase 3 follow-up: if we previously asked "which capability?" and
        # this turn didn't itself resolve one, combine the remembered request
        # with this reply and re-parse. Skipped when the user clearly moved
        # on (a fresh approve/cancel).
        pending_capclar = self._pending_capability_clarifications.get(session_id)
        if pending_capclar is not None and parsed.action not in {"approve", "cancel"}:
            if not (parsed.action == "draft" and parsed.template):
                combined = f"{pending_capclar['original_message']} {message}".strip()
                recombined = parse_routine_authoring(
                    combined,
                    candidate_packages=candidate_packages,
                    router=_get_semantic_router(),
                    llm_caller=self.routine_authoring_llm_caller(),
                    origin_channel=channel,
                )
                if recombined.action == "draft" and recombined.template:
                    self._pending_capability_clarifications.pop(session_id, None)
                    parsed = recombined
                    message = combined
        if parsed.action == "none":
            if pending_clarification is not None:
                return self._handle_routine_clarification_reply(
                    message,
                    session_id=session_id,
                    pending=pending_clarification,
                    span=span,
                )
            if pending_request is not None and refinements:
                return self._handle_pending_routine_refinement(
                    message,
                    session_id=session_id,
                    pending_request=pending_request,
                    refinements=refinements,
                    span=span,
                )
            if refinements:
                # Phase B: a refinement message ("deliver to telegram",
                # "3 lines per news") arriving after the routine is
                # already SCHEDULED updates that routine in place
                # rather than complain about a missing draft.
                recent_approved = self._host.routine_store.list_recent_approved_by_session(
                    session_id
                )
                # Precision guard: a bare tone adjective ("detailed", "formal")
                # is the one refinement signal weak enough to collide with
                # ordinary language. With NO routine context — no routine this
                # session, no routine/brief reference in the message — it is not
                # a routine turn; fall through to intent classification rather
                # than swallow it with a "no draft" reply. (A self-harm probe
                # reached this intercept via the word "detailed".)
                if (
                    set(refinements) <= {"tone"}
                    and not recent_approved
                    and not _ROUTINE_REFERENCE_RE.search(message)
                ):
                    return None
                if len(recent_approved) == 1:
                    return self._handle_post_approval_refinement(
                        message,
                        session_id=session_id,
                        target=recent_approved[0],
                        refinements=refinements,
                        span=span,
                    )
                if len(recent_approved) > 1:
                    return self._handle_post_approval_refinement_ambiguous(
                        message,
                        session_id=session_id,
                        candidates=recent_approved,
                        refinements=refinements,
                        span=span,
                    )
                return self._routine_chat_result(
                    message=message,
                    session_id=session_id,
                    response=(
                        "I can update delivery or content style, but I do not have a "
                        "routine draft in progress for this chat. Start by asking me to "
                        "create the routine, then send the refinement."
                    ),
                    metadata={"routine_action": "missing_pending_routine_refinement"},
                    span=span,
                )
            # Phase B slice 3: no value-refinement extracted, but the
            # message may still ask to rebind a routine to a different
            # skill ("switch to morning briefing"). Only propose when
            # there's exactly one SCHEDULED routine in the session;
            # multi-routine swap goes through the same disambiguation
            # path as value refinements when we wire it later.
            swap_target = detect_capability_swap_intent(
                message,
                candidate_packages,
                _get_semantic_router(),
            )
            if swap_target is not None:
                recent_approved = self._host.routine_store.list_recent_approved_by_session(
                    session_id
                )
                if (
                    len(recent_approved) == 1
                    and swap_target.manifest.name != recent_approved[0].template
                ):
                    return self._handle_capability_swap_proposal(
                        message,
                        session_id=session_id,
                        target=recent_approved[0],
                        swap_to=swap_target,
                        span=span,
                    )
            return None
        if parsed.action == "approve":
            if pending_request is None:
                if pending_clarification is not None:
                    pending_brief = self._brief_for_routine(pending_clarification)
                    slot_labels = (
                        ", ".join(brief_tool_slot_keys(pending_brief))
                        if pending_brief is not None
                        else "the sections you want"
                    )
                    response = (
                        "I still need what the briefing should include before I can "
                        f"ask for approval. Reply with one or more: {slot_labels}; or `all`."
                    )
                    return self._routine_chat_result(
                        message=message,
                        session_id=session_id,
                        response=response,
                        metadata={
                            "routine_action": "clarify",
                            "routine_id": pending_clarification.id,
                            "missing_slots": list(
                                pending_clarification.metadata.get("missing_slots", [])
                            ),
                        },
                        span=span,
                    )
                return None
            spec = self._host.routine_store.load(pending_request.routine_id)
            if spec is None:
                self._host.routine_store.resolve_approval_request(
                    pending_request.id,
                    RoutineApprovalRequestStatus.CANCELLED,
                )
                return self._routine_chat_result(
                    message=message,
                    session_id=session_id,
                    response="I could not find the routine draft to approve. Please draft it again.",
                    metadata={
                        "routine_action": "missing_pending_draft",
                        "approval_request_id": pending_request.id,
                    },
                    span=span,
                    has_errors=True,
                    error_summary="pending routine draft missing",
                )
            updated = self._host.routine_store.save(
                spec.with_status(RoutineApprovalStatus.SCHEDULED)
            )
            self._host.routine_store.resolve_approval_request(
                pending_request.id,
                RoutineApprovalRequestStatus.APPROVED,
            )
            updated_brief = self._brief_for_routine(updated)
            sections = normalize_briefing_sections(
                updated.metadata.get("briefing_sections"),
                updated_brief,
            )
            section_text = (
                f"- Briefing sections: {format_brief_sections(sections)}\n" if sections else ""
            )
            tool_callbacks = updated.metadata.get("tool_callbacks")
            callback_items = (
                [str(item) for item in tool_callbacks] if isinstance(tool_callbacks, list) else []
            )
            callback_text = (
                f"- Tool callbacks: {', '.join(callback_items)}\n" if callback_items else ""
            )
            content_style_text = _content_style_line(updated.metadata)
            response = (
                f"Scheduled routine `{updated.title}`.\n"
                f"- ID: {updated.id}\n"
                f"- Schedule: {updated.schedule}\n"
                f"- Routine type: {_routine_template_label(updated.template, brief_package=updated_brief)}\n"
                f"{section_text}"
                f"{content_style_text}"
                f"{callback_text}"
                f"- Delivery: {updated.delivery_channel}\n"
                f"- Approval request: {pending_request.id}"
            )
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response=response,
                metadata={
                    "routine_action": "approved",
                    "routine_id": updated.id,
                    "approval_request_id": pending_request.id,
                    "briefing_sections": list(sections),
                    "tool_callbacks": callback_items,
                    "delivery_channel": updated.delivery_channel,
                    "content_style": updated.metadata.get("content_style", ""),
                },
                span=span,
            )
        if parsed.action == "cancel":
            if pending_request is None:
                if pending_clarification is not None:
                    updated = self._host.routine_store.save(
                        pending_clarification.with_status(RoutineApprovalStatus.RETIRED)
                    )
                    return self._routine_chat_result(
                        message=message,
                        session_id=session_id,
                        response=f"Retired routine clarification `{updated.title}`.",
                        metadata={"routine_action": "cancelled", "routine_id": updated.id},
                        span=span,
                    )
                return None
            spec = self._host.routine_store.load(pending_request.routine_id)
            self._host.routine_store.resolve_approval_request(
                pending_request.id,
                RoutineApprovalRequestStatus.CANCELLED,
            )
            if spec is None:
                return self._routine_chat_result(
                    message=message,
                    session_id=session_id,
                    response="Discarded the pending routine draft.",
                    metadata={
                        "routine_action": "cancelled_missing_draft",
                        "approval_request_id": pending_request.id,
                    },
                    span=span,
                )
            updated = self._host.routine_store.save(spec.with_status(RoutineApprovalStatus.RETIRED))
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response=f"Retired draft routine `{updated.title}`.",
                metadata={
                    "routine_action": "cancelled",
                    "routine_id": updated.id,
                    "approval_request_id": pending_request.id,
                },
                span=span,
            )
        if parsed.action == "clarify":
            if parsed.template and parsed.schedule:
                self._retire_pending_routine_clarifications(session_id)
                clarification = self._host.routine_store.save(
                    parsed.to_draft_spec()
                    .with_status(RoutineApprovalStatus.CLARIFY)
                    .model_copy(
                        update={
                            "metadata": {
                                **parsed.metadata,
                                "session_id": session_id,
                                "missing_slots": list(parsed.missing_slots),
                            }
                        }
                    )
                )
                brief = self._brief_for_routine(clarification)
                response = (
                    f"I can set up `{_routine_template_label(clarification.template, brief_package=brief)}` "
                    "for that schedule, but I need more detail.\n"
                    f"- ID: {clarification.id}\n"
                    f"- Schedule: {clarification.schedule}\n"
                    f"- What it does: {_routine_template_summary(clarification.template, brief_package=brief)}\n"
                    f"- Delivery: {clarification.delivery_channel}\n"
                    f"{_content_style_line(clarification.metadata)}"
                    f"- Next step: {parsed.reason}"
                )
                return self._routine_chat_result(
                    message=message,
                    session_id=session_id,
                    response=response,
                    metadata={
                        "routine_action": "clarify",
                        "routine_id": clarification.id,
                        "missing_slots": list(parsed.missing_slots),
                    },
                    span=span,
                )
            # Remember the request so the next turn naming a capability
            # resolves it (Phase 3 follow-up: stateful capability clarify).
            self._pending_capability_clarifications[session_id] = {
                "original_message": message,
            }
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response=(
                    f"I can draft that routine, but {parsed.reason} "
                    "Just reply with the capability name and I'll combine it with "
                    "what you already told me."
                ),
                metadata={
                    "routine_action": "clarify",
                    "missing_slots": list(parsed.missing_slots),
                    "awaiting": "capability",
                },
                span=span,
            )
        if parsed.action != "draft":
            return None

        self._retire_pending_routine_clarifications(session_id)
        # Stamp session_id on the draft so Phase B's
        # list_recent_approved_by_session can find it post-approval.
        draft_spec = parsed.to_draft_spec()
        # Update-don't-accumulate (roadmap "next slice"): if this session already has an
        # in-progress draft for the same template, reuse its id so re-stating the routine
        # UPDATES that draft instead of piling up near-duplicates.
        dup = self._host.routine_store.find_session_duplicate(
            session_id, template=draft_spec.template
        )
        save_update: dict[str, Any] = {
            "metadata": {**draft_spec.metadata, "session_id": session_id}
        }
        if dup is not None:
            save_update["id"] = dup.id
        draft = self._host.routine_store.save(draft_spec.model_copy(update=save_update))
        self._host.routine_store.supersede_pending_approval_requests(session_id)
        approval_request = self._host.routine_store.create_approval_request(
            routine_id=draft.id,
            session_id=session_id,
            prompt=message,
            metadata={"source": "chat", "routine_title": draft.title},
        )
        draft_brief = self._brief_for_routine(draft)
        sections = normalize_briefing_sections(
            draft.metadata.get("briefing_sections"),
            draft_brief,
        )
        tool_callbacks = draft.metadata.get("tool_callbacks")
        callback_items = (
            [str(item) for item in tool_callbacks] if isinstance(tool_callbacks, list) else []
        )
        callback_text = ""
        if callback_items:
            callback_text = f"- Tool callbacks: {', '.join(callback_items)}\n"
        section_text = (
            f"- Briefing sections: {format_brief_sections(sections)}\n" if sections else ""
        )
        content_style_text = _content_style_line(draft.metadata)
        response = (
            f"Drafted routine `{draft.title}` and left it unapproved.\n"
            f"- ID: {draft.id}\n"
            f"- Approval request: {approval_request.id}\n"
            f"- Schedule: {draft.schedule}\n"
            f"- Routine type: {_routine_template_label(draft.template, brief_package=draft_brief)}\n"
            f"- What it does: {_routine_template_summary(draft.template, brief_package=draft_brief)}\n"
            f"{section_text}"
            f"{content_style_text}"
            f"{callback_text}"
            f"- Delivery: {draft.delivery_channel}\n"
            "Reply `sample` to preview the real output, `approve` to schedule it, "
            "or `cancel` to retire the draft."
        )
        return self._routine_chat_result(
            message=message,
            session_id=session_id,
            response=response,
            metadata={
                "routine_action": "drafted",
                "routine_id": draft.id,
                "approval_request_id": approval_request.id,
                "routine_status": str(draft.approval_status),
                "briefing_sections": list(sections),
                "tool_callbacks": callback_items,
                "delivery_channel": draft.delivery_channel,
                "content_style": draft.metadata.get("content_style", ""),
            },
            span=span,
        )

    def _handle_pending_routine_refinement(
        self,
        message: str,
        *,
        session_id: str,
        pending_request: RoutineApprovalRequest,
        refinements: dict[str, object],
        span: Any = None,
    ) -> ChatResult:
        spec = self._host.routine_store.load(pending_request.routine_id)
        if spec is None:
            self._host.routine_store.resolve_approval_request(
                pending_request.id,
                RoutineApprovalRequestStatus.CANCELLED,
            )
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response="I could not find the routine draft to update. Please draft it again.",
                metadata={
                    "routine_action": "missing_pending_draft",
                    "approval_request_id": pending_request.id,
                },
                span=span,
                has_errors=True,
                error_summary="pending routine draft missing",
            )
        updated = self._host.routine_store.save(_apply_routine_refinements(spec, refinements))
        response = (
            f"Updated {_routine_refinement_summary(refinements)} for `{updated.title}`.\n"
            f"- ID: {updated.id}\n"
            f"- Approval request: {pending_request.id}\n"
            f"- Routine type: {_routine_template_label(updated.template)}\n"
            f"- Delivery: {updated.delivery_channel}\n"
            f"{_content_style_line(updated.metadata)}"
            "Reply `sample` to preview the real output, `approve` to schedule it, "
            "or `cancel` to retire the draft."
        )
        return self._routine_chat_result(
            message=message,
            session_id=session_id,
            response=response,
            metadata={
                "routine_action": "updated",
                "routine_id": updated.id,
                "approval_request_id": pending_request.id,
                "delivery_channel": updated.delivery_channel,
                "content_style": updated.metadata.get("content_style", ""),
            },
            span=span,
        )

    def _handle_post_approval_refinement(
        self,
        message: str,
        *,
        session_id: str,
        target: RoutineSpec,
        refinements: dict[str, object],
        span: Any = None,
    ) -> ChatResult:
        """Apply a refinement to an already-SCHEDULED routine in place.

        Phase B: messages like "deliver to telegram" arriving after the
        user has already approved the routine must update *that* routine
        rather than spawn a new draft or complain about a missing one.
        This handles the unambiguous case — exactly one scheduled routine
        in the session. Captures a 5-minute undo snapshot so the user
        can roll the change back if the refinement was a mistake.
        """

        self._recent_refinement_snapshots[session_id] = {
            "snapshot": target,
            "expires_at": datetime.now(UTC) + _REFINEMENT_UNDO_WINDOW,
        }
        updated = self._host.routine_store.save(_apply_routine_refinements(target, refinements))
        response = (
            f"Updated {_routine_refinement_summary(refinements)} for `{updated.title}`.\n"
            f"- ID: {updated.id}\n"
            f"- Routine type: {_routine_template_label(updated.template)}\n"
            f"- Delivery: {updated.delivery_channel}\n"
            f"{_content_style_line(updated.metadata)}"
            "Reply `undo` within 5 minutes to revert."
        )
        return self._routine_chat_result(
            message=message,
            session_id=session_id,
            response=response,
            metadata={
                "routine_action": "updated",
                "routine_id": updated.id,
                "delivery_channel": updated.delivery_channel,
                "content_style": updated.metadata.get("content_style", ""),
            },
            span=span,
        )

    def _handle_post_approval_refinement_ambiguous(
        self,
        message: str,
        *,
        session_id: str,
        candidates: Sequence[RoutineSpec],
        refinements: dict[str, object],
        span: Any = None,
    ) -> ChatResult:
        """Multiple scheduled routines match — ask the user to pick.

        Stores the refinement and candidate ids on the session so the
        next turn (handled by ``_handle_refinement_pick_reply``) can
        resolve the user's pick and apply the refinement.
        """

        self._pending_refinement_picks[session_id] = {
            "refinements": dict(refinements),
            "candidate_ids": [c.id for c in candidates],
        }
        summary = _routine_refinement_summary(refinements)
        lines = [
            f"Refine which routine? Reply with the ID, position (e.g. `1`), "
            f"or part of the title — then I'll apply {summary}.",
        ]
        for index, spec in enumerate(candidates, start=1):
            lines.append(
                f"  {index}. `{spec.id}` — {spec.title} ({spec.schedule}, "
                f"delivery {spec.delivery_channel})"
            )
        return self._routine_chat_result(
            message=message,
            session_id=session_id,
            response="\n".join(lines),
            metadata={
                "routine_action": "clarify",
                "missing_slots": ["refinement_target"],
                "candidate_routine_ids": [c.id for c in candidates],
                "pending_refinements": dict(refinements),
            },
            span=span,
        )

    def _handle_refinement_pick_reply(
        self,
        message: str,
        *,
        session_id: str,
        pending_pick: dict[str, Any],
        parsed_action: str,
        span: Any = None,
    ) -> ChatResult | None:
        """Resolve the user's reply against a pending refinement pick.

        Returns ``None`` when the user clearly moved on to a fresh
        intent (a new routine draft, an approval, etc.) — the caller
        clears the pending pick and falls through to normal handling.
        Otherwise consumes the turn: applies the refinement to the
        chosen routine, cancels on negative intent, or re-asks when
        the reply is ambiguous.
        """

        # Fresh-intent escape hatch: the user typed something that
        # looks like a brand new routine request or an approval —
        # drop the pending pick and let the normal flow handle it.
        if parsed_action in {"draft", "clarify", "approve"}:
            self._pending_refinement_picks.pop(session_id, None)
            return None

        refinements_raw = pending_pick.get("refinements") or {}
        refinements: dict[str, object] = (
            dict(refinements_raw) if isinstance(refinements_raw, dict) else {}
        )
        candidate_ids_raw = pending_pick.get("candidate_ids") or []
        candidates: list[RoutineSpec] = []
        for rid in candidate_ids_raw:
            spec = self._host.routine_store.load(str(rid))
            if spec is not None:
                candidates.append(spec)

        if not candidates:
            self._pending_refinement_picks.pop(session_id, None)
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response=(
                    "The routines I asked about are no longer available. "
                    "Try the refinement again — I'll pick the right one this time."
                ),
                metadata={"routine_action": "missing_pending_refinement_target"},
                span=span,
            )

        if parsed_action == "cancel":
            self._pending_refinement_picks.pop(session_id, None)
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response="Cancelled the refinement.",
                metadata={
                    "routine_action": "refinement_cancelled",
                    "candidate_routine_ids": [c.id for c in candidates],
                },
                span=span,
            )

        chosen = resolve_routine_pick(message, candidates)
        if chosen is None:
            lines = [
                "I still need which routine to update. Reply with the ID, "
                "position (e.g. `1`), or part of the title."
            ]
            for index, spec in enumerate(candidates, start=1):
                lines.append(
                    f"  {index}. `{spec.id}` — {spec.title} ({spec.schedule}, "
                    f"delivery {spec.delivery_channel})"
                )
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response="\n".join(lines),
                metadata={
                    "routine_action": "clarify",
                    "missing_slots": ["refinement_target"],
                    "candidate_routine_ids": [c.id for c in candidates],
                },
                span=span,
            )

        self._pending_refinement_picks.pop(session_id, None)
        return self._handle_post_approval_refinement(
            message,
            session_id=session_id,
            target=chosen,
            refinements=refinements,
            span=span,
        )

    def _consume_recent_refinement_snapshot(self, session_id: str) -> RoutineSpec | None:
        """Return the unexpired pre-refinement snapshot for *session_id*.

        Pops the snapshot regardless of expiry so callers don't see a
        stale entry on a subsequent turn. Returns ``None`` when nothing
        was captured or the 5-minute window already elapsed.
        """

        entry = self._recent_refinement_snapshots.pop(session_id, None)
        if entry is None:
            return None
        expires_at = entry.get("expires_at")
        if isinstance(expires_at, datetime) and datetime.now(UTC) > expires_at:
            return None
        spec = entry.get("snapshot")
        return spec if isinstance(spec, RoutineSpec) else None

    def _handle_refinement_undo(
        self,
        message: str,
        *,
        session_id: str,
        span: Any = None,
    ) -> ChatResult:
        """Restore the most recent post-approval refinement target.

        Phase B slice 4: the agent promises a 5-minute undo window
        after every post-approval refinement; this is the handler.
        Also clears any pending pick / swap state so the session
        returns to a clean post-approval baseline.
        """

        snapshot = self._consume_recent_refinement_snapshot(session_id)
        if snapshot is None:
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response=(
                    "Nothing to undo in this chat — there's no recent "
                    "refinement within the 5-minute window."
                ),
                metadata={"routine_action": "undo_unavailable"},
                span=span,
            )

        restored = self._host.routine_store.save(
            snapshot.model_copy(update={"updated_at": datetime.now(UTC)})
        )
        # Any in-progress pick / swap state was implicitly about the
        # post-refinement spec — drop it so subsequent turns aren't
        # confused.
        self._pending_refinement_picks.pop(session_id, None)
        self._pending_capability_swaps.pop(session_id, None)
        return self._routine_chat_result(
            message=message,
            session_id=session_id,
            response=(
                f"Reverted `{restored.title}` to its previous state.\n"
                f"- ID: {restored.id}\n"
                f"- Routine type: {_routine_template_label(restored.template)}\n"
                f"- Delivery: {restored.delivery_channel}\n"
                f"{_content_style_line(restored.metadata)}"
            ),
            metadata={
                "routine_action": "undone",
                "routine_id": restored.id,
                "delivery_channel": restored.delivery_channel,
                "content_style": restored.metadata.get("content_style", ""),
            },
            span=span,
        )

    def _handle_capability_swap_proposal(
        self,
        message: str,
        *,
        session_id: str,
        target: RoutineSpec,
        swap_to: Any,
        span: Any = None,
    ) -> ChatResult:
        """Propose a capability swap and wait for explicit confirmation.

        Phase B slice 3: rebinding a routine to a different skill is
        large — it changes the tool callbacks, source preferences, and
        the user's mental model. We never apply silently. The pending
        state stores the routine id + target skill name; the next turn
        is handled by ``_handle_capability_swap_confirmation``.
        """

        new_template = swap_to.manifest.name
        self._pending_capability_swaps[session_id] = {
            "routine_id": target.id,
            "template": new_template,
        }
        response = (
            f"This will rebind `{target.title}` from `{target.template}` "
            f"to `{new_template}` — proceed? Reply `yes` to apply, "
            "anything else to cancel."
        )
        return self._routine_chat_result(
            message=message,
            session_id=session_id,
            response=response,
            metadata={
                "routine_action": "clarify",
                "missing_slots": ["capability_swap_confirmation"],
                "routine_id": target.id,
                "proposed_template": new_template,
                "current_template": target.template,
            },
            span=span,
        )

    def _handle_capability_swap_confirmation(
        self,
        message: str,
        *,
        session_id: str,
        pending: dict[str, Any],
        parsed_action: str,
        span: Any = None,
    ) -> ChatResult | None:
        """Resolve the user's reply to a pending capability-swap proposal.

        Returns ``None`` only when the user starts a fresh routine
        intent (draft/clarify) — the caller clears the pending state
        and falls through. Otherwise consumes the turn: applies on
        approve, drops on anything else.
        """

        if parsed_action in {"draft", "clarify"}:
            self._pending_capability_swaps.pop(session_id, None)
            return None

        routine_id = str(pending.get("routine_id", ""))
        new_template = str(pending.get("template", ""))
        target = self._host.routine_store.load(routine_id) if routine_id else None
        self._pending_capability_swaps.pop(session_id, None)

        if target is None or not new_template:
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response="The routine I asked about is no longer available.",
                metadata={"routine_action": "missing_pending_refinement_target"},
                span=span,
            )

        if parsed_action != "approve":
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response=(f"Kept `{target.title}` bound to `{target.template}` — " "no change."),
                metadata={
                    "routine_action": "refinement_cancelled",
                    "routine_id": target.id,
                },
                span=span,
            )

        return self._handle_post_approval_refinement(
            message,
            session_id=session_id,
            target=target,
            refinements={"template": new_template},
            span=span,
        )

    def _pending_routine_clarification(self, session_id: str) -> RoutineSpec | None:
        for routine in self._host.routine_store.list_by_status(RoutineApprovalStatus.CLARIFY):
            if routine.metadata.get("session_id") == session_id:
                return routine
        return None

    def _loaded_capability_packages(self) -> tuple[Any, ...]:
        """Return every loadable skill package — any kind, any toolset.

        Routines bind to arbitrary capabilities (briefs, tool skills,
        and any future kinds), so the candidate set for capability
        matching must include all loadable packages. Brief-render code
        paths still filter by ``kind == "brief"`` at their own
        boundaries.
        """

        try:
            packages = self._host.skill_registry.list_packages(only_loadable=True)
        except Exception:
            logger.debug("capability package discovery failed", exc_info=True)
            return ()
        return tuple(packages)

    def routine_authoring_llm_caller(self) -> LLMCaller:
        """Return a Tier-2 callable for routine authoring (Q-A1 / Q-A3).

        The LLM client is constructed on each invocation rather than at
        runtime build time so a misconfigured ``llm_tiers.yaml`` or a
        downed Ollama service doesn't break bootstrap. Downstream
        helpers (``extract_user_named_title``, ``parse_tool_arg_reply``)
        already catch exceptions and degrade to deterministic fallbacks
        so a runtime failure inside this caller is safe.

        Uses ``task_planning`` intent — that's mapped to Tier-2 in the
        default ``llm_tiers.yaml`` and structurally matches what the
        helpers do (extract structured choices from natural language).
        """

        tier_router = self._host.tier_router

        def call(prompt: str) -> str:
            from iris_harness.llm.client import (
                CodingLLMClient,
                CodingLLMConfig,
            )

            cfg = tier_router.get_llm_config("task_planning")
            client = CodingLLMClient(CodingLLMConfig(**vars(cfg)))
            return client.invoke(system_prompt="", user_prompt=prompt)

        return call

    def _capability_for_routine(self, routine: RoutineSpec) -> Any | None:
        """Resolve the skill package a routine is bound to (any kind)."""

        return _capability_package_for_template(routine.template, self._host.skill_registry)

    def _brief_for_routine(self, routine: RoutineSpec) -> Any | None:
        """Resolve the routine's bound package only when it is a brief.

        Display/clarification code paths that expect brief slots check
        for ``None`` to detect non-brief routines and fall through to
        the kind-appropriate flow.
        """

        package = self._capability_for_routine(routine)
        if package is None:
            return None
        if package.manifest.kind != "brief" or package.manifest.brief is None:
            return None
        return package

    def _retire_pending_routine_clarifications(self, session_id: str) -> None:
        self._pending_capability_clarifications.pop(session_id, None)
        for routine in self._host.routine_store.list_by_status(RoutineApprovalStatus.CLARIFY):
            if routine.metadata.get("session_id") == session_id:
                self._host.routine_store.save(routine.with_status(RoutineApprovalStatus.RETIRED))

    def _handle_routine_clarification_reply(
        self,
        message: str,
        *,
        session_id: str,
        pending: RoutineSpec,
        span: Any = None,
    ) -> ChatResult:
        refinements = extract_routine_refinements(message)
        # Phase A-2: tool-args clarifications use a different shape than
        # brief-section clarifications (rich ToolArg list rather than a
        # slot menu). Route those through a dedicated handler before
        # falling into the brief-section flow.
        pending_tool_args_meta = pending.metadata.get("pending_tool_args")
        if isinstance(pending_tool_args_meta, list) and pending_tool_args_meta:
            return self._handle_tool_args_clarification_reply(
                message,
                session_id=session_id,
                pending=pending,
                pending_tool_args_meta=pending_tool_args_meta,
                refinements=refinements,
                span=span,
            )
        brief = self._brief_for_routine(pending)
        sections: tuple[str, ...] = ()
        if brief is not None:
            sections = extract_briefing_sections(message, brief, _get_semantic_router())
        if not sections:
            if refinements:
                updated = self._host.routine_store.save(
                    _apply_routine_refinements(pending, refinements)
                )
                slot_labels = (
                    ", ".join(brief_tool_slot_keys(brief))
                    if brief is not None
                    else "the sections you want"
                )
                return self._routine_chat_result(
                    message=message,
                    session_id=session_id,
                    response=(
                        f"Updated {_routine_refinement_summary(refinements)}. I still need "
                        "what the briefing should include.\n"
                        f"- ID: {updated.id}\n"
                        f"- Delivery: {updated.delivery_channel}\n"
                        f"{_content_style_line(updated.metadata)}"
                        f"- Next step: choose {slot_labels}; or `all`."
                    ),
                    metadata={
                        "routine_action": "clarify",
                        "routine_id": updated.id,
                        "missing_slots": ["briefing_sections", "template_confirmation"],
                        "delivery_channel": updated.delivery_channel,
                        "content_style": updated.metadata.get("content_style", ""),
                    },
                    span=span,
                )
            slot_labels = (
                ", ".join(brief_tool_slot_keys(brief))
                if brief is not None
                else "the sections you want"
            )
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response=(
                    "I still need what the briefing should include. Reply with one or more: "
                    f"{slot_labels}; or `all`."
                ),
                metadata={
                    "routine_action": "clarify",
                    "routine_id": pending.id,
                    "missing_slots": ["briefing_sections", "template_confirmation"],
                },
                span=span,
            )
        pending = _apply_routine_refinements(pending, refinements)
        callbacks = brief_slot_callbacks(brief, sections) if brief is not None else ()
        capabilities = (
            brief_slot_capabilities(brief, sections) if brief is not None else ("channel_delivery",)
        )
        metadata = {
            **pending.metadata,
            "briefing_sections": list(sections),
            "tool_callbacks": list(callbacks),
            "template_confirmed": True,
            "clarified_from": message,
        }
        updated = pending.model_copy(
            update={
                "approval_status": RoutineApprovalStatus.DRAFT,
                "source_preferences": sections,
                "required_capabilities": capabilities,
                "metadata": metadata,
                "updated_at": datetime.now(UTC),
            }
        )
        draft = self._host.routine_store.save(updated)
        self._host.routine_store.supersede_pending_approval_requests(session_id)
        approval_request = self._host.routine_store.create_approval_request(
            routine_id=draft.id,
            session_id=session_id,
            prompt=message,
            metadata={
                "source": "chat",
                "routine_title": draft.title,
                "clarification_for": pending.id,
            },
        )
        callback_text = f"- Tool callbacks: {', '.join(callbacks)}\n" if callbacks else ""
        response = (
            f"Confirmed `{_routine_template_label(draft.template, brief_package=brief)}` "
            "and drafted the routine.\n"
            f"- ID: {draft.id}\n"
            f"- Approval request: {approval_request.id}\n"
            f"- Schedule: {draft.schedule}\n"
            f"- What it does: {_routine_template_summary(draft.template, brief_package=brief)}\n"
            f"- Briefing sections: {format_brief_sections(sections)}\n"
            f"{_content_style_line(draft.metadata)}"
            f"{callback_text}"
            f"- Delivery: {draft.delivery_channel}\n"
            "Reply `sample` to preview the real output, `approve` to schedule it, "
            "or `cancel` to retire the draft."
        )
        return self._routine_chat_result(
            message=message,
            session_id=session_id,
            response=response,
            metadata={
                "routine_action": "drafted",
                "routine_id": draft.id,
                "approval_request_id": approval_request.id,
                "routine_status": str(draft.approval_status),
                "briefing_sections": list(sections),
                "tool_callbacks": list(callbacks),
                "delivery_channel": draft.delivery_channel,
                "content_style": draft.metadata.get("content_style", ""),
            },
            span=span,
        )

    def _handle_tool_args_clarification_reply(
        self,
        message: str,
        *,
        session_id: str,
        pending: RoutineSpec,
        pending_tool_args_meta: list[dict[str, Any]],
        refinements: dict[str, object],
        span: Any = None,
    ) -> ChatResult:
        """Resolve a user's reply to a multi-arg routine clarification.

        Implements the Q-A1 contract: the user's natural-language reply
        is sent (with the args schema) to the Tier-2 LLM, which returns
        structured values. Successful args are merged into the routine
        spec; unresolved args trigger a follow-up that re-asks ONLY the
        still-missing ones (not the full set). When the LLM is
        unavailable the helper falls back to ``key=value`` / positional
        parsing so the flow degrades cleanly rather than crashing.
        """

        pending_args = tuple(ToolArg.model_validate(entry) for entry in pending_tool_args_meta)
        prior_resolved_raw = pending.metadata.get("resolved_tool_args") or {}
        prior_resolved: dict[str, Any] = (
            dict(prior_resolved_raw) if isinstance(prior_resolved_raw, dict) else {}
        )

        values, missing_names = parse_tool_arg_reply(
            message,
            pending_args,
            self.routine_authoring_llm_caller(),
            prior_answers=prior_resolved,
        )

        merged_resolved: dict[str, Any] = {**prior_resolved, **values}
        pending = _apply_routine_refinements(pending, refinements)

        if missing_names:
            still_pending = tuple(arg for arg in pending_args if arg.name in set(missing_names))
            still_pending_meta = [arg.model_dump(mode="json") for arg in still_pending]
            new_metadata = {
                **pending.metadata,
                "pending_tool_args": still_pending_meta,
                "resolved_tool_args": merged_resolved,
                "clarified_from": message,
            }
            updated = self._host.routine_store.save(
                pending.model_copy(
                    update={
                        "metadata": new_metadata,
                        "updated_at": datetime.now(UTC),
                    }
                )
            )
            confirmed_line = (
                "Got " + ", ".join(f"{k}={v}" for k, v in values.items()) + ".\n" if values else ""
            )
            reask_lines = [confirmed_line + "I still need:"]
            reask_lines.extend(format_tool_arg_prompt_line(arg) for arg in still_pending)
            return self._routine_chat_result(
                message=message,
                session_id=session_id,
                response="\n".join(reask_lines),
                metadata={
                    "routine_action": "clarify",
                    "routine_id": updated.id,
                    "missing_slots": ["tool_args"],
                    "resolved_tool_args": merged_resolved,
                    "pending_tool_args": [arg.name for arg in still_pending],
                    "delivery_channel": updated.delivery_channel,
                    "content_style": updated.metadata.get("content_style", ""),
                },
                span=span,
            )

        # All args resolved → draft the routine.
        bound_tool = str(pending.metadata.get("bound_tool") or "")
        new_metadata = {
            **pending.metadata,
            "resolved_tool_args": merged_resolved,
            "tool_call_args": merged_resolved,
            "template_confirmed": True,
            "clarified_from": message,
        }
        new_metadata.pop("pending_tool_args", None)

        source_preferences: tuple[str, ...] = (bound_tool,) if bound_tool else ()
        updated_spec = pending.model_copy(
            update={
                "approval_status": RoutineApprovalStatus.DRAFT,
                "source_preferences": source_preferences,
                "metadata": new_metadata,
                "updated_at": datetime.now(UTC),
            }
        )
        draft = self._host.routine_store.save(updated_spec)
        self._host.routine_store.supersede_pending_approval_requests(session_id)
        approval_request = self._host.routine_store.create_approval_request(
            routine_id=draft.id,
            session_id=session_id,
            prompt=message,
            metadata={
                "source": "chat",
                "routine_title": draft.title,
                "clarification_for": pending.id,
            },
        )
        args_line = (
            "- Args: " + ", ".join(f"{k}={v}" for k, v in merged_resolved.items()) + "\n"
            if merged_resolved
            else ""
        )
        callbacks = draft.metadata.get("tool_callbacks") or []
        callback_items = [str(item) for item in callbacks] if isinstance(callbacks, list) else []
        callback_text = f"- Tool callbacks: {', '.join(callback_items)}\n" if callback_items else ""
        response = (
            f"Confirmed `{draft.template}` and drafted the routine.\n"
            f"- ID: {draft.id}\n"
            f"- Approval request: {approval_request.id}\n"
            f"- Schedule: {draft.schedule}\n"
            f"- Tool: {bound_tool or 'n/a'}\n"
            f"{args_line}"
            f"{callback_text}"
            f"{_content_style_line(draft.metadata)}"
            f"- Delivery: {draft.delivery_channel}\n"
            "Reply `sample` to preview the real output, `approve` to schedule it, "
            "or `cancel` to retire the draft."
        )
        return self._routine_chat_result(
            message=message,
            session_id=session_id,
            response=response,
            metadata={
                "routine_action": "drafted",
                "routine_id": draft.id,
                "approval_request_id": approval_request.id,
                "routine_status": str(draft.approval_status),
                "resolved_tool_args": merged_resolved,
                "tool_callbacks": callback_items,
                "delivery_channel": draft.delivery_channel,
                "content_style": draft.metadata.get("content_style", ""),
            },
            span=span,
        )

    def _routine_chat_result(
        self,
        *,
        message: str,
        session_id: str,
        response: str,
        metadata: dict[str, object],
        span: Any = None,
        has_errors: bool = False,
        error_summary: str | None = None,
    ) -> ChatResult:
        intent_result = IntentResult(
            intent="routine_authoring",
            agent_type="system",
            confidence=0.95,
            raw_query=message,
        )
        curated = CuratedResponse(
            text=response,
            sources=["routines"],
            has_errors=has_errors,
            error_summary=error_summary,
            metadata={"total_latency_ms": 0.0, **metadata},
        )
        self._host.sessions.record_turn(session_id, message, response)
        self._host.capture.record_signal(
            intent_result,
            curated,
            latency_ms=0.0,
            model="deterministic",
            provider="local",
            query=message,
        )
        if span is not None:
            try:
                span.set_attribute("output.value", response)
            except Exception:  # noqa: BLE001, S110
                pass
            span.set_attribute("intent", intent_result.intent)
            span.set_attribute("agent_type", intent_result.agent_type)
            span.set_attribute("session_id", session_id)
            span.set_attribute("model", "deterministic")
            span.set_attribute("provider", "local")
            span.set_attribute("latency_ms", 0.0)
            span.set_attribute("has_errors", has_errors)
        return ChatResult(
            response=response,
            intent=intent_result.intent,
            agent_type=intent_result.agent_type,
            sources=tuple(curated.sources),
            has_errors=has_errors,
            error_summary=error_summary,
            metadata={
                **curated.metadata,
                "is_multi_step": False,
                "plan_size": 0,
                "session_id": session_id,
                "model": "deterministic",
                "provider": "local",
                "router_model": "deterministic",
                "router_provider": "local",
            },
        )

    # Backwards-compatible alias retained until call sites migrate.
    _loaded_brief_packages = _loaded_capability_packages

    def has_active_routine_conversation(self, session_id: str) -> bool:
        """True when a routine authoring/clarification conversation is mid-flight.

        Mirrors the session-state the routine-authoring handler consumes, so the
        on-demand brief intercept can step aside and let a continuation turn reach it.
        """
        if self._host.routine_store.get_pending_approval_request(session_id) is not None:
            return True
        if self._pending_routine_clarification(session_id) is not None:
            return True
        return bool(
            self._pending_refinement_picks.get(session_id)
            or self._pending_capability_swaps.get(session_id)
            or self._pending_capability_clarifications.get(session_id)
        )


# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_semantic_router_singleton", "_semantic_router_init_failed")
