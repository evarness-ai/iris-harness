"""The local-skill half of the general handler, as its own factory.

Gate-1 extraction (OSS plan M5.7), slice 4b. Five functions that turn the loaded skill
packages into tools the handler can bind, and into the direct answers a matched skill or
brief produces without an LLM call at all.

**Why a factory and not five functions taking ``skill_registry``.** The measurement said
this cluster closes over exactly one name, so threading it through as a parameter was the
obvious move — five signatures, and every call between them passing it along. That would
have been a behaviour-preserving refactor rather than a relocation, and the only thing
standing behind it would have been the suite. Keeping the closure and moving *it* is a
move: the nested definitions are byte-identical to what they were inside
``_make_general_handler``, and the change at the call site is one line. The closure was
never the problem; living inside a 1,100-line function was.

``record_skill_pending_actions`` travels with them because
``_direct_local_skill_response`` is its only caller.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iris_harness.agent.agent_executor import AgentTask
from iris_harness.foundation.clock import local_now
from iris_harness.foundation.paths import data_dir
from iris_harness.kernel.governance.external_content import add_redaction_notice
from iris_harness.llm.errors import friendly_llm_error as _friendly_llm_error
from iris_harness.runtime.brief_tools import make_brief_render_tool, make_brief_runner
from iris_harness.runtime.external_text import redact_external_text
from iris_harness.runtime.handlers.general_support import local_skill_input_schema
from iris_harness.runtime.routine_authoring import _get_semantic_router
from iris_harness.runtime.skill_matching import (
    SKILL_MATCH_MIN_SCORE,
    best_matching_skill_package,
    extract_skill_arguments,
    format_direct_skill_result,
    score_skill_package,
)
from iris_harness.runtime.skill_tool_specs import json_skill_call, skill_tool_spec
from iris_harness.tools.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

#: The caller of a skill tool the general lane runs for a matched request, before any model: code,
#: not a model (``core:`` callers need no caller-policy entry; only ``plugin:`` callers are checked).
DIRECT_CALLER = "core:general_lane"


def record_skill_pending_actions(result: object, *, data_dir: Path) -> int:
    """Persist pending actions from known skill proposal payload shapes.

    This is the bridge from model-callable skill outputs (proposal dicts) into
    Action Center tasks. The helper is intentionally conservative: unknown shapes
    are ignored and the caller proceeds with normal response rendering.
    """
    if not isinstance(result, dict):
        return 0

    kind = str(result.get("kind") or "").strip().lower()
    if not kind:
        return 0

    from iris_harness.services.tasks import TaskStore
    from iris_harness.services.tasks.models import TaskAction

    store = TaskStore(db_path=data_dir / "tasks.db")
    store.ensure_schema()

    if kind == "rag_ingest":
        proposal = result.get("proposal")
        if not isinstance(proposal, dict):
            return 0
        resolved_path = str(proposal.get("resolved_path") or proposal.get("path") or "").strip()
        content_sha = str(proposal.get("content_sha") or "").strip()
        source_id = content_sha or resolved_path
        if not source_id:
            return 0
        title_name = Path(resolved_path).name if resolved_path else "document"
        description = str(
            proposal.get("reason")
            or result.get("summary")
            or "Approve to ingest this document into the RAG index; reject to skip it."
        )
        # Execution re-proposes from the live path (TOCTOU re-scan), so the action
        # must carry the path. Without one, fall back to a read-only review.
        executable = bool(resolved_path)
        store.upsert(
            dedup_key=f"rag-ingest:{source_id}",
            title=f"Approve RAG ingest: {title_name}",
            description=description,
            source_kind="rag-ingest",
            source_id=source_id,
            action=TaskAction(
                kind="execute" if executable else "review",
                label="Approve & ingest into RAG" if executable else "Review RAG ingest",
                target_id=resolved_path or source_id,
                safe=True,
            ),
        )
        return 1

    if kind == "photos_albums":
        plan = result.get("plan")
        if not isinstance(plan, dict):
            return 0
        created_at = str(plan.get("created_at") or "").strip()
        raw_groups = plan.get("groups")
        groups = raw_groups if isinstance(raw_groups, list) else []
        group_names = [
            str(g.get("name") or "").strip()
            for g in groups
            if isinstance(g, dict) and str(g.get("name") or "").strip()
        ]
        source_id = created_at or f"groups-{len(group_names)}"
        preview = ", ".join(group_names[:3])
        description = f"Approve to create {len(group_names)} album(s) in Apple Photos" + (
            f": {preview}. Rejecting makes no changes."
            if preview
            else ". Rejecting makes no changes."
        )
        store.upsert(
            dedup_key=f"photos-albums:{source_id}",
            title="Approve proposed photo albums",
            description=description,
            source_kind="photos-albums",
            source_id=source_id,
            action=TaskAction(
                kind="execute",
                label="Approve & create albums",
                target_id=source_id,
                safe=True,
            ),
        )
        return 1

    if kind == "filemanager_vault":
        proposal = result.get("quarantine_proposal")
        if not isinstance(proposal, dict):
            return 0
        proposal_kind = str(proposal.get("kind") or "").strip().lower()
        if proposal_kind != "filemanager-quarantine":
            return 0
        source_id = str(proposal.get("vault_file_id") or result.get("file_id") or "").strip()
        source_path = str(proposal.get("path") or "").strip()
        dedup_seed = source_id or source_path
        if not dedup_seed:
            return 0
        reason = str(proposal.get("reason") or "quarantine original after vaulting")
        description = f"{reason}. Source: {source_path or 'unknown'}"
        # Quarantine needs the original path; without it, fall back to review.
        executable = bool(source_path)
        store.upsert(
            dedup_key=f"filemanager-quarantine:{dedup_seed}",
            title="Approve file quarantine",
            description=description,
            source_kind="filemanager-quarantine",
            source_id=dedup_seed,
            action=TaskAction(
                kind="execute" if executable else "review",
                label="Approve & quarantine original" if executable else "Review quarantine",
                target_id=source_path or dedup_seed,
                safe=True,
            ),
        )
        return 1

    return 0


@dataclass(frozen=True)
class LocalSkills:
    """The four local-skill entry points the general handler binds or calls."""

    tools: Any
    relevant_tools: Any
    direct_skill_response: Any
    direct_brief_response: Any


def make_local_skills(
    skill_registry: SkillRegistry | None, runtime_holder: list[Any] | None = None
) -> LocalSkills:
    """Bind the local-skill helpers to one registry, which may be absent.

    ``None`` is a real case — the factory's own parameter is optional and
    ``_local_skill_packages`` returns an empty tuple for it, so a profile with no skill
    registry gets helpers that correctly find nothing.

    The bodies below are unchanged from their nested originals — this function is the
    closure they already had, with a name and a module of its own.
    """

    def _tool_service() -> Any:
        """The runtime's ``ToolService``, or None when this handler has no runtime."""
        runtime = runtime_holder[0] if runtime_holder else None
        return getattr(runtime, "tool_service", None)

    def _local_skill_packages() -> tuple[Any, ...]:
        if skill_registry is None:
            return ()

        try:
            skill_registry.discover()
            return skill_registry.list_packages(only_loadable=True)
        except Exception:
            logger.exception("local skill discovery failed for general handler")
            return ()

    def _local_skill_tools() -> dict[str, tuple[type[Any], str, dict[str, Any]]]:
        packages = _local_skill_packages()
        if not packages:
            return {}

        reserved_tools = {
            "research",
            "memory_search",
            "memory_graph",
            "wiki_search",
            "code_exec",
            "propose_skill_from_sandbox",
        }
        tools: dict[str, tuple[type[Any], str, dict[str, Any]]] = {}
        for package in packages:
            for manifest_tool, tool_class in zip(
                package.manifest.tools,
                package.tool_classes,
                strict=False,
            ):
                if manifest_tool.name in reserved_tools or manifest_tool.name in tools:
                    continue
                tools[manifest_tool.name] = (
                    tool_class,
                    manifest_tool.description,
                    local_skill_input_schema(tool_class),
                )
            if package.manifest.kind == "brief" and package.manifest.brief is not None:
                assert skill_registry is not None  # guaranteed: packages is non-empty
                brief_tool_name, brief_tool_class, brief_description = make_brief_render_tool(
                    package.manifest.name,
                    package.manifest.description,
                    make_brief_runner(skill_registry, package.manifest.name),
                )
                if brief_tool_name in reserved_tools or brief_tool_name in tools:
                    continue
                tools[brief_tool_name] = (
                    brief_tool_class,
                    brief_description,
                    local_skill_input_schema(brief_tool_class),
                )
        return tools

    def _direct_local_skill_response(task: AgentTask) -> tuple[str, dict[str, object]] | None:
        package = best_matching_skill_package(task.query, _local_skill_packages())
        if package is None or not package.tool_classes:
            return None

        tool_class = package.tool_classes[0]
        tool_name = package.manifest.tools[0].name if package.manifest.tools else tool_class().name
        arguments = extract_skill_arguments(task.query, tool_class)
        if arguments is None:
            return None

        service = _tool_service()
        if service is not None and package.manifest.tools:
            # Through governance like any other tool call (#155): PRE/POST_TOOL_USE, the
            # external-content floor, the audit rows. The tool's result is serialised to JSON
            # inside the governed call and parsed back out of the governed text, so the
            # formatter and the pending-action recorder still read the structured value.
            spec = skill_tool_spec(
                package,
                package.manifest.tools[0],
                tool_class,
                call=json_skill_call(tool_class),
            )
            outcome = service.call_core_tool(DIRECT_CALLER, spec, arguments)
            if not outcome.ok:
                # Governance held or refused it (an effectful tool the owner has not approved:
                # nothing ran), or the tool raised: the owner is told, the turn does not guess.
                text = (
                    outcome.text
                    if outcome.held
                    else f"The `{tool_name}` skill failed: {outcome.text}"
                )
                return (
                    text,
                    {"used_skill": True, "skill_tool": tool_name, "skill_error": True},
                )
            try:
                result: object = json.loads(outcome.text)
                parsed = True
            except ValueError:
                # What governance let through is not the tool's JSON (a result it replaced or
                # withheld): show it as the text it is, and record no pending actions from it.
                logger.warning(
                    "direct local skill %s: governed result is not JSON; shown as text, "
                    "pending actions not recorded",
                    tool_name,
                )
                result, parsed = outcome.text, False
            recorded = 0
            if parsed:
                try:
                    recorded = record_skill_pending_actions(result, data_dir=data_dir())
                except Exception:
                    logger.exception("failed to record pending actions from skill result")
            answer = format_direct_skill_result(tool_name, result)
            if outcome.external:
                # The floor already redacted the text for this owner-facing caller (no
                # envelope); say so once when it cut a span out (issue #139).
                answer = add_redaction_notice(answer)
            return (
                answer,
                {
                    "used_skill": True,
                    "skill_tool": tool_name,
                    "pending_actions_recorded": recorded,
                },
            )

        # No runtime to govern the call (a handler built on its own, as a unit test does):
        # the tool runs in-process and its text gets the owner-channel tripwire, as before.
        try:
            result = tool_class().invoke(arguments)
        except Exception as exc:
            logger.exception("direct local skill execution failed: %s", tool_name)
            return (
                f"The `{tool_name}` skill failed: {_friendly_llm_error(exc)}",
                {"used_skill": True, "skill_tool": tool_name, "skill_error": True},
            )
        recorded = 0
        try:
            pending_data_dir = data_dir()
            recorded = record_skill_pending_actions(result, data_dir=pending_data_dir)
        except Exception:
            logger.exception("failed to record pending actions from skill result")
        answer = format_direct_skill_result(tool_name, result)
        if package.manifest.tools and package.manifest.tools[0].content == "external":
            # The answer is the tool's text, shown to the owner as it is: tripwire, no envelope.
            answer = redact_external_text(answer, skill=package.manifest.name, tool=tool_name)
            # Say so once when the floor cut a span out (issue #139); the turn's own notice
            # stage does the same for any answer and does not repeat it.
            answer = add_redaction_notice(answer)
        return (
            answer,
            {
                "used_skill": True,
                "skill_tool": tool_name,
                "pending_actions_recorded": recorded,
            },
        )

    def _direct_local_brief_response(task: AgentTask) -> tuple[str, dict[str, object]] | None:
        """Render matching ``kind: brief`` skills directly for chat turns.

        Brief skills don't expose tool classes themselves, so they would
        otherwise fall through to generic LLM chat after skill-intent routing.
        """
        package = best_matching_skill_package(task.query, _local_skill_packages())
        if package is None or package.manifest.kind != "brief" or package.manifest.brief is None:
            return None
        if skill_registry is None:
            return None

        try:
            from iris_harness.runtime.handlers.skill_brief import (
                build_slot_context,
                render_layout,
                resolve_slot,
            )

            spec = package.manifest.brief
            ctx = build_slot_context(skill_registry, now=local_now())
            resolved = {
                name: resolve_slot(slot, ctx, spec.uses) for name, slot in spec.slots.items()
            }
            body = render_layout(spec.layout, resolved)
        except Exception as exc:
            logger.exception("direct local brief execution failed: %s", package.manifest.name)
            return (
                f"The `{package.manifest.name}` brief failed: {_friendly_llm_error(exc)}",
                {
                    "used_skill": True,
                    "skill_tool": package.manifest.name,
                    "skill_kind": "brief",
                    "skill_error": True,
                },
            )

        return (
            body,
            {
                "used_skill": True,
                "skill_tool": package.manifest.name,
                "skill_kind": "brief",
            },
        )

    def _relevant_skill_tools(query: str) -> dict[str, tuple[type[Any], str, dict[str, Any]]]:
        """Return only skills whose manifest is relevant to *query*.

        Uses the semantic router (cosine similarity over manifest text)
        when available, falling back to the legacy keyword scorer otherwise.
        When query is empty every loaded skill is returned (used for direct
        execution lookup).
        """
        packages = _local_skill_packages()
        if not packages:
            return {}
        reserved = {
            "research",
            "memory_search",
            "memory_graph",
            "wiki_search",
            "code_exec",
            "propose_skill_from_sandbox",
        }
        tools: dict[str, tuple[type[Any], str, dict[str, Any]]] = {}
        explicit_local_helper_request = bool(
            re.search(r"\b(local helper|local skill|promoted skill)\b", query.lower())
        )
        router = _get_semantic_router()

        def _is_relevant(package: Any) -> bool:
            if not query or explicit_local_helper_request:
                return True
            if router is not None:
                return bool(router.score(query, package) >= router.threshold)
            return score_skill_package(query, package) >= SKILL_MATCH_MIN_SCORE

        for package in packages:
            if not package.is_loadable:
                continue
            if not _is_relevant(package):
                continue
            for manifest_tool, tool_class in zip(
                package.manifest.tools, package.tool_classes, strict=False
            ):
                if manifest_tool.name in reserved or manifest_tool.name in tools:
                    continue
                tools[manifest_tool.name] = (
                    tool_class,
                    manifest_tool.description,
                    local_skill_input_schema(tool_class),
                )
            if package.manifest.kind == "brief" and package.manifest.brief is not None:
                assert skill_registry is not None  # guaranteed: packages is non-empty
                brief_tool_name, brief_tool_class, brief_description = make_brief_render_tool(
                    package.manifest.name,
                    package.manifest.description,
                    make_brief_runner(skill_registry, package.manifest.name),
                )
                if brief_tool_name in reserved or brief_tool_name in tools:
                    continue
                tools[brief_tool_name] = (
                    brief_tool_class,
                    brief_description,
                    local_skill_input_schema(brief_tool_class),
                )
        return tools

    return LocalSkills(
        tools=_local_skill_tools,
        relevant_tools=_relevant_skill_tools,
        direct_skill_response=_direct_local_skill_response,
        direct_brief_response=_direct_local_brief_response,
    )
