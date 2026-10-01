"""Matching a user's turn to a local skill package, and shaping the result.

Gate-1 extraction (OSS plan M5.7). Token overlap scoring, the semantic-router
tie-break, argument extraction from the turn, and the direct-result formatter — the
deterministic half of skill dispatch, with no LLM call and no state.

Extracted ahead of the local-skills cluster that needs it. That cluster's measurement
found these helpers and ``brief_tools`` were prerequisites rather than successors, so the
plan's slice order was inverted here instead of reaching back into bootstrap for them.
Four of the five are also called by ``_is_relevant``, ``_skills_to_react_tools`` and
``_resolve_skill_intent``, which stay in bootstrap — they were module-level there already.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from iris_harness.runtime.routine_authoring import _get_semantic_router

logger = logging.getLogger(__name__)


SKILL_MATCH_MIN_SCORE = 2


_SKILL_TOKEN_ALIASES = {
    "git": "github",
    "repos": "repo",
    "repository": "repo",
    "repositories": "repo",
    "briefing": "brief",
    "briefings": "brief",
}


_SKILL_TOKEN_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "for",
        "from",
        "give",
        "me",
        "of",
        "please",
        "show",
        "the",
        "today",
        "what",
        "with",
    }
)


def _skill_tokens(text: str) -> set[str]:
    """Return normalized keyword tokens for matching natural language to skills."""
    tokens: set[str] = set()
    for raw in re.findall(r"[a-z0-9]+", text.lower()):
        token = _SKILL_TOKEN_ALIASES.get(raw) or raw
        if len(token) < 3 or token in _SKILL_TOKEN_STOPWORDS:
            continue
        tokens.add(token)
        if token in {"remind", "reminder", "reminders"}:
            tokens.update({"calendar", "reminder"})
        if token in {"schedule", "scheduled", "event", "events", "appointment"}:
            tokens.add("calendar")
    return tokens


def score_skill_package(query: str, package: Any) -> int:
    """Score how strongly *query* matches a loaded skill package."""
    manifest = package.manifest
    skill_text = " ".join(
        (
            manifest.name.replace("-", " ").replace("_", " "),
            manifest.description,
            " ".join(tool.name.replace("_", " ") for tool in manifest.tools),
            " ".join(tool.description for tool in manifest.tools),
            package.agent_context or "",
        )
    )
    query_tokens = _skill_tokens(query)
    skill_tokens = _skill_tokens(skill_text)
    if not query_tokens or not skill_tokens:
        return 0

    overlap = query_tokens & skill_tokens
    score = len(overlap)
    normalized_query = re.sub(r"[\s_]+", "-", query.lower())
    if manifest.name.lower() in normalized_query:
        score += 3
    for tool in manifest.tools:
        if tool.name.lower() in query.lower():
            score += 3
    return score


def best_matching_skill_package(query: str, packages: tuple[Any, ...]) -> Any | None:
    """Return the best loadable skill package for *query*, if it is a clear match.

    Prefers the embedding-based semantic router; falls back to the keyword
    scorer only when the router cannot be initialised (e.g. embedding model
    unavailable). A semantic match below ``router.threshold`` returns
    ``None`` — we don't double-fall-back to keyword on a rejected query,
    since that defeats the threshold's purpose.
    """
    router = _get_semantic_router()
    if router is not None:
        return router.best_match(query, packages)
    best_package: Any | None = None
    best_score = 0
    for package in packages:
        if not package.is_loadable:
            continue
        score = score_skill_package(query, package)
        if score > best_score:
            best_package = package
            best_score = score
    return best_package if best_score >= SKILL_MATCH_MIN_SCORE else None


def extract_skill_arguments(query: str, tool_class: type[Any]) -> dict[str, Any] | None:
    """Extract simple defaultable arguments for direct local-skill execution."""
    try:
        input_model = tool_class().get_input_schema()
    except Exception:
        logger.debug("local skill input schema lookup failed", exc_info=True)
        return None

    fields = getattr(input_model, "model_fields", {})
    arguments: dict[str, Any] = {}
    lowered = query.lower()
    for name, field_info in fields.items():
        if name == "limit":
            match = re.search(r"\b([1-9][0-9]?)\b", lowered)
            if match:
                arguments[name] = int(match.group(1))
        elif name == "since":
            if re.search(r"\b(week|weekly)\b", lowered):
                arguments[name] = "weekly"
            elif re.search(r"\b(month|monthly)\b", lowered):
                arguments[name] = "monthly"
            elif re.search(r"\b(today|daily)\b", lowered):
                arguments[name] = "daily"

        is_required = getattr(field_info, "is_required", None)
        if callable(is_required) and is_required() and name not in arguments:
            return None
    return arguments


def format_direct_skill_result(tool_name: str, result: object) -> str:
    """Format a direct local-skill result without another LLM round trip."""
    if isinstance(result, list) and all(isinstance(item, dict) for item in result):
        if not result:
            return f"Used skill `{tool_name}`.\n\nNo results returned."
        lines = [f"Used skill `{tool_name}`.", "", "Top repositories:"]
        for index, item in enumerate(result, start=1):
            repo = str(item.get("repo") or item.get("name") or f"Result {index}")
            url = str(item.get("url") or "")
            description = str(item.get("description") or "").strip()
            language = str(item.get("language") or "").strip()
            stars_today = item.get("stars_today")
            stars_total = item.get("stars_total")
            forks = item.get("forks")
            title = f"{index}. {repo}"
            if url:
                title += f" - {url}"
            lines.append(title)
            if description:
                lines.append(f"   Description: {description}")
            details = []
            if language:
                details.append(f"language: {language}")
            if stars_today is not None:
                details.append(f"stars today: {stars_today}")
            if stars_total is not None:
                details.append(f"total stars: {stars_total}")
            if forks is not None:
                details.append(f"forks: {forks}")
            if details:
                lines.append(f"   {'; '.join(details)}")
        return "\n".join(lines)

    if isinstance(result, dict | list):
        return f"Used skill `{tool_name}`.\n\n```json\n{json.dumps(result, indent=2, default=str)}\n```"
    return f"Used skill `{tool_name}`.\n\n{result}"
