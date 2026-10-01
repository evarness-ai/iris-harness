"""Deterministic task specification for code-execution and artifact tasks.

A :class:`TaskSpec` is built by code (no LLM) from the user's query. It captures
what the user actually asked for — particularly the *shape* of the deliverable
(file extensions, whether an artifact is required at all) — so the runtime can
verify model output against an objective reference instead of trusting the
model to self-report success.

This is Phase 1 of the tier-aware planner architecture:

* For every tier we *build* a spec and *verify* against it (cheap safety net).
* For small/local tiers (Phase 3) we will additionally *render the system
  prompt from the spec* instead of relying on a freeform prompt the model can
  misinterpret.
* For large/capable tiers the spec stays a verifier only; their existing rich
  system prompt continues to drive the planner.

The helpers here previously lived inline in ``iris_harness.runtime.bootstrap``; they
are extracted unchanged in behaviour so the existing planner loop keeps
working bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

__all__ = [
    "TaskSpec",
    "Verdict",
    "build_task_spec",
    "verify",
    "describe_expected_artifact",
    "expected_artifact_extensions",
    "user_visible_artifacts",
    "artifacts_satisfy_request",
    "build_missing_artifact_answer",
    "render_code_exec_system_prompt_for_small",
]


# ---------------------------------------------------------------------------
# Heuristics — single source of truth for "what counts as a document request"
# ---------------------------------------------------------------------------

_DOCUMENT_REQUEST_TERMS: tuple[str, ...] = (
    "article",
    "document",
    "one pager",
    "one-pager",
    "one page",
    "one-page",
    "report",
    "research summary",
    "write-up",
    "writeup",
)

_DOCUMENT_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".markdown", ".pdf", ".html", ".htm", ".txt"}
)


def expected_artifact_extensions(task_query: str) -> set[str]:
    """Infer deliverable extensions the user explicitly or implicitly requested.

    Returns an empty set when the request does not clearly imply a deliverable
    file (e.g. a pure conversational or calculation query).
    """
    query = f" {task_query.lower()} "
    if "pdf" in query:
        return {".pdf"}
    if "csv" in query:
        return {".csv"}
    if "excel" in query or "spreadsheet" in query or ".xlsx" in query:
        return {".xlsx", ".xls"}
    if "markdown" in query or ".md" in query:
        return {".md", ".markdown"}
    if "html" in query:
        return {".html", ".htm"}
    if any(term in query for term in _DOCUMENT_REQUEST_TERMS):
        return set(_DOCUMENT_EXTENSIONS)
    return set()


def describe_expected_artifact(task_query: str) -> str:
    """Friendly, user-facing description of the deliverable shape."""
    expected = expected_artifact_extensions(task_query)
    if not expected:
        return "the requested output"
    if expected == {".pdf"}:
        return "a PDF artifact"
    if expected == {".csv"}:
        return "a CSV artifact"
    if expected == {".xlsx", ".xls"}:
        return "an Excel artifact"
    if expected == {".md", ".markdown"}:
        return "a Markdown document artifact"
    if expected == {".html", ".htm"}:
        return "an HTML artifact"
    return "a document artifact (.md, .pdf, .html, or .txt)"


def _artifact_matches_expected(path: str, expected: set[str]) -> bool:
    if not expected:
        return True
    return Path(path).suffix.lower() in expected


def artifacts_satisfy_request(task_query: str, artifacts: list[str]) -> bool:
    """True when at least one artifact matches the inferred deliverable shape."""
    expected = expected_artifact_extensions(task_query)
    if not expected:
        return True
    return any(_artifact_matches_expected(path, expected) for path in artifacts)


def user_visible_artifacts(task_query: str, artifacts: list[str]) -> list[str]:
    """Hide implementation-only files when the user asked for a deliverable.

    Helper scripts (``script.py`` for a "create a one-page article" request)
    are not what the user wanted; they are means, not ends. We surface only
    files whose extension matches the inferred deliverable shape.
    """
    unique = list(dict.fromkeys(artifacts))
    expected = expected_artifact_extensions(task_query)
    if not expected:
        return unique
    return [path for path in unique if _artifact_matches_expected(path, expected)]


def build_missing_artifact_answer(
    *,
    task_query: str,
    artifacts: list[str],
    workspace_path: str,
    iterations: int,
) -> str:
    """User-facing answer when execution never produced the requested file."""
    expected = describe_expected_artifact(task_query)
    lines = [
        "Stopped before completing the requested output.",
        f"Asked: {task_query}",
        f"Missing: {expected} was not created.",
        "What happened: the sandbox only produced intermediate files or output that "
        "does not match the requested deliverable.",
        f"Tool loop: {iterations} iteration(s).",
        f"Workspace: {workspace_path}",
        "Use /trace to inspect the internal tool calls and intermediate files.",
    ]
    visible = user_visible_artifacts(task_query, artifacts)
    if visible:
        names = ", ".join(Path(path).name for path in visible)
        lines.append(f"Artifacts ready: {names}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Spec + verifier API (Phase 1 surface, used by Phase 2-3 tier branching)
# ---------------------------------------------------------------------------


class Verdict(str, Enum):
    """Outcome of verifying produced artifacts against a :class:`TaskSpec`."""

    SATISFIED = "satisfied"
    MISSING_ARTIFACT = "missing_artifact"


@dataclass(frozen=True)
class TaskSpec:
    """Deterministic, code-built description of what the user asked for.

    Phase 1 only models the deliverable shape — that is enough to subsume the
    existing inline helpers. Later phases will extend this with allowed tools,
    ``ask_user`` policy, default values applied, and an explicit task type.
    """

    query: str
    required_extensions: frozenset[str]
    expected_description: str

    @property
    def must_create_artifact(self) -> bool:
        return bool(self.required_extensions)


def build_task_spec(query: str) -> TaskSpec:
    """Build a :class:`TaskSpec` from the raw user query (deterministic, no LLM)."""
    extensions = frozenset(expected_artifact_extensions(query))
    return TaskSpec(
        query=query,
        required_extensions=extensions,
        expected_description=describe_expected_artifact(query),
    )


def verify(spec: TaskSpec, artifacts: list[str]) -> Verdict:
    """Check produced artifacts against the spec.

    Returns :attr:`Verdict.SATISFIED` when the spec imposes no artifact
    requirement *or* at least one artifact has a matching extension; otherwise
    :attr:`Verdict.MISSING_ARTIFACT`.
    """
    if not spec.required_extensions:
        return Verdict.SATISFIED
    expected = set(spec.required_extensions)
    for path in artifacts:
        if _artifact_matches_expected(path, expected):
            return Verdict.SATISFIED
    return Verdict.MISSING_ARTIFACT


# ---------------------------------------------------------------------------
# Tier-aware prompt rendering — Phase 3
# ---------------------------------------------------------------------------


def _format_required_extensions(spec: TaskSpec) -> str:
    """Comma-separated, deterministic ordering for prompt rendering."""
    return ", ".join(sorted(spec.required_extensions))


def render_code_exec_system_prompt_for_small(spec: TaskSpec) -> str:
    """Build a tight, slot-filled system prompt for SMALL-tier models.

    SMALL-tier models (local non-reasoning) need a deterministic prompt with
    minimal prose — they tend to drift, narrate, or lose the JSON contract
    when given the rich freeform prompt MID/LARGE tiers receive. We render
    only the essentials: tool schema, the exact JSON shape, the required
    deliverable extension(s) (when any), and the strictest do/don't rules.
    """
    deliverable_block: str
    if spec.must_create_artifact:
        exts = _format_required_extensions(spec)
        deliverable_block = (
            "REQUIRED DELIVERABLE\n"
            f"  - Task: {spec.query}\n"
            f"  - Output: {spec.expected_description}\n"
            f"  - Allowed extensions: {exts}\n"
            "  - You MUST end with a real file in /workspace whose extension\n"
            "    matches one of those. A helper script alone is NOT completion.\n"
        )
    else:
        deliverable_block = (
            "TASK\n"
            f"  - {spec.query}\n"
            "  - No specific file deliverable is required; complete the task\n"
            "    and report results in the final prose answer.\n"
        )

    return (
        "You are IRIS code-execution agent. Solve the user's request by emitting\n"
        "JSON tool calls inside a sandboxed Docker container.\n"
        "\n"
        "TOOL — run_shell(cmd: string, timeout?: int)\n"
        "  - Runs cmd via `bash -lc` in /workspace (Python 3.12 + reportlab,\n"
        "    pandas, requests, beautifulsoup4, pypdf, openpyxl pre-installed).\n"
        "  - Files in /workspace persist across calls in this session.\n"
        "  - NEVER run apt/yum/brew. NEVER use `python -c` for multi-line code.\n"
        "    Always write code via heredoc, then run it:\n"
        "      cat > /workspace/script.py << 'PYEOF'\\n<code>\\nPYEOF\\npython /workspace/script.py\n"
        "  - Default timeout 30s, max 300s.\n"
        "  - ask_user is disabled for SMALL-tier models. Choose reasonable\n"
        "    defaults instead of pausing for clarification.\n"
        "\n"
        f"{deliverable_block}"
        "\n"
        "RESPONSE FORMAT — every turn, respond with EXACTLY ONE of:\n"
        "  (A) one JSON tool call on its own (no fences, no prose):\n"
        '      {"tool":"run_shell","args":{"cmd":"...","timeout":30},'
        '"progress":"short status"}\n'
        "  (B) a final prose answer summarising what you did.\n"
        "      Only emit (B) AFTER the required file (if any) exists in /workspace.\n"
        "\n"
        "RULES\n"
        "  - If a tool call fails, READ the error and FIX the script. Retry with\n"
        "    a corrected call. NEVER repeat an identical failing command.\n"
        "  - For PDF use reportlab. For CSV use pandas. For Excel use openpyxl.\n"
        "    For Markdown write the .md file directly.\n"
        "  - Maximum 8 tool calls per request.\n"
    )
