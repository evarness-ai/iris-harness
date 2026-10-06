"""The sandboxed ``code_exec`` agent, lifted out of ``bootstrap.py`` (OSS plan M4.6).

A bounded LLM ↔ sandbox loop: the model emits a ``run_shell`` JSON tool call, the
sandbox runs it, the result is fed back, and the cycle repeats until the model
emits a prose final answer or the iteration cap is hit. Each session gets its own
sandbox host so files persist across turns.

Verbatim from the runtime except for its imports: everything it reaches for —
``core.task_spec``, ``llm.client``, ``sandbox``, ``tools.sandbox_tools``,
the tool-call parsers — is core and stays core, so this plugin imports no runtime
internals (release gate 2). The two helpers it borrowed from ``bootstrap``'s module
scope (``_friendly_llm_error``, ``_safe_int_env``) came with it.
"""

from __future__ import annotations

import contextvars
import dataclasses
import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

from iris_harness.sdk.content import redact_external_content, wrap_external_content
from iris_harness.sdk.llm import friendly_llm_error as _friendly_llm_error
from iris_harness.sdk.logging import agent_scope, log_tool_run
from iris_harness.sdk.parsing import (
    _PROSE_CUTOFF_LOOKBACK,
    _find_degenerate_repetition,
    _find_first_cutoff,
    _looks_truncated_tool_call,
    _one_line_preview,
    _parse_tool_call,
    _sanitize_prose,
)
from iris_harness.sdk.services import LessonService, TierRouterService
from iris_harness.sdk.types import (
    ActivityChunk,
    AgentTask,
    HandlerResult,
    StreamChunk,
    TraceChunk,
    Verdict,
)
from iris_harness.sdk.types import (
    build_missing_artifact_answer as _build_missing_artifact_answer,
)
from iris_harness.sdk.types import (
    build_task_spec as _build_task_spec,
)
from iris_harness.sdk.types import (
    describe_expected_artifact as _describe_expected_artifact,
)
from iris_harness.sdk.types import (
    render_code_exec_system_prompt_for_small as _render_code_exec_prompt_small,
)
from iris_harness.sdk.types import (
    user_visible_artifacts as _user_visible_artifacts,
)
from iris_harness.sdk.types import (
    verify as _verify_task_spec,
)

logger = logging.getLogger(__name__)


#: Matching calls per run that go through the SDK helper (one counts-only ledger row each).
_AUDITED_MATCHES_PER_RUN = 3
_MEMO_MAX_CHARS = 4096
_ENV_TAG = "external_content"


class _RunRedaction:
    """Per-run bookkeeping so one hostile run writes a bounded number of ledger rows.

    ``redact_external_content`` writes one row per matching call and this plugin calls it
    per streamed line, hint and trace line. The first ``_AUDITED_MATCHES_PER_RUN`` matches
    of a run go through it (so the ledger shows the run was hostile, with counts); later
    matches are redacted by the same scan without a row and only counted. Identical texts
    are memoised. What is redacted is the same either way.
    """

    def __init__(self) -> None:
        self.audited = 0
        self.quiet_hits = 0
        self.memo: dict[str, str] = {}


_RUN: contextvars.ContextVar[_RunRedaction | None] = contextvars.ContextVar(
    "code_exec_run_redaction", default=None
)


def _begin_run() -> None:
    """Start a fresh budget; called at the top of every handler run."""
    _RUN.set(_RunRedaction())


def _quiet_scan(text: str) -> str:
    """The SDK's tripwire without a ledger row: ``wrap_external_content`` minus its envelope."""
    wrapped = wrap_external_content(text, source="code_exec", tool="redact")
    body = wrapped.split("\n", 1)[1].rsplit(f"\n</{_ENV_TAG}>", 1)[0]
    return body


def _redact_owner_text(text: str) -> str:
    """The ONE entry point for scanning text this plugin shows the owner or logs.

    Every owner-facing site goes through here (streamed prose, the answer, activity hints,
    trace lines, the logged command, artifact names), so what scans it can change in one
    place. It is the SDK's tripwire without the envelope, which honours the floor setting;
    per run the ledger rows are bounded (see :class:`_RunRedaction`).
    """
    if not text:
        return text
    run = _RUN.get()
    if run is None:
        run = _RunRedaction()
        _RUN.set(run)
    if text in run.memo:
        return run.memo[text]
    # The quiet path only for text it reproduces exactly: a literal envelope tag in the
    # text would be escaped or unwrapped by the wrapper, so that text always takes the
    # audited path.
    if run.audited < _AUDITED_MATCHES_PER_RUN or _ENV_TAG in text:
        out = redact_external_content(text)
        if out is not text:
            run.audited += 1
    else:
        out = _quiet_scan(text)
        if out != text:
            run.quiet_hits += 1
            if run.quiet_hits == 1:
                logger.warning(
                    "code_exec: further instruction-like spans redacted this run "
                    "(ledger rows capped at %d per run)",
                    _AUDITED_MATCHES_PER_RUN,
                )
    if len(text) <= _MEMO_MAX_CHARS and len(run.memo) < 1024:
        run.memo[text] = out
    return out


#: Complete lines kept back from emission, so a phrase split across up to two line breaks
#: is scanned whole before any of it leaves.
_HOLD_LINES = 2
#: A buffer with no line break is force-scanned and flushed at this size.
_FLUSH_CAP = 64 * 1024
#: The tail kept (unemitted) after a forced flush, so a phrase straddling the cut is caught.
_OVERLAP_CHARS = 2 * 1024


class _ProseRedactor:
    """The tripwire over prose that is streamed live.

    The planner's final prose reaches the owner as it is generated, and the final tuple
    then carries nothing (the answer was already streamed), so the answer-level redaction
    cannot see it. A phrase can straddle chunks AND line breaks, so text is scanned with a
    sliding overlap: the last ``_HOLD_LINES`` complete lines are held back from emission
    and rescanned together with each new line (what was already emitted is never touched),
    so a phrase split across up to two line breaks is caught. A line with no break is
    force-scanned at ``_FLUSH_CAP`` bytes, keeping the last ``_OVERLAP_CHARS`` as context.
    :meth:`flush` scans and emits what is left.

    The cost is latency (the owner sees text two lines behind) and a phrase split across
    more than two line breaks, or one longer than the overlap, is not caught by this layer.
    Feeding is linear: only the new chunk is searched for a line break.
    """

    def __init__(self) -> None:
        self._held = ""  # scanned, unemitted text
        self._partial: list[str] = []  # the current unterminated line, as pieces
        self._partial_len = 0

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        cut = chunk.rfind("\n")
        if cut < 0:
            self._partial.append(chunk)
            self._partial_len += len(chunk)
            return self._force() if self._partial_len >= _FLUSH_CAP else ""
        tail = chunk[cut + 1 :]
        text = "".join(self._partial) + chunk[: cut + 1]
        self._partial = [tail] if tail else []
        self._partial_len = len(tail)
        out = ""
        for line in text.split("\n")[:-1]:
            self._held = _redact_owner_text(self._held + line + "\n")
            out += self._emit()
        return out + (self._force() if self._partial_len >= _FLUSH_CAP else "")

    def flush(self) -> str:
        rest = _redact_owner_text(self._held + "".join(self._partial))
        self._held, self._partial, self._partial_len = "", [], 0
        return rest

    def _emit(self) -> str:
        """Emit everything but the last ``_HOLD_LINES`` lines (or the overlap, if huge)."""
        held = self._held
        idx, keep = len(held) - 1, 0
        for _ in range(_HOLD_LINES):
            idx = held.rfind("\n", 0, idx)
            if idx < 0:
                keep = 0
                break
            keep = idx + 1
        if len(held) - keep > _FLUSH_CAP:
            keep = len(held) - _OVERLAP_CHARS
        out, self._held = held[:keep], held[keep:]
        return out

    def _force(self) -> str:
        """Scan a long unterminated buffer and emit all but the overlap tail."""
        self._held = _redact_owner_text(self._held + "".join(self._partial))
        self._partial, self._partial_len = [], 0
        keep = max(0, len(self._held) - _OVERLAP_CHARS)
        out, self._held = self._held[:keep], self._held[keep:]
        return out


def _safe_int_env(name: str, default: int, *, min_value: int = 1, max_value: int = 128) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value))


_CODE_EXEC_SYSTEM_PROMPT = (
    "You are IRIS Code Execution agent. You solve the user's request by running\n"
    "shell commands inside a sandboxed Docker container.\n"
    "You DO have access to the run_shell tool in this mode. Never claim you\n"
    "lack tool access or sandbox capability.\n"
    "\n"
    "TOOL — run_shell(cmd: string, timeout?: int)\n"
    "  • Runs cmd via `bash -lc` in /workspace.\n"
    "  • Container has Python 3.12, git, curl, jq, pandoc, plus these Python\n"
    "    libraries pre-installed: reportlab weasyprint markdown pandas numpy\n"
    "    matplotlib pypdf pillow openpyxl beautifulsoup4 requests httpx lxml\n"
    "    feedparser gnews.\n"
    "  • Files written to /workspace/<name> persist across calls in this\n"
    "    session and surface back as artifact paths.\n"
    "  • IMPORTANT: each run_shell call is a FRESH container — pip installs do NOT\n"
    "    persist. If you must install a missing package, do it in the SAME call as\n"
    "    the script: `pip install <pkg> -q && python /workspace/script.py`\n"
    "  • PREFER pre-installed packages. For PDF use reportlab (NOT fpdf/fpdf2,\n"
    "    NOT a `pdfkit` shell command — there is no such CLI in the sandbox).\n"
    "    For HTTP use requests or httpx. For HTML use beautifulsoup4. For feeds\n"
    "    use feedparser. For news use gnews. Avoid unnecessary pip installs.\n"
    "  • NEVER run `apt-get`, `apt`, `yum`, `dpkg`, `brew`, or any OS package\n"
    "    manager — the sandbox image is fixed and cannot install OS packages.\n"
    "    If a binary is missing, switch to a Python library instead.\n"
    "  • Default timeout 30s, max 300s.\n"
    "\n"
    "TOOL — ask_user(question: string)\n"
    "  • Ask exactly one concise clarification question when a missing\n"
    "    user-specific detail blocks safe completion. Use this rarely.\n"
    "  • Do not use ask_user for format defaults, filenames, or implementation\n"
    "    choices; make reasonable defaults and continue with run_shell.\n"
    "  • This tool is allowed only when the tier policy permits it.\n"
    "\n"
    "TOOL INVOCATION INTENTS\n"
    "  • If the user asks to use sandbox, run a script, execute code, generate\n"
    "    a file/artifact (PDF/CSV/Excel/image/chart), or fetch/process live data,\n"
    "    you MUST respond with a run_shell JSON tool call first.\n"
    "  • Do not ask for confirmation when requirements are already clear; start\n"
    "    with a tool call and iterate until done.\n"
    "  • Only return prose final answer after successful execution or after\n"
    "    hitting the max call limit.\n"
    "  • If prior turns already produced an artifact and the user asks to\n"
    "    tweak format/layout/style/content, treat it as a continuation task\n"
    "    and immediately issue run_shell calls to revise and regenerate.\n"
    "\n"
    "WRITING PYTHON SCRIPTS\n"
    "  NEVER use `python -c '...'` for anything beyond a single expression.\n"
    "  ALWAYS write multi-line code to a file with a heredoc, then run it:\n"
    "    cmd: \"cat > /workspace/script.py << 'PYEOF'\\n<code>\\nPYEOF\\npython /workspace/script.py\"\n"
    "  This avoids all quoting and indentation problems that cause SyntaxError.\n"
    "  A helper script is only an implementation detail. If the user asked for\n"
    "  a document/report/article/one-pager/research summary, the final artifact\n"
    "  must be a document file such as .md, .pdf, .html, or .txt — never only\n"
    "  script.py. When no specific format is requested, prefer a polished .md\n"
    "  file with the requested content.\n"
    "\n"
    "PDF GENERATION TEMPLATE — when the user asks for a PDF, copy this pattern\n"
    "(adapt the `items` list and title to the actual content):\n"
    "    cat > /workspace/script.py << 'PYEOF'\n"
    "    from reportlab.lib.pagesizes import letter\n"
    "    from reportlab.lib.styles import getSampleStyleSheet\n"
    "    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer\n"
    "    items = ['line 1', 'line 2', 'line 3']\n"
    "    doc = SimpleDocTemplate('/workspace/output.pdf', pagesize=letter)\n"
    "    styles = getSampleStyleSheet()\n"
    "    flow = [Paragraph('Title', styles['Title']), Spacer(1, 12)]\n"
    "    flow += [Paragraph(line, styles['BodyText']) for line in items]\n"
    "    doc.build(flow)\n"
    "    PYEOF\n"
    "    python /workspace/script.py\n"
    "  The artifact MUST be a real .pdf written by reportlab — never a .txt file\n"
    "  renamed to .pdf, never a plain `cat > out.txt` substitute.\n"
    "\n"
    "RESPONSE FORMAT — every turn, respond with EXACTLY ONE of:\n"
    "\n"
    "(A) A tool call as a single JSON object on its own. No prose, no fences.\n"
    '    {"tool": "run_shell", "args": {"cmd": "...", "timeout": 30}, '
    '"progress": "writing the report and verifying the artifact"}\n'
    '    {"tool": "ask_user", "args": {"question": "..."}, '
    '"progress": "checking one missing requirement"}\n'
    "    The optional progress field is a short status line shown to the user\n"
    "    while the tool runs. Use it to summarize what this call is doing and\n"
    "    the immediate outcome you expect. Do not include secrets.\n"
    "\n"
    "(B) A final answer in prose summarising what you did and where the output\n"
    "    files are. Do NOT include any JSON in a final answer.\n"
    "\n"
    "RULES\n"
    "  • Plan tightly: write the code, run it, verify, report. Don't side-quest.\n"
    "  • Reference artifact paths by their absolute host path when reporting.\n"
    "  • If a tool call fails, READ the error output and FIX the code — keep\n"
    "    retrying with corrected tool calls until the task succeeds or you\n"
    "    reach the call limit. NEVER respond with prose if there are calls left.\n"
    "  • NEVER reissue an identical or near-identical command after it failed.\n"
    "    Diagnose the error and try a *different* approach (different library,\n"
    "    different file path, different invocation). Repeating the same failing\n"
    "    command wastes the call budget.\n"
    "  • If prior conversation context (PRIOR CONVERSATION block) references\n"
    "    content from earlier turns (a list, a result, an answer), treat that\n"
    "    content as the source-of-truth input. Do NOT fabricate placeholder\n"
    '    text like "Hello, World!" — copy the actual content into your script.\n'
    '  • OUTPUT FORMAT MUST MATCH THE USER\'S REQUEST. If they say "PDF",\n'
    "    the artifact extension MUST be .pdf and the file MUST be a real PDF\n"
    "    (use reportlab — see PDF GENERATION TEMPLATE above). If they say\n"
    '    "CSV", write .csv via pandas. If they say "Excel", write .xlsx via\n'
    "    openpyxl. NEVER fall back to writing a plain .txt file when a\n"
    "    structured format was requested — that is a task failure.\n"
    "  • For article, report, research-summary, one-pager, or document requests\n"
    "    without an explicit file type, create a real .md document artifact.\n"
    "    Do not treat a script that could generate the document as completion.\n"
    "  • Maximum 8 tool calls per request.\n"
    "\n"
    "RUN COMPLETION (final-answer turns only)\n"
    "  When emitting a final prose answer for a successful run, append a\n"
    "  fenced ```lesson JSON block AFTER your prose so future runs can learn\n"
    "  from this one. Format:\n"
    "    ```lesson\n"
    '    {"category": "<short-slug>", "summary": "<1-3 sentences on what worked>",\n'
    '     "tools": ["run_shell"], "sources": ["url1", "url2"],\n'
    '     "scripts": ["script_name.py"]}\n'
    "    ```\n"
    '  • category: e.g. "news-fetch-pdf", "data-cleanup-csv".\n'
    "  • summary: what approach succeeded — strategy, key library/flags.\n"
    "  • sources: data URLs/feeds you fetched (NOT secrets).\n"
    "  • scripts: filenames you wrote in /workspace.\n"
    '  • tools: subset of ["run_shell", "ask_user"]. Omit fields that don\'t apply.\n'
    "  • Do NOT add a lesson block on failure or partial runs.\n"
)


# Tool-call + prose parsing utils were extracted to
# iris_harness.runtime.tool_call_parsing (Phase 2) and re-exported at the top.


def _build_code_exec_handoff_answer(
    *,
    task_query: str,
    artifacts: list[str],
    workspace_path: str,
    last_result_summary: str,
    iterations: int,
    warning: str = "",
    last_stdout: str = "",
) -> str:
    """Build a final handoff when the planner fails to emit one itself."""
    unique_artifacts = _user_visible_artifacts(task_query, artifacts)
    result_preview = _one_line_preview(last_stdout) if last_stdout else ""
    lines: list[str]
    if warning:
        lines = [
            "Stopped after sandbox execution produced artifacts, but the last "
            "command reported a shell warning. I am not treating this as a clean completion.",
            f"Asked: {task_query}",
            "Achieved: The sandbox created or updated artifact files, but they should be "
            "reviewed because the final shell run wrote to stderr.",
            f"Warning: {_one_line_preview(warning, limit=260)}",
        ]
    elif result_preview and not unique_artifacts:
        # Compute/answer task with no file deliverable: the sandbox result IS
        # the answer, so lead with it instead of burying it behind a generic
        # handoff notice (exp-006 GAP-11 — silent result loss).
        lines = [
            f"Result: {result_preview}",
            f"Asked: {task_query}",
            "Achieved: Ran the sandbox computation; the result above is the answer.",
        ]
    else:
        lines = [
            "Completed the sandbox work; the planner did not emit its own final prose, "
            "so I am handing off the result directly.",
            f"Asked: {task_query}",
            "Achieved: Generated or updated the requested sandbox output and verified "
            "the shell command completed cleanly.",
        ]
    lines.append(f"Tool loop: {iterations} iteration(s).")
    lines.append(f"Workspace: {workspace_path}")
    if unique_artifacts:
        names = ", ".join(Path(path).name for path in unique_artifacts)
        lines.append(f"Artifacts ready: {names}")
    # Never silently drop the computed result: when it is not already the
    # headline (artifact/warning paths), surface it; fall back to the raw
    # summary only when no clean stdout was captured.
    if result_preview and (unique_artifacts or warning):
        lines.append(f"Result: {result_preview}")
    elif not result_preview and last_result_summary:
        lines.append(f"Last result: {_one_line_preview(last_result_summary)}")
    return "\n".join(lines)


_CODE_EXEC_MAX_ITERATIONS = 8
_CODE_EXEC_MAX_ITERATIONS_CAP = 14
_CODE_EXEC_INVALID_TOOL_CALL_LIMIT = 2
# A tool call cut off mid-JSON is a different failure from a planner that
# ignored the protocol, so it gets its own (slightly roomier) budget: retrying
# with "make the command shorter" is a genuinely different attempt, and it must
# not consume the tighter invalid-protocol budget above.
_CODE_EXEC_TRUNCATED_TOOL_CALL_LIMIT = 3
# How many streamed chars to accumulate between degenerate-repetition checks.
_REPETITION_CHECK_INTERVAL = 512


def _code_exec_budget(task: AgentTask) -> tuple[int, int]:
    """Return ``(base_budget, cap_budget)`` for code_exec loops."""
    base = _safe_int_env("IRIS_CODE_EXEC_MAX_ITERATIONS", _CODE_EXEC_MAX_ITERATIONS)
    cap = _safe_int_env("IRIS_CODE_EXEC_MAX_ITERATIONS_CAP", _CODE_EXEC_MAX_ITERATIONS_CAP)
    if cap < base:
        cap = base
    lowered = task.query.lower()
    continuationish = bool(re.search(r"\b(yes|do that|continue|update|refine|reformat)\b", lowered))
    if continuationish:
        base = min(cap, base + 1)
    return base, cap


# ---------------------------------------------------------------------------
# Skill-proposal detection helpers (code_exec post-loop hook)
# ---------------------------------------------------------------------------

_SAVE_AS_SKILL_RE = re.compile(
    r"\b(save|store|remember|keep|propose|draft|register)\b.{0,80}"
    r"\b(as\s+(a\s+)?(draft\s+|reusable\s+)?skill|as\s+skill|to\s+skill\s+queue)\b",
    re.IGNORECASE,
)

_SKILL_SLUG_RE = re.compile(
    r"\b(call\s+it|name\s+it|named|slug|use\s+slug|with\s+slug)\s+['\"]?([a-zA-Z0-9][a-zA-Z0-9_-]*)['\"]?",
    re.IGNORECASE,
)


def _extract_skill_request(query: str) -> tuple[bool, str | None]:
    """Return ``(wants_skill_proposal, slug_or_None)`` parsed from *query*.

    Returns ``(True, slug)`` or ``(True, None)`` when the query asks to save
    the script as a draft skill (slug may or may not be specified), or
    ``(False, None)`` otherwise.
    """
    if not _SAVE_AS_SKILL_RE.search(query):
        return False, None
    m = _SKILL_SLUG_RE.search(query)
    slug = m.group(2) if m else None
    return True, slug


def _is_docker_available() -> bool:
    """Return True only when the Docker daemon is reachable."""
    docker_exe = shutil.which("docker")
    if docker_exe is None:
        return False
    try:
        proc = subprocess.run(  # noqa: S603
            [docker_exe, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return proc.returncode == 0
    except Exception:  # noqa: BLE001 — a probe that cannot run means unavailable
        return False


def _make_code_exec_handler(
    tier_router: TierRouterService,
    lesson_capture: LessonService | None = None,
    *,
    repo_root: Path | None = None,
) -> tuple[Callable[[AgentTask], HandlerResult], Callable[[AgentTask], Iterator[StreamChunk]]]:
    """Build the sync + streaming handler pair for the ``code_exec`` agent.

    Runs a bounded LLM ↔ sandbox loop: the LLM emits a ``run_shell`` JSON tool
    call, the sandbox runs it, the result is fed back, and the cycle repeats
    until the LLM emits a prose final answer or the iteration cap is hit.

    Each session gets its own sandbox host so files persist across turns.
    """
    from iris_harness.sdk.llm import (
        CodingLLMClient,
        CodingLLMConfig,
        ModelTier,
        model_tier_for,
    )
    from iris_harness.sdk.sandbox import (
        SandboxConfig,
        SandboxToolHost,
        resolve_runtime_name,
    )

    # Phase 6 (6c.2): resolve the sandbox runtime once at startup (default docker;
    # gVisor where runsc is available, else fall back to docker with a warning).
    _sandbox_cfg = SandboxConfig.default()
    if repo_root is not None:
        _sandbox_cfg = SandboxConfig.from_yaml(repo_root / "config" / "governance" / "sandbox.yaml")
    _runtime_name = resolve_runtime_name(_sandbox_cfg) or "docker"
    if _runtime_name != "docker":
        logger.info("code_exec sandbox runtime: %s", _runtime_name)

    _hosts: dict[str, SandboxToolHost] = {}
    _clients: dict[str, CodingLLMClient] = {}

    def _get_host(session_id: str | None) -> SandboxToolHost:
        key = session_id or "default"
        if key not in _hosts:
            _hosts[key] = SandboxToolHost(key, config=_sandbox_cfg, runtime_name=_runtime_name)
        return _hosts[key]

    def _get_client() -> CodingLLMClient:
        # code_exec is pinned to its YAML tier — the global `/provider` choice
        # is *not* honored here. Why: the planner emits strict JSON tool calls
        # for `run_shell`, and small generalist models (e.g. Gemma via LM
        # Studio) regularly produce malformed JSON that aborts the loop. The
        # YAML `code_exec` tier is chosen for reliable JSON tool calling.
        cache_key = "code_exec"
        if cache_key not in _clients:
            cfg = CodingLLMConfig(**vars(tier_router.get_llm_config("code_exec")))
            cfg = cfg.model_copy(update={"temperature": 0.2, "max_tokens": 2048})
            _clients[cache_key] = CodingLLMClient(cfg)
        return _clients[cache_key]

    def _run_loop_gen(
        task: AgentTask,
    ) -> Iterator[StreamChunk]:
        """Generator: yields narration str chunks as they happen, then (answer, meta) tuple last.

        Streams planner LLM output token-by-token. JSON tool calls are silently
        buffered (the user sees the resulting ``▸ run_shell`` status line instead).
        Prose final answers are flushed live so the user sees progressive output
        instead of a long blocking spinner. When prose is streamed live, the
        final tuple yields an empty answer string so the stream handler does
        not double-emit it; the sync handler concatenates narration + answer.
        """
        client = _get_client()
        host = _get_host(task.session_id)

        # Tier-aware spec rendering: SMALL models get a tight slot-filled prompt
        # built from the deterministic TaskSpec; MID/LARGE keep the rich
        # freeform prompt and the spec is used only as a post-hoc verifier via
        # `verify(spec, all_artifacts)`.
        spec = _build_task_spec(task.query)
        code_exec_cfg = CodingLLMConfig(**vars(tier_router.get_llm_config("code_exec")))
        resolved_model = code_exec_cfg.model
        resolved_provider = code_exec_cfg.provider
        model_tier = model_tier_for(resolved_model, resolved_provider)
        if model_tier == ModelTier.SMALL:
            code_exec_system_prompt = _render_code_exec_prompt_small(spec)
        else:
            code_exec_system_prompt = _CODE_EXEC_SYSTEM_PROMPT

        transcript: list[str] = []

        # Carry recent conversation turns so follow-ups like "create a pdf out
        # of it" can resolve "it" to the previous assistant answer instead of
        # fabricating placeholder content.
        ctx = task.memory_context
        if ctx and ctx.recent_turns:
            history_text = "\n".join(ctx.recent_turns[-6:])
            transcript.append(
                "PRIOR CONVERSATION (most recent last) — use as source-of-truth "
                'input for any references like "it", "that", "the list":\n'
                f"{history_text}"
            )

        transcript.append(f"USER REQUEST:\n{task.query}")

        # Prepend prior lessons so the planner can reuse known strategies.
        if lesson_capture is not None:
            try:
                prior = lesson_capture.find_similar(task.query)
                rendered = lesson_capture.render_prior_lessons(prior)
                if rendered:
                    transcript.insert(0, rendered)
            except Exception:
                logger.exception("lesson_capture.find_similar failed; continuing")

        all_artifacts: list[str] = []
        iterations = 0
        final_answer = ""
        last_result_summary = ""
        last_stdout = ""
        last_exit_code: int | None = None
        last_stderr = ""
        prose_was_streamed = False
        # Track (normalized_cmd, exit_code) per call so we can detect a planner
        # stuck on the same failing command and break out instead of burning
        # the whole call budget.
        attempt_history: list[tuple[str, int, tuple[str, ...], str]] = []
        invalid_tool_call_turns = 0
        truncated_tool_call_turns = 0
        ask_user_denied_turns = 0

        budget, budget_cap = _code_exec_budget(task)
        allow_budget_growth = bool(
            re.search(
                r"\b(yes|do that|continue|again|update|refine|reformat|token|\.env)\b",
                task.query.lower(),
            )
        )
        i = 0
        while i < budget:
            iterations = i + 1
            i += 1
            user_prompt = (
                "\n\n".join(transcript)
                + "\n\nRespond with EITHER a JSON tool call OR a final prose answer."
            )
            raw_parts: list[str] = []
            # streaming_prose: None=undecided, True=prose (yield live), False=JSON (buffer silently)
            streaming_prose: bool | None = None
            # When prose contains an internal-marker / inline-tool-call cutoff
            # we hide everything from that point onward. Hold a small tail
            # buffer so we can detect a marker that crosses chunk boundaries.
            held_tail = ""
            cutoff_seen = False
            # Set when the planner collapses into a repetition loop. Abandoning
            # the stream there saves the rest of the generation budget, which
            # would otherwise be spent emitting the same line until the token
            # cap truncates the tool call mid-JSON.
            repetition_unit: str | None = None
            chars_since_repetition_check = 0
            try:
                with agent_scope("code_exec", iterations):
                    for chunk in client.invoke_stream(
                        system_prompt=code_exec_system_prompt,
                        user_prompt=user_prompt,
                    ):
                        raw_parts.append(chunk)
                        chars_since_repetition_check += len(chunk)
                        if chars_since_repetition_check >= _REPETITION_CHECK_INTERVAL:
                            chars_since_repetition_check = 0
                            repetition_unit = _find_degenerate_repetition("".join(raw_parts))
                            if repetition_unit is not None:
                                break
                        if streaming_prose is None:
                            joined = "".join(raw_parts).lstrip()
                            if not joined:
                                continue
                            # JSON tool call always begins with `{` (or a ```json fence).
                            # Anything else is prose — flush buffer and stream live.
                            first = joined[0]
                            if first == "{" or joined.startswith("```"):
                                streaming_prose = False
                            else:
                                streaming_prose = True
                                held_tail = joined  # buffer the start; flush via tail logic below
                        elif streaming_prose:
                            if cutoff_seen:
                                continue
                            held_tail += chunk

                        if streaming_prose and not cutoff_seen:
                            marker_idx, drop_all = _find_first_cutoff(held_tail.lower())
                            if marker_idx != -1:
                                if not drop_all:
                                    # Final-answer marker (e.g. ```lesson) — keep
                                    # prose preceding it.
                                    safe = held_tail[:marker_idx].rstrip()
                                    if safe:
                                        yield safe
                                # drop_all → suppress everything; user already saw
                                # whatever flushed before the marker arrived. The
                                # large lookback exists so this rarely happens.
                                held_tail = ""
                                cutoff_seen = True
                            elif len(held_tail) > _PROSE_CUTOFF_LOOKBACK:
                                # Hold back enough chars to potentially form a marker.
                                flush = held_tail[:-_PROSE_CUTOFF_LOOKBACK]
                                held_tail = held_tail[-_PROSE_CUTOFF_LOOKBACK:]
                                if flush:
                                    yield flush
            except Exception as exc:
                logger.exception("code_exec planner LLM call failed")
                partial_raw = "".join(raw_parts)
                if partial_raw.strip():
                    yield TraceChunk(f"[planner iter={iterations} ERRORED]\n{partial_raw.strip()}")
                final_answer = _friendly_llm_error(exc)
                break
            raw = "".join(raw_parts)
            tool_call = _parse_tool_call(raw)
            final_answer_candidate = _sanitize_prose(raw.strip()) if tool_call is None else ""
            request_satisfied = _verify_task_spec(spec, all_artifacts) is Verdict.SATISFIED
            final_answer_allowed = (
                bool(final_answer_candidate) and last_exit_code == 0 and request_satisfied
            )

            # Surface raw planner output for /trace. The full raw response is
            # also captured in the unified session log (llm_call event) by
            # llm_call_scope inside CodingLLMClient.invoke_stream.
            raw_stripped = raw.strip()
            if raw_stripped:
                yield TraceChunk(f"[planner iter={iterations}]\n{raw_stripped}")

            # Flush any remaining held tail. Skip the flush when this turn ends
            # in a tool call: the held prose is the planner's intermediate
            # narration ("Let me try…", "Here's a fixed version…") and should
            # never reach the user.
            if (
                streaming_prose
                and not cutoff_seen
                and held_tail
                and tool_call is None
                and final_answer_allowed
            ):
                yield held_tail
            if tool_call is None:
                if final_answer_allowed:
                    final_answer = final_answer_candidate
                    if streaming_prose and final_answer:
                        prose_was_streamed = True
                    break

                # LLM answered in prose before calling any tool — this is a
                # knowledge/advice response routed here by mistake. Accept it
                # rather than demanding run_shell and aborting.
                if final_answer_candidate and last_exit_code is None:
                    final_answer = final_answer_candidate
                    if streaming_prose and final_answer:
                        prose_was_streamed = True
                    break

                # The planner DID try to call run_shell, but the response died
                # before the JSON closed — either the token cap cut it off or
                # it collapsed into a repetition loop. Restating the protocol
                # does not help here; a shorter command does. Keep this off the
                # invalid-protocol counter so a cut-off attempt never spends
                # that budget, and don't echo the truncated garbage back into
                # the transcript (it is what overflowed the window to start).
                if repetition_unit is not None or _looks_truncated_tool_call(raw):
                    truncated_tool_call_turns += 1
                    reason = (
                        "planner repeated itself until the output budget ran out"
                        if repetition_unit is not None
                        else "planner hit its output budget mid-JSON"
                    )
                    yield ActivityChunk(
                        f"tool call cut off — {reason} "
                        f"({truncated_tool_call_turns}/{_CODE_EXEC_TRUNCATED_TOOL_CALL_LIMIT})"
                    )
                    if truncated_tool_call_turns >= _CODE_EXEC_TRUNCATED_TOOL_CALL_LIMIT:
                        final_answer = (
                            "Aborted: the planner's run_shell command was too long to finish "
                            f"inside its output budget on {truncated_tool_call_turns} "
                            "consecutive iterations, so the tool call never closed. Ask for "
                            "one smaller step, or raise max_tokens / switch the code_exec "
                            "tier to a stronger model in config/llm_tiers.yaml. "
                            "Use /trace to see the raw planner output."
                        )
                        break
                    transcript.append(
                        "PLANNER GUIDANCE: Your previous run_shell call was cut off before "
                        "the JSON closed — the command was too long to finish inside the "
                        "output budget. Do not repeat it. Emit a much shorter `cmd` that "
                        "makes one small step of progress: create the file with a minimal "
                        "body now and append to it in later calls. Never repeat the same "
                        "line twice."
                    )
                    continue

                invalid_tool_call_turns += 1
                raw_preview = " ".join(raw.strip().split())
                if len(raw_preview) > 240:
                    raw_preview = raw_preview[:237] + "..."
                if final_answer_candidate and last_exit_code == 0 and not request_satisfied:
                    expected = _describe_expected_artifact(task.query)
                    yield ActivityChunk(
                        f"{expected} missing "
                        f"({invalid_tool_call_turns}/{_CODE_EXEC_INVALID_TOOL_CALL_LIMIT})"
                    )
                    if invalid_tool_call_turns >= _CODE_EXEC_INVALID_TOOL_CALL_LIMIT:
                        final_answer = _build_missing_artifact_answer(
                            task_query=task.query,
                            artifacts=all_artifacts,
                            workspace_path=host.workspace_path,
                            iterations=iterations,
                        )
                        break
                    transcript.append(
                        "PLANNER GUIDANCE: Your previous final answer was premature. "
                        f"The user asked for {_describe_expected_artifact(task.query)}, "
                        "but no matching deliverable artifact exists yet. Do not explain "
                        "how to create it. Emit a run_shell JSON call that creates the "
                        "actual requested file. If you need a helper script, run it and "
                        "make it write the final artifact."
                    )
                    if raw_preview:
                        transcript.append(f"PREMATURE FINAL ANSWER:\n{raw_preview}")
                    continue
                yield ActivityChunk(
                    "planner returned no valid run_shell JSON "
                    f"({invalid_tool_call_turns}/{_CODE_EXEC_INVALID_TOOL_CALL_LIMIT})"
                )
                if invalid_tool_call_turns >= _CODE_EXEC_INVALID_TOOL_CALL_LIMIT:
                    final_answer = (
                        "Aborted: the planner produced no valid run_shell JSON tool call "
                        f"for {invalid_tool_call_turns} consecutive iterations. "
                        "Use /trace to see the raw planner output."
                    )
                    break

                transcript.append(
                    "PLANNER GUIDANCE: Your previous response was not a valid "
                    "run_shell JSON tool call. Retry once by emitting exactly one JSON "
                    "object with this shape and no markdown fences or prose: "
                    '{"tool":"run_shell","args":{"cmd":"...","timeout":30},'
                    '"progress":"short status"}. '
                    "If the previous tool failed, change approach instead of repeating it."
                )
                if raw_preview:
                    transcript.append(f"INVALID PLANNER RESPONSE:\n{raw_preview}")
                continue

            invalid_tool_call_turns = 0
            truncated_tool_call_turns = 0

            if tool_call.get("tool") == "ask_user":
                question = str(tool_call["question"]).strip()
                progress = str(tool_call.get("progress") or "").strip()
                if progress:
                    yield ActivityChunk(_redact_owner_text(progress))

                if model_tier == ModelTier.SMALL:
                    ask_user_denied_turns += 1
                    yield ActivityChunk(
                        "ask_user blocked by tier policy "
                        f"({ask_user_denied_turns}/{_CODE_EXEC_INVALID_TOOL_CALL_LIMIT})"
                    )
                    if ask_user_denied_turns >= _CODE_EXEC_INVALID_TOOL_CALL_LIMIT:
                        final_answer = (
                            "Aborted: the planner requested ask_user, but this "
                            "model tier is not allowed to pause for clarification. "
                            "Use /trace to see the raw planner output."
                        )
                        break
                    transcript.append(
                        "PLANNER GUIDANCE: The ask_user tool is disabled for this "
                        "model tier. Do not ask the user for clarification. Choose "
                        "reasonable defaults from the request and emit a run_shell "
                        "JSON call, or provide a final answer only if no execution "
                        "is required."
                    )
                    transcript.append(f"BLOCKED ASK_USER QUESTION:\n{question}")
                    continue

                final_answer = "I need one detail before I can continue:\n\n" + question
                break

            cmd = tool_call["cmd"]
            timeout = tool_call["timeout"]
            progress = str(tool_call.get("progress") or "").strip()

            # Pre-call activity hint — surfaced in the live spinner so the user
            # knows something is happening (replaces the dumb "Thinking..." text).
            if progress:
                yield ActivityChunk(_redact_owner_text(progress))
            else:
                cmd_hint = " ".join(_redact_owner_text(cmd).split())
                if len(cmd_hint) > 60:
                    cmd_hint = cmd_hint[:57] + "..."
                yield ActivityChunk(f"running shell ({timeout}s) - {cmd_hint}")

            # What the script printed can be derived from third-party data (a page it
            # curled, a file it downloaded). Redact it ONCE, here, before any consumer:
            # the planner transcript, the trace and session log, the activity hints and
            # the handoff answer all read this result (issue #140).
            raw_result = host.run_shell(cmd, timeout=timeout)
            # File names the script created are third-party text too (it may name a file
            # after a downloaded title): redacted here, so every later reader (the
            # answer, the artifact block, the meta, the log, the trace) gets clean names.
            result = dataclasses.replace(
                raw_result,
                stdout=_redact_owner_text(raw_result.stdout),
                stderr=_redact_owner_text(raw_result.stderr),
                artifacts=type(raw_result.artifacts)(
                    _redact_owner_text(a) for a in raw_result.artifacts
                ),
            )
            last_result_summary = result.summary()
            last_stdout = result.stdout.strip()
            last_exit_code = result.exit_code
            last_stderr = result.stderr.strip()

            # Outcome activity hint — short, ephemeral. Full detail goes to
            # the trace channel (TraceChunk) for /trace expansion only.
            if result.exit_code == 0 and not last_stderr:
                yield ActivityChunk(f"shell ok {result.duration_ms:.0f}ms")
            elif result.exit_code == 0:
                err_lines = last_stderr.splitlines()
                err_first = err_lines[-1] if err_lines else "stderr warning"
                if len(err_first) > 140:
                    err_first = err_first[:137] + "..."
                yield ActivityChunk(f"shell warning - {err_first}")
            else:
                err_lines = (result.stderr or result.stdout or "").strip().splitlines()
                err_first = err_lines[-1] if err_lines else f"exit={result.exit_code}"
                if len(err_first) > 140:
                    err_first = err_first[:137] + "..."
                yield ActivityChunk(f"shell failed exit={result.exit_code} - {err_first}")

            # Trace chunk: full raw detail buffered client-side for /trace.
            trace_lines = [
                f"$ {_redact_owner_text(cmd)}",
                f"  -> exit={result.exit_code}  duration={result.duration_ms:.0f}ms",
            ]
            if progress:
                trace_lines.insert(0, f"progress: {_redact_owner_text(progress)}")
            if result.artifacts:
                trace_lines.append(f"  -> artifacts: {', '.join(result.artifacts)}")
            if result.stdout:
                trace_lines.append("--- stdout ---")
                trace_lines.append(result.stdout.rstrip())
            if result.stderr:
                trace_lines.append("--- stderr ---")
                trace_lines.append(result.stderr.rstrip())
            yield TraceChunk("\n".join(trace_lines))

            with agent_scope("code_exec", iterations):
                log_tool_run(
                    cmd=_redact_owner_text(cmd),
                    exit_code=result.exit_code,
                    duration_ms=result.duration_ms,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    artifacts=list(result.artifacts),
                )

            all_artifacts.extend(result.artifacts)
            request_satisfied = _verify_task_spec(spec, all_artifacts) is Verdict.SATISFIED
            if (
                allow_budget_growth
                and result.exit_code == 0
                and (result.artifacts or result.stdout.strip())
                and budget < budget_cap
            ):
                budget += 1
            transcript.append(f"ASSISTANT TOOL CALL: {raw.strip()}")
            # The planner is a model: it gets the output as marked third-party data, the
            # way the governed loop hands it an external tool's result.
            transcript.append(
                "TOOL RESULT:\n"
                + wrap_external_content(
                    result.summary(), source="code_exec sandbox", tool="run_shell"
                )
            )

            if result.exit_code == 0 and last_stderr:
                transcript.append(
                    "PLANNER GUIDANCE: The command returned exit code 0 but stderr "
                    "included warnings, so this is not a clean completion. Inspect "
                    "and fix the command before finalizing. If this is a heredoc "
                    "warning, ensure the closing delimiter appears exactly on its own "
                    "line before any follow-up shell commands."
                )
            elif (
                result.exit_code == 0
                and (result.artifacts or result.stdout.strip())
                and request_satisfied
            ):
                transcript.append(
                    "COMPLETION CHECKPOINT: The last tool call succeeded cleanly. "
                    "If the stdout/artifacts satisfy the USER REQUEST, stop calling "
                    "tools now and emit a final answer with: Asked, Achieved, "
                    "Artifacts, and any caveats. Only call run_shell again for a "
                    "specific missing verification or correction. Do not repeat a "
                    "previous successful command."
                )
            elif result.exit_code == 0 and (result.artifacts or result.stdout.strip()):
                transcript.append(
                    "PLANNER GUIDANCE: The command succeeded, but it did not create "
                    f"{_describe_expected_artifact(task.query)}. A helper script or "
                    "stdout alone does not satisfy this request. Continue with a "
                    "run_shell JSON call that creates the final deliverable file."
                )

            attempt_key = (
                " ".join(cmd.split()),
                result.exit_code,
                tuple(result.artifacts),
                last_stderr,
            )
            attempt_history.append(attempt_key)
            identical_outcomes = sum(1 for key in attempt_history if key == attempt_key)
            if result.exit_code != 0:
                if identical_outcomes >= 3:
                    err_lines = (result.stderr or result.stdout or "").strip().splitlines()
                    err_first = err_lines[-1] if err_lines else f"exit={result.exit_code}"
                    if len(err_first) > 200:
                        err_first = err_first[:197] + "..."
                    yield ActivityChunk(
                        f"loop detected - aborted after {identical_outcomes} identical failures"
                    )
                    final_answer = (
                        f"Aborted: the same command failed {identical_outcomes} times "
                        f"with the same exit code. Last error: {err_first}"
                    )
                    break
                if identical_outcomes == 2:
                    transcript.append(
                        "PLANNER GUIDANCE: You just retried a command that "
                        "already failed with the same exit code. STOP repeating "
                        "it. Diagnose the error and try a *different* approach "
                        "— a different library, a different file path, or skip "
                        "the install (the sandbox cannot install OS packages "
                        "via apt-get; pip is the only option, and pre-installed "
                        "Python libs are preferred)."
                    )
            elif last_stderr and identical_outcomes >= 2 and request_satisfied:
                yield ActivityChunk(
                    "loop detected - repeated warning result; handing off artifacts"
                )
                final_answer = _build_code_exec_handoff_answer(
                    task_query=task.query,
                    artifacts=all_artifacts,
                    workspace_path=host.workspace_path,
                    last_result_summary=last_result_summary,
                    last_stdout=last_stdout,
                    iterations=iterations,
                    warning=last_stderr,
                )
                break
            elif last_stderr and identical_outcomes >= 2:
                yield ActivityChunk("loop detected - repeated warning without requested artifact")
                transcript.append(
                    "PLANNER GUIDANCE: You repeated a warning-producing command that "
                    f"still does not create {_describe_expected_artifact(task.query)}. "
                    "Fix the warning and write the actual requested output file."
                )
            elif (
                not last_stderr
                and (result.artifacts or result.stdout.strip())
                and identical_outcomes >= 2
                and request_satisfied
            ):
                yield ActivityChunk("completion checkpoint reached - finalizing successful handoff")
                final_answer = _build_code_exec_handoff_answer(
                    task_query=task.query,
                    artifacts=all_artifacts,
                    workspace_path=host.workspace_path,
                    last_result_summary=last_result_summary,
                    last_stdout=last_stdout,
                    iterations=iterations,
                )
                break
            elif (
                not last_stderr
                and (result.artifacts or result.stdout.strip())
                and identical_outcomes >= 2
            ):
                yield ActivityChunk(
                    "loop detected - repeated intermediate output; requesting deliverable"
                )
                transcript.append(
                    "PLANNER GUIDANCE: You repeated a successful command that still "
                    f"does not create {_describe_expected_artifact(task.query)}. Stop "
                    "repeating it and write the actual requested output file."
                )

        if not final_answer:
            if last_exit_code == 0 and (all_artifacts or last_result_summary):
                if _verify_task_spec(spec, all_artifacts) is Verdict.SATISFIED:
                    final_answer = _build_code_exec_handoff_answer(
                        task_query=task.query,
                        artifacts=all_artifacts,
                        workspace_path=host.workspace_path,
                        last_result_summary=last_result_summary,
                        last_stdout=last_stdout,
                        iterations=iterations,
                        warning=last_stderr,
                    )
                else:
                    final_answer = _build_missing_artifact_answer(
                        task_query=task.query,
                        artifacts=all_artifacts,
                        workspace_path=host.workspace_path,
                        iterations=iterations,
                    )
            else:
                final_answer = (
                    f"Reached the maximum of {budget} tool calls.\n"
                    f"Last sandbox result:\n{last_result_summary}"
                )

        # The answer goes to the owner or channel (intent route) and into the transcript,
        # and a lesson is derived from it: redact it (no envelope: it is not a model-bound
        # external result here) so an instruction the planner repeated does not travel on.
        final_answer = _redact_owner_text(final_answer)

        # Capture lesson (and strip the fenced JSON block from the answer text).
        if lesson_capture is not None:
            try:
                _, cleaned = lesson_capture.handle(
                    query=task.query,
                    answer=final_answer,
                    artifacts=all_artifacts,
                    session_id=task.session_id,
                    iterations=iterations,
                    all_succeeded=(last_exit_code == 0),
                )
                final_answer = cleaned
            except Exception:
                logger.exception("lesson_capture.handle failed; continuing")

        artifact_block = ""
        visible_artifacts = _user_visible_artifacts(task.query, all_artifacts)
        if visible_artifacts:
            unique = list(dict.fromkeys(visible_artifacts))
            # Redacted on its own: it is appended AFTER the final-answer redaction and is
            # the only part of the answer that the non-streamed paths carry unscanned.
            artifact_block = _redact_owner_text(
                "\n\nArtifacts:\n" + "\n".join(f"- {p}" for p in unique)
            )
            final_answer = final_answer.rstrip() + artifact_block

        # If prose was streamed live, emit the artifacts trailer as narration so
        # the user sees it inline; suppress the final-answer yield to avoid dup.
        if prose_was_streamed and artifact_block:
            yield artifact_block

        # --- Skill-proposal hook ---
        # When the user asks to "save the script as a draft skill" (and gives an
        # optional slug like "call it fetch-top-repos"), auto-call
        # propose_skill_from_sandbox so the request is satisfied without forcing
        # the user to re-issue the request through the general handler.
        # Conditions: user asked for a skill, repo_root is wired, a .py artifact
        # was produced, and we can read its content from disk.
        wants_skill, skill_slug = _extract_skill_request(task.query)
        if wants_skill and repo_root is not None and all_artifacts:
            py_artifacts = [p for p in dict.fromkeys(all_artifacts) if p.endswith(".py")]
            if py_artifacts:
                script_path = Path(py_artifacts[0])
                try:
                    script_content = script_path.read_text(encoding="utf-8")
                    from iris_harness.sdk.sandbox import (
                        propose_skill_from_sandbox as _propose_skill,
                    )

                    # Derive a short intent: everything before the first "save …
                    # as skill" clause, capped at 120 chars.
                    save_m = _SAVE_AS_SKILL_RE.search(task.query)
                    intent_text = (
                        task.query[: save_m.start()].strip() if save_m else task.query
                    ).strip()
                    intent_text = (intent_text or task.query)[:120]
                    narrative_text = (
                        f"Auto-proposed from a sandbox code_exec run.\n"
                        f"Original request: {task.query[:500]}"
                    )
                    proposal_result = _propose_skill(
                        repo_root,
                        script=script_content,
                        intent=intent_text,
                        narrative=narrative_text,
                        slug=skill_slug,
                    )
                    if proposal_result.get("ok"):
                        slug_val = proposal_result.get("slug", "")
                        skill_note = (
                            f"\n\nDraft skill saved as `{slug_val}`. "
                            f"Run `/queue` to review, then `/queue promote {slug_val}` to begin promotion."
                        )
                    else:
                        err = proposal_result.get("error", "unknown error")
                        skill_note = (
                            f"\n\n⚠️ Draft skill save failed: {_redact_owner_text(str(err))}"
                        )
                    if prose_was_streamed:
                        yield skill_note
                    else:
                        final_answer = final_answer.rstrip() + skill_note
                except OSError:
                    logger.warning(
                        "skill proposal: could not read script artifact %s; skipping",
                        py_artifacts[0],
                    )
                except Exception:
                    logger.exception(
                        "skill proposal hook failed for query=%r; continuing without proposal",
                        task.query,
                    )

        meta: dict[str, object] = {
            "iterations": iterations,
            "artifacts": [_redact_owner_text(a) for a in dict.fromkeys(all_artifacts)],
            "workspace": host.workspace_path,
            "final_answer": final_answer,
            "prose_streamed": prose_was_streamed,
        }
        # Final sentinel: a tuple (answer, meta) — callers check isinstance(chunk, tuple).
        # When prose streamed live, suppress duplicate by yielding empty answer.
        yield ("" if prose_was_streamed else final_answer, meta)  # type: ignore[misc]

    def handler(task: AgentTask) -> tuple[str, dict[str, object]]:
        _begin_run()
        narration: list[str] = []
        answer = ""
        meta: dict[str, object] = {}
        try:
            for chunk in _run_loop_gen(task):
                if isinstance(chunk, tuple):
                    answer, meta = chunk
                elif isinstance(chunk, str):
                    narration.append(chunk)
                # ActivityChunk / TraceChunk are streaming-only signals; the
                # sync handler returns just the final prose answer.
        except Exception as exc:
            logger.exception("code_exec handler failed for query=%r", task.query)
            return _friendly_llm_error(exc), {}
        if narration:
            # The streamed prose is the answer here, and it never passed the final-answer
            # redaction in the loop: redact the joined text (issue #140).
            answer = _redact_owner_text("".join(narration)) + "\n" + answer
        return answer, meta

    def stream_handler(task: AgentTask) -> Iterator[StreamChunk]:
        _begin_run()
        prose = _ProseRedactor()
        try:
            for chunk in _run_loop_gen(task):
                if isinstance(chunk, tuple):
                    answer, meta = chunk
                    if tail := prose.flush():
                        yield tail
                    yield answer
                    yield meta
                elif isinstance(chunk, str):
                    if ready := prose.feed(chunk):
                        yield ready
                elif isinstance(chunk, TraceChunk):
                    yield TraceChunk(_redact_owner_text(chunk.text))
                else:
                    yield chunk
        except Exception as exc:
            logger.exception("code_exec stream handler failed for query=%r", task.query)
            yield _friendly_llm_error(exc)

    return handler, stream_handler
