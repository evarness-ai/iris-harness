"""Pure parsing/sanitizing utilities for the ReAct code-exec loop.

Best-effort extraction of a JSON tool call from an LLM response and cleanup of
prose that leaks internal scaffolding (tool-call JSON, planner transcript,
markdown code fences). Extracted verbatim from ``bootstrap.py`` (Phase 2
decomposition) and re-exported there for backward compatibility. Mechanism, not
behavior-config — so this stays Python.
"""

from __future__ import annotations

import re
from typing import Any

_TOOL_CALL_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_TOOL_PROGRESS_MAX_CHARS = 160


def _relax_json_strings(text: str) -> str:
    """Escape raw control chars (LF/CR/TAB) that appear inside JSON string values.

    Small models often emit heredoc-style ``cmd`` values with literal newlines
    instead of ``\\n``, which is invalid JSON per spec. This walks the text
    tracking string state and escapes those control chars only inside strings,
    leaving structural whitespace alone.
    """
    out: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue
            if ch == "\\":
                out.append(ch)
                escape = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if ch == "\n":
                out.append("\\n")
                continue
            if ch == "\r":
                out.append("\\r")
                continue
            if ch == "\t":
                out.append("\\t")
                continue
            out.append(ch)
        else:
            out.append(ch)
            if ch == '"':
                in_string = True
    return "".join(out)


def _parse_tool_call(raw: str) -> dict[str, Any] | None:
    """Best-effort extract a supported tool call from an LLM response."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    match = _TOOL_CALL_JSON_RE.search(text)
    if match is None:
        return None
    candidate = match.group(0)
    import json as _json

    try:
        loaded = _json.loads(candidate)
    except Exception:  # noqa: BLE001
        # Retry with relaxed parsing: escape raw control chars inside strings.
        # Handles the common heredoc pattern where ``cmd`` contains real
        # newlines (which JSON spec disallows in string values).
        try:
            loaded = _json.loads(_relax_json_strings(candidate))
        except Exception:  # noqa: BLE001
            return None
    if not isinstance(loaded, dict):
        return None
    tool_name = loaded.get("tool")
    if tool_name not in {"run_shell", "ask_user"}:
        return None
    args = loaded.get("args")
    if not isinstance(args, dict):
        return None

    if tool_name == "ask_user":
        question = args.get("question")
        if not isinstance(question, str) or not question.strip():
            return None
        parsed_question = " ".join(question.split())
        parsed: dict[str, Any] = {"tool": "ask_user", "question": parsed_question}
        progress_raw = loaded.get("progress")
        if not isinstance(progress_raw, str):
            progress_raw = args.get("progress")
        if isinstance(progress_raw, str):
            progress = " ".join(progress_raw.split())
            if len(progress) > _TOOL_PROGRESS_MAX_CHARS:
                progress = progress[: _TOOL_PROGRESS_MAX_CHARS - 3] + "..."
            if progress:
                parsed["progress"] = progress
        return parsed

    cmd = args.get("cmd")
    if not isinstance(cmd, str) or not cmd.strip():
        return None
    timeout_raw = args.get("timeout", 30)
    try:
        timeout = int(timeout_raw)
    except (TypeError, ValueError):
        timeout = 30
    parsed_cmd: dict[str, Any] = {"cmd": cmd, "timeout": timeout}
    progress_raw = loaded.get("progress")
    if not isinstance(progress_raw, str):
        progress_raw = args.get("progress")
    if isinstance(progress_raw, str):
        progress = " ".join(progress_raw.split())
        if len(progress) > _TOOL_PROGRESS_MAX_CHARS:
            progress = progress[: _TOOL_PROGRESS_MAX_CHARS - 3] + "..."
        if progress:
            parsed_cmd["progress"] = progress
    return parsed_cmd


# Markers that end a final-answer turn cleanly: the prose BEFORE the marker is
# the user-visible answer; the marker itself and anything after is internal
# (e.g. a fenced lesson-capture block).
_PROSE_KEEP_PREFIX_MARKERS: tuple[str, ...] = ("```lesson",)

# Markers that signal "this entire turn is intermediate or noise" — a tool-call
# JSON that leaked into prose, or the planner echoing back its own input
# transcript, or (most commonly) a small model misformatting the response as
# markdown code blocks instead of JSON tool calls. Anything emitted in such a
# turn is suppressed from the live user-facing stream (the raw response is
# still buffered in raw_parts so the tool-call extractor and trace log get it).
_PROSE_DROP_ALL_MARKERS: tuple[str, ...] = (
    "ASSISTANT TOOL CALL:",
    "TOOL RESULT:",
    "USER REQUEST:",
    "PRIOR CONVERSATION",
    "PLANNER GUIDANCE:",
    '{"tool":',
    '{"tool" :',
    '{ "tool":',
    '{ "tool" :',
    # Markdown code fences other than ```lesson are a planner-format failure:
    # the model is trying to emit a shell command in prose instead of as a
    # JSON tool call. Drop the whole turn from chat (the next iteration's
    # tool call, if any, is unaffected).
    "```python",
    "```bash",
    "```sh\n",
    "```shell",
    "```javascript",
    "```js\n",
    "```json",
    # Raw shell-command shapes that sometimes leak when the model abandons
    # the JSON protocol mid-response.
    "cat > /workspace",
    "cat >/workspace",
    "<< 'pyeof'",
    "<<'pyeof'",
)

_PROSE_CUTOFF_MARKERS: tuple[str, ...] = _PROSE_KEEP_PREFIX_MARKERS + _PROSE_DROP_ALL_MARKERS

# Hold this many chars of prose in the live-stream tail before flushing. Sized
# so a multi-paragraph retry narration ("Since the previous attempt had an
# error… Here's another attempt:\n\n```python…") is still entirely buffered
# when the ```python fence arrives, allowing the drop-all path to suppress it.
# Real final answers up to 4 KB therefore appear at end-of-stream rather than
# token-by-token; that's an acceptable trade for clean retries.
_PROSE_CUTOFF_LOOKBACK = 4096


def _find_first_cutoff(text_lower: str) -> tuple[int, bool]:
    """Return ``(idx, drop_all)`` for the earliest cutoff marker, or ``(-1, False)``.

    ``drop_all`` is True when the matched marker means the *entire* prose for
    this turn should be suppressed (intermediate-turn noise). It is False for
    "keep prefix" markers like ``​```lesson`` where the prose preceding
    the marker is still the legitimate final answer.
    """
    earliest = -1
    drop_all = False
    for marker in _PROSE_KEEP_PREFIX_MARKERS:
        idx = text_lower.find(marker.lower())
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx
            drop_all = False
    for marker in _PROSE_DROP_ALL_MARKERS:
        idx = text_lower.find(marker.lower())
        if idx != -1 and (earliest == -1 or idx < earliest):
            earliest = idx
            drop_all = True
    return earliest, drop_all


def _sanitize_prose(text: str) -> str:
    """Trim or drop ``text`` based on internal-marker / tool-call cutoffs."""
    if not text:
        return text
    cutoff, drop_all = _find_first_cutoff(text.lower())
    if cutoff == -1:
        return text
    if drop_all:
        return ""
    return text[:cutoff].rstrip()


def _one_line_preview(text: str, *, limit: int = 500) -> str:
    """Return compact single-line text suitable for a user-facing handoff."""
    preview = " ".join((text or "").split())
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3] + "..."


# ---------------------------------------------------------------------------
# Cut-off / degeneration detection
# ---------------------------------------------------------------------------
#
# A planner response that dies mid-JSON is a different failure from a planner
# that ignored the protocol and answered in prose, and it needs a different
# remedy: a shorter command, not a restatement of the tool-call format. These
# helpers let the ReAct loop tell the two apart.


def _tool_call_json_is_open(text: str) -> bool:
    """Return True when ``text`` opens a JSON object that never closes.

    Walks the text tracking string/escape state. True when the scan ends while
    still inside a string literal or with unbalanced braces — the signature of
    a response cut short by the generation token cap.
    """
    depth = 0
    in_string = False
    escape = False
    started = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
            started = True
        elif ch == "}":
            depth -= 1
            if started and depth <= 0:
                return False
    return started and (depth > 0 or in_string)


def _looks_truncated_tool_call(raw: str) -> bool:
    """Return True when ``raw`` is a tool call that was cut off mid-emission."""
    if not isinstance(raw, str):
        return False
    text = raw.strip()
    if not text:
        return False
    if text.startswith("```"):
        text = text.lstrip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.lstrip()
    if not text.startswith("{"):
        return False
    if _parse_tool_call(text) is not None:
        return False
    return _tool_call_json_is_open(text)


# A small model that loses the plot inside a heredoc emits the same line (or
# small block) over and over until the token cap kills the response. Detecting
# it from the streamed tail lets the caller abandon the call early instead of
# spending the whole generation budget producing garbage.
_REPETITION_WINDOW = 4096
_REPETITION_MAX_PERIOD = 256
_REPETITION_MIN_REPEATS = 4
_REPETITION_MIN_RUN_CHARS = 512


def _find_degenerate_repetition(text: str) -> str | None:
    """Return the repeating unit when ``text`` ends in a degenerate loop.

    Scans the tail for the shortest unit that repeats enough times to account
    for at least ``_REPETITION_MIN_RUN_CHARS`` consecutive characters, so a
    long repeated block needs fewer repeats than a single repeated character.
    Returns ``None`` when the tail is not degenerate.
    """
    if not text:
        return None
    tail = text[-_REPETITION_WINDOW:]
    if len(tail) < _REPETITION_MIN_RUN_CHARS:
        return None
    for period in range(1, _REPETITION_MAX_PERIOD + 1):
        repeats = max(_REPETITION_MIN_REPEATS, -(-_REPETITION_MIN_RUN_CHARS // period))
        span = period * repeats
        if span > len(tail):
            continue
        segment = tail[-span:]
        unit = segment[:period]
        if segment == unit * repeats:
            return unit
    return None
