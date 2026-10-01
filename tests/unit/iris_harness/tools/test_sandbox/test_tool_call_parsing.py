"""Tests for the LLM tool-call JSON parser used by the code_exec loop."""

from __future__ import annotations

from iris_harness.runtime.tool_call_parsing import (
    _find_degenerate_repetition,
    _looks_truncated_tool_call,
    _parse_tool_call,
)


def test_parses_clean_json() -> None:
    raw = '{"tool": "run_shell", "args": {"cmd": "echo hi", "timeout": 30}}'
    parsed = _parse_tool_call(raw)
    assert parsed == {"cmd": "echo hi", "timeout": 30}


def test_parses_optional_progress_field() -> None:
    raw = (
        '{"tool": "run_shell", "args": {"cmd": "echo hi", "timeout": 30}, '
        '"progress": "writing the report   and verifying it"}'
    )
    parsed = _parse_tool_call(raw)
    assert parsed == {
        "cmd": "echo hi",
        "timeout": 30,
        "progress": "writing the report and verifying it",
    }


def test_parses_ask_user_tool_call() -> None:
    raw = (
        '{"tool": "ask_user", "args": {"question": "Which account should I use?"}, '
        '"progress": "checking missing account"}'
    )
    parsed = _parse_tool_call(raw)
    assert parsed == {
        "tool": "ask_user",
        "question": "Which account should I use?",
        "progress": "checking missing account",
    }


def test_returns_none_when_ask_user_question_missing() -> None:
    raw = '{"tool": "ask_user", "args": {"question": "   "}}'
    assert _parse_tool_call(raw) is None


def test_parses_json_in_fence() -> None:
    raw = '```json\n{"tool": "run_shell", "args": {"cmd": "ls"}}\n```'
    parsed = _parse_tool_call(raw)
    assert parsed is not None
    assert parsed["cmd"] == "ls"
    assert parsed["timeout"] == 30  # default


def test_returns_none_for_prose() -> None:
    raw = "I have created the PDF at /workspace/report.pdf. Done!"
    assert _parse_tool_call(raw) is None


def test_returns_none_when_tool_name_wrong() -> None:
    raw = '{"tool": "delete_everything", "args": {"cmd": "rm -rf /"}}'
    assert _parse_tool_call(raw) is None


def test_returns_none_when_cmd_missing() -> None:
    raw = '{"tool": "run_shell", "args": {"timeout": 10}}'
    assert _parse_tool_call(raw) is None


def test_returns_none_when_cmd_empty() -> None:
    raw = '{"tool": "run_shell", "args": {"cmd": "   "}}'
    assert _parse_tool_call(raw) is None


def test_invalid_timeout_falls_back_to_default() -> None:
    raw = '{"tool": "run_shell", "args": {"cmd": "ls", "timeout": "abc"}}'
    parsed = _parse_tool_call(raw)
    assert parsed is not None
    assert parsed["timeout"] == 30


def test_returns_none_for_garbage() -> None:
    assert _parse_tool_call("") is None
    assert _parse_tool_call("not json at all") is None
    assert _parse_tool_call("{not valid json") is None


def test_parses_multiline_heredoc_cmd() -> None:
    """Real models emit heredoc cmds with raw newlines (invalid per JSON spec).

    The parser must accept them via a relaxed retry that escapes control chars.
    """
    raw = (
        '{"tool":"run_shell","args":{"cmd":"cat > /workspace/script.py << \'PYEOF\'\n'
        "import pandas as pd\n"
        "print('hello')\n"
        'PYEOF\npython /workspace/script.py","timeout":30},'
        '"progress":"writing markdown report"}'
    )
    parsed = _parse_tool_call(raw)
    assert parsed is not None
    assert parsed["timeout"] == 30
    assert parsed["progress"] == "writing markdown report"
    assert "import pandas as pd" in parsed["cmd"]
    assert "PYEOF" in parsed["cmd"]
    assert "python /workspace/script.py" in parsed["cmd"]


def test_parses_multiline_cmd_with_tabs_and_cr() -> None:
    raw = '{"tool":"run_shell","args":{"cmd":"echo a\r\n\techo b","timeout":5}}'
    parsed = _parse_tool_call(raw)
    assert parsed is not None
    assert parsed["cmd"] == "echo a\r\n\techo b"
    assert parsed["timeout"] == 5


def test_looks_truncated_tool_call_detects_unterminated_string() -> None:
    raw = '{"tool":"run_shell","args":{"cmd":"cat > /workspace/a.py << \'PYEOF\'\\nimport os'
    assert _parse_tool_call(raw) is None
    assert _looks_truncated_tool_call(raw) is True


def test_looks_truncated_tool_call_detects_unbalanced_braces() -> None:
    assert _looks_truncated_tool_call('{"tool":"run_shell","args":{"cmd":"echo hi"') is True


def test_looks_truncated_tool_call_false_for_complete_call() -> None:
    assert _looks_truncated_tool_call('{"tool":"run_shell","args":{"cmd":"echo hi"}}') is False


def test_looks_truncated_tool_call_false_for_prose() -> None:
    assert _looks_truncated_tool_call("Here is how you would do it: install pandas first.") is False
    assert _looks_truncated_tool_call("") is False


def test_looks_truncated_tool_call_false_for_complete_but_unsupported_tool() -> None:
    # Parses fine, just not a tool we support — that is a protocol failure,
    # not a cut-off response, and must not be routed to the truncation path.
    assert _looks_truncated_tool_call('{"tool":"browse","args":{"url":"x"}}') is False


def test_find_degenerate_repetition_catches_repeated_line() -> None:
    text = '{"tool":"run_shell","args":{"cmd":"' + "import dash\\nimport dash_table\\n" * 60
    assert _find_degenerate_repetition(text) is not None


def test_find_degenerate_repetition_catches_repeated_block() -> None:
    block = (
        "\\nfrom reportlab.lib import colors"
        "\\nfrom reportlab.lib.enums import TA_CENTER"
        "\\nfrom reportlab.lib.units import mm"
        "\\nfrom reportlab.platypus import SimpleDocTemplate, Paragraph, Table"
    )
    assert _find_degenerate_repetition("prefix" + block * 8) is not None


def test_find_degenerate_repetition_ignores_healthy_output() -> None:
    healthy = (
        '{"tool":"run_shell","args":{"cmd":"cat > /workspace/report.py << \'PYEOF\'\\n'
        "import csv\\nimport json\\nrows = load_rows()\\nwrite_pdf(rows)\\nPYEOF\\n"
        'python3 /workspace/report.py","timeout":60}}'
    )
    assert _find_degenerate_repetition(healthy) is None
    assert _find_degenerate_repetition("") is None
    assert _find_degenerate_repetition("short text") is None


def test_find_degenerate_repetition_ignores_repeats_below_run_threshold() -> None:
    # Four short repeats is normal code (e.g. a padded table), not degeneration.
    assert _find_degenerate_repetition("x" * 200 + "ab" * 4) is None
