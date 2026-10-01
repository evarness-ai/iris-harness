"""CI guard: no silent swallow of a side-effecting operation on the chat path.

"Logs are gems" — every write / audit / record / commit on the request lifecycle must
leave a trace. This test flags `except` handlers that NEITHER log NOR re-raise when their
`try` performs a side-effecting call. Parse-only fallbacks (json.loads, int(), ...) are
intentionally ignored. To deliberately allow a silent swallow, put `# silent-ok: <reason>`
on the `except` line.

Extend GUARDED_MODULES to bring more of the codebase under the guard.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve()
while not (_ROOT / "src" / "iris_harness").is_dir():
    if _ROOT == _ROOT.parent:
        raise RuntimeError("could not locate repo root (src/iris_harness)")
    _ROOT = _ROOT.parent

# The chat request lifecycle: router -> governance -> tools -> response, + audit writers.
GUARDED_MODULES = (
    "src/iris_harness/agent/response_curator.py",
    "src/iris_harness/agent/intent_router.py",
    "src/iris_harness/agent/task_planner.py",
    "src/iris_harness/agent/agentic_core.py",
    "src/iris_harness/agent/agent_executor.py",
    "src/iris_harness/kernel/governance/kernel.py",
    "src/iris_harness/kernel/governance/audit/log.py",
    "src/iris_harness/runtime/router_audit.py",
    "src/iris_code/llm_client.py",
    "src/iris_harness/runtime/bootstrap.py",
)

_SIDE_EFFECT = re.compile(
    r"\b(record|audit|write|commit|execute|executemany|persist|save|insert|upsert|"
    r"delete|store|emit|publish|append|sendall|log_tool_run|run_shell|fire)\b"
)


def _logs_or_raises(handler: ast.ExceptHandler) -> bool:
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Call):
            try:
                func = ast.unparse(node.func).lower()
            except Exception:  # noqa: BLE001 — best-effort unparse in a pure test helper
                func = ""
            if "log" in func or "print" in func:
                return True
    return False


def _try_has_side_effect(try_node: ast.Try) -> bool:
    for stmt in try_node.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                try:
                    if _SIDE_EFFECT.search(ast.unparse(node.func)):
                        return True
                except Exception:  # noqa: S110, BLE001 - best-effort unparse, pure test helper
                    pass
    return False


def _violations(path: Path) -> list[str]:
    src = path.read_text()
    lines = src.splitlines()
    out: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Try) and _try_has_side_effect(node):
            for handler in node.handlers:
                if _logs_or_raises(handler):
                    continue
                if "# silent-ok:" in lines[handler.lineno - 1]:
                    continue
                out.append(f"{path.name}:{handler.lineno}  {lines[handler.lineno - 1].strip()}")
    return out


def test_no_silent_swallow_of_side_effecting_op_in_chat_path() -> None:
    violations: list[str] = []
    checked = 0
    for module in GUARDED_MODULES:
        path = _ROOT / module
        source_root = _ROOT / Path(module).parts[0] / Path(module).parts[1]
        if not source_root.is_dir():
            # A source root this tree does not carry (the public export ships
            # src/iris_harness and src/memris only). Where the root exists, a missing
            # module is still a failure: that is a move the guard list did not follow.
            continue
        assert path.is_file(), f"guarded module missing: {module}"
        violations.extend(_violations(path))
        checked += 1
    assert checked, "no guarded module found -- the guard checked nothing"
    assert not violations, (
        "Silent swallow of a side-effecting op (write/audit/record/commit) — log the "
        "failure, re-raise, or annotate the `except` line with `# silent-ok: <reason>`:\n  "
        + "\n  ".join(violations)
    )


def test_guard_flags_a_silent_write_swallow(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(
        "def f(conn):\n"
        "    try:\n"
        "        conn.execute('INSERT INTO t VALUES (1)')\n"
        "    except Exception:\n"
        "        pass\n"
    )
    assert _violations(bad), "guard must flag a silently-swallowed write"


def test_guard_ignores_logged_swallow_and_parse_only(tmp_path: Path) -> None:
    ok = tmp_path / "ok.py"
    ok.write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "def f(conn):\n"
        "    try:\n"
        "        conn.execute('INSERT INTO t VALUES (1)')\n"
        "    except Exception:\n"
        "        logger.warning('write failed')\n"
        "    try:\n"
        "        int('not a number')\n"
        "    except ValueError:\n"
        "        pass\n"
    )
    assert not _violations(ok), "guard must ignore logged swallows and parse-only fallbacks"


# The memory and retrieval paths: a broken store or retriever must never read as "no
# results". Every BROAD handler here (``except Exception``/``BaseException``/bare) must
# log or re-raise, or say why it is silent with ``silent-ok: <reason>`` on its line.
# Narrow handlers (``except AttributeError``, ``except OSError`` ...) are not checked.
MEMORY_GUARDED_ROOTS = (
    "src/memris",
    "src/iris_harness/memory",
    "src/iris_harness/services/rag",
    "src/iris_harness/runtime/session_memory.py",
    # The agent's own memory, retrieval and action tools: a failure reaches the agent as
    # "X failed: ..." text, so without a log line operators never see it.
    "src/iris_harness/runtime/react_tools.py",
    # The email plugin's search paths (#668 follow-up). Not in the public tree; a path
    # this tree does not carry is skipped.
    "src/iris_personal/email/semantic_index.py",
    "src/iris_personal/email/agent_tools.py",
)

_BROAD = {"Exception", "BaseException"}


def _is_broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(t, ast.Name) and t.id in _BROAD for t in types)


def _silent_broad_handlers(path: Path) -> list[str]:
    src = path.read_text()
    lines = src.splitlines()
    out: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.ExceptHandler) or not _is_broad(node):
            continue
        if _logs_or_raises(node) or "silent-ok:" in lines[node.lineno - 1]:
            continue
        out.append(f"{path}:{node.lineno}  {lines[node.lineno - 1].strip()}")
    return out


def _memory_modules() -> list[Path]:
    paths: list[Path] = []
    for root in MEMORY_GUARDED_ROOTS:
        target = _ROOT / root
        if target.is_file():
            paths.append(target)
        elif target.is_dir():
            paths.extend(sorted(target.rglob("*.py")))
    return paths


def test_no_silent_broad_except_in_memory_and_retrieval() -> None:
    modules = _memory_modules()
    assert modules, "no memory module found -- the guard checked nothing"
    violations = [v for path in modules for v in _silent_broad_handlers(path)]
    assert not violations, (
        "A broad `except` in the memory/retrieval path neither logs nor re-raises, so a "
        "broken store reads as 'no results'. Log it (operation + exception type, never "
        "memory content), narrow it, or mark the line `silent-ok: <reason>`:\n  "
        + "\n  ".join(violations)
    )


def test_memory_guard_flags_a_silent_broad_except(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"
    bad.write_text(
        "def f(store):\n"
        "    try:\n"
        "        return store.read()\n"
        "    except Exception:\n"
        "        return []\n"
    )
    ok = tmp_path / "ok.py"
    ok.write_text(
        "import logging\n"
        "logger = logging.getLogger(__name__)\n"
        "def f(store):\n"
        "    try:\n"
        "        return store.read()\n"
        "    except AttributeError:\n"
        "        return []\n"
        "    except Exception:\n"
        "        logger.warning('read failed', exc_info=True)\n"
        "        return []\n"
        "def g():\n"
        "    try:\n"
        "        import thing\n"
        "    except Exception:  # silent-ok: a capability probe\n"
        "        return False\n"
    )
    assert _silent_broad_handlers(bad), "guard must flag a silent broad except"
    assert not _silent_broad_handlers(ok), "guard must pass logged, narrow and marked handlers"
