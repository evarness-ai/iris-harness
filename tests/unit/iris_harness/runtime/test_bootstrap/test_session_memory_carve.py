"""The session-memory carve (OSS plan M5.7 track C, slice 16 — the last group).

The conversation window, its reload, its compaction and its context-health view left
``IrisRuntime`` as ``SessionMemory``, held as ``runtime.sessions``, with the six state
members only they wrote. ``TurnHost`` lost its last two private methods for it; the
routine, replies, notices and learning hosts each swapped a private member for
``sessions``. What mypy cannot check about that:

* **The host declaration is exact**, measured by AST from both sides.
* **Nothing reaches for the moved names on the runtime.** A dataclass will not raise on
  ``runtime._conversations[sid] = [...]`` — the API would then read an empty window while
  the pipeline recorded into the collaborator's.
* **The archive drain is consumed once**, the contract behavior mining relies on.

Behaviour is pinned where it was: compaction and reload in ``test_runtime_chat.py``,
notices landing in the window in ``test_activities_async.py`` and
``test_approval_timeout_notice.py``, the drain in ``test_learning_controls.py``, the
self-management tools in ``test_self_management_tools.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from iris_harness.memory.compactor import ConversationTurn
from iris_harness.runtime.session_memory import SessionMemory

MODULE = Path("src/iris_harness/runtime/session_memory.py")
SWEPT = ("src", "services", "tests", "scripts")

# Renamed public on the collaborator: the old name is stale wherever it is read.
RENAMED = {
    "_load_session_if_needed",
    "_build_memory_context",
    "_format_recent_context",
    "_record_turn",
    "_conversations",
}
# Moved from the runtime's fields, same names: valid only off ``.sessions``.
MOVED_STATE = {
    "_session_summaries",
    "_last_compaction",
    "_compaction_archive_buffer",
    "_loaded_sessions",
    "_ephemeral_sessions",
}
MOVED = RENAMED | MOVED_STATE


def test_the_archive_drain_is_consumed_once() -> None:
    sessions = SessionMemory(SimpleNamespace())  # type: ignore[arg-type]
    turns = [
        ConversationTurn(role="user", content="old"),
        ConversationTurn(role="assistant", content="er"),
    ]
    sessions._compaction_archive_buffer.extend(turns)

    assert sessions.drain_compaction_archive() == turns
    assert sessions.drain_compaction_archive() == []
    assert sessions._compaction_archive_buffer == []


def _host_reached() -> set[str]:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "_host"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "self"
    }


def _host_declared() -> set[str]:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SessionMemoryHost"
    )
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)} | {
        n.name for n in cls.body if isinstance(n, ast.FunctionDef)
    }


def test_session_memory_host_declares_exactly_what_the_module_reaches() -> None:
    assert _host_reached() == _host_declared()
    assert _host_declared() == {
        "compactor",
        "learning",
        "learning_store",
        "memory_retriever",
        "memory_store",
        "semantic_index",
        "_last_react_budget",
    }


def _via_sessions(owner: ast.expr) -> bool:
    """``<anything>.sessions`` (or a local named ``sessions``) — the collaborator, where
    the moved state now lives."""
    return (isinstance(owner, ast.Attribute) and owner.attr == "sessions") or (
        isinstance(owner, ast.Name) and owner.id == "sessions"
    )


def _stale(name: str, owner: ast.expr) -> bool:
    return name in RENAMED or (name in MOVED_STATE and not _via_sessions(owner))


def test_nothing_reaches_for_the_moved_session_members_off_a_runtime() -> None:
    leftovers: list[str] = []
    for path in sorted(p for root in SWEPT for p in Path(root).rglob("*.py")):
        if path == MODULE:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute) and _stale(node.attr, node.value):
                leftovers.append(f"{path}:{node.lineno} .{node.attr}")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setattr"
                and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)
                and node.args[1].value in MOVED
                and _stale(node.args[1].value, node.args[0])
            ):
                leftovers.append(f"{path}:{node.lineno} setattr {node.args[1].value!r}")
    assert leftovers == []
