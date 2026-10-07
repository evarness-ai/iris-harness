"""Which tools need the owner's approval once the run has read outside text (issue #149).

A run is tainted when one of its tool results came from a tool declared ``content:
external``. The loop computes that from the run's recorded steps (so a run resumed after an
approval keeps it) and stamps it on the call (``ToolCall.tainted``); the runner then treats
a call of a tool listed here as approved per call, on the same card as any other. This module
only answers "which tools".

The list is ``config/governance/taint-policy.yaml`` (``approval_when_tainted``). A file that
is missing, unreadable or malformed falls back to the default, the three memory writes: the
failure is more approval, never less. A valid file the owner edited is the owner's decision.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

TAINT_POLICY_FILE = "taint-policy.yaml"

#: The memory writes: the realistic poisoning sink, and the default when no file can be read.
DEFAULT_GATED: frozenset[str] = frozenset({"memory_correct", "memory_forget", "memory_restore"})

#: What the owner is told on the approval card, in plain words.
TAINT_REASON = (
    "IRIS read text from outside (a page, an email, a document) earlier in this turn and "
    "now wants to make this change. Approve it only if you asked for it."
)

_WILDCARDS = frozenset("*?[]")
_lock = threading.Lock()
_cache: dict[Path, tuple[int, frozenset[str]]] = {}
_warned: set[tuple[Path, int]] = set()


def taint_policy_path() -> Path:
    from iris_harness.foundation.paths import config_path

    return config_path("governance", TAINT_POLICY_FILE)


def parse_taint_policy(document: Any) -> frozenset[str]:
    """The tool names in a parsed file; raises ``ValueError`` when it is malformed."""
    if not isinstance(document, dict) or set(document) - {"approval_when_tainted"}:
        raise ValueError("the file holds one key, 'approval_when_tainted', a list of tool names")
    names = document.get("approval_when_tainted")
    if not isinstance(names, list):
        raise ValueError("'approval_when_tainted' must be a list")
    out: set[str] = set()
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("every entry must be a non-empty tool name")
        if any(c in name for c in _WILDCARDS):
            raise ValueError(f"{name!r}: wildcards are not allowed")
        out.add(name.strip())
    return frozenset(out)


def taint_gated_tools() -> frozenset[str]:
    """The tools held for approval in a tainted run; the default when the file cannot be used."""
    path = taint_policy_path()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return DEFAULT_GATED
    with _lock:
        hit = _cache.get(path)
        if hit is None or hit[0] != mtime:
            try:
                names = parse_taint_policy(yaml.safe_load(path.read_text(encoding="utf-8")))
            except (OSError, yaml.YAMLError, ValueError) as exc:
                if (path, mtime) not in _warned:
                    _warned.add((path, mtime))
                    logger.warning("taint policy ignored (using the default list): %s", exc)
                names = DEFAULT_GATED
            hit = _cache[path] = (mtime, names)
    return hit[1]


__all__ = [
    "DEFAULT_GATED",
    "TAINT_POLICY_FILE",
    "TAINT_REASON",
    "parse_taint_policy",
    "taint_gated_tools",
    "taint_policy_path",
]
