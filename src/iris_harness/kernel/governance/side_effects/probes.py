# ruff: noqa: S603, S607
"""Built-in verification probes for common side-effecting tools.

Story 12.gov-4.10 / design §10.3.

Each probe is a callable ``(side_effect_id, metadata) -> ProbeResult`` that
inspects external state to determine whether a side effect has already landed.
The ledger runs probes at ``iris-code resume`` / ``iris run resume`` time.

Probe return values:
  - ``"completed"``     — side effect is confirmed; skip at resume
  - ``"not_completed"`` — confirmed absent; safe to re-execute
  - ``"ambiguous"``     — can't tell; enqueue an approval before retrying
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from iris_harness.foundation.process_state import track_globals

logger = logging.getLogger(__name__)

ProbeResult = Literal["completed", "not_completed", "ambiguous"]
ProbeCallable = Callable[[str, dict[str, Any]], ProbeResult]

# Registry: probe_name → callable
_REGISTRY: dict[str, ProbeCallable] = {}


#: The probe name of a side effect nothing can verify (a tool that declares no
#: ``verify:`` probe): resume cannot tell whether it landed, so it is ``ambiguous``.
NO_PROBE = ""


def register_probe(name: str, fn: ProbeCallable) -> None:
    if name == NO_PROBE:
        raise ValueError("a probe needs a name")
    _REGISTRY[name] = fn


def get_probe(name: str) -> ProbeCallable | None:
    return _REGISTRY.get(name)


def probe_names() -> frozenset[str]:
    """Every registered probe's name: what a tool's ``verify:`` may declare."""
    return frozenset(_REGISTRY)


def run_probe(
    probe_name: str,
    side_effect_id: str,
    metadata: dict[str, Any],
) -> ProbeResult:
    """Dispatch to a registered probe; return 'ambiguous' if probe is unknown."""
    if probe_name == NO_PROBE:
        return "ambiguous"
    probe = _REGISTRY.get(probe_name)
    if probe is None:
        logger.warning("no probe registered for %r — treating as ambiguous", probe_name)
        return "ambiguous"
    try:
        return probe(side_effect_id, metadata)
    except Exception as exc:  # noqa: BLE001
        logger.warning("probe %r raised %s — treating as ambiguous", probe_name, exc)
        return "ambiguous"


# ---------------------------------------------------------------------------
# git_commit probe
# side_effect_id = full commit SHA
# metadata = {"repo_path": str}  (optional, defaults to cwd)
# ---------------------------------------------------------------------------


def _probe_git_commit(side_effect_id: str, metadata: dict[str, Any]) -> ProbeResult:
    repo_path = metadata.get("repo_path", ".")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "cat-file", "-t", side_effect_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip() == "commit":
            return "completed"
        return "not_completed"
    except subprocess.TimeoutExpired:
        return "ambiguous"
    except FileNotFoundError:
        return "ambiguous"


register_probe("git_commit", _probe_git_commit)


# ---------------------------------------------------------------------------
# git_push probe
# side_effect_id = "<remote>/<branch>"  (e.g. "origin/feat/my-branch")
# metadata = {"commit_sha": str, "repo_path": str}
# ---------------------------------------------------------------------------


def _probe_git_push(side_effect_id: str, metadata: dict[str, Any]) -> ProbeResult:
    repo_path = metadata.get("repo_path", ".")
    commit_sha = metadata.get("commit_sha", "")
    parts = side_effect_id.split("/", 1)
    if len(parts) != 2:
        logger.warning("git_push probe: expected '<remote>/<branch>', got %r", side_effect_id)
        return "ambiguous"
    remote, branch = parts
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "ls-remote", "--heads", remote, branch],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return "ambiguous"
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            sha, *_ = line.split()
            if commit_sha and sha.startswith(commit_sha[:7]):
                return "completed"
            if not commit_sha and sha:
                return "completed"
        return "not_completed"
    except subprocess.TimeoutExpired:
        return "ambiguous"
    except FileNotFoundError:
        return "ambiguous"


register_probe("git_push", _probe_git_push)


# ---------------------------------------------------------------------------
# github_create_pr probe
# side_effect_id = PR URL (e.g. "https://github.com/org/repo/pull/42")
# metadata = {} (gh CLI resolves from the URL)
# ---------------------------------------------------------------------------


def _probe_github_create_pr(side_effect_id: str, metadata: dict[str, Any]) -> ProbeResult:
    if not side_effect_id.startswith("https://"):
        logger.warning("github_create_pr probe: unexpected side_effect_id %r", side_effect_id)
        return "ambiguous"
    try:
        result = subprocess.run(
            ["gh", "pr", "view", side_effect_id, "--json", "state"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            return "completed"
        # gh returns non-zero when the PR doesn't exist or the URL is wrong
        return "not_completed"
    except subprocess.TimeoutExpired:
        return "ambiguous"
    except FileNotFoundError:
        # gh CLI not installed — can't verify
        return "ambiguous"


register_probe("github_create_pr", _probe_github_create_pr)


# ---------------------------------------------------------------------------
# write_file probe
# side_effect_id = "<path>:<sha256_hex>"  (empty sha = check existence only)
# metadata = {}
# ---------------------------------------------------------------------------


def _probe_write_file(side_effect_id: str, metadata: dict[str, Any]) -> ProbeResult:
    if ":" in side_effect_id:
        path_str, expected_sha = side_effect_id.split(":", 1)
    else:
        path_str = side_effect_id
        expected_sha = ""

    path = Path(path_str)
    if not path.exists():
        return "not_completed"

    if not expected_sha:
        return "completed"

    try:
        content = path.read_bytes()
        actual_sha = hashlib.sha256(content).hexdigest()
        if actual_sha == expected_sha:
            return "completed"
        # File exists but content differs — could be a partial write
        return "ambiguous"
    except OSError:
        return "ambiguous"


register_probe("write_file", _probe_write_file)

# Process-wide state: put back when a harness run ends (foundation/process_state.py).
track_globals(__name__, "_REGISTRY")
