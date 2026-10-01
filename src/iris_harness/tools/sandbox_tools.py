"""Sandbox tool surface exposed to the chat LLM.

A thin wrapper over ``DockerSandbox`` that:
  - lazy-creates per-session workspaces
  - caches one ``DockerSandbox`` per session
  - provides the JSON tool schema the agent loop sends to the LLM
"""

from __future__ import annotations

import logging
import re

from iris_harness.tools.sandbox import (
    DEFAULT_TIMEOUT,
    MAX_TIMEOUT,
    DockerUnavailableError,
    ExecResult,
    SandboxConfig,
    SandboxImageMissingError,
    SessionWorkspace,
    instantiate_runtime,
)
from iris_harness.tools.sandbox.egress import configured_allowlist

logger = logging.getLogger(__name__)

# exp-007 S3b: catastrophic-command guard at the sandbox tool layer (where the
# command string is visible — the governance kernel only sees the chat `code_exec`
# tool's natural-language task, never the Docker-internal `run_shell` cmd). Docker
# already CONTAINS these (ephemeral --rm, cap-drop=ALL, only /workspace mounted),
# so this is defense-in-depth + an audit signal. Patterns are intentionally NARROW
# to never trip legitimate code-exec (e.g. `rm -rf build/` is fine; only root/home/
# device-level destruction and fork-bombs are blocked).
_DESTRUCTIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("no_preserve_root", re.compile(r"--no-preserve-root")),
    ("rm_root", re.compile(r"\brm\s+-[a-zA-Z]*\s*(/|/\*|~|\$HOME)(\s|$|\*|/)")),
    ("fork_bomb", re.compile(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:")),
    ("mkfs", re.compile(r"\bmkfs(\.\w+)?\b")),
    ("wipefs", re.compile(r"\bwipefs\b")),
    ("dd_to_device", re.compile(r"\bdd\b[^\n]*\bof=/dev/")),
    ("redirect_device", re.compile(r">\s*/dev/(sd|nvme|disk|hd)")),
    ("shred_device", re.compile(r"\bshred\b[^\n]*/dev/")),
    ("chmod_root", re.compile(r"\bchmod\s+-?[rR]?\s*777\s+/(\s|$)")),
)


def _destructive_match(cmd: str) -> str | None:
    """Return the name of the first catastrophic pattern matched, else None."""
    normalized = " ".join(cmd.split())
    for name, pattern in _DESTRUCTIVE_PATTERNS:
        if pattern.search(normalized):
            return name
    return None


_RUN_SHELL_SCHEMA = {
    "name": "run_shell",
    "description": (
        "Run a shell command inside a sandboxed Docker container. "
        "The container has Python 3.12, git, curl, jq, pandoc, and common "
        "Python libraries (reportlab, weasyprint, pandas, numpy, matplotlib, "
        "pypdf, pillow, openpyxl, requests, beautifulsoup4, feedparser, gnews) preinstalled. "
        "Workdir is /workspace — files written there persist across calls in "
        "the same session and surface back as artifact paths. Network egress is "
        "restricted to an allowlist (PyPI + GitHub by default); `pip install` works, "
        "but fetches to other hosts are blocked."
    ),
    "args": {
        "cmd": "the shell command to run (passed to bash -lc)",
        "timeout": (f"max seconds to wait (default {DEFAULT_TIMEOUT}, capped at {MAX_TIMEOUT})"),
    },
}


class SandboxToolHost:
    """Per-session sandbox tool host.

    One instance per chat session — owns the workspace and Docker sandbox.
    The agent loop calls ``run_shell`` directly; ``schema()`` returns the
    description the LLM sees in its system prompt.
    """

    def __init__(
        self,
        session_id: str,
        *,
        config: SandboxConfig | None = None,
        runtime_name: str | None = None,
    ) -> None:
        self._session_id = session_id
        self._workspace = SessionWorkspace(session_id)
        cfg = config or SandboxConfig.default()
        # Availability/fallback is resolved once at startup by the caller, which
        # passes the effective ``runtime_name``; default (no arg) → docker, so
        # standalone construction behaves exactly as before.
        self.runtime_name = runtime_name or cfg.resolved_runtime()
        self._sandbox = instantiate_runtime(
            self.runtime_name,
            self._workspace,
            config=cfg,
            egress_allowlist=configured_allowlist(),
        )
        try:
            # Pre-step before tool use: ensure image exists, auto-building if missing.
            self._sandbox.ensure_ready(auto_build_image=True)
        except (DockerUnavailableError, SandboxImageMissingError):
            logger.warning(
                "sandbox preflight failed for session_id=%s",
                session_id,
                exc_info=True,
            )

    @property
    def workspace_path(self) -> str:
        return str(self._workspace.path)

    def run_shell(self, cmd: str, *, timeout: int = DEFAULT_TIMEOUT) -> ExecResult:
        """Execute ``cmd`` in the sandbox; return the structured result."""
        blocked = _destructive_match(cmd)
        if blocked is not None:
            logger.warning(
                "sandbox run_shell blocked catastrophic command (pattern=%s) for session_id=%s",
                blocked,
                self._session_id,
            )
            return ExecResult(
                stdout="",
                stderr=(
                    f"blocked by sandbox command guard: refusing a catastrophic "
                    f"command (matched '{blocked}')."
                ),
                exit_code=126,
                duration_ms=0.0,
            )
        try:
            return self._sandbox.run_shell(cmd, timeout=timeout)
        except DockerUnavailableError as exc:
            return ExecResult(
                stdout="",
                stderr=f"sandbox unavailable: {exc}",
                exit_code=127,
                duration_ms=0.0,
            )
        except SandboxImageMissingError as exc:
            return ExecResult(
                stdout="",
                stderr=str(exc),
                exit_code=127,
                duration_ms=0.0,
            )

    @staticmethod
    def schema() -> dict[str, object]:
        """Return the tool schema dict shown to the LLM."""
        return dict(_RUN_SHELL_SCHEMA)
